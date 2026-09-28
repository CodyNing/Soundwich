"""Edit stems of a completed generation, then refine its video on the edited audio.

    python -m soundwich_h3.edit --run outputs/enchanted_archive-seed74 \
        --edit examples/edits/archive_move_meow.json

Retiming moves one window's clip inside a stem's saved Scene Integration audio
latents; the tokens in between shift to keep the clip length. Replacement runs
Stem Formation for that one stem with a new source prompt and seed and swaps in
its audio latents. A kept stem may get a new source prompt for the refinement;
its audio is unchanged. The edited audio is then held clean and fixed while the
saved Stem Formation video is re-noised to the Scene Integration video sigma and
refined with the original SAM entity masks, the revised windows, and quiet
suppression (no activation carriers).
Outputs: <run>/edits/<edit name>/{edit.json, fixed_audio.pt, takes/<stem>/, edited/}.
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import runtime
from .batch import AUDIO_TOKENS_PER_SECOND
from .generate import (QUIET_GROUP, Pipeline, _key, blend, carrier_root, check_stems, completed, run_stage1,
                       settings, stage1_bank)

EDIT_KEYS = {'stem', 'prompt', 'replace', 'retime'}
TAKE_KEYS = {'prompt', 'seed', 'windows'}


def move_stereo_clip(audio, row, start, end, target):
    """Move a half-open token interval of one stem to start at `target`, in both stereo channels.

    Tokens between the old and new position shift to fill the gap; the clip length is unchanged.
    """
    import torch
    if audio.ndim != 3 or audio.shape[1] % 2 or not 0 <= row < len(audio):
        raise ValueError('Expected [stem, 2*time, channel] channel-major stereo')
    frames = audio.shape[1] // 2
    if not (0 <= start < end <= frames and 0 <= target <= frames-(end-start)):
        raise ValueError('Clip outside audio timeline')
    indices = list(range(frames))
    clip = indices[start:end]
    del indices[start:end]
    indices[target:target] = clip
    order = torch.tensor(indices + [i+frames for i in indices], device=audio.device)
    result = audio.clone()
    result[row] = audio[row].index_select(0, order)
    return result


def replace_stem(audio, row, take):
    """Swap one stem's audio latents for a new take of the same geometry."""
    import torch
    if not 0 <= row < len(audio) or take.shape != audio[row:row+1].shape:
        raise ValueError('Replacement must be one matching native audio stem')
    if not torch.isfinite(take).all():
        raise ValueError('Nonfinite replacement audio')
    result = audio.clone()
    result[row:row+1] = take.to(audio)
    return result


def _bin(seconds):
    return round(float(seconds) * AUDIO_TOKENS_PER_SECOND)


def resolve(scene, spec):
    """Edited scene and the ordered audio operations; per stem, replacement precedes retiming."""
    edited = copy.deepcopy(scene)
    for old, new in spec.get('video_prompt_substitutions', []):
        if old not in edited['video_prompt']:
            raise ValueError(f'Video prompt does not contain {old!r}')
        edited['video_prompt'] = edited['video_prompt'].replace(old, new)
    rows = {s['id']: i for i, s in enumerate(edited['stems'])}
    frames = _bin(scene['num_frames'] / 24)
    operations, seen = [], set()
    for item in spec.get('edits', []):
        unknown = set(item) - EDIT_KEYS
        if unknown or item.get('stem') not in rows or item['stem'] in seen:
            raise ValueError(f'Invalid edit (unknown keys {sorted(unknown)}, unknown or repeated stem): {item}')
        seen.add(item['stem'])
        row = rows[item['stem']]
        stem = edited['stems'][row]
        if 'prompt' in item:
            if 'replace' in item:
                raise ValueError(f'A replaced stem takes its prompt from replace: {item["stem"]}')
            # Conditions the kept audio of this stem on a new source prompt during refinement.
            stem['prompt'] = item['prompt']
        if 'replace' in item:
            take = item['replace']
            if not {'prompt', 'seed'} <= take.keys() or set(take) - TAKE_KEYS:
                raise ValueError(f'replace needs prompt and seed (optional windows): {item["stem"]}')
            stem['prompt'] = take['prompt']
            if 'windows' in take:
                stem['windows'] = copy.deepcopy(take['windows'])
            check_stems(edited)
            take_scene = copy.deepcopy(edited)
            take_scene['seed'] = int(take['seed'])
            take_scene['stems'] = [copy.deepcopy(stem)]
            operations.append(dict(op='replace', stem=stem['id'], row=row, take=take_scene))
        for move in item.get('retime', []):
            windows = stem.get('windows', [])
            index = move.get('window', 0 if len(windows) == 1 else None)
            if index is None or not 0 <= index < len(windows) or set(move) - {'window', 'start'}:
                raise ValueError(f'retime needs a window index and a new start: {item["stem"]}')
            window = windows[index]
            start, end, target = _bin(window['start']), _bin(window['end']), _bin(move['start'])
            if not 0 <= target <= frames - (end - start):
                raise ValueError(f'Retimed window of {stem["id"]!r} leaves the clip')
            low, high = min(start, target), max(end, target + end - start)
            for other, w in enumerate(windows):
                if other != index and _bin(w['start']) < high and _bin(w['end']) > low:
                    raise ValueError(f'Retiming {stem["id"]!r} window {index} would shift its window {other}')
            old = copy.deepcopy(window)
            window.update(start=target / AUDIO_TOKENS_PER_SECOND, end=(target + end - start) / AUDIO_TOKENS_PER_SECOND)
            operations.append(dict(op='retime', stem=stem['id'], row=row, start=start, end=end, target=target,
                                   old_window=old, new_window=copy.deepcopy(window)))
    if not operations:
        raise ValueError('The edit file has no edits')
    check_stems(edited)
    return edited, operations


def fixed_audio_refinement(scene, run):
    """Editing refinement: Scene Integration video sigma and noise seed, clean fixed audio."""
    _, stage2 = settings(scene)
    return dict(source=str((run/'stage1').resolve()), video_sigma=stage2['video_sigma'], audio_sigma=0.,
                noise_seed=int(scene['seed'])+1)


def edit_quiet_spec(reference, stage2):
    return dict(video_sigma=stage2['video_sigma'], audio_sigma=0., fixed_audio=True,
                noise_seed=int(reference['seed'])+1)


def edit_quiet_bank(pipeline, reference, scene):
    """Quiet carrier on the editing schedule: the quiet reference refined with its own audio held fixed."""
    from .carriers import CarrierBank
    from .replay import CaptureCarrier
    from .sampling import generate
    spec = edit_quiet_spec(reference, settings(scene)[1])
    root = carrier_root(QUIET_GROUP, reference, scene)
    bank_dir = root/f'edit-refinement-{_key(spec)}'/'bank'
    if (bank_dir/'carrier.json').is_file():
        bank = CarrierBank.load(bank_dir)
        if bank.metadata.get('reference') != reference or bank.metadata.get('refinement') != spec:
            raise ValueError(f'Cached editing quiet carrier {bank_dir} does not match; remove it')
        print('Using cached editing quiet carrier', flush=True)
        return bank
    if bank_dir.parent.exists():
        raise FileExistsError(f'Incomplete carrier cache; remove: {bank_dir.parent}')
    stage1_bank(pipeline, QUIET_GROUP, reference, scene)
    source = root/'reference'
    ref_scene = json.loads((source/'scene.json').read_text())
    ref_scene['refinement'] = dict(source=str(source.resolve()), fixed_audio=str((source/'latents.pt').resolve()),
                                   video_sigma=spec['video_sigma'], audio_sigma=0., noise_seed=spec['noise_seed'])
    pipe = pipeline.get()
    capture = CaptureCarrier(reference['role'], QUIET_GROUP,
        {**runtime.model_identity(pipe, scene), 'reference': reference, 'refinement': spec}, reference['quantile'])
    print('Capturing editing quiet carrier', flush=True)
    destination = bank_dir.parent/'reference'
    generate(pipe, ref_scene, destination, text_cache=runtime.cache_dir()/'text',
             audio_hook=capture, native_capture=True, decode_video=False)
    _check_fixed(destination, source/'latents.pt')
    capture.bank.save(bank_dir)
    return capture.bank


def _load_audio(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)['audio']


def _check_fixed(output, fixed):
    import torch
    if not torch.equal(_load_audio(Path(output)/'latents.pt'), _load_audio(fixed)):
        raise AssertionError('Fixed audio changed during refinement')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(run):
    run = Path(run).resolve()
    scene_path, plan_path = run/'stage1'/'scene.json', run/'plan.json'
    if not scene_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(f'{run} is not a soundwich_h3.generate output directory')
    scene, plan = json.loads(scene_path.read_text()), json.loads(plan_path.read_text())
    if (plan['scene'], plan['seed']) != (scene['id'], scene['seed']):
        raise ValueError(f'{plan_path} does not belong to {scene_path}')
    return run, scene, plan['references']


def check_sources(run, scene):
    """The base audio must be Scene Integration of this exact Stem Formation output, with its masks."""
    for directory in ('stage1', 'stage2'):
        record = run/directory/'run.json'
        if not record.is_file() or json.loads(record.read_text()).get('status') != 'complete':
            raise ValueError(f'Editing needs a complete {directory}/ in {run}; run soundwich_h3.generate first')
    stage2 = json.loads((run/'stage2'/'run.json').read_text())
    if stage2['refinement']['source_latents_sha256'] != _sha256(run/'stage1'/'latents.pt'):
        raise ValueError('stage2/ belongs to a different Stem Formation output')
    if [s['id'] for s in json.loads((run/'stage2'/'scene.json').read_text())['stems']] != \
            [s['id'] for s in scene['stems']]:
        raise ValueError('stage2/ stem order differs from stage1/')
    if not (run/'sam'/'mask_validation.json').is_file():
        raise ValueError(f'Validated SAM masks are missing: {run/"sam"}')


def plan(run, scene, references, spec, edited, operations, output):
    """Everything that determines the edit, for --dry-run and edit.json."""
    _, stage2 = settings(scene)
    refinement = fixed_audio_refinement(scene, run)
    result = dict(run=str(run), output=str(output), spec=spec,
        operations=[{k: v for k, v in op.items() if k != 'take'} for op in operations],
        takes={op['stem']: dict(seed=op['take']['seed'], prompt=op['take']['stems'][0]['prompt'],
                                windows=op['take']['stems'][0].get('windows', []),
                                carrier_group=op['take']['stems'][0].get('carrier_group'),
                                stage1_blend=asdict(blend(op['take'], 1)))
               for op in operations if op['op'] == 'replace'},
        windows={s['id']: s.get('windows', []) for s in edited['stems']},
        video_prompt_changed=edited['video_prompt'] != scene['video_prompt'],
        stem_prompts_changed=[s['id'] for s, old in zip(edited['stems'], scene['stems']) if s['prompt'] != old['prompt']],
        refinement=dict(video_source=refinement['source'], audio_source=str(run/'stage2'), masks=str(run/'sam'),
                        video_sigma=refinement['video_sigma'], audio_sigma=0., fixed_audio=True,
                        noise_seed=refinement['noise_seed'], model_evaluations=scene['num_inference_steps']-1,
                        activation_injection=False, entity_routing=True,
                        quiet={k: v for k, v in asdict(blend(scene, 2)).items()
                               if k in ('outside_suppression', 'outside_silence_blend', 'feather_seconds')},
                        quiet_reference_refinement=edit_quiet_spec(references[QUIET_GROUP], stage2)))
    if (runtime.checkpoint_dir()/'scheduler'/'scheduler_config.json').is_file():
        from .refinement import shifted_interval
        sigmas = shifted_interval(refinement['video_sigma'], scene['num_inference_steps'],
                                  runtime.scheduler_shifts(runtime.checkpoint_dir())[0])
        result['refinement']['video_sigmas'] = sigmas.tolist()
    return result


def run_edit(run, scene, references, edited, operations, output):
    import torch
    from .masks import mask_policy
    from .replay import QuietReplay
    from .sam import prepare_masks
    from .sampling import generate
    pipeline = Pipeline()
    audio = _load_audio(run/'stage2'/'latents.pt')
    for op in operations:
        if op['op'] == 'replace':
            take_dir = output/'takes'/op['stem']
            groups = {op['take']['stems'][0].get('carrier_group'), QUIET_GROUP} - {None}
            print(f'Stem Formation take for {op["stem"]}', flush=True)
            run_stage1(pipeline, op['take'], {g: references[g] for g in sorted(groups)}, take_dir)
            audio = replace_stem(audio, op['row'], _load_audio(take_dir/'latents.pt'))
        else:
            audio = move_stereo_clip(audio, op['row'], op['start'], op['end'], op['target'])
    fixed = output/'fixed_audio.pt'
    if fixed.is_file():
        if not torch.equal(_load_audio(fixed), audio):
            raise ValueError(f'{fixed} differs from the recomputed edit; remove {output}')
    else:
        torch.save(dict(audio=audio), fixed)
    edited = copy.deepcopy(edited)
    edited['refinement'] = dict(fixed_audio_refinement(scene, run), fixed_audio=str(fixed.resolve()))
    if completed(output/'edited', edited):
        print(f'Using completed edit: {output/"edited"}', flush=True)
        return
    quiet = edit_quiet_bank(pipeline, references[QUIET_GROUP], scene)
    masks, _ = prepare_masks(scene, run/'stage1', run/'sam', mask_policy(scene))
    pipe = pipeline.get()
    replay = QuietReplay(edited, quiet, runtime.model_identity(pipe, edited), blend(scene, 2))
    print('Editing refinement with fixed audio and entity routing', flush=True)
    generate(pipe, edited, output/'edited', text_cache=runtime.cache_dir()/'text', audio_hook=replay,
             entity_masks=masks)
    _check_fixed(output/'edited', fixed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='output directory of soundwich_h3.generate')
    parser.add_argument('--edit', type=Path, required=True, help='edit specification (JSON)')
    parser.add_argument('--output-dir', type=Path, help='default: <run>/edits/<edit file name>')
    parser.add_argument('--dry-run', action='store_true', help='validate the edit and print the resolved plan; no GPU')
    args = parser.parse_args()
    run, scene, references = load_run(args.run)
    spec = json.loads(args.edit.read_text())
    edited, operations = resolve(scene, spec)
    output = (args.output_dir or run/'edits'/args.edit.stem).resolve()
    resolved = plan(run, scene, references, spec, edited, operations, output)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    check_sources(run, scene)
    output.mkdir(parents=True, exist_ok=True)
    record = output/'edit.json'
    if record.is_file() and json.loads(record.read_text()) != resolved:
        raise ValueError(f'{output} holds a different edit; choose another --output-dir')
    record.write_text(json.dumps(resolved, indent=2)+'\n')
    run_edit(run, scene, references, edited, operations, output)
    print(f'OUTPUT={output/"edited"}', flush=True)


if __name__ == '__main__':
    main()
