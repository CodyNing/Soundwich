"""Soundwich on MiniMax-H3: carriers -> Stem Formation -> SAM entity masks -> Scene Integration.

    python -m soundwich_h3.generate --scene examples/the_last_button.json

Carrier banks and text embeddings are cached (SOUNDWICH_H3_CACHE, default ./cache).
Outputs: <output-dir>/{stage1,sam,stage2}/ with per-stem WAVs, mix.wav, sample.mp4, latents.pt.
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import argparse
import copy
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import runtime

# Paper settings. Scenes may override these under "stage1" / "stage2".
STAGE1_DEFAULTS = dict(quiet_blend=0.30, outside_attenuation=0.25)
STAGE2_DEFAULTS = dict(video_sigma=0.95, audio_sigma=0.95, quiet_blend=0.30, outside_attenuation=0.25)
QUIET_GROUP = 'quiet'
QUIET_REFINEMENT_SEED_OFFSET = 1000
GEOMETRY_KEYS = ('width', 'height', 'num_frames', 'num_inference_steps')


def load_scene(path, seed=None):
    path = Path(path)
    scene = json.loads(path.read_text())
    for key in ('id', 'seed', 'video_prompt', 'stems', 'carrier_references', *GEOMETRY_KEYS):
        if key not in scene:
            raise ValueError(f'Scene is missing {key!r}')
    if seed is not None:
        scene['seed'] = seed
    ids = [s['id'] for s in scene['stems']]
    if len(set(ids)) != len(ids) or not all('prompt' in s for s in scene['stems']):
        raise ValueError('Stems need unique ids and a prompt')
    duration = scene['num_frames'] / 24
    for stem in scene['stems']:
        for w in stem.get('windows', []):
            if not 0 <= w['start'] < w['end'] <= duration:
                raise ValueError(f'Window of {stem["id"]!r} must lie within the {duration:.3f}s clip')
        if stem.get('windows') and not stem.get('carrier_group'):
            raise ValueError(f'Controlled stem {stem["id"]!r} needs a carrier_group')
    references = json.loads((path.parent/scene['carrier_references']).read_text())
    groups = sorted({s['carrier_group'] for s in scene['stems'] if s.get('carrier_group')} | {QUIET_GROUP})
    for group in groups:
        ref = references.get(group)
        if ref is None or not {'role', 'prompt', 'seed', 'quantile'} <= ref.keys():
            raise ValueError(f'Carrier reference {group!r} is missing or incomplete')
        if ref['role'] != ('suppression' if group == QUIET_GROUP else 'activation'):
            raise ValueError(f'Carrier reference {group!r} has the wrong role')
    return scene, {g: references[g] for g in groups}


def settings(scene):
    for name, defaults in (('stage1', STAGE1_DEFAULTS), ('stage2', STAGE2_DEFAULTS)):
        unknown = set(scene.get(name, {})) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown {name} settings: {sorted(unknown)}')
    stage1 = {**STAGE1_DEFAULTS, **scene.get('stage1', {})}
    stage2 = {**STAGE2_DEFAULTS, **scene.get('stage2', {})}
    for value in (*stage1.values(), stage2['quiet_blend'], stage2['outside_attenuation']):
        if not 0 <= value <= 1:
            raise ValueError('Blend/attenuation settings must lie in [0, 1]')
    return stage1, stage2


def blend(scene, stage):
    """Carrier replay settings; the boundary feather is 2% of the clip duration."""
    from .replay import BlendConfig
    values = settings(scene)[0 if stage == 1 else 1]
    return BlendConfig(feather_seconds=0.02*scene['num_frames']/24, outside_silence_blend=values['quiet_blend'],
                       outside_suppression=values['outside_attenuation'])


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def carrier_root(group, reference, scene):
    ident = dict(reference=reference, checkpoint_revision=runtime.CHECKPOINT_REVISION,
                 diffusers_revision=runtime.DIFFUSERS_REVISION, **{k: scene[k] for k in GEOMETRY_KEYS})
    return runtime.cache_dir()/'carriers'/f'{group}-{_key(ident)}'


def reference_scene(group, reference, scene):
    """Native single-row reference generation used to capture a carrier."""
    return dict(id=f'carrier_{group}', seed=int(reference['seed']), **{k: scene[k] for k in GEOMETRY_KEYS},
                video_prompt=reference['prompt'],
                stems=[dict(id=group, prompt=reference['prompt'], windows=[], volume=1.0)])


def quiet_refinement(reference, stage2):
    return dict(video_sigma=stage2['video_sigma'], audio_sigma=stage2['audio_sigma'],
                noise_seed=int(reference['seed']) + QUIET_REFINEMENT_SEED_OFFSET)


def plan(scene, references, output, stage1_only):
    """Everything that determines the generated result, for --dry-run and run records."""
    from .masks import mask_policy
    _, stage2 = settings(scene)
    result = dict(
        scene=scene['id'], seed=scene['seed'], output=str(output), checkpoint=str(runtime.checkpoint_dir()),
        cache=str(runtime.cache_dir()), references=references,
        carriers={g: str(carrier_root(g, r, scene)) for g, r in references.items()},
        stage1=dict(rows=1+len(scene['stems']), noise_seed=scene['seed'], num_inference_steps=scene['num_inference_steps'],
                    activation_groups=[s.get('carrier_group') for s in scene['stems']],
                    blend=asdict(blend(scene, 1)), initial_boost=dict(strength=0.50, until_progress=0.10)))
    if not stage1_only:
        result['stage2'] = dict(rows=1+len(scene['stems']), noise_seed=scene['seed']+1,
            video_sigma=stage2['video_sigma'], audio_sigma=stage2['audio_sigma'],
            model_evaluations=scene['num_inference_steps']-1, activation_injection=False,
            quiet={k: v for k, v in asdict(blend(scene, 2)).items()
                   if k in ('outside_suppression', 'outside_silence_blend', 'feather_seconds')},
            quiet_reference_refinement=quiet_refinement(references[QUIET_GROUP], stage2),
            sam_policy=mask_policy(scene))
        shifts_available = (runtime.checkpoint_dir()/'scheduler'/'scheduler_config.json').is_file()
        if shifts_available:
            from .refinement import independent_sigmas
            vs, aus = independent_sigmas(stage2['video_sigma'], stage2['audio_sigma'], scene['num_inference_steps'],
                                         *runtime.scheduler_shifts(runtime.checkpoint_dir()))
            result['stage2'].update(video_sigmas=vs.tolist(), audio_sigmas=aus.tolist())
    return result


class Pipeline:
    """Lazily loaded H3 pipeline that can be released while SAM runs."""
    def __init__(self):
        self.pipe = self.manager = None

    def get(self):
        if self.pipe is None:
            runtime.setup_gpu()
            self.pipe, self.manager = runtime.load_pipeline(runtime.checkpoint_dir())
        return self.pipe

    def release(self):
        if self.pipe is not None:
            import torch
            self.pipe = self.manager = None
            gc.collect()
            torch.cuda.empty_cache()


def completed(directory, scene):
    directory = Path(directory)
    if not directory.exists():
        return False
    run = directory/'run.json'
    if run.is_file() and json.loads(run.read_text()).get('status') == 'complete':
        if json.loads((directory/'scene.json').read_text()) != scene:
            raise ValueError(f'{directory} was generated from a different scene; choose another --output-dir')
        return True
    raise FileExistsError(f'Incomplete output; inspect and remove: {directory}')


def stage1_bank(pipeline, group, reference, scene):
    from .carriers import CarrierBank
    from .replay import CaptureCarrier
    from .sampling import generate
    root = carrier_root(group, reference, scene)
    if (root/'bank'/'carrier.json').is_file():
        bank = CarrierBank.load(root/'bank')
        if bank.metadata.get('reference') != reference:
            raise ValueError(f'Cached carrier {root} does not match its reference; remove it')
        print(f'Using cached carrier: {group}', flush=True)
        return bank
    if root.exists():
        raise FileExistsError(f'Incomplete carrier cache; remove: {root}')
    pipe = pipeline.get()
    capture = CaptureCarrier(reference['role'], group,
                             {**runtime.model_identity(pipe, scene), 'reference': reference}, reference['quantile'])
    print(f'Capturing carrier: {group}', flush=True)
    generate(pipe, reference_scene(group, reference, scene), root/'reference', text_cache=runtime.cache_dir()/'text',
             audio_hook=capture, native_capture=True, decode_video=False)
    capture.bank.save(root/'bank')
    return capture.bank


def stage2_quiet_bank(pipeline, reference, scene, stage2):
    """Quiet carrier captured on the Scene Integration schedule by refining the quiet reference itself."""
    from .carriers import CarrierBank
    from .replay import CaptureCarrier
    from .sampling import generate
    spec = quiet_refinement(reference, stage2)
    root = carrier_root(QUIET_GROUP, reference, scene)
    bank_dir = root/f'refinement-{_key(spec)}'/'bank'
    if (bank_dir/'carrier.json').is_file():
        bank = CarrierBank.load(bank_dir)
        if bank.metadata.get('reference') != reference or bank.metadata.get('refinement') != spec:
            raise ValueError(f'Cached quiet refinement carrier {bank_dir} does not match; remove it')
        print('Using cached Scene Integration quiet carrier', flush=True)
        return bank
    if bank_dir.parent.exists():
        raise FileExistsError(f'Incomplete carrier cache; remove: {bank_dir.parent}')
    source = root/'reference'
    ref_scene = json.loads((source/'scene.json').read_text())
    ref_scene['refinement'] = dict(source=str(source.resolve()), **spec)
    pipe = pipeline.get()
    capture = CaptureCarrier(reference['role'], QUIET_GROUP,
        {**runtime.model_identity(pipe, scene), 'reference': reference, 'refinement': spec}, reference['quantile'])
    print('Capturing Scene Integration quiet carrier', flush=True)
    generate(pipe, ref_scene, bank_dir.parent/'reference', text_cache=runtime.cache_dir()/'text',
             audio_hook=capture, native_capture=True, decode_video=False)
    capture.bank.save(bank_dir)
    return capture.bank


def run_stage1(pipeline, scene, references, output):
    from .replay import ReplayCarriers
    from .sampling import generate
    banks = {g: stage1_bank(pipeline, g, r, scene) for g, r in references.items()}
    if completed(output, scene):
        print(f'Using completed Stem Formation: {output}', flush=True)
        return
    pipe = pipeline.get()
    activation = [banks[s['carrier_group']] if s.get('carrier_group') else None for s in scene['stems']]
    replay = ReplayCarriers(scene, activation, banks[QUIET_GROUP], runtime.model_identity(pipe, scene),
                            blend(scene, 1))
    print('Stem Formation', flush=True)
    generate(pipe, scene, output, text_cache=runtime.cache_dir()/'text', audio_hook=replay)


def run_stage2(pipeline, scene, references, output):
    from .masks import mask_policy
    from .replay import QuietReplay
    from .sam import prepare_masks
    from .sampling import generate
    _, stage2 = settings(scene)
    source = output/'stage1'
    scene2 = copy.deepcopy(scene)
    scene2['refinement'] = dict(source=str(source.resolve()), video_sigma=stage2['video_sigma'],
                                audio_sigma=stage2['audio_sigma'], noise_seed=int(scene['seed'])+1)
    if completed(output/'stage2', scene2):
        print(f'Using completed Scene Integration: {output/"stage2"}', flush=True)
        return
    quiet = stage2_quiet_bank(pipeline, references[QUIET_GROUP], scene, stage2)
    if not (output/'sam'/'mask_validation.json').is_file():
        pipeline.release()
    masks, _ = prepare_masks(scene, source, output/'sam', mask_policy(scene))
    pipe = pipeline.get()
    replay = QuietReplay(scene2, quiet, runtime.model_identity(pipe, scene2), blend(scene, 2))
    print('Scene Integration with entity routing', flush=True)
    generate(pipe, scene2, output/'stage2', text_cache=runtime.cache_dir()/'text', audio_hook=replay,
             entity_masks=masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='default: outputs/<scene id>-seed<seed>')
    parser.add_argument('--seed', type=int, help='override the scene seed (Scene Integration uses seed+1)')
    parser.add_argument('--stage1-only', action='store_true', help='stop after Stem Formation')
    parser.add_argument('--dry-run', action='store_true', help='validate the scene and print the resolved plan; no GPU')
    args = parser.parse_args()
    scene, references = load_scene(args.scene, args.seed)
    output = (args.output_dir or Path('outputs')/f'{scene["id"]}-seed{scene["seed"]}').resolve()
    resolved = plan(scene, references, output, args.stage1_only)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    (output/'plan.json').write_text(json.dumps(resolved, indent=2)+'\n')
    pipeline = Pipeline()
    run_stage1(pipeline, scene, references, output/'stage1')
    if not args.stage1_only:
        run_stage2(pipeline, scene, references, output)
    final = output/('stage1' if args.stage1_only else 'stage2')
    print(f'OUTPUT={final}', flush=True)


if __name__ == '__main__':
    main()
