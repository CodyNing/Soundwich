"""Entity masks from the Stem Formation video via the shared SAM3.1 video backend (separate environment).

Environment:
  SOUNDWICH_SAM3_PYTHON      Python executable of the SAM3 environment (required).
  SOUNDWICH_SAM3_REPO        facebookresearch/sam3 checkout, if not installed in that environment.
  SOUNDWICH_SAM3_CHECKPOINT  local SAM3.1 multiplex checkpoint (otherwise the gated HF download is used).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from .masks import backend_prompt_spec, load_entity_masks, sha256, token_grid

BACKEND = Path(__file__).resolve().parents[2]/'common'/'sam3_video_backend.py'


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sam_environment():
    python = os.environ.get('SOUNDWICH_SAM3_PYTHON')
    if not python or not Path(python).is_file():
        raise RuntimeError('Set SOUNDWICH_SAM3_PYTHON to the python executable of your SAM3 environment '
                           '(see README, "SAM3 environment").')
    if not BACKEND.is_file():
        raise FileNotFoundError(f'SAM3 backend script not found: {BACKEND}')
    return python


def extract_frames(video, destination, count):
    import av
    destination.mkdir(parents=True)
    with av.open(str(video)) as container:
        frames = list(container.decode(video=0))
    if len(frames) != count:
        raise ValueError(f'Source video has {len(frames)} frames, expected {count}: {video}')
    for i, frame in enumerate(frames):
        frame.to_image().save(destination/f'frame_{i:05d}.jpg', quality=95)


def _source_identity(source, mask_root):
    return dict(source_latents_sha256=sha256(source/'latents.pt'),
                source_video_sha256=sha256(source/'sample.mp4'),
                masks_sha256=sha256(mask_root/'sam_masks.json'))


def _validated(scene, source, mask_root, policy):
    if json.loads((mask_root/'mask_policy.json').read_text()) != policy:
        raise ValueError(f'SAM prompt policy changed since masks were made; remove {mask_root}')
    record = json.loads((mask_root/'mask_validation.json').read_text())
    if {k: record.get(k) for k in ('source_latents_sha256', 'source_video_sha256', 'masks_sha256')} != \
            _source_identity(source, mask_root):
        raise ValueError(f'SAM masks do not belong to this Stem Formation output; remove {mask_root}')
    masks, metadata = load_entity_masks(scene, mask_root, policy)
    if metadata != record['mask_metadata']:
        raise ValueError(f'SAM masks changed after validation; remove {mask_root}')
    return masks, metadata


def prepare_masks(scene, source, mask_root, policy):
    """Return cached validated masks, or run SAM3.1 on `source/sample.mp4` and validate them."""
    source, mask_root = Path(source), Path(mask_root)
    if (mask_root/'mask_validation.json').is_file():
        print(f'Using cached SAM masks: {mask_root}', flush=True)
        return _validated(scene, source, mask_root, policy)
    if mask_root.exists():
        raise FileExistsError(f'Incomplete SAM output; inspect and remove: {mask_root}')
    python = sam_environment()
    work = mask_root.with_name(mask_root.name + '.work')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    _write_json(work/'mask_policy.json', policy)
    _write_json(work/'prompts.json', backend_prompt_spec(policy))
    extract_frames(source/'sample.mp4', work/'frames', scene['num_frames'])
    command = [python, str(BACKEND), '--frames-dir', str(work/'frames'),
               '--prompts-json', str(work/'prompts.json'), '--output', str(work/'sam_masks.json'),
               '--token-shape', ','.join(map(str, token_grid(scene))),
               '--frame-count', str(scene['num_frames']),
               '--frame-indices', ','.join(map(str, range(scene['num_frames']))),
               '--predictor-kind', 'sam3.1_multiplex', '--device', 'cuda', '--offload-state-to-cpu']
    if os.environ.get('SOUNDWICH_SAM3_REPO'):
        command += ['--sam3-repo', os.environ['SOUNDWICH_SAM3_REPO']]
    if os.environ.get('SOUNDWICH_SAM3_CHECKPOINT'):
        command += ['--checkpoint-path', os.environ['SOUNDWICH_SAM3_CHECKPOINT']]
    env = dict(os.environ, PYTORCH_ALLOC_CONF='expandable_segments:True')
    print('Running SAM3.1 on the Stem Formation video', flush=True)
    subprocess.run(command, check=True, env=env)
    masks, metadata = load_entity_masks(scene, work, policy)
    _write_json(work/'mask_validation.json', dict(scene_id=scene['id'], source=str(source.resolve()),
                                                   mask_metadata=metadata, **_source_identity(source, work)))
    shutil.rmtree(work/'frames')
    work.rename(mask_root)
    print(f'SAM masks and overlays: {mask_root} (inspect overlays/ before relying on routing)', flush=True)
    return masks, metadata
