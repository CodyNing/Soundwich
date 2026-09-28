"""Pinned model identity, paths, and H3 pipeline loading."""
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_REPO = 'MiniMaxAI/MiniMax-H3'
CHECKPOINT_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
DIFFUSERS_REVISION = '8b3c707ebd3ec4881f4190cf42931da07eaf3b65'


def checkpoint_dir():
    return Path(os.environ.get('SOUNDWICH_H3_CHECKPOINT', BACKEND_ROOT/'checkpoints'/'MiniMax-H3')).expanduser()


def cache_dir():
    return Path(os.environ.get('SOUNDWICH_H3_CACHE', BACKEND_ROOT/'cache')).expanduser()


def scheduler_shifts(checkpoint):
    import json
    return tuple(json.loads((Path(checkpoint)/name/'scheduler_config.json').read_text())['shift']
                 for name in ('scheduler', 'audio_scheduler'))


def setup_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('MiniMax-H3 inference requires a CUDA GPU.')
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total < 79:
        print(f'Warning: GPU 0 has {total:.0f} GiB; a 4-stem 768x448 scene peaks near 73 GiB allocated '
              'with CPU offload, and more stems or larger frames need more (see README). '
              'Expect out-of-memory errors below 80 GiB.', flush=True)
    return torch.device('cuda:0')


def load_pipeline(checkpoint):
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    checkpoint = Path(checkpoint)
    if not (checkpoint/'modular_model_index.json').is_file():
        raise FileNotFoundError(f'MiniMax-H3 checkpoint not found at {checkpoint}; see README (set SOUNDWICH_H3_CHECKPOINT).')
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(checkpoint, workflow='t2va', components_manager=manager, local_files_only=True)
    pipe.load_components(pretrained_model_name_or_path=str(checkpoint), dtype=torch.bfloat16, local_files_only=True)
    manager.enable_auto_cpu_offload(device='cuda:0', memory_reserve_margin='56GB')
    return pipe, manager


def model_identity(pipe, scene):
    """Carrier-bank identity: carriers are only valid for this model, geometry, and hidden width."""
    return dict(checkpoint_revision=CHECKPOINT_REVISION, diffusers_revision=DIFFUSERS_REVISION,
                hidden_size=pipe.transformer.config.hidden_size,
                width=scene['width'], height=scene['height'], num_frames=scene['num_frames'])
