"""Refinement initialization from saved clean latents.

Scene Integration re-noises both modalities on independent intervals. Editing
refinement re-noises only the video and keeps the supplied edited audio clean
and fixed.
"""
import hashlib
import json
from pathlib import Path

import torch


def shifted_interval(start_sigma, points, shift):
    """`points` sigmas from `start_sigma` to 0: uniform in H3's unshifted clock, then shifted."""
    if not 0 < start_sigma < 1 or points < 2 or shift <= 0:
        raise ValueError('Invalid refinement interval')
    base_start = start_sigma / (shift - (shift - 1) * start_sigma)
    base = torch.linspace(base_start, 0, points, dtype=torch.float32)
    return shift * base / (1 + (shift - 1) * base)


def independent_sigmas(video_sigma, audio_sigma, points, video_shift, audio_shift):
    return (shifted_interval(video_sigma, points, video_shift),
            shifted_interval(audio_sigma, points, audio_shift))


def initialize_refinement(pipe, scene, video_noise, audio_noise):
    """Validate the source generation and re-noise both saved latent modalities; no VAE round trip."""
    spec = scene['refinement']
    source = Path(spec['source']).expanduser().resolve()
    source_scene = json.loads((source/'scene.json').read_text())
    source_run = json.loads((source/'run.json').read_text())
    if source_run.get('status') != 'complete':
        raise ValueError('Refinement source is incomplete')
    for key in ('width', 'height', 'num_frames', 'num_inference_steps', 'video_prompt'):
        if source_scene[key] != scene[key]:
            raise ValueError(f'Refinement source mismatch: {key}')
    for key in ('id', 'prompt', 'windows', 'volume'):
        if [s.get(key) for s in source_scene['stems']] != [s.get(key) for s in scene['stems']]:
            raise ValueError(f'Refinement source stem mismatch: {key}')
    path = source/'latents.pt'
    clean = torch.load(path, map_location='cpu', weights_only=True)
    for name, noise in (('video', video_noise), ('audio', audio_noise)):
        if clean[name].shape != noise.shape or not torch.isfinite(clean[name]).all():
            raise ValueError(f'Invalid saved {name} latents')
    vs, aus = independent_sigmas(float(spec['video_sigma']), float(spec['audio_sigma']),
                                 scene['num_inference_steps'], pipe.scheduler.shift, pipe.audio_scheduler.shift)
    pipe.scheduler.set_timesteps(sigmas=vs, device=video_noise.device)
    pipe.audio_scheduler.set_timesteps(sigmas=aus, device=audio_noise.device)
    # x_sigma = (1 - sigma) * x_clean + sigma * noise
    video = pipe.scheduler.scale_noise(clean['video'].to(video_noise), pipe.scheduler.timesteps[0], video_noise)
    audio = pipe.audio_scheduler.scale_noise(clean['audio'].to(audio_noise), pipe.audio_scheduler.timesteps[0], audio_noise)
    provenance = dict(source=str(source), source_latents_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      noise_seed=int(spec['noise_seed']), video_sigma=float(vs[0]), audio_sigma=float(aus[0]),
                      video_sigmas=vs.tolist(), audio_sigmas=aus.tolist(), model_evaluations=len(vs)-1)
    return video, audio, provenance


class FixedAudioScheduler:
    """H3 timestep 1 is clean audio. Predictions never update the supplied audio latents."""
    def __init__(self, steps, device):
        self.timesteps = torch.ones(steps, device=device)

    def step(self, prediction, timestep, sample, return_dict=False):
        if return_dict:
            raise ValueError('FixedAudioScheduler returns tuples only')
        return (sample,)


def initialize_fixed_audio(pipe, scene, video_noise, audio_noise):
    """Re-noise the saved source video to `video_sigma`; the edited audio stays clean for every step.

    Returns (video, audio, audio_scheduler, provenance).
    """
    spec = scene['refinement']
    source = Path(spec['source']).expanduser().resolve()
    source_scene = json.loads((source/'scene.json').read_text())
    if json.loads((source/'run.json').read_text()).get('status') != 'complete':
        raise ValueError('Video source is incomplete')
    for key in ('width', 'height', 'num_frames', 'num_inference_steps'):
        if source_scene[key] != scene[key]:
            raise ValueError(f'Edit geometry mismatch: {key}')
    if [s['id'] for s in source_scene['stems']] != [s['id'] for s in scene['stems']]:
        raise ValueError('Edit stem order changed')
    video = torch.load(source/'latents.pt', map_location='cpu', weights_only=True)['video']
    path = Path(spec['fixed_audio'])
    audio = torch.load(path, map_location='cpu', weights_only=True)['audio']
    for name, clean, noise in (('video', video, video_noise), ('audio', audio, audio_noise)):
        if clean.shape != noise.shape or not torch.isfinite(clean).all():
            raise ValueError(f'Invalid fixed-edit {name} latents')
    sigmas = shifted_interval(float(spec['video_sigma']), scene['num_inference_steps'], pipe.scheduler.shift)
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=video_noise.device)
    audio_scheduler = FixedAudioScheduler(len(pipe.scheduler.timesteps), audio_noise.device)
    video = pipe.scheduler.scale_noise(video.to(video_noise), pipe.scheduler.timesteps[0], video_noise)
    provenance = dict(source=str(source),
                      source_latents_sha256=hashlib.sha256((source/'latents.pt').read_bytes()).hexdigest(),
                      fixed_audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      video_sigma=float(spec['video_sigma']), audio_sigma=0., fixed_audio=True,
                      video_sigmas=sigmas.tolist(), audio_timesteps=audio_scheduler.timesteps.tolist(),
                      model_evaluations=len(sigmas)-1, noise_seed=int(spec['noise_seed']))
    return video, audio.to(audio_noise), audio_scheduler, provenance
