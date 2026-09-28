"""One H3 generation: N+1 batch expansion (or a native single-row carrier reference)."""
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from diffusers.modular_pipelines.minimax_h3.decoders import (
    MiniMaxH3AfterDenoiseStep, MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep)
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames, audio_latent_num_frames, video_latent_num_frames)
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.utils.export_utils import encode_video

from .batch import BatchExpansion, BatchLayout, NativeCapture, pad_contexts
from .refinement import initialize_refinement

FPS = 24


def encode_prompts(pipe, prompts, text_cache, device):
    text_cache.mkdir(parents=True, exist_ok=True)
    contexts = []
    for i, prompt in enumerate(prompts):
        path = text_cache/f'{hashlib.sha256(prompt.encode()).hexdigest()}.pt'
        if path.exists():
            context = torch.load(path, map_location=device, weights_only=True)
        else:
            ids = pipe.tokenizer(prompt, add_special_tokens=False)['input_ids']
            context = get_qwen3vl_prompt_embeds(pipe.text_encoder, pipe.processor, ids, {},
                text_encoder_layer=pipe.text_encoder_layer, device=device, dtype=pipe.text_encoder.dtype)
            torch.save(context.cpu(), path)
        contexts.append(context)
        print(f'Encoded context {i+1}/{len(prompts)}: {context.shape[1]} tokens', flush=True)
    return contexts


@torch.inference_mode()
def generate(pipe, scene, destination, *, text_cache, audio_hook=None, native_capture=False,
             entity_masks=None, decode_video=True, device='cuda:0'):
    """Denoise, save latents, decode each stem, the mix, and the shared video.

    Batch mode: row 0 carries the video prompt, rows 1..N the stem prompts.
    `native_capture`: one unmodified row with the single stem prompt (carrier references).
    `scene['refinement']`: start from a saved generation re-noised to the given sigmas.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'scene.json').write_text(json.dumps(scene, indent=2)+'\n')
    started = time.monotonic()
    device = torch.device(device)
    if native_capture and (len(scene['stems']) != 1 or audio_hook is None or entity_masks is not None):
        raise ValueError('Native capture requires one reference stem, a capture hook, and no routing')
    prompts = ([scene['stems'][0]['prompt']] if native_capture else
               [scene['video_prompt']] + [s['prompt'] for s in scene['stems']])
    text, valid = pad_contexts(encode_prompts(pipe, prompts, Path(text_cache), device))
    frames = align_num_frames(scene['num_frames'], pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
    nf = video_latent_num_frames(frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
    nh, nw = scene['height']//pipe.vae_spatial_compression_ratio, scene['width']//pipe.vae_spatial_compression_ratio
    na = audio_latent_num_frames(frames)
    layout = BatchLayout.create(valid, latent_frames=nf, latent_height=nh, latent_width=nw, audio_frames=na,
                                patch_size=pipe.patch_size)
    generator = torch.Generator('cpu').manual_seed(scene.get('refinement', {}).get('noise_seed', scene['seed']))
    initial = torch.randn(1, pipe.vae_latent_channels, nf, nh, nw, generator=generator).to(device)
    video = patchify_video_latents(initial, pipe.patch_size)[None]
    audio = torch.randn(len(scene['stems']), 2*na, pipe.audio_latent_channels, generator=generator).to(device)
    pipe.scheduler.set_timesteps(scene['num_inference_steps'], device=device)
    pipe.audio_scheduler.set_timesteps(scene['num_inference_steps'], device=device)
    refinement = None
    if scene.get('refinement'):
        video, audio, refinement = initialize_refinement(pipe, scene, video, audio)
    schedule = {'video': pipe.scheduler.timesteps.cpu().tolist(), 'audio': pipe.audio_scheduler.timesteps.cpu().tolist()}
    if audio_hook is not None:
        audio_hook.validate(schedule, na, len(pipe.transformer.transformer_blocks))
    if entity_masks is not None:
        entity_masks = entity_masks.to(device)
    expansion = (NativeCapture(pipe.transformer, layout, audio_hook) if native_capture else
                 BatchExpansion(pipe.transformer, layout, scene['stems'], audio_hook=audio_hook,
                                entity_masks=entity_masks))
    trace = []
    try:
        for step, (vt, at) in enumerate(zip(pipe.scheduler.timesteps, pipe.audio_scheduler.timesteps)):
            tick = time.monotonic()
            vp, ap = expansion(video, audio, text, vt, at, step)
            video = pipe.scheduler.step(vp.float(), vt, video, return_dict=False)[0]
            audio = pipe.audio_scheduler.step(ap.float(), at, audio, return_dict=False)[0]
            trace.append({'step': step, 'seconds': time.monotonic()-tick})
            print(f'Denoise {step+1}/{len(pipe.scheduler.timesteps)} {trace[-1]["seconds"]:.1f}s', flush=True)
    finally:
        expansion.close()
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise RuntimeError('Nonfinite latent.')
    torch.save({'video': video.cpu(), 'audio': audio.cpu(), 'schedule': schedule}, destination/'latents.pt')
    geometries = dict(num_latent_frames=nf, latent_height=nh, latent_width=nw, num_audio_latents=na)
    waveforms = []
    video_frames = None
    for i, stem in enumerate(scene['stems']):
        state = PipelineState(values=dict(latents=video[0], audio_latents=audio[i], output_type='pil', **geometries))
        _, state = MiniMaxH3AfterDenoiseStep()(pipe, state)
        _, state = MiniMaxH3AudioDecodeStep()(pipe, state)
        wave = state.get('audio')[0].float().cpu()
        sample_rate = state.get('sampling_rate')
        sf.write(destination/f'{stem["id"]}.wav', wave.T.numpy(), sample_rate, subtype='FLOAT')
        waveforms.append(wave)
        if i == 0 and decode_video:
            _, state = MiniMaxH3VideoDecodeStep()(pipe, state)
            video_frames = state.get('videos')[0]
        print('Decoded '+stem['id'], flush=True)
    mixed = sum(w*stem.get('volume', 1.) for w, stem in zip(waveforms, scene['stems']))
    peak = mixed.abs().max().item()
    gain = min(1., .98/max(peak, 1e-8))
    mixed *= gain
    sf.write(destination/'mix.wav', mixed.T.numpy(), sample_rate, subtype='FLOAT')
    if video_frames is not None:
        encode_video(video_frames, fps=FPS, output_path=str(destination/'sample.mp4'), audio=mixed,
                     audio_sample_rate=sample_rate)
    report = dict(status='complete',
        mode='native single-row carrier reference' if native_capture else
             'N+1 batch expansion' + (' with entity routing' if entity_masks is not None else ''),
        stems=len(scene['stems']), frames=frames, fps=FPS, duration=frames/FPS,
        refinement=refinement, schedule=schedule, trace=trace, mix_gain=gain,
        elapsed_seconds=time.monotonic()-started,
        peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None,
        carriers=audio_hook.summary() if audio_hook is not None else None)
    (destination/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
