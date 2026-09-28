# Soundwich on LTX-2.5

Soundwich generates one shared video together with separately controlled audio stems on
[LTX-2.5](https://github.com/Lightricks/LTX-2). Each scene is generated in three steps:

1. **Stage 1** (dev model, 40 steps, half resolution): one shared video and one audio lane per stem. Every
   stem has its own positive/negative audio CFG, cached activation/suppression carriers set its timeline,
   and a persistent scene lane gathers the time-controlled stems for the first half of denoising.
2. **SAM3**: each stem's `sam_prompt` is tracked on decoded Stage-1 frames and pooled into Stage-2 video tokens.
3. **Stage 2** (distilled model, 8 steps from sigma 0.95, full resolution): the upscaled video is refined with
   A2V routed by SAM ownership, V2A routed by SAM and the stem timeline, and the scene lane broadcast into
   every stem's audio self-attention.

Every stem is decoded separately, then mixed.

## Requirements

- Linux with one CUDA GPU with 32 GB of memory (developed on an RTX 5090).
- [uv](https://docs.astral.sh/uv/), git, and about 61 GB of disk for the checkpoints.
- A separate Python environment for [SAM 3](https://github.com/facebookresearch/sam3) (SAM 3.1 multiplex
  video predictor).

## Setup

```bash
cd ltx-2.5
./setup.sh
```

`setup.sh` clones LTX-2 at commit `400fd31054597515f47125691032c04b1c3ee24e` into `third_party/LTX-2`, applies
`patches/ltx-2.patch`, and runs `uv sync --extra natten` there. The patch adds the Comfy INT8 ConvRot checkpoint
loader (and the `comfy-kitchen` dependency), batched initial latents for `DiffusionStage`, and a decoder call that
returns both the pre- and post-bandwidth-extension waveforms.

### Checkpoints

Download from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (accept the model terms first):

```bash
hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
    diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir models/ltx-2.5
```

The files are expected under `models/ltx-2.5/` with this layout. Use `--models-dir` or `SOUNDWICH_LTX_MODELS`
to point elsewhere.

| Role | File |
| --- | --- |
| Stage 1 transformer, carrier recording | `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` |
| Stage 2 transformer, Stage-2 suppression carrier | `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| Text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| Video VAE | `vae/ltx-2.5-video-vae-bf16.safetensors` |
| Audio VAE (with bandwidth extension) | `vae/ltx-2.5-audio-vae-bf16.safetensors` |
| x2 spatial upscaler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |

### SAM 3

Install SAM 3 in its own environment following its README, then point Soundwich at it:

```bash
export SOUNDWICH_SAM3_PYTHON=/path/to/sam3-env/bin/python   # required
export SOUNDWICH_SAM3_REPO=/path/to/sam3                     # optional, if sam3 is not installed in that env
export SOUNDWICH_SAM3_CHECKPOINT=/path/to/sam3.1.pt           # optional, otherwise downloaded from Hugging Face
```

SAM runs as a subprocess through the shared `../common/sam3_video_backend.py` after the LTX models have been
released.

## Run the example

```bash
cd ltx-2.5
PY=third_party/LTX-2/.venv/bin/python
$PY -m soundwich_ltx.generate --scene examples/neon_biology_lab.yaml --dry-run   # check config, no GPU
$PY -m soundwich_ltx.generate --scene examples/neon_biology_lab.yaml
```

The first run records the six carriers the scene needs (four activation carriers from the scene's
`entity_groups`, plus the Stage-1 and Stage-2 suppression carriers) and caches them in `outputs/carriers/`.
Later runs reuse them. Other flags: `--output-dir`, `--seed`, `--carrier-cache`, `--stage1-only`.

### Outputs

Everything for one run is written to `outputs/<scene id>_seed<seed>/`:

| Path | Content |
| --- | --- |
| `stage2_mix.mp4` | Final video with the mixed audio |
| `stage2_audio/<stem>.wav`, `stage2_audio/mix.wav` | Final 48 kHz stems and mix |
| `stage2_audio/scene_render.wav` | Scene lane, for inspection only (not in the mix) |
| `stage2_audio/pre_bwe/` | Same stems before bandwidth extension (24 kHz) |
| `stage1_mix.mp4`, `<stem>.wav`, `mix.wav`, `pre_bwe/` | Half-resolution Stage-1 result |
| `sam_outputs/` | SAM prompts, sampled frames, token masks, backend logs |
| `stage1_latents.pt`, `stage2_*_latent.pt`, `run.json` | Latents and the resolved configuration |

## Scene format

See `examples/neon_biology_lab.yaml`. A scene lists:

- `visual`: the shared video prompt and negative; `allowed_entities` are appended as "Visible sound entities".
- `audio_globals`: quality prompts added to every stem.
- `entity_groups`: one activation carrier per sound type (`carrier: {positive, negative, seed}`; optional
  `audio_rescale`, `frames`, `frame_rate`, `quantile`, ...). Carriers are recorded with the scene's step count.
- `sound_entities`: visible sources with their group, `sam_prompt`, and the description used to exclude them
  from other stems' negatives.
- `stems`: prompt, explicit negative (otherwise the other stems' sources are used), `volume`, timeline
  `windows` (seconds), and `scene_context_enabled` (keep continuous sources out of the scene lane).
- `generation`: `seed`, `frames` (8k+1), `frame_rate`, `width`/`height` (final size; Stage 1 runs at half),
  `steps`, CFG scales.
- `method_settings`: overrides of the method defaults in `soundwich_ltx/scene.py` (`DEFAULT_METHOD_SETTINGS`).

Key knobs:

| Setting | Default | Meaning |
| --- | --- | --- |
| `stage1.activation_peak_strength` / `activation_peak_fraction` / `activation_strength` | 0.50 / 0.10 / 0.10 | Carrier strength: 0.50 for the first 10% of denoising, then 0.10 |
| `stage1.outside_suppression`, `outside_silence_blend` | 0.25, 0.60 | Attenuation and suppression-carrier blend outside a stem's windows |
| `stage1.window_feather` | 0.02 | Window feathering as a fraction of `duration_seconds` |
| `stage1.scene_gather_fraction` | 0.50 | Share of Stage-1 steps during which the scene lane gathers the stems |
| `stage2.start_sigma` | 0.95 | Stage-2 starting noise level (8 densified distilled steps) |
| `stage2.scene_to_real_strength`, `real_to_scene_strength` | 1.0, 0.10 | Scene broadcast and scene update in Stage 2 |
| `stage2.a2v_active_gain` | 1.0 | Gain on routed A2V updates for SAM-owned video tokens (the example uses 1.5) |
| `stage2.audio_cfg` | 1.0 | Values above 1 enable per-stem negative audio CFG in Stage 2 |

## License

Soundwich code is released under Apache-2.0. LTX-2 and its checkpoints are covered by the LTX-2.x Community
License; SAM 3 by its own license.
