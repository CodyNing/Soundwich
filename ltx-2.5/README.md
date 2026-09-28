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
Later runs reuse them. The Stage-2 suppression carrier is the recording used for the paper and ships in
`carriers/`; it is copied into the cache instead of being recorded. Other flags: `--output-dir`, `--seed`, `--carrier-cache`, `--stage1-only`,
`--reuse-stage1`.

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
| `stage1_latents.pt`, `stage2_*_latent.pt`, `run.json`, `scene.yaml` | Latents, the resolved configuration, and the scene (used by `edit`) |
| `edits/<edit id>/` | Stem edits (see [Editing stems](#editing-stems)) |

### Reviewing SAM masks

Entity routing is only as good as the masks, and a text prompt alone is often not enough (two similar animals,
people who swap places). Inspect every run:

1. Open `sam_outputs/overlays/` and check that each stem's mask follows its on-screen source for the whole clip.
2. For a wrong or unstable mask, add reviewed clicks to that stem in the scene: `sam_point_frame` (Stage-1 frame
   index) and `sam_points` (normalized `x`/`y` in [0, 1] with `label: positive` or `negative`). Optionally override
   the stem's `sam_prompt`; the text starts tracking and the clicks select the object.
3. Rerun with `--reuse-stage1`. This keeps the Stage-1 result and recomputes only the masks and Stage 2.

The example includes the reviewed prompts and clicks used for the paper's result.

## Editing stems

A finished run can be edited without regenerating the other stems. Editing builds new audio latents, then
refines the video again around them:

- **Retime** moves each window of a stem to a new start. The saved Stage-2 audio latent is cut at the window
  (frame-snapped) and the surrounding latent frames shift to keep the clip length; nothing is re-encoded.
- **Replace** regenerates one stem with Stage 1 (new line or prompt, original visual prompt and clip length, half
  resolution). Its audio latent replaces that stem; the other stems keep their saved Stage-2 latents. A replaced
  stem can also be retimed.
- **Video refinement** restarts from the saved Stage-1 video and SAM masks with the Stage-2 upscale and 8-step
  schedule. Only the video is denoised; the edited audio latents stay clean and fixed. Routing uses the edited
  windows and the original masks, the scene lane is off, and Stage-2 suppression replay stays active outside the
  edited windows.

The example reproduces the paper's speaker-order swap: generate the base scene (Maya speaks first, then Leo),
then move both introductions so Leo speaks first. Both lines keep their generated audio; the video is refined
with Stage-2 seed 94.

```bash
$PY -m soundwich_ltx.generate --scene examples/two_introductions.yaml
$PY -m soundwich_ltx.edit --run outputs/two_introductions_seed99 \
    --edit examples/edits/two_introductions_swap_order.yaml --dry-run   # print the plan, no GPU
$PY -m soundwich_ltx.edit --run outputs/two_introductions_seed99 \
    --edit examples/edits/two_introductions_swap_order.yaml
```

An edit lists stems by id:

```yaml
id: new_line                     # output folder: <run>/edits/new_line/
seed: 94                         # optional Stage-2 seed (default: the run's seed)
stems:
  maya_voice:
    retime: [3.96]               # new start (s) per window, null keeps one; lengths are kept
  leo_voice:
    replace:
      windows: [{text: I'm Leo. I study the tides.}]   # replaces the quoted line in the stem prompt
      # optional: positive, negative, per-window start/end, seed (default: the run's seed)
```

The run folder must contain `scene.yaml` (written by `generate`; otherwise pass `--scene`), `stage1_latents.pt`,
`stage2_audio_latent.pt`, and `sam_outputs/stage2_a2v_mask.json`. The edit writes `stage2_mix.mp4`,
`stage2_audio/` (the decoded edited stems and mix), the fixed `audio_latents.pt`, the edited `scene.yaml` and
`timeline.json`, and for replacements the new Stage-1 take in `takes/<stem>/`.

## Scene format

See `examples/neon_biology_lab.yaml`. A scene lists:

- `visual`: the shared video prompt and negative; `allowed_entities` are appended as "Visible sound entities".
- `audio_globals`: quality prompts added to every stem.
- `entity_groups`: one activation carrier per sound type (`carrier: {positive, negative, seed}`; optional
  `audio_rescale`, `frames`, `frame_rate`, `quantile`, ...). Carriers are recorded with the scene's step count.
- `sound_entities`: visible sources with their group, `sam_prompt`, and the description used to exclude them
  from other stems' negatives.
- `stems`: prompt, explicit negative (otherwise the other stems' sources are used), `volume`, timeline
  `windows` (seconds), `scene_context_enabled` (keep continuous sources out of the scene lane), and optional
  reviewed SAM overrides (`sam_prompt`, `sam_point_frame`, `sam_points`).
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
| `stage1.scene_aggregation` | `raw_sum` | How the scene lane sums the gated stems: `raw_sum`, or `rms_sum` (each stem RMS-normalized, the sum rescaled by the stems' mean RMS). The paper's main comparison used `rms_sum` for most scenes (Stage 2 always uses `raw_sum`) |
| `stage2.start_sigma` | 0.95 | Stage-2 starting noise level (8 densified distilled steps) |
| `stage2.scene_to_real_strength`, `real_to_scene_strength` | 1.0, 0.10 | Scene broadcast and scene update in Stage 2 |
| `stage2.a2v_active_gain` | 1.0 | Gain on routed A2V updates for SAM-owned video tokens (some scenes benefit from a higher value) |
| `stage2.audio_cfg` | 1.0 | Values above 1 enable per-stem negative audio CFG in Stage 2 |

## License

Soundwich code is released under Apache-2.0. LTX-2 and its checkpoints are covered by the LTX-2.x Community
License; SAM 3 by its own license.
