# Soundwich on MiniMax H3

Soundwich generates one shared video together with separately controlled audio stems on the frozen single-stream
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) model (text-to-video+audio, via the diffusers
implementation). For *N* stems, H3's batch axis holds *N*+1 rows: row 0 carries the shared video and the visual
prompt, and each other row carries one stem with its own source prompt. Before every block the shared video is copied into
the stem rows, and the video row attends to the concatenated audio of all stems. H3 attends jointly over text,
video, and audio, so no separate scene stem is used. A scene is generated in three steps:

1. **Stage 1 (Stem Formation)**, 24 scheduler points (23 model evaluations): cached activation carriers are
   replayed inside each stem's windows, and the quiet carrier outside them. Outside its windows, a stem's audio
   reads only its own audio.
2. **SAM 3.1**: each visible stem's `sam_prompt` is tracked on the Stage-1 video and pooled onto H3's packed
   video-token grid.
3. **Stage 2 (Scene Integration)**: the Stage-1 latents are re-noised to σ<sub>v</sub>=σ<sub>a</sub>=0.95 and refined
   for 23 evaluations with entity routing: each stem's audio sees only its owner's video, and owner video sees the
   stem's audio. Quiet-carrier suppression remains active outside windows, and no activation carriers are used.

Every stem is decoded separately, then mixed.

## Requirements

- Linux with one CUDA GPU with at least 80 GB of memory. The 4-stem example peaks at about 73 GiB allocated in
  Stage 1 and 56 GiB in Stage 2, with automatic CPU offload. Plan for about 220 GB of host RAM.
- Python ≥ 3.10 (developed on 3.12), git, and about 135 GB of disk for the checkpoint.
- A separate Python environment for [SAM 3](https://github.com/facebookresearch/sam3) with the SAM 3.1 multiplex
  video predictor.

## Setup

```bash
cd minimax-h3
./setup.sh
```

`setup.sh` clones diffusers at commit `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`, which contains the MiniMax H3
modular pipeline, into `third_party/diffusers`. It then creates `.venv` and installs `requirements.txt`, including
PyTorch 2.11 with CUDA 12.8. No patches are applied.

### Checkpoint

Download [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) at the pinned revision. The
`transformer_ref/` weights (reference-to-video) are not needed:

```bash
.venv/bin/hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
    --exclude "transformer_ref/*" --local-dir checkpoints/MiniMax-H3
```

Use `SOUNDWICH_H3_CHECKPOINT` to point elsewhere.

### SAM 3

Install SAM 3 in its own environment following its README, then point Soundwich at it:

```bash
export SOUNDWICH_SAM3_PYTHON=/path/to/sam3-env/bin/python   # required for Stage 2
export SOUNDWICH_SAM3_REPO=/path/to/sam3                     # optional, if sam3 is not installed in that env
export SOUNDWICH_SAM3_CHECKPOINT=/path/to/sam3.1_multiplex.pt # optional, otherwise downloaded from Hugging Face
```

SAM runs as a subprocess through the shared `../common/sam3_video_backend.py`, after the H3 pipeline has been
released.

## Run the example

```bash
cd minimax-h3
.venv/bin/python -m soundwich_h3.generate --scene examples/the_last_button.json --dry-run   # check config, no GPU
.venv/bin/python -m soundwich_h3.generate --scene examples/the_last_button.json
```

The example is *The Last Button* from the project page: a groom and a tailor with two dialogue turns each, a viola,
and street ambience.

The first run records three carriers, each a short native single-row H3 generation. These are
`male_speech` (activation) and `quiet` on the Stage-1 schedule, plus `quiet` again on the Stage-2 schedule, made
by refining the quiet reference itself. They are cached in `cache/carriers/` (`SOUNDWICH_H3_CACHE`), together
with the text embeddings, and later runs reuse them. The other flags are `--output-dir`, `--seed` (Stage 2 uses
seed + 1), and `--stage1-only`. Completed stages in the output directory are reused.

### Outputs

Everything for one run is written to `outputs/<scene id>-seed<seed>/`:

| Path | Content |
| --- | --- |
| `stage2/sample.mp4` | Final video with the mixed audio |
| `stage2/<stem>.wav`, `stage2/mix.wav` | Final stems and mix |
| `stage1/` | The same files for the Stage-1 result |
| `sam/` | SAM prompts, full-resolution masks, overlays (`overlays/`) and mask validation |
| `*/latents.pt`, `*/scene.json`, `*/run.json` | Latents, the exact scene, and the resolved schedule/settings |

### Reviewing SAM masks

A text prompt alone often does not isolate the right person. Check `sam/overlays/` after every run. For a wrong or
unstable mask, add reviewed clicks to that stem in the scene: `sam_point_frame` (source frame index) and
`sam_points` (normalized `x`/`y` in [0, 1] with `label: positive` or `negative`). Then delete `sam/` and `stage2/`
and rerun; Stage 1 is reused.

## Scene format

See `examples/the_last_button.json`. Geometry must be `num_frames` = 17k+5 and `width`/`height` multiples of 32.
The paper uses 243 frames (10.125 s at 24 fps), 768×448, and `num_inference_steps` 24. A scene has:

- `seed`, `width`, `height`, `num_frames`, `num_inference_steps`, and `video_prompt` (the shared video row's
  prompt).
- `carrier_references`: a JSON file of carrier reference prompts (`role`, `prompt`, `seed`, `quantile`).
  `quiet` is required.
- `stems`: `id`, source `prompt`, and `volume` for the mix. Timeline-controlled stems also have `windows`
  (seconds) and a `carrier_group`. Stems without windows are continuous and uncontrolled. For Stage 2, each stem
  needs either a `sam_prompt` (a string, or a list whose masks are unioned, e.g. performer and instrument) or a
  `video_routing` of `background_only` (the complement of all foreground masks) or `offscreen` (no video
  tokens). A stem with a single `sam_prompt` may add reviewed `sam_points` on `sam_point_frame`.

Key settings (optional scene overrides):

| Setting | Default | Meaning |
| --- | --- | --- |
| `stage1.quiet_blend`, `stage1.outside_attenuation` | 0.30, 0.25 | Quiet-carrier blend and attenuation outside a stem's windows |
| `stage2.video_sigma`, `stage2.audio_sigma` | 0.95, 0.95 | Stage-2 re-noising levels |
| `stage2.quiet_blend`, `stage2.outside_attenuation` | 0.30, 0.25 | Stage-2 quiet-carrier blend and attenuation |

The fixed carrier settings are as follows. Activation strength is 0.50 for the first 10% of denoising, then 0.10,
with a value scale of 0.8 and RMS matching capped at 4. Windows are feathered by 2% of the clip duration. Masks
are thresholded at 0.15 after box pooling.

## License

Soundwich code is released under Apache-2.0. The attention processors in `soundwich_h3/batch.py` and
`routing.py` are adapted from diffusers (Apache-2.0). The MiniMax H3 checkpoint is covered by the license in its
Hugging Face repository, and SAM 3 by its own license.
