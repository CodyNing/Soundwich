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

Every stem is decoded separately, then mixed. A finished run can then be edited per stem (retiming or replacing a
source) and its video refined on the edited audio; see [Editing stems](#editing-stems).

## Requirements

- Linux with one CUDA GPU. Peak GPU allocations measured with automatic CPU offload:

  | Run | Stage 1 | Stage 2 |
  | --- | --- | --- |
  | 4 stems, 768×448 | 73 GiB | 56 GiB |
  | *The Rooftop Reservation* (5 stems, 768×448) | 89 GiB | not measured |
  | *The Wrong Stop* (4 stems, 1280×736) | 122 GiB | not measured |

  4-stem 768×448 scenes run on 80 GB GPUs. The two example peaks were measured on a GPU with 128 GiB available, and
  whether *The Rooftop Reservation* fits in 80 GB is untested. Stage 1 of *The Wrong Stop* takes about 100 s per
  step, against about 25 s at 768×448. Editing refinement at 768×448 peaked at 75–80 GiB (2–3 stems). Plan for
  about 220 GB of host RAM.
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

## Run the examples

```bash
cd minimax-h3
.venv/bin/python -m soundwich_h3.generate --scene examples/the_rooftop_reservation.json --dry-run   # no GPU
.venv/bin/python -m soundwich_h3.generate --scene examples/the_rooftop_reservation.json
.venv/bin/python -m soundwich_h3.generate --scene examples/the_wrong_stop.json   # 1280×736, see Requirements
```

Both examples are from the project page:

- *The Rooftop Reservation* (768×448, seed 3101): a woman and a man with two lines each, a door closing, rain on
  the glass, and a jazz-guitar score.
- *The Wrong Stop* (1280×736, seed 3103): two travelers on a sleeper train with two lines each, train ambience,
  and an acoustic-guitar score. It uses its own carrier references (`carrier_references_wrong_stop.json`, a
  different quiet prompt).

The project page shows the Stage-1 result (`stage1/`) of both.

The first run records the carriers, each a short native single-row H3 generation. These are the activation
carriers of the scene's stems (`female_speech`, `male_speech`, and `door_knock` for the door) and `quiet` on the
Stage-1 schedule, plus `quiet` again on the Stage-2 schedule, made by refining the quiet reference itself. Carriers
depend on the geometry, so each example records its own. They are cached in `cache/carriers/`
(`SOUNDWICH_H3_CACHE`), together with the text embeddings, and later runs reuse them. The other flags are
`--output-dir`, `--seed` (Stage 2 uses seed + 1), and `--stage1-only`. Completed stages in the output directory are
reused.

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

A text prompt alone often does not isolate the right person. The examples provide text prompts only (the entity
descriptions from their video prompts, and `wooden door` for the door close), without reviewed clicks. Check
`sam/overlays/` after every run. For a wrong or unstable mask, add reviewed clicks to that stem in the scene:
`sam_point_frame` (source frame index) and `sam_points` (normalized `x`/`y` in [0, 1] with `label: positive` or
`negative`). Then delete `sam/` and `stage2/` and rerun; Stage 1 is reused.

## Editing stems

A completed run (Stage 1, SAM masks, and Stage 2) can be edited without regenerating the other stems. The two
examples are the paper's H3 edits, each on its own base scene (768×448):

- *The Enchanted Archive* (`examples/enchanted_archive.json`, seed 74): the cat's meowing is moved from 0.4–2.3 s
  to 3.2–5.1 s (`examples/edits/archive_move_meow.json`).
- *Occult Noir* (`examples/occult_noir.json`, seed 71): the detective's line "Then someone was here before us" is
  replaced by "Someone is still watching us", a new take (seed 71) in the same 4.4–8.0 s window
  (`examples/edits/noir_new_reply.json`).

```bash
.venv/bin/python -m soundwich_h3.generate --scene examples/enchanted_archive.json
.venv/bin/python -m soundwich_h3.edit --run outputs/enchanted_archive-seed74 \
    --edit examples/edits/archive_move_meow.json --dry-run   # validate and print the plan, no GPU
.venv/bin/python -m soundwich_h3.edit --run outputs/enchanted_archive-seed74 \
    --edit examples/edits/archive_move_meow.json

.venv/bin/python -m soundwich_h3.generate --scene examples/occult_noir.json
.venv/bin/python -m soundwich_h3.edit --run outputs/occult_noir-seed71 \
    --edit examples/edits/noir_new_reply.json
```

The Archive scene needs a `cat_meow` activation carrier (`carrier_references_archive.json`). The SAM prompts
include the reviewed clicks used for the paper's masks.

An edit file has `edits`, a list with one entry per edited stem (`stem`, plus `retime`, `replace`, and/or
`prompt`):

- `retime`: a list of `{"window": i, "start": seconds}`. The clip of window *i* is moved within the stem's Stage-2
  audio latents (40 tokens per second, both stereo channels), and the tokens between the old and the new position
  shift to fill the gap. The window moves with it. The shifted span may not contain another window of the same
  stem. `window` may be omitted for a stem with a single window.
- `replace`: `{"prompt": ..., "seed": ..., "windows": [...]}` (`windows` optional). Stage 1 is run for this stem
  alone with the new source prompt, seed, and windows (and the scene's Stage-1 settings), and its audio latents
  replace the stem's. A replaced stem can also be retimed.
- `prompt`: a new source prompt for a kept stem during the refinement; its audio is unchanged. The Noir example
  uses it to give the woman's stem the same source-only prompt style as the new take. It only accompanies a
  `retime` or `replace` in the same edit file; an edit file needs at least one of those.

An optional `video_prompt_substitutions` list of `[old, new]` pairs edits the shared video prompt, e.g. to change
a spoken line there too (see `examples/edits/noir_new_reply.json`).

The other stems keep their Stage-2 audio. The edited audio latents are then held clean and fixed, while the saved
Stage-1 video is re-noised to σ<sub>v</sub>=0.95 (seed + 1) and refined for 23 evaluations. This uses the run's SAM
masks, the revised windows, and quiet suppression (0.30/0.25), without activation carriers. The first edit records
one more carrier: `quiet` refined with its own audio fixed. Results are written to
`<run>/edits/<edit file name>/`: `edited/` (video, stems, mix, latents), `fixed_audio.pt`, `takes/<stem>/`
for replacements, and the resolved `edit.json`.

## Scene format

See `examples/the_rooftop_reservation.json`. Geometry must be `num_frames` = 17k+5 and `width`/`height`
multiples of 32. The paper uses 243 frames (10.125 s at 24 fps), 768×448, and `num_inference_steps` 24. A scene has:

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
`routing.py` are adapted from diffusers (Apache-2.0). SAM 3 is covered by its own license.

The MiniMax H3 checkpoint is covered by the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE),
which you accept by downloading or running it. Read it before use. In particular:

- **Territory.** The license applies only outside the European Union, the United Kingdom, the Republic of Korea,
  and the United States. It does not authorize using the weights, or using or displaying their outputs, in those
  regions. Contact MiniMax for a license there.
- **Attribution.** Products or services built with MiniMax H3 must show "Powered by MiniMax H3".
- **Acceptable use.** Its Acceptable Use Policy applies to everything you generate.
