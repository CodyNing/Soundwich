# Soundwich on Ovi 1.1

Runs the Soundwich method on [Ovi 1.1](https://github.com/character-ai/Ovi) (T2V): one shared video and one
audio stem per requested source, with source-specific guidance and carrier-based timeline control. Scene
broadcast and entity routing are not part of this port. Entity-routed refinement did not reliably fix
speaker ownership on Ovi (see the paper's appendix).

## Requirements

- A CUDA GPU. The included config runs the 960x960_10s checkpoint with `qint8` weight quantization and
  CPU offload, which fits the ~11B fused backbone in about 32 GB of VRAM. With more VRAM you can drop
  `qint8`/`cpu_offload` in `soundwich_ovi/generate.py`'s `DEFAULT_CONFIG` for faster generation.
- Python 3.11 and a CUDA 12.8-capable driver. `setup.sh` installs PyTorch 2.8.0 (cu128) and flash-attn 2.8.3.post1,
  which may need to compile against your CUDA toolkit (`CUDA_HOME`).

## Setup

```bash
./setup.sh
.venv/bin/python third_party/Ovi/download_weights.py --output-dir ckpts --models 960x960_10s
```

`setup.sh` clones upstream Ovi at the pinned commit into `third_party/Ovi` (gitignored) and installs its
dependencies plus this package's extras into `.venv`. It is idempotent.

## Run the example

```bash
.venv/bin/python -m soundwich_ovi.generate --scene examples/ticket_counter.yaml --validate-only
.venv/bin/python -m soundwich_ovi.generate --scene examples/ticket_counter.yaml
```

The first run builds and caches the two carrier banks the method needs (an "activation" carrier from
`examples/carriers/general_party_detailed.yaml` and a "suppression" carrier from
`examples/carriers/silence_detailed_black_frame.yaml`) into `outputs/carrier_bank/`, keyed by carrier id,
step count, and a hash of the carrier, sampling, and model settings; later runs with the same settings reuse them.
`--validate-only` parses the scene and prints the resolved config without loading the model (upstream Ovi
queries the CUDA device at import, so a GPU must still be visible).

## Output files

Each run writes a timestamped folder under `outputs/scenes/`:

- `video_mix.mp4` — shared video with the normalized stem mix
- `mix.wav` — normalized sum of all stems
- `stem_<id>.wav` — one normalized audio stem per scene source
- `metadata.json` — prompts, resolution, and the sampling and method settings used

## Key config knobs

Defaults live in `soundwich_ovi/generate.py`'s `DEFAULT_CONFIG` and match the settings used in the paper:
`960x960_10s` model, 50-step UniPC, shift 5.0, audio/video guidance 3.0/4.0, SLG layer 11, 704x1280 frames,
and the carrier-blending strengths (`inside_strength`, `outside_strength`, `outside_suppression`,
`activation_value_scale`, `feather_seconds`). The carriers' token-selection quantiles are set in the carrier YAMLs. CLI flags cover
`--output-dir`, `--carrier-cache-dir`, `--ckpt-dir`, `--sample-steps`, `--seed`, and `--validate-only`.

## Describing a scene

A scene YAML lists a shared visual prompt, one or more audio `stems` (each with its own prompt, negative
prompt, and `windows` giving when it is audible), and optional `sound_entities` naming who each stem
belongs to on screen. A stem's `kind` (set on the stem, or taken from its entity's `group`; default `speech`)
decides whether its window `text` is written into the prompts as spoken lines. `duration_seconds` must be 10
(the clip length of the 960x960_10s model). All stems share the one activation carrier. See
`examples/ticket_counter.yaml`.
