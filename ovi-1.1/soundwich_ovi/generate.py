"""CLI: build any missing carrier banks, then generate a Soundwich Ovi scene.

    python -m soundwich_ovi.generate --scene examples/ticket_counter.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from scipy.io import wavfile

from .carriers import ensure_carrier_bank
from .engine import OviMultiStemEngine
from .scene import SceneSpec, load_scene
from .audio import normalize_and_mix
from . import UPSTREAM_REPO_ROOT

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
DEFAULT_ACTIVATION_CARRIER = EXAMPLES_DIR / "carriers" / "general_party_detailed.yaml"
DEFAULT_SUPPRESSION_CARRIER = EXAMPLES_DIR / "carriers" / "silence_detailed_black_frame.yaml"

# Runtime and method defaults used in the paper (960x960_10s, qint8 + CPU
# offload, 50-step UniPC).
MODEL_CLIP_SECONDS = {"960x960_10s": 10.0}

DEFAULT_CONFIG: dict = {
    "ckpt_dir": "./ckpts",
    "model_name": "960x960_10s",
    "mode": "t2v",
    "qint8": True,
    "fp8": False,
    "cpu_offload": True,
    "sp_size": 1,
    "video_frame_height_width": [704, 1280],
    "solver_name": "unipc",
    "sample_steps": 50,
    "shift": 5.0,
    "video_guidance_scale": 4.0,
    "audio_guidance_scale": 3.0,
    "slg_layer": 11,
    "method": {
        "inside_strength": 0.15,
        "outside_strength": 0.50,
        "outside_suppression": 0.25,
        "activation_value_scale": 0.80,
        "feather_seconds": 0.20,
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Scene YAML to generate")
    parser.add_argument(
        "--output-dir", default="./outputs/scenes", help="Where to write results"
    )
    parser.add_argument(
        "--carrier-cache-dir",
        default="./outputs/carrier_bank",
        help="Where built carrier banks are cached",
    )
    parser.add_argument("--ckpt-dir", default=DEFAULT_CONFIG["ckpt_dir"])
    parser.add_argument(
        "--sample-steps", type=int, default=DEFAULT_CONFIG["sample_steps"]
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the scene's seed"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse the scene and print the resolved config without loading the model",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = {k: v for k, v in DEFAULT_CONFIG.items() if k != "method"}
    config["method"] = dict(DEFAULT_CONFIG["method"])
    config["ckpt_dir"] = args.ckpt_dir
    if args.sample_steps <= 0:
        raise ValueError("--sample-steps must be positive")
    config["sample_steps"] = int(args.sample_steps)
    return config


def _save_scene_result(output_root: Path, scene: SceneSpec, result: dict, config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scene_dir = output_root / f"{stamp}_{scene.id}_seed_{scene.seed}"
    scene_dir.mkdir(parents=True, exist_ok=False)
    normalized, mixed = normalize_and_mix(
        result["stems"], [stem.volume for stem in scene.stems]
    )
    for stem, waveform in zip(scene.stems, normalized, strict=True):
        wavfile.write(scene_dir / f"stem_{stem.id}.wav", 16000, waveform)
    wavfile.write(scene_dir / "mix.wav", 16000, mixed)

    from ovi.utils.io_utils import save_video

    save_video(
        str(scene_dir / "video_mix.mp4"),
        result["video"],
        mixed,
        fps=24,
        sample_rate=16000,
    )
    metadata = {
        "scene_id": scene.id,
        "seed": scene.seed,
        "duration_seconds": scene.duration_seconds,
        "stems": [stem.id for stem in scene.stems],
        "prompts": result["prompts"],
        "resolution": result["resolution"],
        "method": result["method"],
        "sampling": {key: value for key, value in config.items() if key not in {"method", "ckpt_dir"}},
    }
    (scene_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return scene_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    scene = load_scene(args.scene)
    if args.seed is not None:
        scene = dataclasses.replace(scene, seed=args.seed)
    # Timeline windows are normalized by the scene duration, so it must equal the model's clip length.
    if abs(scene.duration_seconds - MODEL_CLIP_SECONDS[config["model_name"]]) > 1e-6:
        raise ValueError(
            f"{scene.id}: duration_seconds={scene.duration_seconds} but {config['model_name']} generates "
            f"{MODEL_CLIP_SECONDS[config['model_name']]} s clips"
        )

    if args.validate_only:
        print(
            f"valid: {scene.id} stems={len(scene.stems)} "
            f"duration={scene.duration_seconds}s seed={scene.seed} "
            f"steps={config['sample_steps']}"
        )
        print(json.dumps(config, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible in this shell. Run this command from the GPU "
            "session where torch.cuda.is_available() is True."
        )
    if int(config["sp_size"]) != 1:
        raise ValueError("Soundwich Ovi currently requires sp_size: 1")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    # Upstream Ovi reads its model configs relative to its repo root, so make
    # user paths absolute and run from there.
    config["ckpt_dir"] = str(Path(config["ckpt_dir"]).resolve())
    carrier_cache_dir = Path(args.carrier_cache_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    os.chdir(UPSTREAM_REPO_ROOT)

    torch.cuda.set_device(0)
    from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state

    initialize_sequence_parallel_state(1)

    engine = OviMultiStemEngine(
        config=OmegaConf.create(config), device=0, target_dtype=torch.bfloat16
    )

    carrier_kwargs = dict(
        cache_dir=carrier_cache_dir,
        sample_steps=config["sample_steps"],
        video_frame_height_width=config["video_frame_height_width"],
        solver_name=config["solver_name"],
        shift=config["shift"],
        video_guidance_scale=config["video_guidance_scale"],
        audio_guidance_scale=config["audio_guidance_scale"],
        slg_layer=config["slg_layer"],
        # Recorded tokens depend on the checkpoint and its quantization, not only the sampling settings.
        model_settings={key: config[key] for key in ("model_name", "qint8", "fp8")},
    )
    activation_path = ensure_carrier_bank(
        engine, carrier_config=DEFAULT_ACTIVATION_CARRIER, **carrier_kwargs
    )
    suppression_path = ensure_carrier_bank(
        engine, carrier_config=DEFAULT_SUPPRESSION_CARRIER, **carrier_kwargs
    )

    result = engine.generate_scene(
        scene,
        video_frame_height_width=config["video_frame_height_width"],
        solver_name=config["solver_name"],
        sample_steps=config["sample_steps"],
        shift=config["shift"],
        video_guidance_scale=config["video_guidance_scale"],
        audio_guidance_scale=config["audio_guidance_scale"],
        slg_layer=config["slg_layer"],
        activation_carrier_paths=[str(activation_path)] * len(scene.stems),
        suppression_carrier_path=str(suppression_path),
        **config["method"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    saved = _save_scene_result(output_root, scene, result, config)
    logging.info("Saved %s", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
