"""Generate a Soundwich scene: carriers (cached) -> Stage 1 -> SAM3 masks -> Stage 2.

    python -m soundwich_ltx.generate --scene examples/neon_biology_lab.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from soundwich_ltx.config import MultiStemConfig
from soundwich_ltx.scene import CompiledScene, carrier_record_config, compile_scene, load_scene

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soundwich layered audio-video generation on LTX-2.5")
    parser.add_argument("--scene", type=Path, required=True, help="Scene YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output root (default: outputs)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("SOUNDWICH_LTX_MODELS", BACKEND_ROOT / "models" / "ltx-2.5")),
        help="LTX-2.5 checkpoint folder (default: $SOUNDWICH_LTX_MODELS or models/ltx-2.5)",
    )
    parser.add_argument(
        "--carrier-cache",
        type=Path,
        help="Folder for recorded carriers (default: <output-dir>/carriers)",
    )
    parser.add_argument("--seed", type=int, help="Override the scene's generation seed")
    parser.add_argument("--stage1-only", action="store_true", help="Stop after Stage 1 (no SAM3 or Stage 2)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the scene and print the resolved configuration without loading any model",
    )
    return parser


def _record_missing_carriers(compiled: CompiledScene, models: dict[str, str], cache_root: Path) -> None:
    from ltx_pipelines.utils.helpers import cleanup_memory  # noqa: PLC0415

    from soundwich_ltx.pipeline import MultiStemPipeline  # noqa: PLC0415

    for spec in compiled.carrier_specs():
        destination = spec.directory(cache_root)
        if (destination / "carrier.json").is_file():
            logging.info("carrier %s: cached at %s", spec.name, destination)
            continue
        logging.info("carrier %s: recording into %s", spec.name, destination)
        config = MultiStemConfig.from_dict(carrier_record_config(spec, models))
        pipeline = MultiStemPipeline(config)
        try:
            pipeline.record_carrier(destination)
        finally:
            del pipeline
            cleanup_memory()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    carrier_cache = (args.carrier_cache or output_dir / "carriers").expanduser().resolve()
    compiled = compile_scene(
        load_scene(args.scene),
        models_dir=args.models_dir.expanduser().resolve(),
        carrier_cache=carrier_cache,
        seed=args.seed,
    )
    run_root = output_dir / f"{compiled.raw['id']}_seed{compiled.raw['generation']['seed']}"
    config = compiled.config()
    carrier_configs = [
        MultiStemConfig.from_dict(carrier_record_config(spec, config.model.__dict__))
        for spec in compiled.carrier_specs()
    ]
    if args.dry_run:
        payload = {
            "run_directory": str(run_root),
            "generation": config.summary(),
            "carriers": [
                {
                    "directory": str(spec.directory(carrier_cache)),
                    "cached": (spec.directory(carrier_cache) / "carrier.json").is_file(),
                    **carrier.summary(),
                }
                for spec, carrier in zip(compiled.carrier_specs(), carrier_configs, strict=True)
            ],
            "missing_model_files": [str(path) for path in config.missing_model_paths()],
            "sam3_missing_settings": [] if args.stage1_only else config.sam3.missing_settings(),
        }
        print(json.dumps(payload, indent=2))  # noqa: T201
        return

    missing = config.missing_model_paths()
    if missing:
        raise FileNotFoundError(
            "missing LTX-2.5 checkpoints (see README):\n" + "\n".join(f"  - {path}" for path in missing)
        )
    if not args.stage1_only and config.sam3.missing_settings():
        raise RuntimeError(
            "SAM3 is not configured:\n" + "\n".join(f"  - {item}" for item in config.sam3.missing_settings())
        )
    _record_missing_carriers(compiled, dict(config.model.__dict__), carrier_cache)

    from soundwich_ltx.pipeline import MultiStemPipeline  # noqa: PLC0415

    result = MultiStemPipeline(config).generate(run_root, stage1_only=args.stage1_only)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
