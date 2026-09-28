"""Generate a Soundwich scene: carriers (cached) -> Stage 1 -> SAM3 masks -> Stage 2.

    python -m soundwich_ltx.generate --scene examples/neon_biology_lab.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import yaml

from soundwich_ltx.config import MultiStemConfig
from soundwich_ltx.scene import CompiledScene, carrier_record_config, compile_scene, load_scene

BACKEND_ROOT = Path(__file__).resolve().parents[1]
# Recorded carriers shipped with the code, named like cache entries (<name>-<spec hash>). The Stage-2
# suppression carrier used for the paper predates this code and cannot be re-recorded bit-exactly.
BUNDLED_CARRIERS = BACKEND_ROOT / "carriers"


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
        "--reuse-stage1",
        action="store_true",
        help="Keep the saved Stage-1 result and recompute only the SAM3 masks and Stage 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the scene and print the resolved configuration without loading any model",
    )
    return parser


def record_missing_carriers(compiled: CompiledScene, models: dict[str, str], cache_root: Path) -> None:
    from ltx_pipelines.utils.helpers import cleanup_memory  # noqa: PLC0415

    from soundwich_ltx.pipeline import MultiStemPipeline  # noqa: PLC0415

    for spec in compiled.carrier_specs():
        destination = spec.directory(cache_root)
        if (destination / "carrier.json").is_file():
            logging.info("carrier %s: cached at %s", spec.name, destination)
            continue
        bundled = BUNDLED_CARRIERS / destination.name
        if (bundled / "carrier.json").is_file():
            logging.info("carrier %s: copying the bundled recording into %s", spec.name, destination)
            shutil.copytree(bundled, destination)
            continue
        logging.info("carrier %s: recording into %s", spec.name, destination)
        config = MultiStemConfig.from_dict(carrier_record_config(spec, models))
        pipeline = MultiStemPipeline(config)
        try:
            pipeline.record_carrier(destination)
        finally:
            del pipeline
            cleanup_memory()


# Scene fields that may change when reusing a saved Stage 1: SAM prompts/clicks and Stage-2 settings.
_REUSE_MUTABLE_KEYS = {"sam_prompt", "sam_points", "sam_point_frame"}


def _stage1_view(scene: dict) -> dict:
    view = {key: value for key, value in scene.items() if key not in {"sound_entities", "stems", "method_settings"}}
    for key in ("sound_entities", "stems"):
        view[key] = [
            {k: v for k, v in item.items() if k not in _REUSE_MUTABLE_KEYS} for item in scene.get(key) or []
        ]
    view["method_settings"] = {k: v for k, v in (scene.get("method_settings") or {}).items() if k != "stage2"}
    return view


def _check_stage1_reuse(run_root: Path, scene: dict, stem_ids: list[str]) -> None:
    """Refuse to reuse a saved Stage 1 that was generated from different stems or Stage-1 settings."""
    manifest_path = run_root / "stage1_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no saved Stage-1 result to reuse in {run_root}")
    saved_ids = json.loads(manifest_path.read_text(encoding="utf-8")).get("stem_ids")
    if saved_ids != stem_ids:
        raise ValueError(f"--reuse-stage1: saved stems {saved_ids} do not match the scene's stems {stem_ids}")
    saved_scene_path = run_root / "scene.yaml"
    if saved_scene_path.is_file():
        saved = yaml.safe_load(saved_scene_path.read_text(encoding="utf-8"))
        if _stage1_view(saved) != _stage1_view(scene):
            raise ValueError(
                "--reuse-stage1: the scene changed beyond SAM prompts/clicks and Stage-2 settings; "
                "rerun Stage 1 (drop --reuse-stage1) or use a new --output-dir"
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    carrier_cache = (args.carrier_cache or output_dir / "carriers").expanduser().resolve()
    scene = load_scene(args.scene)
    compiled = compile_scene(
        scene,
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
    if args.stage1_only and args.reuse_stage1:
        raise ValueError("--stage1-only and --reuse-stage1 are mutually exclusive")
    saved_scene = {**scene, "generation": {**(scene.get("generation") or {}), "seed": config.generation.seed}}
    if args.reuse_stage1:
        _check_stage1_reuse(run_root, saved_scene, [stem.id for stem in config.stems])
    record_missing_carriers(compiled, dict(config.model.__dict__), carrier_cache)
    # Keep the scene (with the effective seed) next to the run; `soundwich_ltx.edit` recompiles it.
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "scene.yaml").write_text(yaml.safe_dump(saved_scene, sort_keys=False, allow_unicode=True), encoding="utf-8")

    from soundwich_ltx.pipeline import MultiStemPipeline  # noqa: PLC0415

    result = MultiStemPipeline(config).generate(
        run_root, stage1_only=args.stage1_only, reuse_stage1=args.reuse_stage1
    )
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
