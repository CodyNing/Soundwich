"""Edit the stems of a finished run: retime and/or replace stems, then refine the video around the fixed audio.

    python -m soundwich_ltx.edit --run outputs/neon_biology_lab_seed1096 --edit examples/edits/neon_biology_lab_swap_turns.yaml

Retiming moves each window of a stem's saved Stage-2 audio latent, shifting the surrounding latent frames so
the clip length is unchanged. Replacement regenerates one stem with Stage 1 (new source prompt, original visual
prompt and clip length); its audio latent replaces that stem while the others keep their saved Stage-2 latents.
The video is then refined again from the saved Stage-1 video and SAM masks with the Stage-2 schedule: only the
video is denoised, the edited audio latents stay clean and fixed, routing follows the edited windows, the scene
lane is off, and Stage-2 suppression replay stays active outside the edited windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from soundwich_ltx.config import MultiStemConfig
from soundwich_ltx.generate import BACKEND_ROOT, record_missing_carriers
from soundwich_ltx.scene import CompiledScene, compile_scene, load_scene
from soundwich_ltx.timeline_edit import (
    apply_replacement,
    create_timeline,
    move_clip,
    remapped_clips,
    reorder_audio_latents_by_segments,
    scene_with_timeline,
    stem_clip_ids,
)

RUN_FILES = ("stage1_latents.pt", "stage2_audio_latent.pt", "sam_outputs/stage2_a2v_mask.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retime or replace stems of a finished Soundwich run")
    parser.add_argument("--run", type=Path, required=True, help="Run folder written by soundwich_ltx.generate")
    parser.add_argument("--edit", type=Path, required=True, help="Edit YAML")
    parser.add_argument("--scene", type=Path, help="Scene YAML of the run (default: <run>/scene.yaml)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("SOUNDWICH_LTX_MODELS", BACKEND_ROOT / "models" / "ltx-2.5")),
        help="LTX-2.5 checkpoint folder (default: $SOUNDWICH_LTX_MODELS or models/ltx-2.5)",
    )
    parser.add_argument("--carrier-cache", type=Path, help="Carrier folder (default: <run>/../carriers)")
    parser.add_argument("--dry-run", action="store_true", help="Print the edit plan without loading any model")
    return parser


def load_edit(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not str(spec.get("id") or "").strip():
        raise ValueError(f"{path}: an edit needs an id")
    stems = spec.get("stems")
    if not isinstance(stems, dict) or not stems:
        raise ValueError(f"{path}: stems must map stem ids to retime/replace edits")
    for stem_id, item in stems.items():
        if not isinstance(item, dict) or not set(item) or set(item) - {"retime", "replace"}:
            raise ValueError(f"{path}: stems.{stem_id} takes only retime and/or replace")
        if "retime" in item and not isinstance(item["retime"], list):
            raise ValueError(f"{path}: stems.{stem_id}.retime must list one new start (seconds or null) per window")
        replace = item.get("replace")
        if replace is not None and (
            not isinstance(replace, dict) or set(replace) - {"positive", "negative", "windows", "seed"}
        ):
            raise ValueError(f"{path}: stems.{stem_id}.replace takes positive, negative, windows, seed")
    return spec


def plan_edit(scene: dict[str, Any], spec: dict[str, Any], *, total_frames: int, frame_rate: float) -> dict[str, Any]:
    """Apply replacements to the scene, then retime windows on the per-stem timelines."""
    scene_ids = [str(stem["id"]) for stem in scene["stems"]]
    unknown = sorted(set(spec["stems"]) - set(scene_ids))
    if unknown:
        raise ValueError(f"edit names unknown stems: {unknown}")
    replaced = deepcopy(scene)
    takes: dict[str, dict[str, Any]] = {}
    for stem_id, item in spec["stems"].items():
        if item.get("replace") is None:
            continue
        index = scene_ids.index(stem_id)
        stem, seed = apply_replacement(replaced["stems"][index], item["replace"], replaced)
        replaced["stems"][index] = stem
        takes[stem_id] = {"stem": stem, "seed": seed}
    timeline = create_timeline(replaced, total_frames=total_frames, frame_rate=frame_rate)
    for stem_id, item in spec["stems"].items():
        starts = item.get("retime") or []
        clip_ids = stem_clip_ids(timeline, stem_id)
        if len(starts) > len(clip_ids):
            raise ValueError(f"{stem_id} has {len(clip_ids)} windows but retime lists {len(starts)} starts")
        for clip_id, start in zip(clip_ids, starts):
            if start is not None:
                timeline = move_clip(timeline, clip_id, round(float(start) * frame_rate))
    return {"scene": scene_with_timeline(replaced, timeline), "replaced_scene": replaced, "timeline": timeline, "takes": takes}


def take_config(
    scene: dict[str, Any], stem: dict[str, Any], seed: int, *, models_dir: Path, carrier_cache: Path
) -> tuple[CompiledScene, MultiStemConfig]:
    """Stage-1-only config that regenerates one stem at half resolution."""
    take_scene = deepcopy(scene)
    take_scene["stems"] = [stem]
    compiled = compile_scene(take_scene, models_dir=models_dir, carrier_cache=carrier_cache, seed=seed)
    raw = deepcopy(compiled.raw)
    raw["generation"]["width"] //= 2
    raw["generation"]["height"] //= 2
    raw["stage2"]["enabled"] = False
    raw["scene_coupling"] = {"enabled": False}
    return compiled, MultiStemConfig.from_dict(raw)


def refinement_config(
    scene: dict[str, Any], seed: int | None, *, models_dir: Path, carrier_cache: Path
) -> tuple[CompiledScene, MultiStemConfig]:
    """Stage-2 config for video-only refinement around fixed audio."""
    refine_scene = deepcopy(scene)
    stage2 = refine_scene.setdefault("method_settings", {}).setdefault("stage2", {})
    stage2.update({"audio_cfg": 1.0, "suppression": True, "scene_coupling": False})
    compiled = compile_scene(refine_scene, models_dir=models_dir, carrier_cache=carrier_cache, seed=seed)
    raw = deepcopy(compiled.raw)
    raw["stage2"]["freeze_audio"] = True
    return compiled, MultiStemConfig.from_dict(raw)


def prepare_audio_latents(
    base: torch.Tensor, stem_ids: list[str], timeline: dict[str, Any], takes: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Swap in replacement latents, then permute every stem with windows along its edited timeline."""
    prepared = []
    for index, stem_id in enumerate(stem_ids):
        latent = base[index : index + 1].detach().cpu()
        if stem_id in takes:
            take = takes[stem_id].detach().cpu()
            if take.shape != latent.shape or take.dtype != latent.dtype:
                raise ValueError(
                    f"replacement latent for {stem_id} is {tuple(take.shape)}/{take.dtype}, "
                    f"expected {tuple(latent.shape)}/{latent.dtype}"
                )
            latent = take
        stem_timeline = timeline["stem_timelines"].get(stem_id)
        if stem_timeline:
            latent = reorder_audio_latents_by_segments(
                latent, stem_timeline["segments"], stem_timeline["segment_order"], int(timeline["total_frames"])
            )
        prepared.append(latent)
    result = torch.cat(prepared, dim=0)
    if result.shape != base.shape:
        raise ValueError(f"edited audio latents changed shape from {tuple(base.shape)} to {tuple(result.shape)}")
    return result


def _windows(scene: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {
        str(stem["id"]): [[round(float(w["start"]), 3), round(float(w["end"]), 3)] for w in stem.get("windows") or []]
        for stem in scene["stems"]
    }


def _missing_carriers(compiled: list[CompiledScene], cache_root: Path) -> list[str]:
    return sorted(
        {
            str(spec.directory(cache_root))
            for item in compiled
            for spec in item.carrier_specs()
            if not (spec.directory(cache_root) / "carrier.json").is_file()
        }
    )


def main() -> None:  # noqa: PLR0915
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    run_root = args.run.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    carrier_cache = (args.carrier_cache or run_root.parent / "carriers").expanduser().resolve()
    missing_files = [str(run_root / name) for name in RUN_FILES if not (run_root / name).is_file()]
    if missing_files:
        raise FileNotFoundError("not a completed Soundwich run:\n" + "\n".join(f"  - {path}" for path in missing_files))
    scene = load_scene(args.scene or run_root / "scene.yaml")
    spec = load_edit(args.edit)
    generation = compile_scene(scene, models_dir=models_dir, carrier_cache=carrier_cache).config().generation
    plan = plan_edit(scene, spec, total_frames=generation.frames, frame_rate=generation.frame_rate)
    refine_compiled, refine = refinement_config(
        plan["scene"], spec.get("seed"), models_dir=models_dir, carrier_cache=carrier_cache
    )
    takes = {
        stem_id: take_config(
            plan["replaced_scene"], item["stem"], item["seed"], models_dir=models_dir, carrier_cache=carrier_cache
        )
        for stem_id, item in plan["takes"].items()
    }
    edit_root = run_root / "edits" / str(spec["id"])
    all_compiled = [refine_compiled, *(compiled for compiled, _config in takes.values())]

    if args.dry_run:
        payload = {
            "edit_directory": str(edit_root),
            "windows_before": _windows(scene),
            "windows_after": _windows(plan["scene"]),
            "moved_windows": [
                {key: clip[key] for key in ("stem_id", "text", "source_start_frame", "destination_start_frame")}
                for clip in remapped_clips(plan["timeline"])
                if clip["source_start_frame"] != clip["destination_start_frame"]
            ],
            "replacements": {
                stem_id: {"stage1": config.summary()["stems"][0], "seed": config.generation.seed}
                for stem_id, (_compiled, config) in takes.items()
            },
            "refinement": {"seed": refine.generation.seed, "stage2": refine.summary()["stage2"]},
            "missing_model_files": [str(path) for path in refine.missing_model_paths()],
            "missing_carriers": _missing_carriers(all_compiled, carrier_cache),
        }
        print(json.dumps(payload, indent=2))  # noqa: T201
        return

    missing = refine.missing_model_paths()
    if missing:
        raise FileNotFoundError(
            "missing LTX-2.5 checkpoints (see README):\n" + "\n".join(f"  - {path}" for path in missing)
        )
    if edit_root.exists():
        raise FileExistsError(f"{edit_root} exists; choose a new edit id")
    from ltx_pipelines.utils.helpers import cleanup_memory  # noqa: PLC0415

    from soundwich_ltx.pipeline import MultiStemPipeline  # noqa: PLC0415

    for compiled in all_compiled:
        record_missing_carriers(compiled, dict(refine.model.__dict__), carrier_cache)
    edit_root.mkdir(parents=True)
    (edit_root / "edit.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (edit_root / "scene.yaml").write_text(
        yaml.safe_dump(plan["scene"], sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (edit_root / "timeline.json").write_text(json.dumps(plan["timeline"], indent=2) + "\n", encoding="utf-8")

    take_latents: dict[str, torch.Tensor] = {}
    for stem_id, (_compiled, config) in takes.items():
        logging.info("replacing %s with a new Stage-1 take (seed %d)", stem_id, config.generation.seed)
        pipeline = MultiStemPipeline(config)
        try:
            result = pipeline.run_stage1(edit_root / "takes" / stem_id)
        finally:
            del pipeline
            cleanup_memory()
        take_latents[stem_id] = torch.load(result["stage1_latents"], map_location="cpu", weights_only=True)["audio"][:1]

    base = torch.load(run_root / "stage2_audio_latent.pt", map_location="cpu", weights_only=True)
    stem_ids = [stem.id for stem in refine.stems]
    if [str(value) for value in base["stem_ids"]] != stem_ids:
        raise ValueError(f"saved Stage-2 audio has stems {base['stem_ids']}, the scene has {stem_ids}")
    audio = prepare_audio_latents(base["audio"], stem_ids, plan["timeline"], take_latents)
    torch.save({"audio": audio, "stem_ids": stem_ids}, edit_root / "audio_latents.pt")

    (edit_root / "run.json").write_text(json.dumps(refine.summary(), indent=2) + "\n", encoding="utf-8")
    result = MultiStemPipeline(refine).run_stage2(run_root, audio_latents=audio, output_root=edit_root)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
