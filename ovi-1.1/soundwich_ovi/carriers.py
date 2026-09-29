"""Build and cache the activation/suppression carrier banks used for blending."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .engine import OviMultiStemEngine

DEFAULT_CARRIER_SEED = 42

_REQUIRED_CARRIER_KEYS = {"id", "role", "visual_prompt", "audio_prompt", "quantile"}


def load_carrier_spec(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED_CARRIER_KEYS - data.keys())
    if missing:
        raise ValueError(f"carrier config missing: {', '.join(missing)}")
    if data["role"] not in {"activation", "suppression"}:
        raise ValueError("carrier role must be activation or suppression")
    return data


def carrier_cache_path(cache_dir: str | Path, spec: dict[str, Any], settings: dict[str, Any]) -> Path:
    """Cache file named by carrier id, step count, and a hash of everything that shapes the recording."""
    payload = json.dumps({"spec": spec, "settings": settings}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{spec['id']}_steps{settings['sample_steps']}_{digest}.pt"


def ensure_carrier_bank(
    engine: OviMultiStemEngine,
    *,
    carrier_config: str | Path,
    cache_dir: str | Path,
    sample_steps: int,
    video_frame_height_width: list[int],
    solver_name: str,
    shift: float,
    video_guidance_scale: float,
    audio_guidance_scale: float,
    slg_layer: int,
    model_settings: dict[str, Any],
) -> Path:
    """Return the cached carrier-bank path, building it with ``engine`` if missing."""
    spec = load_carrier_spec(carrier_config)
    settings = {
        "sample_steps": sample_steps,
        "video_frame_height_width": list(video_frame_height_width),
        "solver_name": solver_name,
        "shift": shift,
        "video_guidance_scale": video_guidance_scale,
        "audio_guidance_scale": audio_guidance_scale,
        "slg_layer": slg_layer,
        "model": model_settings,
    }
    cache_path = carrier_cache_path(cache_dir, spec, settings)
    if cache_path.exists():
        return cache_path

    logging.info("Building carrier bank %s (steps=%d)", spec["id"], sample_steps)
    result = engine.generate_carrier(
        carrier_id=str(spec["id"]),
        role=str(spec["role"]),
        seed=int(spec.get("seed", DEFAULT_CARRIER_SEED)),
        quantile=float(spec["quantile"]),
        visual_prompt=str(spec["visual_prompt"]),
        audio_prompt=str(spec["audio_prompt"]),
        video_negative_prompt=str(spec.get("video_negative_prompt") or ""),
        audio_negative_prompt=str(spec.get("audio_negative_prompt") or ""),
        video_frame_height_width=video_frame_height_width,
        solver_name=solver_name,
        sample_steps=sample_steps,
        shift=shift,
        video_guidance_scale=video_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        slg_layer=slg_layer,
        a2v_enabled=True,
        v2a_enabled=True,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run never leaves a partial cache entry.
    partial = cache_path.with_suffix(".partial")
    result["bank"].save(partial)
    partial.replace(cache_path)
    return cache_path
