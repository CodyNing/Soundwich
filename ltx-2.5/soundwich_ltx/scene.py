"""Scene YAML schema: entities, stems, timelines, and carrier prompts compiled into runtime configs."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from soundwich_ltx.config import MultiStemConfig


class SceneError(ValueError):
    """Raised when a scene cannot be compiled."""


MODEL_FILES: dict[str, str] = {
    "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    "stage2_transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upscaler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

DEFAULT_GENERATION: dict[str, Any] = {
    "seed": 42,
    "width": 768,
    "height": 448,
    "frames": 241,
    "frame_rate": 24.0,
    "steps": 40,
    "video_cfg": 3.0,
    "audio_cfg": 7.0,
    "video_rescale": 0.0,
    "audio_rescale": 0.0,
}

DEFAULT_METHOD_SETTINGS: dict[str, Any] = {
    "stage1": {
        # Carrier activation: `activation_peak_strength` for the first
        # `activation_peak_fraction` of denoising, then `activation_strength`.
        "activation_strength": 0.10,
        "activation_peak_strength": 0.50,
        "activation_peak_fraction": 0.10,
        "value_scale": 0.80,
        "outside_suppression": 0.25,
        "outside_silence_blend": 0.60,
        # Window feathering as a fraction of the scene's `duration_seconds`.
        "window_feather": 0.02,
        # Stems without an explicit negative use the other stems' sources.
        "auto_negative": True,
        # Scene lane: gather the real stems into it for the first
        # `scene_gather_fraction` of denoising.
        "scene_coupling": True,
        "scene_gather_fraction": 0.50,
        "scene_gather_strength": 1.0,
        "real_to_scene_strength": 0.0,
        # How the gathered stems are summed: raw_sum or rms_sum (RMS-balanced).
        "scene_aggregation": "raw_sum",
    },
    "stage2": {
        "start_sigma": 0.95,
        "audio_cfg": 1.0,
        "suppression": True,
        # Scene lane: broadcast it into every real stem's self-attention.
        "scene_coupling": True,
        "scene_to_real_strength": 1.0,
        "real_to_scene_strength": 0.10,
        # A2V routed by SAM ownership; V2A by SAM and the stem timeline.
        "a2v_hard_route": True,
        "v2a_hard_route": True,
        "a2v_active_gain": 1.0,
        "mask_threshold": 0.05,
    },
}

# Carrier recordings are ordinary single-branch AV generations. Unless a scene
# overrides them, they use these settings with the scene's step count.
DEFAULT_CARRIER_RECORDING: dict[str, Any] = {
    "width": 768,
    "height": 448,
    "frames": 241,
    "frame_rate": 24.0,
    "video_cfg": 3.0,
    "audio_cfg": 7.0,
    "video_rescale": 0.0,
    "audio_rescale": 0.0,
    "quantile": 0.70,
    "a2v_enabled": True,
    "v2a_enabled": True,
}

# Method-level suppression carriers shared by every scene. Stage 1 replays a
# 40-step dev-model carrier; Stage 2 replays an 8-step distilled-model carrier.
STAGE1_SUPPRESSION_CARRIER: dict[str, Any] = {
    "positive": (
        "A uniform pure black frame remains perfectly still. Audio: digital silence, inaudible noise floor"
    ),
    "negative": (
        "camera cuts, text, blur, low quality. Audio: speech, music, singing, loud sound, random noise"
    ),
    "seed": 101,
    "audio_rescale": 0.7,
}
STAGE2_SUPPRESSION_CARRIER: dict[str, Any] = {
    "positive": (
        "A uniform pure black frame fills the entire image from edge to edge. The featureless black frame "
        "remains perfectly still and unchanged throughout the full clip. Audio: digital silence throughout "
        "the full clip, zero spoken words, zero narration, zero music, zero sound effects, zero ambience, "
        "inaudible noise floor."
    ),
    "negative": (
        "people, faces, mouths, objects, text, light, motion, camera movement, human voice, woman speaking, "
        "man speaking, narration, dialogue, spoken words, singing, music, sound effects, ambience, room tone, "
        "hum, hiss, buzz"
    ),
    "seed": 42,
    "transformer": "stage2_transformer",
    "schedule": "distilled",
    "steps": 8,
}

DEFAULT_SAM3: dict[str, Any] = {
    "predictor_kind": "sam3.1_multiplex",
    "device": "cuda",
    "frame_stride": 8,
    "max_frames": 32,
    "prompt_frame_index": 0,
    "score_threshold": 0.50,
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _peak_schedule(*, base_strength: float, peak_strength: float, peak_fraction: float) -> list[dict[str, float]]:
    """Constant `peak_strength` up to `peak_fraction` of denoising, then `base_strength`."""
    if peak_fraction <= 0:
        return []
    if peak_fraction >= 1:
        return [{"end_progress": 1.0, "start_strength": peak_strength, "end_strength": peak_strength}]
    return [
        {"end_progress": peak_fraction, "start_strength": peak_strength, "end_strength": peak_strength},
        {"end_progress": 1.0, "start_strength": base_strength, "end_strength": base_strength},
    ]


def _gather_schedule(fraction: float) -> list[dict[str, float]]:
    """Scene gathering is on while denoising progress is within `fraction`."""
    if fraction <= 0:
        return [{"end_progress": 1.0, "start_strength": 0.0, "end_strength": 0.0}]
    if fraction >= 1:
        return []
    return [
        {"end_progress": fraction, "start_strength": 1.0, "end_strength": 1.0},
        {"end_progress": 1.0, "start_strength": 0.0, "end_strength": 0.0},
    ]


@dataclass(frozen=True)
class CarrierSpec:
    """Everything that determines a recorded carrier; its hash names the cache entry."""

    name: str
    role: str
    raw: dict[str, Any]

    @property
    def key(self) -> str:
        payload = json.dumps({"role": self.role, **self.raw}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def directory(self, cache_root: Path) -> Path:
        return cache_root / f"{self.name}-{self.key}"


def carrier_spec(
    name: str,
    role: str,
    raw: dict[str, Any],
    *,
    steps: int,
) -> CarrierSpec:
    if not isinstance(raw, dict) or not _clean(raw.get("positive")):
        raise SceneError(f"carrier {name!r} needs a positive prompt")
    unknown = set(raw) - {
        "positive",
        "negative",
        "seed",
        "transformer",
        "schedule",
        "steps",
        *DEFAULT_CARRIER_RECORDING,
    }
    if unknown:
        raise SceneError(f"carrier {name!r} has unknown keys: {sorted(unknown)}")
    merged = {
        **DEFAULT_CARRIER_RECORDING,
        "transformer": "transformer",
        "schedule": "standard",
        "steps": steps,
        **raw,
    }
    merged["positive"] = str(merged["positive"]).strip()
    merged["negative"] = str(merged.get("negative") or "").strip()
    merged["seed"] = int(merged.get("seed", 42))
    if merged["transformer"] not in {"transformer", "stage2_transformer"}:
        raise SceneError(f"carrier {name!r}: transformer must be transformer or stage2_transformer")
    return CarrierSpec(name=name, role=role, raw=merged)


def carrier_record_config(spec: CarrierSpec, models: dict[str, str]) -> dict[str, Any]:
    """Runtime config that records one carrier."""
    raw = spec.raw
    return {
        "id": spec.name,
        "task": "record_carrier",
        "model": {
            "transformer": models[raw["transformer"]],
            "text_encoder": models["text_encoder"],
            "video_vae": models["video_vae"],
            "audio_vae": models["audio_vae"],
        },
        "generation": {
            "seed": raw["seed"],
            "width": raw["width"],
            "height": raw["height"],
            "frames": raw["frames"],
            "frame_rate": raw["frame_rate"],
            "steps": raw["steps"],
            "schedule": raw["schedule"],
        },
        "guidance": {
            "video_cfg": raw["video_cfg"],
            "audio_cfg": raw["audio_cfg"],
            "video_rescale": raw["video_rescale"],
            "audio_rescale": raw["audio_rescale"],
        },
        "carrier": {
            "id": spec.name,
            "role": spec.role,
            "group": spec.name,
            "positive": raw["positive"],
            "negative": raw["negative"],
            "quantile": raw["quantile"],
            "a2v_enabled": raw["a2v_enabled"],
            "v2a_enabled": raw["v2a_enabled"],
        },
    }


@dataclass(frozen=True)
class CompiledScene:
    raw: dict[str, Any]
    activation_carriers: dict[str, CarrierSpec]
    stage1_suppression: CarrierSpec
    stage2_suppression: CarrierSpec | None

    def carrier_specs(self) -> list[CarrierSpec]:
        specs = [*self.activation_carriers.values(), self.stage1_suppression]
        if self.stage2_suppression is not None:
            specs.append(self.stage2_suppression)
        return specs

    def config(self) -> MultiStemConfig:
        return MultiStemConfig.from_dict(self.raw)


def load_scene(path: str | Path) -> dict[str, Any]:
    scene = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise SceneError(f"{path}: a scene must be a mapping")
    return scene


def validate_scene(scene: dict[str, Any]) -> list[str]:  # noqa: PLR0912
    """Return validation errors; an empty list means the scene can compile."""
    errors: list[str] = []
    if not _clean(scene.get("id")):
        errors.append("id is required")
    visual = scene.get("visual")
    if not isinstance(visual, dict) or not _clean(visual.get("positive")):
        errors.append("visual.positive is required")
    elif "allowed_entities" in visual and not isinstance(visual.get("allowed_entities"), list):
        errors.append("visual.allowed_entities must be a list")
    groups = scene.get("entity_groups") or {}
    if not isinstance(groups, dict):
        errors.append("entity_groups must be a mapping")
        groups = {}
    for group_id, group in groups.items():
        if not isinstance(group, dict) or not isinstance(group.get("carrier"), dict):
            errors.append(f"entity_groups.{group_id}.carrier must be a mapping of carrier prompts")
    entities = scene.get("sound_entities")
    entity_ids: set[str] = set()
    if not isinstance(entities, list) or not entities:
        errors.append("sound_entities must be a non-empty list")
    else:
        for index, entity in enumerate(entities):
            entity_id = _clean(entity.get("id")) if isinstance(entity, dict) else ""
            if not entity_id:
                errors.append(f"sound_entities[{index}].id is required")
                continue
            if entity_id in entity_ids:
                errors.append(f"duplicate sound entity id: {entity_id}")
            entity_ids.add(entity_id)
            group_id = _clean(entity.get("group"))
            if not group_id or group_id not in groups:
                errors.append(f"sound entity {entity_id} must reference one of entity_groups")
    stems = scene.get("stems")
    if not isinstance(stems, list) or not stems:
        errors.append("stems must be a non-empty list")
        return errors
    stem_ids: set[str] = set()
    for index, stem in enumerate(stems):
        if not isinstance(stem, dict):
            errors.append(f"stems[{index}] must be a mapping")
            continue
        stem_id = str(stem.get("id") or "")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", stem_id) is None:
            errors.append(f"stems[{index}].id must use only letters, numbers, underscore, hyphen, or dot")
        elif stem_id in stem_ids:
            errors.append(f"duplicate stem id: {stem_id}")
        stem_ids.add(stem_id)
        if _clean(stem.get("entity")) not in entity_ids:
            errors.append(f"stems[{index}].entity must reference a sound entity")
        if not _clean(stem.get("positive")):
            errors.append(f"stems[{index}].positive is required")
        if not 0.0 <= float(stem.get("volume", 1.0)) <= 1.0:
            errors.append(f"stems[{index}].volume must be between 0 and 1")
        windows = stem.get("windows") or []
        if not isinstance(windows, list):
            errors.append(f"stems[{index}].windows must be a list")
            continue
        for w_index, window in enumerate(windows):
            try:
                start, end = float(window["start"]), float(window["end"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"stems[{index}].windows[{w_index}] needs numeric start/end")
                continue
            if start < 0 or end <= start:
                errors.append(f"stems[{index}].windows[{w_index}] must satisfy 0 <= start < end")
    return errors


def compile_scene(  # noqa: PLR0915
    scene: dict[str, Any],
    *,
    models_dir: Path,
    carrier_cache: Path,
    seed: int | None = None,
) -> CompiledScene:
    """Build the runtime generation config and the carrier recordings a scene needs."""
    errors = validate_scene(scene)
    if errors:
        raise SceneError(f"scene {scene.get('id', '<unknown>')} failed validation: " + "; ".join(errors))

    generation = {**DEFAULT_GENERATION, **(scene.get("generation") or {})}
    if seed is not None:
        generation["seed"] = int(seed)
    settings = _deep_merge(DEFAULT_METHOD_SETTINGS, scene.get("method_settings") or {})
    for stage_name, stage_settings in settings.items():
        unknown = set(stage_settings) - set(DEFAULT_METHOD_SETTINGS.get(stage_name, {}))
        if unknown:
            raise SceneError(f"method_settings.{stage_name} has unknown keys: {sorted(unknown)}")
    stage1 = settings["stage1"]
    stage2 = settings["stage2"]
    models = {key: str(models_dir / relative) for key, relative in MODEL_FILES.items()}
    steps = int(generation["steps"])

    visual = scene["visual"]
    visual_prompt = _clean(visual.get("positive"))
    allowed_entities = sorted({str(item) for item in visual.get("allowed_entities", [])})
    if allowed_entities:
        visual_prompt = f"{visual_prompt} Visible sound entities: {', '.join(allowed_entities)}."
    audio_globals = scene.get("audio_globals") or {}
    groups = scene.get("entity_groups") or {}
    entities = {str(entity["id"]): entity for entity in scene["sound_entities"]}

    stems: list[dict[str, Any]] = []
    for stem in scene["stems"]:
        entity = entities[str(stem["entity"])]
        item: dict[str, Any] = {
            "id": str(stem["id"]),
            "positive": str(stem.get("positive") or ""),
            "negative": str(stem.get("negative") or ""),
            "negative_source": _clean(
                stem.get("negative_source_audio") or entity.get("negative_source_audio") or stem.get("positive")
            ),
            "reference_group": _clean(entity.get("group")),
            "sam_prompt": _clean(stem.get("sam_prompt") or entity.get("sam_prompt") or entity.get("visual_entity")),
            "sam_points": deepcopy(stem.get("sam_points") or entity.get("sam_points") or []),
            "sam_point_frame": stem.get("sam_point_frame", entity.get("sam_point_frame")),
            "volume": float(stem.get("volume", 1.0)),
            "windows": deepcopy(stem.get("windows") or []),
            "auto_negative": bool(stage1["auto_negative"]),
            "scene_context_enabled": bool(stem.get("scene_context_enabled", True)),
        }
        overrides = {"activation_strength", "activation_peak_strength", "activation_peak_fraction"} & set(stem)
        if overrides:
            base_strength = float(stem.get("activation_strength", stage1["activation_strength"]))
            item["blend_strength"] = base_strength
            item["blend_schedule"] = _peak_schedule(
                base_strength=base_strength,
                peak_strength=float(stem.get("activation_peak_strength", stage1["activation_peak_strength"])),
                peak_fraction=float(stem.get("activation_peak_fraction", stage1["activation_peak_fraction"])),
            )
        stems.append(item)

    used_groups = sorted({item["reference_group"] for item in stems})
    activation_specs = {
        group: carrier_spec(group, "activation", groups[group]["carrier"], steps=steps) for group in used_groups
    }
    stage1_suppression = carrier_spec("suppression", "suppression", STAGE1_SUPPRESSION_CARRIER, steps=steps)
    stage2_suppression = (
        carrier_spec("stage2_suppression", "suppression", STAGE2_SUPPRESSION_CARRIER, steps=8)
        if stage2["suppression"]
        else None
    )

    scene_context_stems = sum(bool(item["scene_context_enabled"]) for item in stems)
    scene_lane_possible = len(stems) >= 2 and scene_context_stems > 0
    duration_seconds = float(scene.get("duration_seconds") or 10.0)
    raw = {
        "id": str(scene["id"]),
        "task": "generate",
        "model": models,
        "generation": {key: generation[key] for key in ("seed", "width", "height", "frames", "frame_rate", "steps")},
        "guidance": {key: generation[key] for key in ("video_cfg", "audio_cfg", "video_rescale", "audio_rescale")},
        "prompts": {
            "visual_positive": visual_prompt,
            "visual_negative": _clean(visual.get("negative")) or "blurry, low quality",
            "audio_global_positive": str(audio_globals.get("positive") or ""),
            "audio_global_negative": str(audio_globals.get("negative") or ""),
        },
        "carriers": {
            "activation": {group: str(spec.directory(carrier_cache)) for group, spec in activation_specs.items()},
            "suppression": str(stage1_suppression.directory(carrier_cache)),
            "blend": {
                "strength": float(stage1["activation_strength"]),
                "value_scale": float(stage1["value_scale"]),
                "outside_suppression": float(stage1["outside_suppression"]),
                "outside_silence_blend": float(stage1["outside_silence_blend"]),
                "feather_seconds": float(stage1["window_feather"]) * duration_seconds,
                "activation_schedule": _peak_schedule(
                    base_strength=float(stage1["activation_strength"]),
                    peak_strength=float(stage1["activation_peak_strength"]),
                    peak_fraction=float(stage1["activation_peak_fraction"]),
                ),
            },
        },
        "scene_coupling": {
            "enabled": bool(stage1["scene_coupling"]) and scene_lane_possible,
            "self_attention_direction": "real_to_scene",
            "scene_to_real_strength": 0.0,
            "real_to_scene_strength": float(stage1["real_to_scene_strength"]),
            "scene_self_attention_strength": float(stage1["scene_gather_strength"]),
            "scene_self_attention_schedule": _gather_schedule(float(stage1["scene_gather_fraction"])),
            "aggregation": str(stage1["scene_aggregation"]),
        },
        "stems": stems,
        "sam3": {**DEFAULT_SAM3, **(scene.get("sam") or {})},
        "stage2": {
            "enabled": True,
            "start_sigma": float(stage2["start_sigma"]),
            "audio_cfg": float(stage2["audio_cfg"]),
            "suppression_blend_enabled": stage2_suppression is not None,
            "suppression_carrier": str(stage2_suppression.directory(carrier_cache)) if stage2_suppression else "",
            "scene_coupling": {
                "enabled": bool(stage2["scene_coupling"]) and scene_lane_possible,
                "self_attention_direction": "scene_to_real",
                "scene_to_real_strength": float(stage2["scene_to_real_strength"]),
                "real_to_scene_strength": float(stage2["real_to_scene_strength"]),
            },
            "a2v_hard_route": bool(stage2["a2v_hard_route"]),
            "v2a_hard_route": bool(stage2["v2a_hard_route"]),
            "a2v_active_gain": float(stage2["a2v_active_gain"]),
            "mask_threshold": float(stage2["mask_threshold"]),
        },
    }
    return CompiledScene(
        raw=raw,
        activation_carriers=activation_specs,
        stage1_suppression=stage1_suppression,
        stage2_suppression=stage2_suppression,
    )
