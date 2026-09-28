"""Resolved runtime configuration for Soundwich on LTX-2.5."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Task = Literal["generate", "record_carrier"]
CarrierRole = Literal["activation", "suppression"]
SceneSelfAttentionDirection = Literal["real_to_scene", "scene_to_real"]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonempty(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


@dataclass(frozen=True)
class ModelConfig:
    transformer: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    stage2_transformer: str = ""
    spatial_upscaler: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        return cls(
            transformer=_nonempty(raw.get("transformer"), "model.transformer"),
            text_encoder=_nonempty(raw.get("text_encoder"), "model.text_encoder"),
            video_vae=_nonempty(raw.get("video_vae"), "model.video_vae"),
            audio_vae=_nonempty(raw.get("audio_vae"), "model.audio_vae"),
            stage2_transformer=str(raw.get("stage2_transformer") or "").strip(),
            spatial_upscaler=str(raw.get("spatial_upscaler") or "").strip(),
        )

    def paths(self) -> tuple[str, ...]:
        required = (self.transformer, self.text_encoder, self.video_vae, self.audio_vae)
        if self.spatial_upscaler:
            required = (*required, self.spatial_upscaler)
        if self.stage2_transformer:
            required = (*required, self.stage2_transformer)
        return required


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    width: int = 768
    height: int = 448
    frames: int = 241
    frame_rate: float = 24.0
    steps: int = 40
    schedule: Literal["standard", "distilled"] = "standard"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerationConfig":
        schedule = str(raw.get("schedule", "standard")).strip().lower()
        if schedule not in {"standard", "distilled"}:
            raise ValueError("generation.schedule must be standard or distilled")
        value = cls(
            seed=int(raw.get("seed", 42)),
            width=int(raw.get("width", 768)),
            height=int(raw.get("height", 448)),
            frames=int(raw.get("frames", 241)),
            frame_rate=float(raw.get("frame_rate", 24.0)),
            steps=int(raw.get("steps", 40)),
            schedule=schedule,  # type: ignore[arg-type]
        )
        if value.width <= 0 or value.height <= 0 or value.frames <= 0 or value.frame_rate <= 0 or value.steps <= 0:
            raise ValueError("generation dimensions, frame rate, and steps must be positive")
        if (value.frames - 1) % 8 != 0:
            raise ValueError("generation.frames must follow the LTX temporal grid 8k+1")
        if value.schedule == "distilled" and value.steps != 8:
            raise ValueError("the LTX-2.5 distilled schedule has exactly 8 denoising steps")
        return value

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.frame_rate


@dataclass(frozen=True)
class GuidanceConfig:
    video_cfg: float = 3.0
    audio_cfg: float = 7.0
    video_rescale: float = 0.0
    audio_rescale: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuidanceConfig":
        value = cls(
            video_cfg=float(raw.get("video_cfg", 3.0)),
            audio_cfg=float(raw.get("audio_cfg", 7.0)),
            video_rescale=float(raw.get("video_rescale", 0.0)),
            audio_rescale=float(raw.get("audio_rescale", 0.0)),
        )
        if value.video_cfg < 1 or value.audio_cfg < 1:
            raise ValueError("CFG scales must be at least 1")
        if not 0 <= value.video_rescale <= 1 or not 0 <= value.audio_rescale <= 1:
            raise ValueError("guidance rescale values must be in [0, 1]")
        return value


@dataclass(frozen=True)
class ScheduleSegment:
    end_progress: float
    start_strength: float
    end_strength: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScheduleSegment":
        value = cls(
            end_progress=float(raw["end_progress"]),
            start_strength=float(raw["start_strength"]),
            end_strength=float(raw["end_strength"]),
        )
        if not 0 < value.end_progress <= 1:
            raise ValueError("schedule end_progress must be in (0, 1]")
        if not 0 <= value.start_strength <= 1 or not 0 <= value.end_strength <= 1:
            raise ValueError("schedule strengths must be in [0, 1]")
        return value


@dataclass(frozen=True)
class StrengthSchedule:
    """Piecewise-linear strength over denoising progress in [0, 1]."""

    segments: tuple[ScheduleSegment, ...] = ()

    @classmethod
    def from_list(cls, raw: object) -> "StrengthSchedule":
        if not raw:
            return cls()
        if not isinstance(raw, list):
            raise ValueError("a strength schedule must be a list")
        segments = tuple(ScheduleSegment.from_dict(_mapping(item, "schedule segment")) for item in raw)
        previous_end = 0.0
        for segment in segments:
            if segment.end_progress <= previous_end:
                raise ValueError("schedule end_progress values must increase")
            previous_end = segment.end_progress
        if segments[-1].end_progress != 1.0:
            raise ValueError("schedule must end at progress 1.0")
        return cls(segments=segments)

    def strength_at(self, progress: float, *, default: float) -> float:
        if not self.segments:
            return default
        position = min(max(float(progress), 0.0), 1.0)
        start_progress = 0.0
        for segment in self.segments:
            if position <= segment.end_progress:
                width = segment.end_progress - start_progress
                amount = 0.0 if width <= 0 else (position - start_progress) / width
                return segment.start_strength + amount * (segment.end_strength - segment.start_strength)
            start_progress = segment.end_progress
        return self.segments[-1].end_strength

    def summary(self) -> list[dict[str, float]]:
        return [
            {
                "end_progress": segment.end_progress,
                "start_strength": segment.start_strength,
                "end_strength": segment.end_strength,
            }
            for segment in self.segments
        ]


@dataclass(frozen=True)
class BlendConfig:
    """Carrier replay inside and outside each stem's timeline windows."""

    strength: float = 0.10
    value_scale: float = 0.80
    outside_suppression: float = 0.25
    outside_silence_blend: float = 0.60
    feather_seconds: float = 0.20
    rms_scale_limit: float = 4.0
    activation_schedule: StrengthSchedule = field(default_factory=StrengthSchedule)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BlendConfig":
        value = cls(
            strength=float(raw.get("strength", 0.10)),
            value_scale=float(raw.get("value_scale", 0.80)),
            outside_suppression=float(raw.get("outside_suppression", 0.25)),
            outside_silence_blend=float(raw.get("outside_silence_blend", 0.60)),
            feather_seconds=float(raw.get("feather_seconds", 0.20)),
            rms_scale_limit=float(raw.get("rms_scale_limit", 4.0)),
            activation_schedule=StrengthSchedule.from_list(raw.get("activation_schedule")),
        )
        for name in ("strength", "outside_suppression", "outside_silence_blend"):
            if not 0 <= getattr(value, name) <= 1:
                raise ValueError(f"blend.{name} must be in [0, 1]")
        if value.value_scale < 0 or value.feather_seconds < 0 or value.rms_scale_limit <= 0:
            raise ValueError("blend value scale and feather must be non-negative; RMS limit must be positive")
        return value

    def activation_strength(self, *, step: int, total_steps: int) -> float:
        progress = step / max(total_steps - 1, 1)
        return self.activation_schedule.strength_at(progress, default=self.strength)


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    text: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, duration: float) -> "Window":
        value = cls(start=float(raw["start"]), end=float(raw["end"]), text=str(raw.get("text") or "").strip())
        if value.start < 0 or value.end <= value.start or value.end > duration:
            raise ValueError(f"timeline window [{value.start}, {value.end}] must stay inside 0-{duration:.3f}s")
        return value


@dataclass(frozen=True)
class StemConfig:
    id: str
    positive: str
    negative: str
    negative_source: str
    reference_group: str
    sam_prompt: str = ""
    windows: tuple[Window, ...] = ()
    volume: float = 1.0
    blend_strength: float | None = None
    blend_schedule: StrengthSchedule | None = None
    auto_negative: bool = True
    scene_context_enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, duration: float) -> "StemConfig":
        windows = tuple(
            Window.from_dict(_mapping(item, "stem window"), duration=duration) for item in raw.get("windows", [])
        )
        volume = float(raw.get("volume", 1.0))
        if not 0 <= volume <= 10:
            raise ValueError("stem volume must be in [0, 10]")
        blend_strength = raw.get("blend_strength")
        if blend_strength is not None and not 0 <= float(blend_strength) <= 1:
            raise ValueError("stem blend_strength must be in [0, 1]")
        return cls(
            id=_nonempty(raw.get("id"), "stem.id"),
            positive=_nonempty(raw.get("positive"), "stem.positive"),
            negative=str(raw.get("negative") or "").strip(),
            negative_source=str(raw.get("negative_source") or raw.get("positive") or "").strip(),
            reference_group=_nonempty(raw.get("reference_group", "general"), "stem.reference_group"),
            sam_prompt=str(raw.get("sam_prompt") or "").strip(),
            windows=windows,
            volume=volume,
            blend_strength=float(blend_strength) if blend_strength is not None else None,
            blend_schedule=StrengthSchedule.from_list(raw["blend_schedule"]) if "blend_schedule" in raw else None,
            auto_negative=bool(raw.get("auto_negative", True)),
            scene_context_enabled=bool(raw.get("scene_context_enabled", True)),
        )


@dataclass(frozen=True)
class CarrierReplayConfig:
    activation: dict[str, str] = field(default_factory=dict)
    suppression: str = ""
    blend: BlendConfig = field(default_factory=BlendConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CarrierReplayConfig":
        activation_raw = _mapping(raw.get("activation", {}), "carriers.activation")
        activation = {str(key): _nonempty(value, f"carriers.activation.{key}") for key, value in activation_raw.items()}
        return cls(
            activation=activation,
            suppression=str(raw.get("suppression") or "").strip(),
            blend=BlendConfig.from_dict(_mapping(raw.get("blend", {}), "carriers.blend")),
        )


@dataclass(frozen=True)
class SceneCouplingConfig:
    """Persistent scene lane shared through audio self-attention.

    Stage 1 gathers: the scene lane queries its own K/V concatenated with the
    timeline-gated sum of the real stems (``real_to_scene`` direction), for the
    steps enabled by ``scene_self_attention_schedule``. Stage 2 broadcasts: every
    real stem queries its own K/V concatenated with the scene K/V
    (``scene_to_real`` direction). After self-attention, ``real_to_scene_strength``
    blends the gated real-stem sum back into the scene lane.
    """

    enabled: bool = False
    self_attention_direction: SceneSelfAttentionDirection = "real_to_scene"
    scene_to_real_strength: float = 0.0
    real_to_scene_strength: float = 0.0
    scene_self_attention_strength: float = 1.0
    scene_self_attention_schedule: StrengthSchedule = field(default_factory=StrengthSchedule)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SceneCouplingConfig":
        direction = str(raw.get("self_attention_direction", "real_to_scene")).strip().lower()
        if direction not in {"real_to_scene", "scene_to_real"}:
            raise ValueError("scene_coupling.self_attention_direction must be real_to_scene or scene_to_real")
        value = cls(
            enabled=bool(raw.get("enabled", False)),
            self_attention_direction=direction,  # type: ignore[arg-type]
            scene_to_real_strength=float(raw.get("scene_to_real_strength", 0.0)),
            real_to_scene_strength=float(raw.get("real_to_scene_strength", 0.0)),
            scene_self_attention_strength=float(raw.get("scene_self_attention_strength", 1.0)),
            scene_self_attention_schedule=StrengthSchedule.from_list(raw.get("scene_self_attention_schedule", [])),
        )
        for name in ("scene_to_real_strength", "real_to_scene_strength", "scene_self_attention_strength"):
            if not 0 <= getattr(value, name) <= 1:
                raise ValueError(f"scene_coupling.{name} must be in [0, 1]")
        return value

    def scene_self_attention_enabled_at(self, *, step: int, total_steps: int) -> bool:
        """Return whether the scene lane gathers the real stems at this denoising step."""
        if not self.gathers_real_into_scene_attention:
            return False
        progress = step / max(total_steps - 1, 1)
        return self.scene_self_attention_schedule.strength_at(progress, default=1.0) >= 0.5

    @property
    def uses_scene_lane(self) -> bool:
        return self.enabled

    @property
    def gathers_real_into_scene_attention(self) -> bool:
        return self.enabled and self.self_attention_direction == "real_to_scene"

    @property
    def broadcasts_scene_into_real_attention(self) -> bool:
        return self.enabled and self.self_attention_direction == "scene_to_real"

    def summary(self) -> dict[str, Any]:
        value: dict[str, Any] = {"enabled": self.enabled}
        if self.enabled:
            value.update(
                {
                    "self_attention_direction": self.self_attention_direction,
                    "scene_to_real_strength": self.scene_to_real_strength,
                    "real_to_scene_strength": self.real_to_scene_strength,
                    "scene_self_attention_strength": self.scene_self_attention_strength,
                    "scene_self_attention_schedule": self.scene_self_attention_schedule.summary(),
                }
            )
        return value


@dataclass(frozen=True)
class CarrierRecordConfig:
    id: str
    role: CarrierRole
    group: str
    positive: str
    negative: str
    quantile: float = 0.70
    a2v_enabled: bool = True
    v2a_enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CarrierRecordConfig":
        role = str(raw.get("role") or "").strip().lower()
        if role not in {"activation", "suppression"}:
            raise ValueError("carrier.role must be activation or suppression")
        quantile = float(raw.get("quantile", 0.70))
        if not 0 <= quantile < 1:
            raise ValueError("carrier.quantile must be in [0, 1)")
        return cls(
            id=_nonempty(raw.get("id"), "carrier.id"),
            role=role,  # type: ignore[arg-type]
            group=_nonempty(raw.get("group", "suppression" if role == "suppression" else "general"), "carrier.group"),
            positive=_nonempty(raw.get("positive"), "carrier.positive"),
            negative=str(raw.get("negative") or "").strip(),
            quantile=quantile,
            a2v_enabled=bool(raw.get("a2v_enabled", True)),
            v2a_enabled=bool(raw.get("v2a_enabled", True)),
        )


def default_sam3_backend() -> str:
    """Shared SAM3 video backend script shipped at the repository root."""
    return str(Path(__file__).resolve().parents[2] / "common" / "sam3_video_backend.py")


@dataclass(frozen=True)
class Sam3Config:
    """SAM3 runs in its own Python environment, configured through the environment."""

    python: str = ""
    repo: str = ""
    checkpoint: str = ""
    backend_script: str = ""
    predictor_kind: Literal["sam3.1_multiplex", "sam3"] = "sam3.1_multiplex"
    device: str = "cuda"
    frame_stride: int = 8
    max_frames: int = 32
    prompt_frame_index: int = 0
    score_threshold: float = 0.50

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Sam3Config":
        predictor = str(raw.get("predictor_kind", "sam3.1_multiplex")).strip()
        if predictor not in {"sam3.1_multiplex", "sam3"}:
            raise ValueError("sam3.predictor_kind must be sam3.1_multiplex or sam3")
        value = cls(
            python=str(raw.get("python") or os.environ.get("SOUNDWICH_SAM3_PYTHON", "")).strip(),
            repo=str(raw.get("repo") or os.environ.get("SOUNDWICH_SAM3_REPO", "")).strip(),
            checkpoint=str(raw.get("checkpoint") or os.environ.get("SOUNDWICH_SAM3_CHECKPOINT", "")).strip(),
            backend_script=str(
                raw.get("backend_script") or os.environ.get("SOUNDWICH_SAM3_BACKEND", "") or default_sam3_backend()
            ).strip(),
            predictor_kind=predictor,  # type: ignore[arg-type]
            device=str(raw.get("device") or "cuda").strip(),
            frame_stride=int(raw.get("frame_stride", 8)),
            max_frames=int(raw.get("max_frames", 32)),
            prompt_frame_index=int(raw.get("prompt_frame_index", 0)),
            score_threshold=float(raw.get("score_threshold", 0.50)),
        )
        if value.frame_stride <= 0 or value.max_frames <= 0 or value.prompt_frame_index < 0:
            raise ValueError("SAM3 frame stride/max frames must be positive and prompt frame must be non-negative")
        if not 0 <= value.score_threshold <= 1:
            raise ValueError("sam3.score_threshold must be in [0, 1]")
        return value

    def missing_settings(self) -> list[str]:
        missing = []
        if not self.python:
            missing.append("SOUNDWICH_SAM3_PYTHON (python executable of the SAM3 environment)")
        elif not Path(self.python).expanduser().exists():
            missing.append(f"SAM3 python not found: {self.python}")
        if self.repo and not Path(self.repo).expanduser().exists():
            missing.append(f"SAM3 repo not found: {self.repo}")
        if self.checkpoint and not Path(self.checkpoint).expanduser().exists():
            missing.append(f"SAM3 checkpoint not found: {self.checkpoint}")
        if not Path(self.backend_script).expanduser().exists():
            missing.append(f"SAM3 backend script not found: {self.backend_script}")
        return missing


@dataclass(frozen=True)
class Stage2Config:
    enabled: bool = True
    start_sigma: float = 0.95
    audio_cfg: float = 1.0
    suppression_blend_enabled: bool = True
    suppression_carrier: str = ""
    scene_coupling: SceneCouplingConfig = field(default_factory=SceneCouplingConfig)
    a2v_hard_route: bool = True
    v2a_hard_route: bool = True
    a2v_active_gain: float = 1.0
    mask_threshold: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage2Config":
        value = cls(
            enabled=bool(raw.get("enabled", True)),
            start_sigma=float(raw.get("start_sigma", 0.95)),
            audio_cfg=float(raw.get("audio_cfg", 1.0)),
            suppression_blend_enabled=bool(raw.get("suppression_blend_enabled", True)),
            suppression_carrier=str(raw.get("suppression_carrier") or "").strip(),
            scene_coupling=SceneCouplingConfig.from_dict(
                {
                    "self_attention_direction": "scene_to_real",
                    **_mapping(raw.get("scene_coupling", {}), "stage2.scene_coupling"),
                }
            ),
            a2v_hard_route=bool(raw.get("a2v_hard_route", True)),
            v2a_hard_route=bool(raw.get("v2a_hard_route", True)),
            a2v_active_gain=float(raw.get("a2v_active_gain", 1.0)),
            mask_threshold=float(raw.get("mask_threshold", 0.05)),
        )
        if value.a2v_active_gain < 0:
            raise ValueError("stage2.a2v_active_gain must be non-negative")
        if not 0.909375 <= value.start_sigma <= 1.0:
            raise ValueError("stage2.start_sigma must be in [0.909375, 1.0]")
        if not 1 <= value.audio_cfg <= 30:
            raise ValueError("stage2.audio_cfg must be in [1, 30]")
        if not 0 <= value.mask_threshold <= 1:
            raise ValueError("stage2.mask_threshold must be in [0, 1]")
        return value

    @property
    def negative_audio_cfg(self) -> bool:
        """Per-stem negative audio CFG runs only when the Stage-2 audio CFG exceeds one."""
        return self.audio_cfg > 1.0


@dataclass(frozen=True)
class MultiStemConfig:
    id: str
    task: Task
    model: ModelConfig
    generation: GenerationConfig
    guidance: GuidanceConfig
    visual_positive: str
    visual_negative: str
    audio_global_positive: str
    audio_global_negative: str
    stems: tuple[StemConfig, ...]
    carriers: CarrierReplayConfig
    scene_coupling: SceneCouplingConfig
    carrier: CarrierRecordConfig | None
    sam3: Sam3Config
    stage2: Stage2Config

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MultiStemConfig":
        task = str(raw.get("task", "generate")).strip()
        if task not in {"generate", "record_carrier"}:
            raise ValueError("task must be generate or record_carrier")
        generation = GenerationConfig.from_dict(_mapping(raw.get("generation", {}), "generation"))
        prompts = _mapping(raw.get("prompts", {}), "prompts")
        stems = tuple(
            StemConfig.from_dict(_mapping(item, "stem"), duration=generation.duration_seconds)
            for item in raw.get("stems", [])
        )
        carrier_raw = raw.get("carrier")
        carrier = CarrierRecordConfig.from_dict(_mapping(carrier_raw, "carrier")) if carrier_raw is not None else None
        visual_positive = str(prompts.get("visual_positive") or "").strip()
        visual_negative = str(prompts.get("visual_negative") or "").strip()
        if task == "generate":
            visual_positive = _nonempty(visual_positive, "prompts.visual_positive")
            visual_negative = _nonempty(visual_negative, "prompts.visual_negative")
        value = cls(
            id=_nonempty(raw.get("id"), "id"),
            task=task,  # type: ignore[arg-type]
            model=ModelConfig.from_dict(_mapping(raw.get("model", {}), "model")),
            generation=generation,
            guidance=GuidanceConfig.from_dict(_mapping(raw.get("guidance", {}), "guidance")),
            visual_positive=visual_positive,
            visual_negative=visual_negative,
            audio_global_positive=str(prompts.get("audio_global_positive") or "").strip(),
            audio_global_negative=str(prompts.get("audio_global_negative") or "").strip(),
            stems=stems,
            carriers=CarrierReplayConfig.from_dict(_mapping(raw.get("carriers", {}), "carriers")),
            scene_coupling=SceneCouplingConfig.from_dict(_mapping(raw.get("scene_coupling", {}), "scene_coupling")),
            carrier=carrier,
            sam3=Sam3Config.from_dict(_mapping(raw.get("sam3", {}), "sam3")),
            stage2=Stage2Config.from_dict(_mapping(raw.get("stage2", {}), "stage2")),
        )
        value.validate_task()
        return value

    def validate_task(self) -> None:
        if self.task == "record_carrier":
            if self.carrier is None:
                raise ValueError("record_carrier task requires carrier settings")
            return
        if not self.stems:
            raise ValueError("generate task requires at least one stem")
        if self.scene_coupling.enabled and len(self.stems) < 2:
            raise ValueError("scene coupling requires at least two real stems")
        ids = [stem.id for stem in self.stems]
        if len(ids) != len(set(ids)):
            raise ValueError("stem ids must be unique")
        missing = sorted({stem.reference_group for stem in self.stems} - self.carriers.activation.keys())
        if missing:
            raise ValueError(f"missing activation carriers for groups: {missing}")
        if not self.carriers.suppression:
            raise ValueError("generate task requires a suppression carrier")
        if self.stage2.enabled:
            if not self.model.spatial_upscaler:
                raise ValueError("Stage 2 requires model.spatial_upscaler")
            if not self.model.stage2_transformer:
                raise ValueError("Stage 2 requires model.stage2_transformer for post-upscale refinement")
            if self.stage2.suppression_blend_enabled and not self.stage2.suppression_carrier:
                raise ValueError("Stage-2 suppression blending requires stage2.suppression_carrier")
            if self.stage2.scene_coupling.enabled and len(self.stems) < 2:
                raise ValueError("Stage-2 scene coupling requires at least two real stems")
            if self.stage2.scene_coupling.uses_scene_lane and not self.scene_coupling.uses_scene_lane:
                raise ValueError("Stage-2 scene coupling requires the Stage-1 scene lane")

    def missing_model_paths(self) -> list[Path]:
        return [Path(value) for value in self.model.paths() if not Path(value).exists()]

    def effective_positive(self, stem: StemConfig) -> str:
        return ". ".join(part for part in (stem.positive, self.audio_global_positive) if part)

    def effective_negative(self, stem: StemConfig) -> str:
        if stem.negative:
            parts = (self.audio_global_negative, stem.negative)
        elif stem.auto_negative:
            other_sources = [other.negative_source for other in self.stems if other.id != stem.id]
            parts = (self.audio_global_negative, "; ".join(other_sources))
        else:
            parts = (self.audio_global_negative,)
        return ". ".join(part for part in parts if part)

    def scene_positive(self) -> str:
        """Joint positive prompt for the persistent scene lane."""
        return ". ".join(
            part
            for part in (
                self.visual_positive,
                *[stem.positive for stem in self.stems if stem.scene_context_enabled],
                self.audio_global_positive,
            )
            if part
        )

    def scene_negative(self) -> str:
        """Global audio-quality negative prompt without per-stem exclusions."""
        return self.audio_global_negative

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            "model": {key: Path(value).name for key, value in self.model.__dict__.items() if value},
            "generation": {
                "seed": self.generation.seed,
                "resolution": [self.generation.width, self.generation.height],
                "frames": self.generation.frames,
                "frame_rate": self.generation.frame_rate,
                "duration_seconds": round(self.generation.duration_seconds, 3),
                "steps": self.generation.steps,
                "schedule": self.generation.schedule,
            },
            "guidance": dict(self.guidance.__dict__),
        }
        if self.task == "record_carrier":
            summary["carrier"] = dict(self.carrier.__dict__) if self.carrier is not None else None
            return summary
        blend = self.carriers.blend
        summary["stage1"] = {
            "activation_strength": blend.strength,
            "activation_schedule": blend.activation_schedule.summary(),
            "value_scale": blend.value_scale,
            "outside_suppression": blend.outside_suppression,
            "outside_silence_blend": blend.outside_silence_blend,
            "feather_seconds": blend.feather_seconds,
            "rms_scale_limit": blend.rms_scale_limit,
            "activation_carriers": dict(sorted(self.carriers.activation.items())),
            "suppression_carrier": self.carriers.suppression,
            "scene_coupling": self.scene_coupling.summary(),
        }
        summary["stage2"] = {
            "enabled": self.stage2.enabled,
            "start_sigma": self.stage2.start_sigma,
            "audio_cfg": self.stage2.audio_cfg,
            "negative_audio_cfg": self.stage2.negative_audio_cfg,
            "suppression_blend": self.stage2.suppression_blend_enabled,
            "suppression_carrier": self.stage2.suppression_carrier,
            "scene_coupling": self.stage2.scene_coupling.summary(),
            "a2v_hard_route": self.stage2.a2v_hard_route,
            "v2a_hard_route": self.stage2.v2a_hard_route,
            "a2v_active_gain": self.stage2.a2v_active_gain,
            "mask_threshold": self.stage2.mask_threshold,
        }
        summary["sam3"] = {
            key: value
            for key, value in self.sam3.__dict__.items()
            if key not in {"python", "repo", "checkpoint", "backend_script"}
        }
        summary["prompts"] = {
            "visual_positive": self.visual_positive,
            "visual_negative": self.visual_negative,
            "scene_positive": self.scene_positive() if self.scene_coupling.uses_scene_lane else "",
            "scene_negative": self.scene_negative() if self.scene_coupling.uses_scene_lane else "",
        }
        summary["stems"] = [
            {
                "id": stem.id,
                "reference_group": stem.reference_group,
                "scene_context_enabled": stem.scene_context_enabled,
                "windows": [[window.start, window.end] for window in stem.windows],
                "positive": self.effective_positive(stem),
                "negative": self.effective_negative(stem),
                "volume": stem.volume,
                "sam_prompt": stem.sam_prompt,
                "blend_strength": stem.blend_strength,
                "blend_schedule": stem.blend_schedule.summary() if stem.blend_schedule is not None else None,
            }
            for stem in self.stems
        ]
        return summary
