"""Scene loading and Ovi prompt adaptation for multi-stem inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StemSpec:
    id: str
    entity: str
    kind: str
    positive: str
    negative: str
    windows: tuple[tuple[float, float], ...]
    dialogue: tuple[str, ...]
    volume: float
    carrier_group: str


@dataclass(frozen=True)
class SceneSpec:
    id: str
    duration_seconds: float
    seed: int
    visual_positive: str
    visual_negative: str
    audio_positive: str
    audio_negative: str
    stems: tuple[StemSpec, ...]


def _dialogue_from_stem(stem: dict[str, Any]) -> tuple[str, ...]:
    windows = stem.get("windows") or []
    texts = [str(window.get("text") or "").strip() for window in windows]
    texts = [text for text in texts if text]
    if texts:
        return tuple(texts)
    positive = str(stem.get("positive") or "")
    quoted = re.findall(r'["“]([^"”]+)["”]', positive)
    return tuple(text.strip() for text in quoted if text.strip())


def load_scene(path: str | Path) -> SceneSpec:
    scene_path = Path(path)
    raw = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scene must be a mapping: {scene_path}")
    duration = float(raw.get("duration_seconds") or 10.0)
    stems: list[StemSpec] = []
    entities = {
        str(entity.get("id")): entity for entity in raw.get("sound_entities") or []
    }
    for stem in raw.get("stems") or []:
        stem_id = str(stem.get("id") or "").strip()
        if not stem_id:
            raise ValueError(f"scene {scene_path} contains a stem without id")
        windows = tuple(
            (float(window["start"]), float(window["end"]))
            for window in stem.get("windows") or []
        )
        for start, end in windows:
            if start < 0 or end <= start or end > duration:
                raise ValueError(
                    f"scene {scene_path}: invalid window {start}-{end} for {stem_id}"
                )
        entity_id = str(stem.get("entity") or "")
        entity = entities.get(entity_id) or {}
        kind = str(stem.get("kind") or entity.get("group") or "speech").strip()
        negative_parts = [
            str((raw.get("audio_globals") or {}).get("negative") or "").strip(),
            str(stem.get("negative") or "").strip(),
        ]
        stems.append(
            StemSpec(
                id=stem_id,
                entity=str(entity.get("visual_entity") or entity_id or stem_id),
                kind=kind,
                positive=str(stem.get("positive") or "").strip(),
                negative=", ".join(part for part in negative_parts if part),
                windows=windows,
                dialogue=_dialogue_from_stem(stem),
                volume=float(stem.get("volume", 1.0)),
                carrier_group=str(
                    stem.get("carrier_group") or kind or "general"
                ).strip(),
            )
        )
    if not stems:
        raise ValueError(f"scene has no stems: {scene_path}")
    return SceneSpec(
        id=str(raw.get("id") or scene_path.stem),
        duration_seconds=duration,
        seed=int((raw.get("generation") or {}).get("seed", 33)),
        visual_positive=str((raw.get("visual") or {}).get("positive") or "").strip(),
        visual_negative=str((raw.get("visual") or {}).get("negative") or "").strip(),
        audio_positive=str((raw.get("audio_globals") or {}).get("positive") or "").strip(),
        audio_negative=str((raw.get("audio_globals") or {}).get("negative") or "").strip(),
        stems=tuple(stems),
    )


def _speech_tags(dialogue: tuple[str, ...]) -> str:
    return " ".join(f"<S>{text}<E>" for text in dialogue)


def build_prompts(scene: SceneSpec) -> dict[str, Any]:
    """Build one shared video prompt and positive/negative prompts per stem."""
    ordered_events: list[tuple[float, str]] = []
    for stem in scene.stems:
        if stem.kind != "speech":
            continue
        for index, text in enumerate(stem.dialogue):
            start = stem.windows[index][0] if index < len(stem.windows) else 0.0
            ordered_events.append(
                (start, f"The {stem.entity} says <S>{text}<E>.")
            )
    ordered_events.sort(key=lambda item: item[0])
    event_text = " Then ".join(text for _, text in ordered_events)
    audio_summary = ", ".join(stem.positive for stem in scene.stems)
    video_prompt = " ".join(
        part
        for part in (
            scene.visual_positive,
            event_text,
            f"Audio: {scene.audio_positive}. {audio_summary}",
        )
        if part
    )

    stem_positive: list[str] = []
    stem_negative: list[str] = []
    for stem in scene.stems:
        speech = _speech_tags(stem.dialogue) if stem.kind == "speech" else ""
        positive = " ".join(
            part
            for part in (
                speech,
                f"Audio: {scene.audio_positive}. {stem.positive}",
            )
            if part
        )
        stem_positive.append(positive)
        stem_negative.append(
            " ".join(
                part
                for part in (
                    f"Audio: {stem.negative}",
                )
                if part
            )
        )

    return {
        "video_positive": video_prompt,
        "video_negative": scene.visual_negative,
        "audio_positive": stem_positive,
        "audio_negative": stem_negative,
    }
