"""Frame-snapped stem timelines for editing: move windows, permute saved audio latents, replace a stem's line.

Every stem with timeline windows gets an independent timeline: a partition of the clip into segments (one per
window, plus the gaps between them) and a playback order. Moving a window cuts its segment and inserts it at
the target frame, so the surrounding segments shift and the clip length is preserved. The saved audio latents
are then permuted along time with the same segments; no value is interpolated or re-encoded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


class TimelineEditError(ValueError):
    """Raised when a timeline edit violates the fixed-window contract."""


def create_timeline(scene: dict[str, Any], *, total_frames: int, frame_rate: float) -> dict[str, Any]:
    """Build the unedited per-stem timelines from the scene's windows (seconds, snapped to video frames)."""
    clips: list[dict[str, Any]] = []
    stem_ids: list[str] = []
    clips_by_stem: dict[str, list[dict[str, Any]]] = {}
    for stem in scene.get("stems", []):
        stem_id = str(stem["id"])
        stem_ids.append(stem_id)
        previous_end = -1
        for index, window in enumerate(stem.get("windows") or []):
            start = _seconds_to_frame(window.get("start"), frame_rate, total_frames)
            end = _seconds_to_frame(window.get("end"), frame_rate, total_frames)
            if end <= start:
                raise TimelineEditError(f"{stem_id} window {index + 1} has no frame-snapped duration")
            if start < previous_end:
                raise TimelineEditError(f"windows overlap within stem {stem_id}")
            previous_end = end
            clip = {
                "id": f"clip_{len(clips) + 1:03d}",
                "stem_id": stem_id,
                "source_start_frame": start,
                "source_end_frame": end,
                "text": str(window.get("text") or ""),
            }
            clips.append(clip)
            clips_by_stem.setdefault(stem_id, []).append(clip)
    timeline = {
        "frame_rate": float(frame_rate),
        "total_frames": int(total_frames),
        "clips": clips,
        "stem_timelines": {
            stem_id: _build_stem_timeline(stem_id, stem_clips, total_frames)
            for stem_id, stem_clips in clips_by_stem.items()
        },
    }
    validate_timeline(timeline)
    return timeline


def stem_clip_ids(timeline: dict[str, Any], stem_id: str) -> list[str]:
    """Clip ids of one stem in window order."""
    return [str(clip["id"]) for clip in timeline["clips"] if clip["stem_id"] == stem_id]


def move_clip(timeline: dict[str, Any], clip_id: str, target_start_frame: int) -> dict[str, Any]:
    """Cut one complete window and insert it at a frame in its own stem."""
    validate_timeline(timeline)
    item = deepcopy(timeline)
    clip = next((candidate for candidate in item["clips"] if candidate["id"] == clip_id), None)
    if clip is None:
        raise TimelineEditError(f"unknown timeline clip: {clip_id}")
    stem_id = str(clip["stem_id"])
    stem_timeline = item["stem_timelines"][stem_id]
    segments = {str(segment["id"]): segment for segment in stem_timeline["segments"]}
    clip_segment_id = next(
        segment_id for segment_id, segment in segments.items() if str(segment.get("clip_id") or "") == clip_id
    )
    duration = _segment_length(segments[clip_segment_id])
    total_frames = int(item["total_frames"])
    target = int(target_start_frame)
    if target < 0 or target + duration > total_frames:
        raise TimelineEditError(f"moving {clip_id} to frame {target} would leave the clip")
    if _segment_destination_starts(stem_timeline)[clip_segment_id] == target:
        return item

    remaining_order = [segment_id for segment_id in stem_timeline["segment_order"] if segment_id != clip_segment_id]
    insertion = _snap_away_from_phrase(target, remaining_order, segments)
    if insertion < 0 or insertion > total_frames - duration:
        raise TimelineEditError(f"moving {clip_id} next to another window would leave the clip")
    remaining_order, segment_list = _split_gap_at_destination(
        stem_id, remaining_order, list(segments.values()), insertion
    )
    segments = {str(segment["id"]): segment for segment in segment_list}
    remaining_order.insert(_boundary_index(remaining_order, segments, insertion), clip_segment_id)
    stem_timeline["segments"] = list(segments.values())
    stem_timeline["segment_order"] = remaining_order
    validate_timeline(item)
    return item


def validate_timeline(timeline: dict[str, Any]) -> None:
    total_frames = int(timeline.get("total_frames", 0))
    if total_frames <= 0:
        raise TimelineEditError("timeline has no duration")
    clips = timeline.get("clips") or []
    clip_ids = {str(clip.get("id")) for clip in clips}
    if len(clip_ids) != len(clips):
        raise TimelineEditError("timeline clip ids must be unique")
    timelines = timeline.get("stem_timelines") or {}
    controlled_stems = {str(clip.get("stem_id")) for clip in clips}
    if set(timelines) != controlled_stems:
        raise TimelineEditError("every stem with windows needs one independent timeline")

    seen_clip_segments: set[str] = set()
    for stem_id, stem_timeline in timelines.items():
        _validate_partition(stem_timeline, total_frames, stem_id)
        for segment in stem_timeline["segments"]:
            segment_clip = str(segment.get("clip_id") or "")
            if not segment_clip:
                continue
            if segment_clip not in clip_ids or segment_clip in seen_clip_segments:
                raise TimelineEditError("each window must own one segment")
            clip = next(value for value in clips if value["id"] == segment_clip)
            if str(clip["stem_id"]) != str(stem_id):
                raise TimelineEditError("window segment belongs to the wrong stem")
            if (
                int(segment["source_start_frame"]) != int(clip["source_start_frame"])
                or int(segment["source_end_frame"]) != int(clip["source_end_frame"])
            ):
                raise TimelineEditError("window segment boundaries cannot change")
            seen_clip_segments.add(segment_clip)
    if seen_clip_segments != clip_ids:
        raise TimelineEditError("every window must remain whole in its stem timeline")

    remapped = remapped_clips(timeline, validate=False)
    for stem_id in controlled_stems:
        previous_end = 0
        for clip in sorted(
            (clip for clip in remapped if clip["stem_id"] == stem_id),
            key=lambda value: value["destination_start_frame"],
        ):
            start = int(clip["destination_start_frame"])
            end = int(clip["destination_end_frame"])
            if start < previous_end or start < 0 or end > total_frames:
                raise TimelineEditError(f"{stem_id} edited windows overlap or leave the clip")
            previous_end = end


def remapped_clips(timeline: dict[str, Any], *, validate: bool = True) -> list[dict[str, Any]]:
    """Return clips with destination spans from their independent stem orders."""
    if validate:
        validate_timeline(timeline)
    result: list[dict[str, Any]] = []
    clips = {str(clip["id"]): clip for clip in timeline.get("clips") or []}
    for stem_timeline in (timeline.get("stem_timelines") or {}).values():
        starts = _segment_destination_starts(stem_timeline)
        for segment in stem_timeline.get("segments") or []:
            clip_id = str(segment.get("clip_id") or "")
            if not clip_id:
                continue
            clip = deepcopy(clips[clip_id])
            start = int(starts[str(segment["id"])])
            clip["destination_start_frame"] = start
            clip["destination_end_frame"] = start + _segment_length(segment)
            result.append(clip)
    return sorted(
        result,
        key=lambda clip: (clip["stem_id"], clip["destination_start_frame"], clip["destination_end_frame"]),
    )


def scene_with_timeline(scene: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    """Move the scene's windows to their edited positions (used by the video refinement routing)."""
    fps = float(timeline["frame_rate"])
    clips_by_stem: dict[str, list[dict[str, Any]]] = {}
    for clip in remapped_clips(timeline):
        clips_by_stem.setdefault(clip["stem_id"], []).append(clip)
    source_clips_by_stem: dict[str, list[dict[str, Any]]] = {}
    for clip in timeline.get("clips") or []:
        source_clips_by_stem.setdefault(str(clip["stem_id"]), []).append(clip)
    result = deepcopy(scene)
    for stem in result.get("stems") or []:
        stem_id = str(stem.get("id") or "")
        if stem_id not in clips_by_stem:
            continue
        original_windows = deepcopy(stem.get("windows") or [])
        stem["windows"] = [
            {
                "start": clip["destination_start_frame"] / fps,
                "end": clip["destination_end_frame"] / fps,
                "text": clip.get("text", ""),
            }
            for clip in clips_by_stem[stem_id]
        ]
        stem["positive"] = _positive_with_texts(
            str(stem.get("positive") or ""), original_windows, source_clips_by_stem.get(stem_id, [])
        )
    return result


def reorder_audio_latents_by_segments(
    latents: torch.Tensor,
    segments: list[dict[str, Any]],
    segment_order: list[str],
    total_frames: int,
) -> torch.Tensor:
    """Permute [B,C,T,F] latent time bins, preserving every value exactly.

    Video-frame boundaries are rounded onto the audio latent time grid once. Shared boundaries keep the
    intervals a partition even when the frame rates differ.
    """
    if latents.ndim != 4 or latents.shape[2] == 0 or total_frames <= 0:
        raise ValueError("Audio latents must be nonempty [B,C,T,F] with positive total_frames")
    _validate_partition({"segments": segments, "segment_order": segment_order}, total_frames, "audio")
    by_id = {str(segment["id"]): segment for segment in segments}
    count = latents.shape[2]
    pieces = []
    for segment_id in segment_order:
        segment = by_id[str(segment_id)]
        start = round(int(segment["source_start_frame"]) * count / total_frames)
        end = round(int(segment["source_end_frame"]) * count / total_frames)
        pieces.append(latents[:, :, start:end, :])
    return torch.cat(pieces, dim=2)


def apply_replacement(
    stem: dict[str, Any], request: dict[str, Any], scene: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Apply a replacement request to one scene stem without changing its number of windows.

    ``request`` may set ``positive``, ``negative``, ``seed`` (default: the scene seed) and ``windows``
    (one mapping per existing window with optional ``start``/``end``/``text``). A changed window text
    replaces the old text in the stem prompt, or is appended when the prompt does not contain it.
    """
    stem = deepcopy(stem)
    original_windows = deepcopy(stem.get("windows") or [])
    for key in ("positive", "negative"):
        if key in request:
            stem[key] = str(request[key])
    seed = int(request.get("seed", (scene.get("generation") or {}).get("seed", 42)))
    requested_windows = request.get("windows")
    if isinstance(requested_windows, list):
        windows = stem.get("windows") or []
        if len(requested_windows) != len(windows):
            raise TimelineEditError(f"replacement windows for {stem.get('id')} must match its {len(windows)} windows")
        for target, edit in zip(windows, requested_windows, strict=True):
            if not isinstance(edit, dict):
                continue
            for key in ("start", "end"):
                if key in edit:
                    target[key] = float(edit[key])
            if "text" in edit:
                target["text"] = str(edit["text"])
    stem["positive"] = _positive_with_texts(str(stem.get("positive") or ""), original_windows, stem.get("windows") or [])
    return stem, seed


def _positive_with_texts(positive: str, original_windows: list[dict[str, Any]], updated: list[dict[str, Any]]) -> str:
    for original, window in zip(original_windows, updated):
        old_text = str(original.get("text") or "").strip()
        new_text = str(window.get("text") or "").strip()
        if old_text and new_text and old_text != new_text and old_text in positive:
            positive = positive.replace(old_text, new_text)
        if new_text and new_text.casefold() not in positive.casefold():
            positive = f"{positive.rstrip(' .')}. {new_text}" if positive else new_text
    return positive


def _validate_partition(value: dict[str, Any], total_frames: int, stem_id: str) -> None:
    segments = value.get("segments")
    order = value.get("segment_order")
    if not isinstance(segments, list) or not isinstance(order, list) or not segments:
        raise TimelineEditError(f"{stem_id} timeline needs non-empty segments and segment_order")
    by_id: dict[str, dict[str, Any]] = {}
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        if not segment_id or segment_id in by_id:
            raise TimelineEditError(f"{stem_id} timeline segment ids must be unique and non-empty")
        start = int(segment["source_start_frame"])
        end = int(segment["source_end_frame"])
        if start < 0 or end <= start or end > total_frames:
            raise TimelineEditError(f"{stem_id} timeline segment leaves the valid frame range")
        by_id[segment_id] = segment
    order_ids = [str(item) for item in order]
    if len(order_ids) != len(by_id) or set(order_ids) != set(by_id):
        raise TimelineEditError(f"{stem_id} segment_order must contain every segment exactly once")
    cursor = 0
    for segment in sorted(by_id.values(), key=lambda item: int(item["source_start_frame"])):
        if int(segment["source_start_frame"]) != cursor:
            raise TimelineEditError(f"{stem_id} segments must partition the complete frame range")
        cursor = int(segment["source_end_frame"])
    if cursor != total_frames:
        raise TimelineEditError(f"{stem_id} segments must end at total_frames")


def _build_stem_timeline(stem_id: str, clips: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    cursor = 0
    for clip in clips:
        start = int(clip["source_start_frame"])
        end = int(clip["source_end_frame"])
        if start > cursor:
            segments.append(_new_segment(stem_id, segments, cursor, start))
        segments.append(_new_segment(stem_id, segments, start, end, clip_id=str(clip["id"])))
        cursor = end
    if cursor < total_frames:
        segments.append(_new_segment(stem_id, segments, cursor, total_frames))
    if not segments:
        segments.append(_new_segment(stem_id, segments, 0, total_frames))
    return {"segments": segments, "segment_order": [segment["id"] for segment in segments]}


def _new_segment(
    stem_id: str, existing: list[dict[str, Any]], start: int, end: int, *, clip_id: str = ""
) -> dict[str, Any]:
    used = {str(segment.get("id")) for segment in existing}
    index = 1
    while f"{stem_id}_segment_{index:03d}" in used:
        index += 1
    return {
        "id": f"{stem_id}_segment_{index:03d}",
        "kind": "clip" if clip_id else "gap",
        "source_start_frame": int(start),
        "source_end_frame": int(end),
        "clip_id": clip_id,
    }


def _segment_destination_starts(stem_timeline: dict[str, Any]) -> dict[str, int]:
    segments = {str(segment["id"]): segment for segment in stem_timeline.get("segments") or []}
    starts: dict[str, int] = {}
    cursor = 0
    for segment_id in stem_timeline.get("segment_order") or []:
        starts[str(segment_id)] = cursor
        cursor += _segment_length(segments[str(segment_id)])
    return starts


def _snap_away_from_phrase(target: int, order: list[str], segments: dict[str, dict[str, Any]]) -> int:
    """A target inside another window of the same stem snaps to that window's nearer edge."""
    cursor = 0
    for segment_id in order:
        segment = segments[segment_id]
        end = cursor + _segment_length(segment)
        if cursor < target < end and segment.get("clip_id"):
            midpoint = cursor + _segment_length(segment) / 2
            return cursor if target < midpoint else end
        cursor = end
    return target


def _split_gap_at_destination(
    stem_id: str, order: list[str], segments: list[dict[str, Any]], destination: int
) -> tuple[list[str], list[dict[str, Any]]]:
    by_id = {str(segment["id"]): segment for segment in segments}
    cursor = 0
    for order_index, segment_id in enumerate(order):
        segment = by_id[segment_id]
        end = cursor + _segment_length(segment)
        if cursor < destination < end:
            if segment.get("clip_id"):
                raise TimelineEditError("cannot split a complete window")
            source_start = int(segment["source_start_frame"])
            split_source = source_start + destination - cursor
            existing = list(by_id.values())
            left = _new_segment(stem_id, existing, source_start, split_source)
            existing.append(left)
            right = _new_segment(stem_id, existing, split_source, int(segment["source_end_frame"]))
            by_id.pop(segment_id)
            by_id[left["id"]] = left
            by_id[right["id"]] = right
            order = [*order[:order_index], left["id"], right["id"], *order[order_index + 1 :]]
            break
        cursor = end
    return order, list(by_id.values())


def _boundary_index(order: list[str], segments: dict[str, dict[str, Any]], destination: int) -> int:
    cursor = 0
    if destination == 0:
        return 0
    for index, segment_id in enumerate(order):
        cursor += _segment_length(segments[segment_id])
        if cursor == destination:
            return index + 1
        if cursor > destination:
            break
    if destination == cursor:
        return len(order)
    raise TimelineEditError("timeline target is not a frame boundary")


def _seconds_to_frame(value: Any, fps: float, total_frames: int) -> int:
    return min(total_frames, max(0, round(float(value) * fps)))


def _segment_length(segment: dict[str, Any]) -> int:
    return int(segment["source_end_frame"]) - int(segment["source_start_frame"])
