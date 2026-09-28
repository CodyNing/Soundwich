#!/usr/bin/env python3
"""SAM3 video backend for Stage-2 A2V token masks.

The Soundwich pipelines write Stage-1 frames to a JPEG folder and call this script in a
subprocess. For each stem prompt, this backend runs SAM3/SAM3.1 video
segmentation, propagates tracked masklets through the frame sequence, writes
full-resolution masks/overlays for review, and downsamples the result to the
Stage-2 video-token grid consumed by the A2V hook.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-shape", required=True, help="T,H,W")
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--frame-indices", required=True, help="comma-separated original frame indices")
    parser.add_argument("--sam3-repo", default="")
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("SAM3_CHECKPOINT_PATH", ""),
        help=(
            "Local SAM3 checkpoint. When set, the predictor loads it directly "
            "instead of downloading from the gated Hugging Face repo. Also "
            "readable from SAM3_CHECKPOINT_PATH."
        ),
    )
    parser.add_argument(
        "--predictor-kind",
        choices=["sam3.1_multiplex", "sam3"],
        default="sam3.1_multiplex",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep SAM3 tracking state on CPU between frame operations. This is "
            "slower, but avoids GPU memory accumulation when a scene has "
            "multiple stem prompts."
        ),
    )
    return parser.parse_args()


def _parse_token_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("--token-shape must be T,H,W")
    return (max(parts[0], 1), max(parts[1], 1), max(parts[2], 1))


def _parse_indices(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_prompt_spec(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("stems"), list):
        raise ValueError("prompts JSON must be {'stems': [...]}")
    return data


def _normalize_prompt(text: str) -> str:
    prompt = " ".join(str(text or "").split())
    if prompt and not prompt.endswith("."):
        prompt += "."
    return prompt


def _safe_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return safe.strip("_") or "stem"


def _palette(index: int) -> tuple[int, int, int]:
    colors = [
        (230, 57, 70),
        (29, 185, 84),
        (0, 119, 182),
        (255, 183, 3),
        (131, 56, 236),
        (251, 133, 0),
        (0, 180, 216),
        (255, 0, 110),
    ]
    return colors[index % len(colors)]


def _require_runtime_modules(repo: str):
    if repo:
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.exists():
            raise RuntimeError(f"SAM3 repo does not exist: {repo_path}")
        sys.path.insert(0, str(repo_path))

    try:
        import numpy as np
        from PIL import Image
        import torch
        from sam3.model_builder import (
            build_sam3_multiplex_video_predictor,
            build_sam3_video_predictor,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SAM3 backend dependencies are missing. Install the official "
            "facebookresearch/sam3 repo in the Python environment used by this "
            "backend and authenticate to Hugging Face for the SAM3 checkpoints."
        ) from exc

    return {
        "np": np,
        "Image": Image,
        "torch": torch,
        "build_sam3_multiplex_video_predictor": build_sam3_multiplex_video_predictor,
        "build_sam3_video_predictor": build_sam3_video_predictor,
    }


def _frame_paths(frames_dir: Path) -> list[Path]:
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No .jpg or .png frames found in {frames_dir}")
    return paths


def _load_frames(frames_dir: Path, Image) -> list[Any]:
    return [Image.open(path).convert("RGB") for path in _frame_paths(frames_dir)]


def _to_numpy(value, np):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _build_predictor(
    modules: dict[str, Any],
    predictor_kind: str,
    device: str,
    checkpoint_path: str = "",
):
    torch = modules["torch"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is false")
    # An explicit local checkpoint skips the Hugging Face download entirely,
    # which matters because the official SAM3 weights are gated.
    extra: dict[str, Any] = {}
    if checkpoint_path:
        extra["checkpoint_path"] = checkpoint_path
    if predictor_kind == "sam3.1_multiplex":
        return modules["build_sam3_multiplex_video_predictor"](use_fa3=False, **extra)
    return modules["build_sam3_video_predictor"](**extra)


def _clear_cuda_cache(modules: dict[str, Any]) -> None:
    torch = modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _filter_model_init_state_kwargs(predictor) -> None:
    """Adapt SAM3 predictor wrappers to version-specific model init signatures."""
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if model is None or init_state is None:
        return

    signature = inspect.signature(init_state)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return
    valid_names = set(signature.parameters)

    def filtered_init_state(*args, **kwargs):
        filtered = {key: value for key, value in kwargs.items() if key in valid_names}
        return init_state(*args, **filtered)

    model.init_state = filtered_init_state


def _propagate_prompt(
    predictor,
    session_id: str,
    prompt: str,
    prompt_frame_index: int,
    score_threshold: float,
    frame_count: int,
    np,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    point_frame_index: int | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    predictor.handle_request(
        request={
            "type": "reset_session",
            "session_id": session_id,
        }
    )

    propagation_start = max(int(prompt_frame_index), 0)
    labels = list(point_labels or [1] * len(points or []))
    if points:
        if len(labels) != len(points):
            raise ValueError("point_labels length must match points length")
        propagation_start = (
            propagation_start
            if point_frame_index is None
            else max(0, min(int(point_frame_index), frame_count - 1))
        )

    # Text discovery always runs first. When the user clicked a different
    # frame, run discovery on that frame so its object masks align with the
    # point coordinates shown in the editor.
    prompt_response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": propagation_start,
            "text": prompt,
            "output_prob_thresh": float(score_threshold),
        }
    )

    def collect_propagation(selected_object_id: int | None = None):
        masks = [None] * frame_count
        detections = [
            {"local_frame": idx, "objects": 0, "max_score": 0.0}
            for idx in range(frame_count)
        ]
        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": propagation_start,
                "max_frame_num_to_track": frame_count,
                "output_prob_thresh": float(score_threshold),
            }
        ):
            frame_idx = int(response["frame_index"])
            if frame_idx < 0 or frame_idx >= frame_count:
                continue
            outputs = response["outputs"]
            out_masks = _to_numpy(outputs.get("out_binary_masks", []), np)
            out_scores = _to_numpy(outputs.get("out_probs", []), np)
            out_ids = _to_numpy(outputs.get("out_obj_ids", []), np)
            if out_masks.size == 0:
                continue
            if out_masks.ndim == 2:
                out_masks = out_masks[None, ...]
            keep = (
                out_scores >= float(score_threshold)
                if out_scores.size
                else np.ones(len(out_masks), dtype=bool)
            )
            if selected_object_id is not None:
                keep &= (
                    out_ids == int(selected_object_id)
                    if out_ids.size
                    else np.zeros(len(out_masks), dtype=bool)
                )
            kept_masks = out_masks[keep]
            kept_scores = (
                out_scores[keep]
                if out_scores.size
                else np.asarray([], dtype="float32")
            )
            kept_ids = (
                out_ids[keep]
                if out_ids.size
                else np.asarray([], dtype="int64")
            )
            if kept_masks.size == 0:
                continue
            union = kept_masks.astype(bool).any(axis=0).astype("float32")
            masks[frame_idx] = union
            detections[frame_idx] = {
                "local_frame": frame_idx,
                "objects": int(len(kept_masks)),
                "object_ids": [int(v) for v in kept_ids.tolist()],
                "max_score": (
                    float(kept_scores.max()) if kept_scores.size else 1.0
                ),
            }
        return masks, detections

    baseline_masks = None
    baseline_detections = None
    selected_object_id = None
    if points:
        # SAM3.1 point refinement expects an established text track. Propagate
        # text discovery first, then refine the selected object and propagate
        # the updated track. This also gives us a safe fallback if SAM2 loses
        # the refined object on an individual frame. Once an object is selected,
        # exclude other text-discovered instances from the refined stem mask.
        selected_object_id = _select_prompt_object_id(
            prompt_response.get("outputs", {}),
            points,
            labels,
            np,
        )
        try:
            baseline_masks, baseline_detections = collect_propagation(
                selected_object_id
            )
        except RuntimeError as exc:
            if str(exc) != "No points are provided; please add points first":
                raise
            # The SAM3.1 multiplex text tracker can lose its conditioning
            # state before point refinement is applied. In that one case,
            # skip the optional text-only fallback and let the supplied point
            # initialize the selected object directly.
            baseline_masks = None
            baseline_detections = None
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": propagation_start,
                "points": points,
                "point_labels": labels,
                "clear_old_points": True,
                "obj_id": int(selected_object_id),
                "rel_coordinates": True,
                "output_prob_thresh": float(score_threshold),
            }
        )

    masks, detections = collect_propagation(selected_object_id)
    if baseline_masks is not None and baseline_detections is not None:
        for frame_idx in range(frame_count):
            if masks[frame_idx] is None and baseline_masks[frame_idx] is not None:
                masks[frame_idx] = baseline_masks[frame_idx]
                detections[frame_idx] = baseline_detections[frame_idx]

    first_shape = None
    for mask in masks:
        if mask is not None:
            first_shape = mask.shape
            break
    if first_shape is None:
        return [np.zeros((1, 1), dtype="float32") for _ in range(frame_count)], detections
    return [
        mask if mask is not None else np.zeros(first_shape, dtype="float32")
        for mask in masks
    ], detections


def _select_prompt_object_id(outputs, points, point_labels, np) -> int:
    object_ids = _to_numpy(outputs.get("out_obj_ids", []), np).reshape(-1)
    masks = _to_numpy(outputs.get("out_binary_masks", []), np)
    scores = _to_numpy(outputs.get("out_probs", []), np).reshape(-1)
    if object_ids.size == 0:
        return 1
    if masks.ndim == 2:
        masks = masks[None, ...]
    positive_points = [
        point for point, label in zip(points, point_labels) if int(label) == 1
    ]
    candidates: list[tuple[int, float]] = []
    for index, object_id in enumerate(object_ids.tolist()):
        hit_count = 0
        if index < len(masks):
            height, width = masks[index].shape[-2:]
            for x_norm, y_norm in positive_points:
                x = max(0, min(int(round(float(x_norm) * (width - 1))), width - 1))
                y = max(0, min(int(round(float(y_norm) * (height - 1))), height - 1))
                if masks[index][y, x] > 0:
                    hit_count += 1
        score = float(scores[index]) if index < len(scores) else 0.0
        candidates.append((int(object_id), hit_count * 10.0 + score))
    return max(candidates, key=lambda item: item[1])[0]


def _resize_mask(mask, size: tuple[int, int], Image, np, resample) -> Any:
    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype("uint8"))
    image = image.resize(size, resample)
    return np.asarray(image).astype("float32") / 255.0


def _video_token_mask(
    frame_masks,
    original_frame_indices: list[int],
    frame_count: int,
    token_shape: tuple[int, int, int],
    Image,
    np,
) -> list[float]:
    token_t, token_h, token_w = token_shape
    if not frame_masks:
        return [0.0] * (token_t * token_h * token_w)
    spatial = [
        _resize_mask(mask, (token_w, token_h), Image, np, Image.Resampling.BOX)
        for mask in frame_masks
    ]
    sample_positions = np.asarray(original_frame_indices, dtype="float32")
    token_positions = (
        (np.arange(token_t, dtype="float32") + 0.5)
        * float(max(frame_count, 1))
        / float(token_t)
        - 0.5
    )
    nearest = np.abs(token_positions[:, None] - sample_positions[None, :]).argmin(axis=1)
    token_masks = np.stack([spatial[int(idx)] for idx in nearest], axis=0)
    return np.clip(token_masks, 0.0, 1.0).reshape(-1).tolist()


def _overlay_image(image, mask, color: tuple[int, int, int], Image, np, alpha: float = 0.45):
    base = np.asarray(image.convert("RGB")).astype("float32")
    if mask.shape[:2] != base.shape[:2]:
        mask = _resize_mask(mask, image.size, Image, np, Image.Resampling.NEAREST)
    mask = np.clip(mask.astype("float32"), 0.0, 1.0)
    tint = np.asarray(color, dtype="float32").reshape(1, 1, 3)
    mix = (mask[..., None] * float(alpha)).clip(0.0, 1.0)
    out = base * (1.0 - mix) + tint * mix
    return out.clip(0, 255).astype("uint8")


def _write_review_artifacts(
    work_dir: Path,
    frame_images,
    original_frame_indices: list[int],
    stem_masks: dict[str, list[Any]],
    stem_prompts: dict[str, str],
    Image,
    np,
) -> dict[str, Any]:
    masks_root = work_dir / "masks"
    overlay_root = work_dir / "overlays"
    per_stem_overlay_root = overlay_root / "per_stem"
    combined_root = overlay_root / "combined"
    masks_root.mkdir(parents=True, exist_ok=True)
    per_stem_overlay_root.mkdir(parents=True, exist_ok=True)
    combined_root.mkdir(parents=True, exist_ok=True)

    legend = {"stems": {}, "combined_frames": []}
    labels = list(stem_masks)
    for stem_idx, label in enumerate(labels):
        color = _palette(stem_idx)
        safe = _safe_name(label)
        stem_mask_dir = masks_root / safe
        stem_overlay_dir = per_stem_overlay_root / safe
        stem_mask_dir.mkdir(parents=True, exist_ok=True)
        stem_overlay_dir.mkdir(parents=True, exist_ok=True)
        legend["stems"][label] = {
            "color_rgb": list(color),
            "prompt": stem_prompts.get(label, ""),
            "mask_directory": str(stem_mask_dir.relative_to(work_dir)),
            "overlay_directory": str(stem_overlay_dir.relative_to(work_dir)),
            "frames": [],
        }
        for original_idx, image, mask in zip(
            original_frame_indices, frame_images, stem_masks[label]
        ):
            if mask.shape[:2] != np.asarray(image).shape[:2]:
                mask = _resize_mask(mask, image.size, Image, np, Image.Resampling.NEAREST)
            mask_path = stem_mask_dir / f"frame_{original_idx:05d}.png"
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype("uint8")).save(mask_path)
            overlay = _overlay_image(image, mask, color, Image, np)
            overlay_path = stem_overlay_dir / f"frame_{original_idx:05d}.png"
            Image.fromarray(overlay).save(overlay_path)
            legend["stems"][label]["frames"].append(
                {
                    "frame": int(original_idx),
                    "mask": str(mask_path.relative_to(work_dir)),
                    "overlay": str(overlay_path.relative_to(work_dir)),
                }
            )

    for frame_pos, (original_idx, image) in enumerate(
        zip(original_frame_indices, frame_images)
    ):
        combined = np.asarray(image.convert("RGB")).astype("float32")
        for stem_idx, label in enumerate(labels):
            mask = stem_masks[label][frame_pos]
            if mask.shape[:2] != combined.shape[:2]:
                mask = _resize_mask(mask, image.size, Image, np, Image.Resampling.NEAREST)
            color = np.asarray(_palette(stem_idx), dtype="float32").reshape(1, 1, 3)
            mix = (np.clip(mask, 0.0, 1.0)[..., None] * 0.35).clip(0.0, 1.0)
            combined = combined * (1.0 - mix) + color * mix
        combined_path = combined_root / f"frame_{original_idx:05d}.png"
        Image.fromarray(combined.clip(0, 255).astype("uint8")).save(combined_path)
        legend["combined_frames"].append(str(combined_path.relative_to(work_dir)))

    legend_path = overlay_root / "legend.json"
    legend_path.write_text(json.dumps(legend, indent=2) + "\n", encoding="utf-8")
    return {
        "mask_root": str(masks_root),
        "overlay_root": str(overlay_root),
        "legend": str(legend_path),
        "combined_dir": str(combined_root),
        "per_stem_dir": str(per_stem_overlay_root),
    }


def main() -> int:
    args = parse_args()
    token_shape = _parse_token_shape(args.token_shape)
    original_frame_indices = _parse_indices(args.frame_indices)
    prompt_spec = _load_prompt_spec(Path(args.prompts_json))
    modules = _require_runtime_modules(args.sam3_repo)
    np = modules["np"]
    Image = modules["Image"]
    predictor = _build_predictor(
        modules, args.predictor_kind, args.device, args.checkpoint_path
    )
    _filter_model_init_state_kwargs(predictor)

    frames_dir = Path(args.frames_dir)
    frame_images = _load_frames(frames_dir, Image)
    if len(frame_images) != len(original_frame_indices):
        raise RuntimeError(
            f"Frame count mismatch: {len(frame_images)} images but "
            f"{len(original_frame_indices)} frame indices"
        )

    stems_out: dict[str, dict[str, Any]] = {}
    stem_masks: dict[str, list[Any]] = {}
    stem_prompts: dict[str, str] = {}

    for stem_idx, stem in enumerate(prompt_spec.get("stems", [])):
        label = str(stem.get("id") or f"stem_{stem_idx}")
        prompt = _normalize_prompt(stem.get("prompt") or stem.get("entity") or "")
        if not prompt:
            empty_masks = [np.zeros(np.asarray(image).shape[:2], dtype="float32") for image in frame_images]
            stems_out[label] = {
                "token_mask": [0.0] * (token_shape[0] * token_shape[1] * token_shape[2]),
                "prompt": "",
                "source": "disabled:no_prompt",
                "detections": [],
                "point_refinement": {"frame": None, "points": []},
            }
            stem_masks[label] = empty_masks
            stem_prompts[label] = ""
            continue
        response = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(frames_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": bool(args.offload_state_to_cpu),
            }
        )
        session_id = response["session_id"]
        try:
            point_items = stem.get("points") or []
            points = [
                [float(item["x"]), float(item["y"])]
                for item in point_items
                if isinstance(item, dict) and "x" in item and "y" in item
            ]
            point_labels = [
                0 if str(item.get("label", "positive")).lower() == "negative" else 1
                for item in point_items
                if isinstance(item, dict) and "x" in item and "y" in item
            ]
            requested_frame = stem.get("point_frame_index")
            point_frame_index = None
            if requested_frame is not None and original_frame_indices:
                point_frame_index = min(
                    range(len(original_frame_indices)),
                    key=lambda index: abs(
                        int(original_frame_indices[index]) - int(requested_frame)
                    ),
                )
            masks, detections = _propagate_prompt(
                predictor,
                session_id,
                prompt,
                min(max(int(args.prompt_frame_index), 0), max(len(frame_images) - 1, 0)),
                float(args.score_threshold),
                len(frame_images),
                np,
                points=points,
                point_labels=point_labels,
                point_frame_index=point_frame_index,
            )
            token_mask = _video_token_mask(
                masks,
                original_frame_indices,
                args.frame_count,
                token_shape,
                Image,
                np,
            )
            stems_out[label] = {
                "token_mask": token_mask,
                "prompt": prompt,
                "source": f"sam3_video_backend:{args.predictor_kind}",
                "detections": [
                    {**item, "frame": int(original_frame_indices[item["local_frame"]])}
                    for item in detections
                ],
                "point_refinement": {
                    "frame": (
                        int(requested_frame)
                        if requested_frame is not None
                        else None
                    ),
                    "points": point_items,
                },
            }
            stem_masks[label] = masks
            stem_prompts[label] = prompt
        finally:
            predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
            _clear_cuda_cache(modules)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = _write_review_artifacts(
        out_path.parent,
        frame_images,
        original_frame_indices,
        stem_masks,
        stem_prompts,
        Image,
        np,
    )
    output = {
        "format": "stage2_a2v_token_masks_v1",
        "case_id": prompt_spec.get("case_id", ""),
        "token_shape": list(token_shape),
        "token_count": token_shape[0] * token_shape[1] * token_shape[2],
        "frame_count": int(args.frame_count),
        "sampled_frame_indices": original_frame_indices,
        "sam3": {
            "predictor_kind": args.predictor_kind,
            "prompt_frame_index": int(args.prompt_frame_index),
            "score_threshold": float(args.score_threshold),
        },
        "overlays": artifacts,
        "stems": stems_out,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
