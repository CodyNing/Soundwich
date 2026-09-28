"""Stage-1 frame export and the isolated SAM3 subprocess that produces entity masks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image

from soundwich_ltx.config import MultiStemConfig, StemConfig


def sample_indices(frame_count: int, stride: int, max_frames: int) -> list[int]:
    indices = list(range(0, frame_count, max(stride, 1)))
    if not indices or indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if len(indices) > max_frames:
        step = (len(indices) - 1) / max(max_frames - 1, 1)
        indices = [indices[round(index * step)] for index in range(max_frames)]
    return sorted(set(indices))


def export_sampled_frames(
    chunks: Iterator[torch.Tensor],
    *,
    frames_dir: Path,
    indices: list[int],
) -> Iterator[torch.Tensor]:
    """Save selected decoded frames while forwarding every chunk to the video encoder."""
    # Start from an empty folder so frames from an earlier run of the same scene cannot linger.
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    selected = set(indices)
    frame_index = 0
    saved_index = 0
    for chunk in chunks:
        cpu = chunk.detach().float().cpu().clamp(0, 1)
        for frame in cpu:
            if frame_index in selected:
                pixels = (frame.numpy() * 255).round().astype("uint8")
                Image.fromarray(pixels).save(frames_dir / f"{saved_index:05d}.jpg", quality=95)
                saved_index += 1
            frame_index += 1
        yield chunk
    if saved_index != len(indices):
        raise RuntimeError(f"decoded {frame_index} frames but exported {saved_index}/{len(indices)} SAM frames")


def read_token_masks(config: MultiStemConfig, path: Path) -> dict[int, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = payload.get("stems", {})
    masks: dict[int, list[float]] = {}
    for index, stem in enumerate(config.stems):
        item = by_id.get(stem.id, {}) if isinstance(by_id, dict) else {}
        values = item.get("token_mask") if isinstance(item, dict) else None
        if not isinstance(values, list) or not values:
            raise ValueError(f"SAM3 output has no token mask for {stem.id}")
        masks[index] = [float(value) for value in values]
    return masks


def _stem_prompt(stem: StemConfig) -> dict[str, object]:
    """Text prompt plus optional reviewed click points (normalized x/y on one Stage-1 frame)."""
    item: dict[str, object] = {"id": stem.id, "prompt": stem.sam_prompt}
    if stem.sam_points:
        item["points"] = [dict(point) for point in stem.sam_points]
        item["point_frame_index"] = stem.sam_point_frame
    return item


def run_sam3(
    config: MultiStemConfig,
    *,
    run_root: Path,
    frames_dir: Path,
    frame_indices: list[int],
    token_shape: tuple[int, int, int],
) -> tuple[dict[int, list[float]], Path]:
    """Segment every stem's ``sam_prompt`` (and reviewed points) on the Stage-1 frames and pool to Stage-2 tokens."""
    missing = config.sam3.missing_settings()
    if missing:
        raise RuntimeError("SAM3 is not configured:\n" + "\n".join(f"  - {item}" for item in missing))
    sam_root = run_root / "sam_outputs"
    sam_root.mkdir(parents=True, exist_ok=True)
    prompt_path = sam_root / "prompts.json"
    output_path = sam_root / "stage2_a2v_mask.json"
    prompts = {"case_id": config.id, "stems": [_stem_prompt(stem) for stem in config.stems]}
    prompt_path.write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")
    command = [
        config.sam3.python,
        config.sam3.backend_script,
        "--frames-dir",
        str(frames_dir),
        "--prompts-json",
        str(prompt_path),
        "--output",
        str(output_path),
        "--token-shape",
        ",".join(str(value) for value in token_shape),
        "--frame-count",
        str(config.generation.frames),
        "--frame-indices",
        ",".join(str(value) for value in frame_indices),
        "--predictor-kind",
        config.sam3.predictor_kind,
        "--device",
        config.sam3.device,
        "--prompt-frame-index",
        str(config.sam3.prompt_frame_index),
        "--score-threshold",
        str(config.sam3.score_threshold),
    ]
    env = os.environ.copy()
    if config.sam3.repo:
        command.extend(["--sam3-repo", config.sam3.repo])
        env["PYTHONPATH"] = config.sam3.repo + os.pathsep + env.get("PYTHONPATH", "")
    if config.sam3.checkpoint:
        command.extend(["--checkpoint-path", config.sam3.checkpoint])
    result = subprocess.run(command, env=env, cwd=str(run_root), text=True, capture_output=True, check=False)
    (sam_root / "backend_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (sam_root / "backend_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"SAM3 failed with code {result.returncode}; inspect {sam_root}")
    return read_token_masks(config, output_path), output_path
