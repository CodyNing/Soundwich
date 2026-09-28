"""Output helpers for decoded multi-stem audio."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_and_mix(
    stems: Sequence[np.ndarray],
    volumes: Sequence[float] | None = None,
    *,
    target_peak: float = 0.90,
    silence_peak: float = 1e-4,
) -> tuple[list[np.ndarray], np.ndarray]:
    if not stems:
        raise ValueError("at least one stem is required")
    arrays = [np.asarray(stem, dtype=np.float32).reshape(-1) for stem in stems]
    length = min(array.shape[0] for array in arrays)
    arrays = [array[:length] for array in arrays]
    weights = list(volumes) if volumes is not None else [1.0] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError("volume count must match stem count")

    normalized: list[np.ndarray] = []
    for array, volume in zip(arrays, weights, strict=True):
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if peak >= silence_peak:
            array = array * (target_peak / peak)
        normalized.append(array * float(volume))
    mixed = np.sum(np.stack(normalized, axis=0), axis=0)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return normalized, mixed.astype(np.float32, copy=False)
