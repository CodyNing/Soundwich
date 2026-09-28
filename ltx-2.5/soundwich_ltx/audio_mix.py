"""Safe per-stem waveform normalization and mixing."""

from __future__ import annotations

import torch


def mix_waveforms(
    waveforms: torch.Tensor,
    volumes: list[float],
    *,
    target_peak: float = 0.95,
    maximum_gain: float = 4.0,
    silence_floor: float = 1e-3,
) -> torch.Tensor:
    """Peak-normalize real stems without boosting failed near-silent lanes."""
    if waveforms.ndim != 3 or waveforms.shape[0] != len(volumes):
        raise ValueError("waveforms and volumes must contain one row per stem")
    shape = (waveforms.shape[0], 1, 1)
    peaks = waveforms.abs().reshape(waveforms.shape[0], -1).amax(dim=1).reshape(shape)
    target = torch.as_tensor(target_peak, dtype=waveforms.dtype, device=waveforms.device)
    gain = (target / peaks.clamp_min(1e-8)).clamp(max=maximum_gain)
    gain = torch.where(peaks >= silence_floor, gain, torch.ones_like(gain))
    normalized = torch.where(peaks > 1e-8, waveforms * gain, waveforms)
    weights = torch.tensor(volumes, dtype=waveforms.dtype, device=waveforms.device).reshape(shape)
    mixed = (normalized * weights).sum(dim=0)
    peak = mixed.abs().amax()
    if bool(peak > target):
        mixed = mixed * (target / peak.clamp_min(1e-8))
    return mixed
