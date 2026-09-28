"""Pure tensor utilities for mean-carrier timeline control."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _soft_window(
    positions: torch.Tensor,
    start: float,
    end: float,
    feather: float,
) -> torch.Tensor:
    start = max(0.0, min(float(start), 1.0))
    end = max(0.0, min(float(end), 1.0))
    if end <= start:
        return torch.zeros_like(positions)
    feather = max(float(feather), 0.0)
    if feather == 0.0:
        return ((positions >= start) & (positions <= end)).to(positions.dtype)
    left = ((positions - (start - feather)) / feather).clamp(0.0, 1.0)
    right = (((end + feather) - positions) / feather).clamp(0.0, 1.0)
    return torch.minimum(left, right)


def build_timeline_gates(
    windows_by_stem: Sequence[Sequence[tuple[float, float]]],
    sequence_length: int,
    *,
    duration_seconds: float,
    feather_seconds: float = 0.20,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return in-window and controlled-outside gates with shape ``[N, L]``.

    Window inputs are expressed in seconds. A stem without windows is treated as
    continuous and receives neither activation nor suppression blending.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    positions = (
        torch.arange(sequence_length, device=device, dtype=torch.float32) + 0.5
    ) / float(sequence_length)
    feather = max(float(feather_seconds), 0.0) / float(duration_seconds)
    rows: list[torch.Tensor] = []
    controlled: list[float] = []
    for windows in windows_by_stem:
        row = torch.zeros_like(positions)
        valid_count = 0
        for start_seconds, end_seconds in windows:
            start = float(start_seconds) / float(duration_seconds)
            end = float(end_seconds) / float(duration_seconds)
            if end <= start:
                continue
            row = torch.maximum(row, _soft_window(positions, start, end, feather))
            valid_count += 1
        rows.append(row)
        controlled.append(1.0 if valid_count else 0.0)

    if not rows:
        empty = torch.empty(0, sequence_length, device=device, dtype=dtype)
        return empty, empty
    inside = torch.stack(rows, dim=0).to(dtype=dtype)
    controlled_tensor = torch.tensor(
        controlled, device=inside.device, dtype=dtype
    ).unsqueeze(1)
    outside = (1.0 - inside) * controlled_tensor
    return inside, outside


def select_mean_token(
    hidden: torch.Tensor,
    *,
    quantile: float,
    mode: str,
) -> torch.Tensor:
    """Select tokens by hidden-state RMS and return their mean ``[1, C]``.

    ``mode='top'`` selects energy at or above the requested quantile.
    ``mode='bottom'`` selects energy at or below the requested quantile.
    """
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, channels]")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if mode not in {"top", "bottom"}:
        raise ValueError("mode must be 'top' or 'bottom'")

    energy = hidden.detach().float().pow(2).mean(dim=-1).sqrt()
    threshold = torch.quantile(energy, float(quantile))
    selected = energy >= threshold if mode == "top" else energy <= threshold
    if not bool(selected.any()):
        selected[energy.argmax() if mode == "top" else energy.argmin()] = True
    return hidden[selected].mean(dim=0, keepdim=True)


def blend_mean_carriers(
    hidden: torch.Tensor,
    *,
    real_stem_count: int,
    activation_index: int,
    suppression_index: int,
    inside_gate: torch.Tensor,
    outside_gate: torch.Tensor,
    activation_quantile: float = 0.70,
    suppression_quantile: float = 0.30,
    inside_strength: float = 0.10,
    outside_strength: float = 0.50,
    outside_suppression: float = 0.25,
    activation_value_scale: float = 0.80,
) -> torch.Tensor:
    """Blend repeated activation/suppression mean tokens into real stems.

    The activation carrier is RMS-matched to each real stem inside its active
    timeline, then added as a residual. The suppression carrier keeps its own
    naturally low-energy scale and is never RMS-matched to the real stem.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [branches, tokens, channels]")
    branch_count, sequence_length, _ = hidden.shape
    if not 0 < real_stem_count <= branch_count:
        raise ValueError("invalid real_stem_count")
    if activation_index >= branch_count or suppression_index >= branch_count:
        raise ValueError("carrier index outside hidden batch")
    expected = (real_stem_count, sequence_length)
    if tuple(inside_gate.shape) != expected or tuple(outside_gate.shape) != expected:
        raise ValueError(f"timeline gates must have shape {expected}")

    alpha = max(0.0, min(float(inside_strength), 1.0))
    beta = max(0.0, min(float(outside_strength), 1.0))
    suppress = max(0.0, min(float(outside_suppression), 1.0))
    value_scale = max(0.0, float(activation_value_scale))
    if alpha == 0.0 and beta == 0.0 and suppress == 0.0:
        return hidden

    activation = select_mean_token(
        hidden[activation_index], quantile=activation_quantile, mode="top"
    )
    suppression = select_mean_token(
        hidden[suppression_index], quantile=suppression_quantile, mode="bottom"
    )
    result = hidden.clone()
    result[:real_stem_count] = blend_cached_carriers(
        result[:real_stem_count],
        activation=activation,
        suppression=suppression,
        inside_gate=inside_gate,
        outside_gate=outside_gate,
        inside_strength=inside_strength,
        outside_strength=outside_strength,
        outside_suppression=outside_suppression,
        activation_value_scale=activation_value_scale,
    )
    return result


def blend_cached_carriers(
    real: torch.Tensor,
    *,
    activation: torch.Tensor,
    suppression: torch.Tensor,
    inside_gate: torch.Tensor,
    outside_gate: torch.Tensor,
    inside_strength: float = 0.10,
    outside_strength: float = 0.50,
    outside_suppression: float = 0.25,
    activation_value_scale: float = 0.80,
) -> torch.Tensor:
    """Blend saved ``[1, C]`` carrier tokens into real audio branches."""
    if real.ndim != 3:
        raise ValueError("real must have shape [branches, tokens, channels]")
    branch_count, sequence_length, channels = real.shape
    if tuple(activation.shape) == (1, channels):
        activation = activation.expand(branch_count, -1)
    elif tuple(activation.shape) != (branch_count, channels):
        raise ValueError(
            "activation carrier must have shape [1, channels] or "
            "[branches, channels]"
        )
    if tuple(suppression.shape) == (1, channels):
        suppression = suppression.expand(branch_count, -1)
    elif tuple(suppression.shape) != (branch_count, channels):
        raise ValueError(
            "suppression carrier must have shape [1, channels] or "
            "[branches, channels]"
        )
    expected = (branch_count, sequence_length)
    if tuple(inside_gate.shape) != expected or tuple(outside_gate.shape) != expected:
        raise ValueError(f"timeline gates must have shape {expected}")

    alpha = max(0.0, min(float(inside_strength), 1.0))
    beta = max(0.0, min(float(outside_strength), 1.0))
    suppress = max(0.0, min(float(outside_suppression), 1.0))
    value_scale = max(0.0, float(activation_value_scale))
    if alpha == 0.0 and beta == 0.0 and suppress == 0.0:
        return real

    activation = activation.to(device=real.device, dtype=real.dtype)
    suppression = suppression.to(device=real.device, dtype=real.dtype)
    inside = inside_gate.to(device=real.device, dtype=real.dtype).unsqueeze(-1)
    outside = outside_gate.to(device=real.device, dtype=real.dtype).unsqueeze(-1)
    real = real.clone()

    # Repeat the mean activation token over each requested timeline. Match its
    # RMS to the corresponding real stem only inside that timeline. This is an
    # additive residual: attenuating the real state here compounds across all
    # transformer blocks and can erase the requested sound entirely.
    activation_guide = activation.unsqueeze(1) * inside
    gate = inside.detach().float()
    denom = (gate.sum(dim=(1, 2), keepdim=True) * real.shape[-1]).clamp_min(1.0)
    target_rms = (
        (real.detach().float().pow(2) * gate).sum(dim=(1, 2), keepdim=True)
        .div(denom)
        .sqrt()
        .to(device=real.device, dtype=real.dtype)
    ).clamp_min(1e-6)
    guide_rms = (
        (activation_guide.detach().float().pow(2) * gate)
        .sum(dim=(1, 2), keepdim=True)
        .div(denom)
        .sqrt()
        .to(device=real.device, dtype=real.dtype)
    ).clamp_min(1e-6)
    activation_guide = (
        activation_guide
        * (target_rms / guide_rms).clamp(0.0, 4.0)
        * value_scale
    )

    # Apply multiplicative outside suppression first, then interpolate toward
    # the quiet carrier at its original low RMS.
    real = real * (1.0 - suppress * outside)
    silence_mix = beta * outside
    real = real * (1.0 - silence_mix) + suppression.unsqueeze(1) * silence_mix
    real = real + alpha * activation_guide
    return real
