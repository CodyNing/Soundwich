"""Timeline gates and carrier replay math independent of model loading."""

from __future__ import annotations

import torch

from soundwich_ltx.carrier_bank import CarrierRecord
from soundwich_ltx.config import BlendConfig, Window


def scene_union_windows(stem_windows: list[tuple[Window, ...]]) -> tuple[Window, ...]:
    """Return the scene-active union; an uncontrolled stem makes the scene full-range."""
    if any(not windows for windows in stem_windows):
        return ()
    return tuple(window for windows in stem_windows for window in windows)


def normalized_audio_positions(
    positions: torch.Tensor | None,
    *,
    token_count: int,
    duration_seconds: float,
    device: torch.device,
) -> torch.Tensor:
    """Return token-center positions in [0, 1], preferring native audio coordinates."""
    if positions is not None and positions.ndim == 4 and positions.shape[1] == 1 and positions.shape[2] == token_count:
        bounds = positions[0, 0].to(device=device, dtype=torch.float32)
        return bounds.mean(dim=-1).div(max(duration_seconds, 1e-6)).clamp(0, 1)
    return (torch.arange(token_count, device=device, dtype=torch.float32) + 0.5) / token_count


def timeline_gate(
    windows: tuple[Window, ...],
    positions: torch.Tensor,
    *,
    duration_seconds: float,
    feather_seconds: float,
) -> torch.Tensor:
    gate = torch.zeros_like(positions)
    feather = max(feather_seconds / max(duration_seconds, 1e-6), 0.0)
    for window in windows:
        start = window.start / duration_seconds
        end = window.end / duration_seconds
        if feather > 0:
            left = ((positions - (start - feather)) / feather).clamp(0, 1)
            right = (((end + feather) - positions) / feather).clamp(0, 1)
            value = torch.minimum(left, right)
        else:
            value = ((positions >= start) & (positions <= end)).float()
        gate = torch.maximum(gate, value)
    return gate


def _suppression_token(record: CarrierRecord, hidden: torch.Tensor) -> torch.Tensor:
    return record.restore(device=hidden.device, dtype=hidden.dtype, own_rms=True).view(1, hidden.shape[-1])


def replay_carriers(
    hidden: torch.Tensor,
    *,
    activation: list[CarrierRecord],
    suppression: CarrierRecord,
    windows: list[tuple[Window, ...]],
    positions: torch.Tensor | None,
    duration_seconds: float,
    blend: BlendConfig,
    stem_blend_strengths: list[float | None] | None = None,
    activation_strength: float | None = None,
) -> torch.Tensor:
    """Apply activation inside each controlled window and suppression outside it."""
    if hidden.ndim != 3 or hidden.shape[0] != len(windows) or len(activation) != len(windows):
        raise ValueError("hidden, activation carriers, and windows must have one row per stem")
    stems, token_count, dim = hidden.shape
    token_positions = normalized_audio_positions(
        positions,
        token_count=token_count,
        duration_seconds=duration_seconds,
        device=hidden.device,
    )
    output = hidden.clone()
    suppression_token = _suppression_token(suppression, hidden)

    strengths = stem_blend_strengths or [None] * stems
    if len(strengths) != stems:
        raise ValueError("stem_blend_strengths must have one value per stem")
    for stem_index in range(stems):
        if not windows[stem_index]:
            continue
        gate = (
            timeline_gate(
                windows[stem_index],
                token_positions,
                duration_seconds=duration_seconds,
                feather_seconds=blend.feather_seconds,
            )
            .to(dtype=hidden.dtype)
            .view(token_count, 1)
        )
        outside = 1 - gate
        carrier = (
            activation[stem_index]
            .restore(
                device=hidden.device,
                dtype=hidden.dtype,
                own_rms=False,
            )
            .view(1, dim)
        )
        moved = carrier.expand(token_count, dim) * gate
        denominator = (gate.float().sum() * dim).clamp_min(1.0)
        target_rms = ((hidden[stem_index].float().square() * gate.float()).sum() / denominator).sqrt().clamp_min(1e-6)
        guide_rms = ((moved.float().square() * gate.float()).sum() / denominator).sqrt().clamp_min(1e-6)
        rms_scale = (target_rms / guide_rms).clamp(0, blend.rms_scale_limit).to(hidden.dtype)
        guide = moved * rms_scale * blend.value_scale

        base = hidden[stem_index] * (1 - blend.outside_suppression * outside)
        silence_mix = blend.outside_silence_blend * outside
        base = base * (1 - silence_mix) + suppression_token * silence_mix
        scheduled_strength = blend.strength if activation_strength is None else float(activation_strength)
        strength = scheduled_strength if strengths[stem_index] is None else float(strengths[stem_index])
        output[stem_index] = base + strength * gate * guide
    return output


def replay_suppression(
    hidden: torch.Tensor,
    *,
    suppression: CarrierRecord,
    windows: list[tuple[Window, ...]],
    positions: torch.Tensor | None,
    duration_seconds: float,
    blend: BlendConfig,
) -> torch.Tensor:
    """Keep native hidden states in-window and blend the quiet carrier outside."""
    if hidden.ndim != 3 or hidden.shape[0] != len(windows):
        raise ValueError("hidden and windows must have one row per stem")
    _stems, token_count, _dim = hidden.shape
    token_positions = normalized_audio_positions(
        positions,
        token_count=token_count,
        duration_seconds=duration_seconds,
        device=hidden.device,
    )
    suppression_token = _suppression_token(suppression, hidden)
    output = hidden.clone()
    for stem_index, stem_windows in enumerate(windows):
        if not stem_windows:
            continue
        gate = timeline_gate(
            stem_windows,
            token_positions,
            duration_seconds=duration_seconds,
            feather_seconds=blend.feather_seconds,
        ).to(dtype=hidden.dtype).view(token_count, 1)
        outside = 1 - gate
        base = hidden[stem_index] * (1 - blend.outside_suppression * outside)
        silence_mix = blend.outside_silence_blend * outside
        output[stem_index] = base * (1 - silence_mix) + suppression_token * silence_mix
    return output
