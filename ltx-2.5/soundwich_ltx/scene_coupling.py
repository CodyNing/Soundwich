"""Scene-lane gather/broadcast attention shared by the Stage-1 and Stage-2 hooks."""

from __future__ import annotations

from typing import Protocol

import torch

from soundwich_ltx.config import SceneCouplingConfig, Window
from soundwich_ltx.timeline import normalized_audio_positions, timeline_gate


class NativeAttention(Protocol):
    heads: int
    to_q: torch.nn.Module
    to_k: torch.nn.Module
    to_v: torch.nn.Module
    to_out: torch.nn.Module
    to_gate_logits: torch.nn.Module | None

    def preattention_function(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        module: torch.nn.Module,
        mask: torch.Tensor | None,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def attention_function(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor: ...

    def gated_attention_function(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        module: torch.nn.Module,
    ) -> torch.Tensor: ...


def _select_pe_rows(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    rows: list[int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    selected: list[torch.Tensor] = []
    for tensor in pe:
        if tensor.shape[0] == 1:
            selected.append(tensor)
        else:
            index = torch.tensor(rows, device=tensor.device, dtype=torch.long)
            selected.append(tensor.index_select(0, index))
    return selected[0], selected[1]


def _concat_pe_rows(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    rows: list[int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Concatenate positional encodings from several lanes along the token axis."""
    if pe is None:
        return None
    concatenated: list[torch.Tensor] = []
    for tensor in pe:
        if tensor.shape[0] == 1:
            selected = tensor.expand(len(rows), *tensor.shape[1:])
        else:
            index = torch.tensor(rows, device=tensor.device, dtype=torch.long)
            selected = tensor.index_select(0, index)
        if selected.ndim == 4:
            lanes, heads, tokens, dim = selected.shape
            selected = selected.permute(1, 0, 2, 3).reshape(1, heads, lanes * tokens, dim)
        elif selected.ndim == 3:
            lanes, tokens, dim = selected.shape
            selected = selected.reshape(1, lanes * tokens, dim)
        else:
            raise ValueError(f"unsupported positional encoding shape: {tuple(selected.shape)}")
        concatenated.append(selected)
    return concatenated[0], concatenated[1]


def _pair_pe_rows(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    real_rows: list[int],
    scene_row: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pair every real lane's positional encoding with the shared scene lane."""
    if pe is None:
        return None
    paired: list[torch.Tensor] = []
    for tensor in pe:
        if tensor.shape[0] == 1:
            real = tensor.expand(len(real_rows), *tensor.shape[1:])
            scene = real
        else:
            index = torch.tensor(real_rows, device=tensor.device, dtype=torch.long)
            real = tensor.index_select(0, index)
            scene = tensor[scene_row : scene_row + 1].expand(len(real_rows), *tensor.shape[1:])
        paired.append(torch.cat([real, scene], dim=-2))
    return paired[0], paired[1]


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)


def _magnitude_preserving_mix(
    base: torch.Tensor,
    update: torch.Tensor,
    *,
    strength: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    if strength <= 0:
        return base
    base_rms = _rms(base)
    update = update * (base_rms / _rms(update)).to(update.dtype)
    if gate is not None:
        update = update * gate
    mixed = base + float(strength) * update
    return mixed * (base_rms / _rms(mixed)).to(mixed.dtype)


def _magnitude_preserving_interpolate(
    base: torch.Tensor,
    update: torch.Tensor,
    *,
    strength: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Interpolate to a full shared-attention replacement at strength one."""
    if strength <= 0:
        return base
    base_rms = _rms(base)
    update = update * (base_rms / _rms(update)).to(update.dtype)
    amount: float | torch.Tensor = float(strength)
    if gate is not None:
        amount = amount * gate
    mixed = base + amount * (update - base)
    return mixed * (base_rms / _rms(mixed)).to(mixed.dtype)


class SceneCoupler:
    """Gather real stems into the persistent scene lane and broadcast it back."""

    def __init__(
        self,
        *,
        config: SceneCouplingConfig,
        num_real_stems: int,
        stem_windows: list[tuple[Window, ...]],
        stem_scene_context: list[bool] | None = None,
        duration_seconds: float,
        feather_seconds: float,
    ) -> None:
        self.config = config
        self.num_real_stems = num_real_stems
        self.stem_windows = stem_windows
        self.stem_scene_context = (
            list(stem_scene_context) if stem_scene_context is not None else [True] * num_real_stems
        )
        if len(self.stem_scene_context) != num_real_stems:
            raise ValueError("scene-context participation must match the real stem count")
        self.duration_seconds = duration_seconds
        self.feather_seconds = feather_seconds

    @property
    def uses_scene_lane(self) -> bool:
        return self.config.uses_scene_lane

    @property
    def audio_lane_count(self) -> int:
        return self.num_real_stems + int(self.uses_scene_lane)

    @property
    def participating_windows(self) -> list[tuple[Window, ...]]:
        return [
            windows
            for windows, enabled in zip(self.stem_windows, self.stem_scene_context, strict=True)
            if enabled
        ]

    def _gates(self, hidden: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """Per-stem timeline gates; stems excluded from scene context get zero."""
        token_positions = normalized_audio_positions(
            positions,
            token_count=hidden.shape[1],
            duration_seconds=self.duration_seconds,
            device=hidden.device,
        )
        gates = []
        for index, windows in enumerate(self.stem_windows):
            if windows:
                gate = timeline_gate(
                    windows,
                    token_positions,
                    duration_seconds=self.duration_seconds,
                    feather_seconds=self.feather_seconds,
                )
            else:
                gate = torch.ones_like(token_positions)
            if not self.stem_scene_context[index]:
                gate = torch.zeros_like(gate)
            gates.append(gate.to(dtype=hidden.dtype))
        return torch.stack(gates, dim=0).unsqueeze(-1)

    def _scene_reference(self, real_hidden: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum the gated real-stem states, raw or RMS-balanced (``aggregation``)."""
        if self.config.aggregation == "rms_sum":
            stem_rms = _rms(real_hidden)
            normalized = real_hidden / stem_rms.to(real_hidden.dtype)
        else:
            stem_rms = None
            normalized = real_hidden
        scene = (normalized * gates).sum(dim=0, keepdim=True)
        active_union = gates.amax(dim=0, keepdim=True)
        if stem_rms is not None:
            scene = scene * stem_rms.mean(dim=0, keepdim=True).to(scene.dtype)
        return scene, active_union

    def _scene_self_attention(
        self,
        module: NativeAttention,
        *,
        input_hidden: torch.Tensor,
        gates: torch.Tensor,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Run scene queries against scene K/V concatenated with the gated real-stem sum K/V."""
        scene_index = self.num_real_stems
        scene_hidden = input_hidden[scene_index : scene_index + 1]
        reference, _active_union = self._scene_reference(input_hidden[: self.num_real_stems], gates)
        source = torch.cat([scene_hidden, reference], dim=1)
        query = module.to_q(scene_hidden)
        key = module.to_k(source)
        value = module.to_v(source)
        query, key = module.preattention_function(
            query,
            key,
            module,  # type: ignore[arg-type]
            None,
            _select_pe_rows(pe, [scene_index]),
            _concat_pe_rows(pe, [scene_index, scene_index]),
        )
        attended = module.attention_function(query, key, value, module.heads)
        if module.to_gate_logits is not None:
            attended = module.gated_attention_function(scene_hidden, attended, module)  # type: ignore[arg-type]
        return module.to_out(attended)

    def _real_self_attention(
        self,
        module: NativeAttention,
        *,
        input_hidden: torch.Tensor,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Run each real query against its own K/V concatenated with the scene K/V."""
        scene_index = self.num_real_stems
        real_hidden = input_hidden[:scene_index]
        scene_hidden = input_hidden[scene_index : scene_index + 1].expand(self.num_real_stems, -1, -1)
        source = torch.cat([real_hidden, scene_hidden], dim=1)
        real_rows = list(range(self.num_real_stems))
        query = module.to_q(real_hidden)
        key = module.to_k(source)
        value = module.to_v(source)
        query, key = module.preattention_function(
            query,
            key,
            module,  # type: ignore[arg-type]
            None,
            _select_pe_rows(pe, real_rows),
            _pair_pe_rows(pe, real_rows=real_rows, scene_row=scene_index),
        )
        attended = module.attention_function(query, key, value, module.heads)
        if module.to_gate_logits is not None:
            attended = module.gated_attention_function(real_hidden, attended, module)  # type: ignore[arg-type]
        return module.to_out(attended)

    def inject_scene_context(
        self,
        module: NativeAttention,
        *,
        input_hidden: torch.Tensor,
        attention_output: torch.Tensor,
        positions: torch.Tensor | None,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        branch: str,
        scene_self_attention_enabled: bool = True,
    ) -> torch.Tensor:
        """Replace the native audio self-attention output with scene-shared attention."""
        if not self.config.enabled or self.num_real_stems < 2:
            return attention_output
        if branch not in {"audio_positive", "audio_negative"}:
            return attention_output
        if input_hidden.shape[0] != self.audio_lane_count:
            raise ValueError(
                f"scene coupling expected {self.audio_lane_count} audio rows, got {input_hidden.shape[0]}"
            )
        gates = self._gates(input_hidden[: self.num_real_stems], positions)
        result = attention_output.clone()
        if self.config.broadcasts_scene_into_real_attention:
            shared_output = self._real_self_attention(module, input_hidden=input_hidden, pe=pe)
            result[: self.num_real_stems] = _magnitude_preserving_interpolate(
                attention_output[: self.num_real_stems],
                shared_output,
                strength=self.config.scene_to_real_strength,
                gate=gates,
            )
        if (
            self.config.gathers_real_into_scene_attention
            and scene_self_attention_enabled
            and branch == "audio_positive"
        ):
            shared_output = self._scene_self_attention(module, input_hidden=input_hidden, gates=gates, pe=pe)
            strength = self.config.scene_self_attention_strength
            native_output = attention_output[self.num_real_stems : self.num_real_stems + 1]
            result[self.num_real_stems : self.num_real_stems + 1] = (
                (1.0 - strength) * native_output + strength * shared_output
            )
        return result

    def update_scene_lane(
        self,
        *,
        controlled_output: torch.Tensor,
        positions: torch.Tensor | None,
        branch: str,
    ) -> torch.Tensor:
        """Blend the controlled real-stem state into the scene lane after self-attention."""
        if not self.uses_scene_lane or branch != "audio_positive":
            return controlled_output
        real_output = controlled_output[: self.num_real_stems]
        gates = self._gates(real_output, positions)
        scene_reference, active_union = self._scene_reference(real_output, gates)
        scene_index = self.num_real_stems
        scene_output = _magnitude_preserving_mix(
            controlled_output[scene_index : scene_index + 1],
            scene_reference,
            strength=self.config.real_to_scene_strength,
            gate=active_union,
        )
        result = controlled_output.clone()
        result[scene_index : scene_index + 1] = scene_output
        return result
