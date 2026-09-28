"""Stage-2 SAM-owned cross-modal routing and scene broadcast."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from soundwich_ltx.carrier_bank import CarrierBank
from soundwich_ltx.config import BlendConfig, SceneCouplingConfig, Window
from soundwich_ltx.hooks import concat_stem_pe, slice_batch, slice_pe
from soundwich_ltx.scene_coupling import SceneCoupler
from soundwich_ltx.timeline import (
    normalized_audio_positions,
    replay_suppression,
    scene_union_windows,
    timeline_gate,
)


def _mask_tensor(
    masks: dict[int, list[float]],
    stem: int,
    tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    values = masks.get(stem)
    if not values:
        return None
    mask = torch.tensor(values, device=device, dtype=torch.float32).clamp(0, 1)
    if mask.numel() != tokens:
        mask = torch.nn.functional.interpolate(
            mask.reshape(1, 1, -1), size=tokens, mode="linear", align_corners=False
        ).reshape(-1)
    return mask.to(dtype=dtype)


def build_a2v_route(
    masks: dict[int, list[float]],
    *,
    stems: int,
    video_tokens: int,
    audio_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    threshold: float,
    active_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route each SAM-owned video query to the complete matching audio stem."""
    per_stem = []
    valid_stems: list[bool] = []
    for stem in range(stems):
        mask = _mask_tensor(masks, stem, video_tokens, device=device, dtype=dtype)
        value = mask if mask is not None else torch.zeros(video_tokens, device=device, dtype=dtype)
        per_stem.append(value)
        valid_stems.append(bool((value > threshold).any()))
    stacked = torch.stack(per_stem)
    owner_values, owners = stacked.max(dim=0)
    entity = owner_values > threshold
    bias = torch.zeros(video_tokens, stems * audio_tokens, device=device, dtype=dtype)
    excluded = torch.finfo(dtype).min
    for stem, valid in enumerate(valid_stems):
        if not valid:
            bias[:, stem * audio_tokens : (stem + 1) * audio_tokens] = excluded
    if not any(valid_stems):
        return torch.zeros_like(bias).reshape(1, 1, video_tokens, stems * audio_tokens), torch.zeros(
            video_tokens, device=device, dtype=dtype
        )
    for stem in range(stems):
        queries = entity & (owners == stem)
        if not bool(queries.any()):
            continue
        bias[queries] = excluded
        start = stem * audio_tokens
        bias[queries, start : start + audio_tokens] = 0
    gate = torch.where(
        entity,
        torch.full((video_tokens,), max(float(active_gain), 0.0), device=device, dtype=dtype),
        torch.ones(video_tokens, device=device, dtype=dtype),
    )
    return bias.reshape(1, 1, video_tokens, stems * audio_tokens), gate


def build_v2a_route(  # noqa: PLR0913
    masks: dict[int, list[float]],
    *,
    windows: list[tuple[Window, ...]],
    audio_positions: torch.Tensor | None,
    duration_seconds: float,
    feather_seconds: float,
    stems: int,
    audio_tokens: int,
    video_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict V2A keys by SAM and its residual to each stem's timeline."""
    bias = torch.zeros(stems, 1, 1, video_tokens, device=device, dtype=dtype)
    excluded = torch.finfo(dtype).min
    valid_stems: list[bool] = []
    for stem in range(stems):
        mask = _mask_tensor(masks, stem, video_tokens, device=device, dtype=dtype)
        if mask is None:
            valid_stems.append(False)
            continue
        allowed = mask > threshold
        valid_stems.append(bool(allowed.any()))
        if bool(allowed.any()):
            bias[stem, 0, 0, ~allowed] = excluded

    positions = normalized_audio_positions(
        audio_positions,
        token_count=audio_tokens,
        duration_seconds=duration_seconds,
        device=device,
    )
    timeline_gates = []
    for stem_windows in windows:
        if stem_windows:
            gate = timeline_gate(
                stem_windows,
                positions,
                duration_seconds=duration_seconds,
                feather_seconds=feather_seconds,
            )
        else:
            gate = torch.ones(audio_tokens, device=device, dtype=torch.float32)
        timeline_gates.append(gate.to(dtype=dtype))
    combined = [
        timeline_gates[index]
        if valid_stems[index]
        else torch.zeros(audio_tokens, device=device, dtype=dtype)
        for index in range(stems)
    ]
    return bias, torch.stack(combined)


class Stage2RoutingHookController:
    """Keep one shared video while hard-routing native A2V and V2A attention."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_stems: int,
        token_masks: dict[int, list[float]],
        stem_windows: list[tuple[Window, ...]],
        stem_scene_context: list[bool] | None = None,
        duration_seconds: float,
        feather_seconds: float,
        a2v_hard_route: bool,
        v2a_hard_route: bool,
        a2v_active_gain: float,
        threshold: float,
        blend: BlendConfig,
        scene_coupling: SceneCouplingConfig | None = None,
        suppression_bank: CarrierBank | None = None,
    ) -> None:
        self.num_stems = num_stems
        self.token_masks = token_masks
        self.stem_windows = stem_windows
        self.duration_seconds = duration_seconds
        self.feather_seconds = feather_seconds
        self.a2v_hard_route = a2v_hard_route
        self.v2a_hard_route = v2a_hard_route
        self.a2v_active_gain = a2v_active_gain
        self.threshold = threshold
        self.blend = blend
        self.suppression_bank = suppression_bank
        self.scene_coupler = SceneCoupler(
            config=scene_coupling or SceneCouplingConfig(),
            num_real_stems=num_stems,
            stem_windows=stem_windows,
            stem_scene_context=stem_scene_context,
            duration_seconds=duration_seconds,
            feather_seconds=feather_seconds,
        )
        self.scene_windows = [
            scene_union_windows(self.scene_coupler.participating_windows)
        ]
        self.step = 0
        self.branch = "off"
        self.audio_positions: torch.Tensor | None = None
        self._model_id: int | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._a2v_gate: torch.Tensor | None = None
        self._v2a_gate: torch.Tensor | None = None

    def bind(self, transformer: torch.nn.Module) -> None:
        velocity_model = transformer.velocity_model
        if id(velocity_model) == self._model_id:
            return
        self.close()
        self._model_id = id(velocity_model)
        for block_index, block in enumerate(velocity_model.transformer_blocks):
            self._install(block, block_index)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._model_id = None

    @contextmanager
    def call(
        self,
        *,
        branch: str,
        step: int,
        audio_positions: torch.Tensor | None,
    ) -> Iterator[None]:
        previous = (self.branch, self.step, self.audio_positions)
        self.branch = branch
        self.step = step
        self.audio_positions = audio_positions
        try:
            yield
        finally:
            self.branch, self.step, self.audio_positions = previous

    def _install(self, block: torch.nn.Module, block_index: int) -> None:  # noqa: PLR0915
        stems = self.num_stems
        audio_lanes = self.scene_coupler.audio_lane_count

        def expand(_module: torch.nn.Module, _args: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
            return output.expand(audio_lanes, *output.shape[1:]).contiguous()

        def attn1_pre(
            _module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            updated["pe"] = slice_pe(updated.get("pe"))  # type: ignore[arg-type]
            for name in ("mask", "perturbation_mask"):
                value = updated.get(name)
                if isinstance(value, torch.Tensor):
                    updated[name] = slice_batch(value)
            return (args[0][:1], *args[1:]), updated

        def attn2_pre(
            _module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            context = updated.get("context")
            if isinstance(context, torch.Tensor):
                updated["context"] = context[:1]
            mask = updated.get("mask")
            if isinstance(mask, torch.Tensor):
                updated["mask"] = slice_batch(mask)
            return (args[0][:1], *args[1:]), updated

        def ff_pre(_module: torch.nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
            return (args[0][:1], *args[1:])

        def a2v_pre(
            _module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            context = updated.get("context")
            if not isinstance(context, torch.Tensor) or context.shape[0] != audio_lanes:
                raise ValueError("Stage-2 A2V needs one audio context row per stem")
            video_tokens = args[0].shape[1]
            audio_tokens = context.shape[1]
            real_context = context[:stems]
            updated["context"] = real_context.reshape(1, stems * audio_tokens, context.shape[2])
            updated["pe"] = slice_pe(updated.get("pe"))  # type: ignore[arg-type]
            k_pe = updated.get("k_pe")
            if isinstance(k_pe, tuple):
                k_pe = tuple(tensor[:stems] for tensor in k_pe)
            updated["k_pe"] = concat_stem_pe(k_pe, stems)  # type: ignore[arg-type]
            self._a2v_gate = None
            if self.a2v_hard_route:
                bias, self._a2v_gate = build_a2v_route(
                    self.token_masks,
                    stems=stems,
                    video_tokens=video_tokens,
                    audio_tokens=audio_tokens,
                    device=context.device,
                    dtype=context.dtype,
                    threshold=self.threshold,
                    active_gain=self.a2v_active_gain,
                )
                updated["mask"] = bias
            return (args[0][:1], *args[1:]), updated

        def a2v_post(_module: torch.nn.Module, _args: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
            if self._a2v_gate is not None:
                output = output * self._a2v_gate.reshape(1, -1, 1)
            return output.expand(audio_lanes, *output.shape[1:]).contiguous()

        def v2a_pre(
            _module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            context = updated.get("context")
            if not isinstance(context, torch.Tensor) or context.shape[0] != audio_lanes:
                raise ValueError("Stage-2 V2A needs one shared-video row per stem")
            self._v2a_gate = None
            if self.v2a_hard_route:
                bias, self._v2a_gate = build_v2a_route(
                    self.token_masks,
                    windows=self.stem_windows,
                    audio_positions=self.audio_positions,
                    duration_seconds=self.duration_seconds,
                    feather_seconds=self.feather_seconds,
                    stems=stems,
                    audio_tokens=args[0].shape[1],
                    video_tokens=context.shape[1],
                    device=context.device,
                    dtype=context.dtype,
                    threshold=self.threshold,
                )
                if audio_lanes > stems:
                    scene_bias = torch.zeros(
                        audio_lanes - stems,
                        1,
                        1,
                        context.shape[1],
                        device=context.device,
                        dtype=context.dtype,
                    )
                    bias = torch.cat([bias, scene_bias], dim=0)
                    scene_gate = torch.ones(
                        audio_lanes - stems,
                        args[0].shape[1],
                        device=context.device,
                        dtype=context.dtype,
                    )
                    self._v2a_gate = torch.cat([self._v2a_gate, scene_gate], dim=0)
                updated["mask"] = bias
            return args, updated

        def v2a_post(_module: torch.nn.Module, _args: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
            if self._v2a_gate is not None:
                output = output * self._v2a_gate.unsqueeze(-1)
            return output

        def audio_attn1_post(
            module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: torch.Tensor,
        ) -> torch.Tensor:
            pe = kwargs.get("pe")
            controlled = self.scene_coupler.inject_scene_context(
                module,  # type: ignore[arg-type]
                input_hidden=args[0],  # type: ignore[arg-type]
                attention_output=output,
                positions=self.audio_positions,
                pe=pe if isinstance(pe, tuple) else None,
                branch=self.branch,
            )
            suppression = None
            if self.branch == "audio_positive" and self.suppression_bank is not None:
                suppression = self.suppression_bank.record(self.step, block_index)
                real = replay_suppression(
                    controlled[:stems],
                    suppression=suppression,
                    windows=self.stem_windows,
                    positions=self.audio_positions,
                    duration_seconds=self.duration_seconds,
                    blend=self.blend,
                )
                controlled = controlled.clone()
                controlled[:stems] = real
            controlled = self.scene_coupler.update_scene_lane(
                controlled_output=controlled,
                positions=self.audio_positions,
                branch=self.branch,
            )
            if (
                self.branch == "audio_positive"
                and suppression is not None
                and self.scene_coupler.uses_scene_lane
            ):
                controlled = controlled.clone()
                controlled[stems : stems + 1] = replay_suppression(
                    controlled[stems : stems + 1],
                    suppression=suppression,
                    windows=self.scene_windows,
                    positions=self.audio_positions,
                    duration_seconds=self.duration_seconds,
                    blend=self.blend,
                )
            return controlled

        self._handles.extend(
            [
                block.attn1.register_forward_pre_hook(attn1_pre, with_kwargs=True),
                block.attn1.register_forward_hook(expand),
                block.attn2.register_forward_pre_hook(attn2_pre, with_kwargs=True),
                block.attn2.register_forward_hook(expand),
                block.ff.register_forward_pre_hook(ff_pre),
                block.ff.register_forward_hook(expand),
                block.audio_to_video_attn.register_forward_pre_hook(a2v_pre, with_kwargs=True),
                block.audio_to_video_attn.register_forward_hook(a2v_post),
                block.video_to_audio_attn.register_forward_pre_hook(v2a_pre, with_kwargs=True),
                block.video_to_audio_attn.register_forward_hook(v2a_post),
                block.audio_attn1.register_forward_hook(audio_attn1_post, with_kwargs=True),
            ]
        )
