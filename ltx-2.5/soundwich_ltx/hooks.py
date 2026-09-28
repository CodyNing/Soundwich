"""Stage-1 transformer hooks: shared video, carrier replay, and scene gathering."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from soundwich_ltx.carrier_bank import CarrierBank
from soundwich_ltx.config import BlendConfig, SceneCouplingConfig, StrengthSchedule, Window
from soundwich_ltx.scene_coupling import SceneCoupler
from soundwich_ltx.timeline import replay_carriers, replay_suppression, scene_union_windows


def slice_batch(value: torch.Tensor | None, batch: int = 1) -> torch.Tensor | None:
    if value is None or value.shape[0] <= batch:
        return value
    return value[:batch]


def slice_pe(pe: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    return pe[0][:1], pe[1][:1]


def _slice_pe_rows(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    return pe[0][:rows], pe[1][:rows]


def concat_stem_pe(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    num_stems: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    values = []
    for tensor in pe:
        if tensor.shape[0] != num_stems:
            values.append(tensor)
        elif tensor.ndim == 4:
            _, heads, tokens, dim = tensor.shape
            values.append(tensor.permute(1, 0, 2, 3).reshape(1, heads, num_stems * tokens, dim))
        elif tensor.ndim == 3:
            _, tokens, dim = tensor.shape
            values.append(tensor.reshape(1, num_stems * tokens, dim))
        else:
            raise ValueError(f"unsupported RoPE shape for stem concatenation: {tuple(tensor.shape)}")
    return values[0], values[1]


class MultiStemHookController:
    """Binds once to a native LTX transformer and changes behavior per model call."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_stems: int,
        duration_seconds: float,
        stem_groups: list[str],
        stem_windows: list[tuple[Window, ...]],
        stem_scene_context: list[bool] | None = None,
        blend: BlendConfig,
        scene_coupling: SceneCouplingConfig | None = None,
        stem_blend_strengths: list[float | None] | None = None,
        stem_blend_schedules: list[StrengthSchedule | None] | None = None,
        activation_banks: dict[str, CarrierBank] | None = None,
        suppression_bank: CarrierBank | None = None,
        capture_bank: CarrierBank | None = None,
    ) -> None:
        self.num_stems = num_stems
        self.duration_seconds = duration_seconds
        self.stem_groups = stem_groups
        self.stem_windows = stem_windows
        self.blend = blend
        self.stem_blend_strengths = stem_blend_strengths or [None] * num_stems
        self.stem_blend_schedules = stem_blend_schedules or [None] * num_stems
        self.activation_banks = activation_banks or {}
        self.suppression_bank = suppression_bank
        self.capture_bank = capture_bank
        recorded_steps = {step for bank in self.activation_banks.values() for step, _block in bank.records}
        self.total_steps = max(recorded_steps, default=0) + 1
        self.scene_coupler = SceneCoupler(
            config=scene_coupling or SceneCouplingConfig(),
            num_real_stems=num_stems,
            stem_windows=stem_windows,
            stem_scene_context=stem_scene_context,
            duration_seconds=duration_seconds,
            feather_seconds=blend.feather_seconds,
        )
        self.scene_windows = [scene_union_windows(self.scene_coupler.participating_windows)]
        self.step = 0
        self.branch = "off"
        self.audio_positions: torch.Tensor | None = None
        self._model_id: int | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def bind(self, transformer: torch.nn.Module) -> None:
        velocity_model = transformer.velocity_model
        if id(velocity_model) == self._model_id:
            return
        self.close()
        self._model_id = id(velocity_model)
        for block_index, block in enumerate(velocity_model.transformer_blocks):
            self._install_video_hooks(block)
            self._handles.append(
                block.audio_attn1.register_forward_hook(self._carrier_hook(block_index), with_kwargs=True)
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._model_id = None

    @contextmanager
    def call(self, *, branch: str, step: int, audio_positions: torch.Tensor | None) -> Iterator[None]:
        previous = (self.branch, self.step, self.audio_positions)
        self.branch = branch
        self.step = step
        self.audio_positions = audio_positions
        try:
            yield
        finally:
            self.branch, self.step, self.audio_positions = previous

    def _install_video_hooks(self, block: torch.nn.Module) -> None:
        """Compute the video stream once and broadcast it to every audio lane."""
        audio_lanes = self.scene_coupler.audio_lane_count
        # The scene lane is an audio-only context aggregator and is kept out of A2V.
        a2v_lanes = self.num_stems

        def expand(_module: torch.nn.Module, _args: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
            return output.expand(audio_lanes, *output.shape[1:]).contiguous()

        def attn1_pre(
            _module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            updated["pe"] = slice_pe(updated.get("pe"))  # type: ignore[arg-type]
            for name in ("mask", "perturbation_mask"):
                value = updated.get(name)
                if isinstance(value, torch.Tensor):
                    updated[name] = slice_batch(value)
            return (args[0][:1], *args[1:]), updated

        def attn2_pre(
            _module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
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
            _module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            updated = dict(kwargs)
            context = updated.get("context")
            if not isinstance(context, torch.Tensor) or context.shape[0] != audio_lanes:
                raise ValueError("A2V context must contain one batch row per audio lane")
            a2v_context = context[:a2v_lanes]
            updated["context"] = a2v_context.reshape(1, a2v_lanes * a2v_context.shape[1], a2v_context.shape[2])
            updated["pe"] = slice_pe(updated.get("pe"))  # type: ignore[arg-type]
            updated["k_pe"] = concat_stem_pe(
                _slice_pe_rows(updated.get("k_pe"), a2v_lanes),  # type: ignore[arg-type]
                a2v_lanes,
            )
            return (args[0][:1], *args[1:]), updated

        self._handles.extend(
            [
                block.attn1.register_forward_pre_hook(attn1_pre, with_kwargs=True),
                block.attn1.register_forward_hook(expand),
                block.attn2.register_forward_pre_hook(attn2_pre, with_kwargs=True),
                block.attn2.register_forward_hook(expand),
                block.ff.register_forward_pre_hook(ff_pre),
                block.ff.register_forward_hook(expand),
                block.audio_to_video_attn.register_forward_pre_hook(a2v_pre, with_kwargs=True),
                block.audio_to_video_attn.register_forward_hook(expand),
            ]
        )

    def _stem_strengths(self) -> list[float | None]:
        progress = self.step / max(self.total_steps - 1, 1)
        return [
            (
                schedule.strength_at(progress, default=self.blend.strength if base is None else base)
                if schedule is not None
                else base
            )
            for base, schedule in zip(self.stem_blend_strengths, self.stem_blend_schedules, strict=True)
        ]

    def _carrier_hook(
        self,
        block: int,
    ) -> Callable[[torch.nn.Module, tuple[object, ...], dict[str, object], torch.Tensor], torch.Tensor]:
        def hook(
            module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if self.branch == "capture_positive" and self.capture_bank is not None:
                self.capture_bank.capture(step=self.step, block=block, hidden=output[:1])
                return output
            pe = kwargs.get("pe")
            suppression = self.suppression_bank.record(self.step, block) if self.suppression_bank is not None else None
            # 1. Scene gathering (scene lane only, positive audio branch).
            controlled = self.scene_coupler.inject_scene_context(
                module,  # type: ignore[arg-type]
                input_hidden=args[0],  # type: ignore[arg-type]
                attention_output=output,
                positions=self.audio_positions,
                pe=pe if isinstance(pe, tuple) else None,
                branch=self.branch,
                scene_self_attention_enabled=self.scene_coupler.config.scene_self_attention_enabled_at(
                    step=self.step,
                    total_steps=self.total_steps,
                ),
            )
            # 2. Carrier replay on the real stems.
            if self.branch == "audio_positive" and suppression is not None:
                activation = [self.activation_banks[group].record(self.step, block) for group in self.stem_groups]
                real = replay_carriers(
                    controlled[: self.num_stems],
                    activation=activation,
                    suppression=suppression,
                    windows=self.stem_windows,
                    positions=self.audio_positions,
                    duration_seconds=self.duration_seconds,
                    blend=self.blend,
                    stem_blend_strengths=self._stem_strengths(),
                    activation_strength=self.blend.activation_strength(step=self.step, total_steps=self.total_steps),
                )
                controlled = controlled.clone()
                controlled[: self.num_stems] = real
            # 3. Optional hidden-state update of the scene lane.
            controlled = self.scene_coupler.update_scene_lane(
                controlled_output=controlled,
                positions=self.audio_positions,
                branch=self.branch,
            )
            # 4. One suppression pass on the scene lane over the union of its stems' windows.
            if self.branch == "audio_positive" and suppression is not None and self.scene_coupler.uses_scene_lane:
                controlled = controlled.clone()
                controlled[self.num_stems : self.num_stems + 1] = replay_suppression(
                    controlled[self.num_stems : self.num_stems + 1],
                    suppression=suppression,
                    windows=self.scene_windows,
                    positions=self.audio_positions,
                    duration_seconds=self.duration_seconds,
                    blend=self.blend,
                )
            return controlled

        return hook
