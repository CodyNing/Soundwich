"""Stage-1 four-call CFG, Stage-2 routed, and carrier-capture denoisers."""

from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import LatentState
from ltx_pipelines.utils.types import DenoisedLatentResult
from soundwich_ltx.hooks import MultiStemHookController
from soundwich_ltx.stage2_routing import Stage2RoutingHookController


def repeat_state(state: LatentState, repeats: int) -> LatentState:
    def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        sizes = [1] * value.ndim
        sizes[0] = repeats
        return value.repeat(*sizes)

    return LatentState(
        latent=repeat(state.latent),
        denoise_mask=repeat(state.denoise_mask),
        positions=repeat(state.positions),
        clean_latent=repeat(state.clean_latent),
        attention_mask=repeat(state.attention_mask),
        keyframes_mask=repeat(state.keyframes_mask),
        generated_keyframe_layout=state.generated_keyframe_layout,
        generated_keyframes=repeat(state.generated_keyframes),
        frozen=state.frozen,
    )


class MultiStemGaussianNoiser:
    """Share one video noise sample while drawing independent noise for every audio stem."""

    def __init__(self, generator: torch.Generator, num_stems: int) -> None:
        self.generator = generator
        self.num_stems = num_stems

    def __call__(self, state: LatentState, noise_scale: float) -> LatentState:
        is_video = state.positions.shape[1] == 3
        if is_video:
            if state.latent.shape[0] not in {1, self.num_stems}:
                raise ValueError("shared video must contain one row or one repeated row per stem")
            expanded = state
        elif state.latent.shape[0] == 1:
            expanded = repeat_state(state, self.num_stems)
        elif state.latent.shape[0] == self.num_stems:
            expanded = state
        else:
            raise ValueError("audio latent must contain either one base row or one row per stem")
        noise_shape = (1, *expanded.latent.shape[1:]) if is_video else expanded.latent.shape
        noise = torch.randn(
            noise_shape,
            generator=self.generator,
            device=expanded.latent.device,
            dtype=expanded.latent.dtype,
        )
        if is_video and expanded.latent.shape[0] > 1:
            noise = noise.expand_as(expanded.latent)
        latent = torch.lerp(expanded.latent.float(), noise.float(), noise_scale)
        latent = torch.lerp(expanded.clean_latent.float(), latent, expanded.denoise_mask)
        noised = replace(expanded, latent=latent.to(expanded.latent.dtype))
        return repeat_state(noised, self.num_stems) if is_video and noised.latent.shape[0] == 1 else noised


class Stage2MultiStemDenoiser:
    """Refine routed video and optionally apply per-stem negative audio CFG."""

    def __init__(
        self,
        *,
        video_context: torch.Tensor,
        audio_positive_context: torch.Tensor,
        audio_negative_context: torch.Tensor,
        audio_cfg: float,
        audio_rescale: float,
        negative_audio_cfg: bool,
        hooks: Stage2RoutingHookController,
    ) -> None:
        self.video_context = video_context
        self.audio_positive_context = audio_positive_context
        self.audio_negative_context = audio_negative_context
        self.audio_cfg = audio_cfg
        self.audio_rescale = audio_rescale
        self.negative_audio_cfg = negative_audio_cfg
        self.hooks = hooks

    def _call(
        self,
        transformer: torch.nn.Module,
        *,
        video_state: LatentState,
        audio_state: LatentState,
        sigma: torch.Tensor,
        audio_context: torch.Tensor,
        branch: str,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = audio_state.latent.shape[0]
        video = _modality_from_state(video_state, _context_batch(self.video_context, rows), sigma)
        audio = _modality_from_state(audio_state, _context_batch(audio_context, rows), sigma)
        with self.hooks.call(
            branch=branch,
            step=step,
            audio_positions=audio_state.positions,
        ):
            denoised_video, denoised_audio = transformer(video=video, audio=audio, perturbations=None)
        if denoised_video is None or denoised_audio is None:
            raise RuntimeError("Stage-2 routing did not return both modalities")
        return denoised_video, denoised_audio

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult, DenoisedLatentResult]:
        if video_state is None or audio_state is None:
            raise ValueError("Stage-2 routing requires both video and audio states")
        rows = audio_state.latent.shape[0]
        if video_state.latent.shape[0] != rows:
            raise ValueError("Stage-2 video and audio batches must both contain one row per stem")
        self.hooks.bind(transformer)
        sigma = sigmas[step_index].expand(rows)
        positive_video, positive_audio = self._call(
            transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigma,
            audio_context=self.audio_positive_context,
            branch="audio_positive",
            step=step_index,
        )
        if not self.negative_audio_cfg:
            return DenoisedLatentResult(denoised=positive_video), DenoisedLatentResult(denoised=positive_audio)
        _negative_video, negative_audio = self._call(
            transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigma,
            audio_context=self.audio_negative_context,
            branch="audio_negative",
            step=step_index,
        )
        return (
            DenoisedLatentResult(denoised=positive_video),
            DenoisedLatentResult(
                denoised=_guided(
                    positive_audio,
                    negative_audio,
                    scale=self.audio_cfg,
                    rescale=self.audio_rescale,
                ),
                cond=positive_audio,
                uncond=negative_audio,
            ),
        )


def _context_batch(context: torch.Tensor, rows: int) -> torch.Tensor:
    if context.shape[0] == rows:
        return context
    if context.shape[0] != 1:
        raise ValueError(f"context batch must be 1 or {rows}, got {context.shape[0]}")
    return context.expand(rows, *context.shape[1:]).contiguous()


def _modality_from_state(state: LatentState, context: torch.Tensor, sigma: torch.Tensor) -> Modality:
    if state.frozen:
        sigma = torch.zeros_like(sigma)
    sigma_view = sigma.view(-1, *([1] * (state.denoise_mask.ndim - 1)))
    return Modality(
        enabled=True,
        latent=state.latent,
        sigma=sigma,
        timesteps=state.denoise_mask * sigma_view,
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


def _guided(cond: torch.Tensor, uncond: torch.Tensor, *, scale: float, rescale: float) -> torch.Tensor:
    prediction = cond.float() + (scale - 1.0) * (cond.float() - uncond.float())
    if rescale > 0:
        factor = cond.float().std().div(prediction.std().clamp_min(1e-6))
        prediction = prediction * (rescale * factor + (1 - rescale))
    return prediction.to(cond.dtype)


class MultiStemDenoiser:
    """Run isolated audio CFG and shared-video CFG as four native transformer calls."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        video_positive: torch.Tensor,
        video_negative: torch.Tensor,
        shared_audio_positive: torch.Tensor,
        shared_audio_negative: torch.Tensor,
        stem_audio_positive: torch.Tensor,
        stem_audio_negative: torch.Tensor,
        video_cfg: float,
        audio_cfg: float,
        video_rescale: float,
        audio_rescale: float,
        hooks: MultiStemHookController,
    ) -> None:
        self.video_positive = video_positive
        self.video_negative = video_negative
        self.shared_audio_positive = shared_audio_positive
        self.shared_audio_negative = shared_audio_negative
        self.stem_audio_positive = stem_audio_positive
        self.stem_audio_negative = stem_audio_negative
        self.video_cfg = video_cfg
        self.audio_cfg = audio_cfg
        self.video_rescale = video_rescale
        self.audio_rescale = audio_rescale
        self.hooks = hooks

    def _modalities(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigma: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[Modality, Modality]:
        rows = video_state.latent.shape[0]
        sigma_batch = sigma.expand(rows)
        return (
            _modality_from_state(video_state, _context_batch(video_context, rows), sigma_batch),
            _modality_from_state(audio_state, _context_batch(audio_context, rows), sigma_batch),
        )

    def _call(
        self,
        transformer: torch.nn.Module,
        *,
        branch: str,
        step: int,
        sigma: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video, audio = self._modalities(video_state, audio_state, sigma, video_context, audio_context)
        with self.hooks.call(branch=branch, step=step, audio_positions=audio_state.positions):
            denoised_video, denoised_audio = transformer(video=video, audio=audio, perturbations=None)
        if denoised_video is None or denoised_audio is None:
            raise RuntimeError("multi-stem denoiser requires both audio and video outputs")
        return denoised_video, denoised_audio

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult, DenoisedLatentResult]:
        if video_state is None or audio_state is None:
            raise ValueError("multi-stem generation requires video and audio states")
        rows = audio_state.latent.shape[0]
        if rows != self.stem_audio_positive.shape[0]:
            raise ValueError("audio state batch does not match configured stem count")
        self.hooks.bind(transformer)
        sigma = sigmas[step_index]

        _, audio_positive = self._call(
            transformer,
            branch="audio_positive",
            step=step_index,
            sigma=sigma,
            video_state=video_state,
            audio_state=audio_state,
            video_context=self.video_positive,
            audio_context=self.stem_audio_positive,
        )
        _, audio_negative = self._call(
            transformer,
            branch="audio_negative",
            step=step_index,
            sigma=sigma,
            video_state=video_state,
            audio_state=audio_state,
            video_context=self.video_positive,
            audio_context=self.stem_audio_negative,
        )
        video_positive, _ = self._call(
            transformer,
            branch="video_positive",
            step=step_index,
            sigma=sigma,
            video_state=video_state,
            audio_state=audio_state,
            video_context=self.video_positive,
            audio_context=self.shared_audio_positive,
        )
        video_negative, _ = self._call(
            transformer,
            branch="video_negative",
            step=step_index,
            sigma=sigma,
            video_state=video_state,
            audio_state=audio_state,
            video_context=self.video_negative,
            audio_context=self.shared_audio_negative,
        )

        shared_video = _guided(
            video_positive[:1],
            video_negative[:1],
            scale=self.video_cfg,
            rescale=self.video_rescale,
        ).expand(rows, *video_positive.shape[1:])
        guided_audio = _guided(
            audio_positive,
            audio_negative,
            scale=self.audio_cfg,
            rescale=self.audio_rescale,
        )
        return (
            DenoisedLatentResult(
                denoised=shared_video,
                cond=video_positive[:1],
                uncond=video_negative[:1],
            ),
            DenoisedLatentResult(
                denoised=guided_audio,
                cond=audio_positive,
                uncond=audio_negative,
            ),
        )


def cross_modal_perturbations(
    *,
    transformer: torch.nn.Module,
    reference: Modality,
    a2v_enabled: bool,
    v2a_enabled: bool,
) -> BatchedPerturbationConfig | None:
    perturbations = []
    if not a2v_enabled:
        perturbations.append(Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None))
    if not v2a_enabled:
        perturbations.append(Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None))
    if not perturbations:
        return None
    config = PerturbationConfig(perturbations)
    return BatchedPerturbationConfig(
        [config] * reference.latent.shape[0],
        num_blocks=transformer.num_blocks,
        device=reference.latent.device,
        dtype=reference.latent.dtype,
    )


class CarrierCaptureDenoiser:
    """Generate one normal AV branch and record one compact audio token per step/block."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        video_positive: torch.Tensor,
        video_negative: torch.Tensor,
        audio_positive: torch.Tensor,
        audio_negative: torch.Tensor,
        video_cfg: float,
        audio_cfg: float,
        video_rescale: float,
        audio_rescale: float,
        a2v_enabled: bool,
        v2a_enabled: bool,
        hooks: MultiStemHookController,
    ) -> None:
        self.video_positive = video_positive
        self.video_negative = video_negative
        self.audio_positive = audio_positive
        self.audio_negative = audio_negative
        self.video_cfg = video_cfg
        self.audio_cfg = audio_cfg
        self.video_rescale = video_rescale
        self.audio_rescale = audio_rescale
        self.a2v_enabled = a2v_enabled
        self.v2a_enabled = v2a_enabled
        self.hooks = hooks

    def _call(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState,
        audio_state: LatentState,
        sigma: torch.Tensor,
        *,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        branch: str,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_batch = sigma.expand(video_state.latent.shape[0])
        video = _modality_from_state(video_state, video_context, sigma_batch)
        audio = _modality_from_state(audio_state, audio_context, sigma_batch)
        perturbations = cross_modal_perturbations(
            transformer=transformer,
            reference=video,
            a2v_enabled=self.a2v_enabled,
            v2a_enabled=self.v2a_enabled,
        )
        with self.hooks.call(branch=branch, step=step, audio_positions=audio_state.positions):
            result = transformer(video=video, audio=audio, perturbations=perturbations)
        if result[0] is None or result[1] is None:
            raise RuntimeError("carrier generation requires both audio and video outputs")
        return result[0], result[1]

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult, DenoisedLatentResult]:
        if video_state is None or audio_state is None:
            raise ValueError("carrier generation requires video and audio states")
        self.hooks.bind(transformer)
        sigma = sigmas[step_index]
        positive_v, positive_a = self._call(
            transformer,
            video_state,
            audio_state,
            sigma,
            video_context=self.video_positive,
            audio_context=self.audio_positive,
            branch="capture_positive",
            step=step_index,
        )
        negative_v, negative_a = self._call(
            transformer,
            video_state,
            audio_state,
            sigma,
            video_context=self.video_negative,
            audio_context=self.audio_negative,
            branch="capture_negative",
            step=step_index,
        )
        return (
            DenoisedLatentResult(
                denoised=_guided(positive_v, negative_v, scale=self.video_cfg, rescale=self.video_rescale),
                cond=positive_v,
                uncond=negative_v,
            ),
            DenoisedLatentResult(
                denoised=_guided(positive_a, negative_a, scale=self.audio_cfg, rescale=self.audio_rescale),
                cond=positive_a,
                uncond=negative_a,
            ),
        )
