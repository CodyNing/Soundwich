"""Two-stage Soundwich runner on native LTX-2.5 components."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_pipelines.distilled import ANCESTRAL_ETA, ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_S_NOISE
from ltx_pipelines.utils.blocks import AudioDecoder, DiffusionStage, PromptEncoder, VideoDecoder, VideoUpsampler
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import encode_audio, encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode
from soundwich_ltx.audio_mix import mix_waveforms
from soundwich_ltx.carrier_bank import CarrierBank
from soundwich_ltx.config import MultiStemConfig
from soundwich_ltx.denoiser import (
    CarrierCaptureDenoiser,
    MultiStemDenoiser,
    MultiStemGaussianNoiser,
    Stage2MultiStemDenoiser,
)
from soundwich_ltx.hooks import MultiStemHookController
from soundwich_ltx.sam3 import export_sampled_frames, read_token_masks, run_sam3, sample_indices
from soundwich_ltx.stage2_routing import Stage2RoutingHookController
from soundwich_ltx.stage2_schedule import densified_stage2_sigmas


class MultiStemPipeline:
    """Own native LTX components and execute carrier recording or two-stage generation."""

    def __init__(self, config: MultiStemConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        model_paths = ModelPaths.from_split(
            transformer_path=config.model.transformer,
            text_encoder_path=config.model.text_encoder,
            video_vae_path=config.model.video_vae,
            audio_vae_path=config.model.audio_vae,
        )
        # The INT8 ConvRot checkpoints run resident on the GPU without runtime quantization.
        offload_mode = OffloadMode.NONE
        self.prompt_encoder = PromptEncoder(
            model_paths, dtype=self.dtype, device=self.device, offload_mode=offload_mode
        )
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            dtype=self.dtype,
            device=self.device,
            quantization=None,
            offload_mode=offload_mode,
        )
        two_stage = config.task == "generate" and config.stage2.enabled
        self.stage_2 = (
            DiffusionStage.from_checkpoint(
                config.model.stage2_transformer,
                dtype=self.dtype,
                device=self.device,
                quantization=None,
                offload_mode=offload_mode,
            )
            if two_stage
            else None
        )
        self.video_decoder = VideoDecoder(model_paths.video_vae(), dtype=self.dtype, device=self.device)
        self.audio_decoder = AudioDecoder(model_paths.audio_vae(), dtype=self.dtype, device=self.device)
        if not self.audio_decoder.uses_bandwidth_extension:
            raise ValueError("Soundwich on LTX-2.5 requires an audio VAE checkpoint with bundled BWE")
        self.upsampler = (
            VideoUpsampler(model_paths.video_vae(), config.model.spatial_upscaler, dtype=self.dtype, device=self.device)
            if two_stage
            else None
        )
        self.scheduler = LTX2Scheduler()

    # ------------------------------------------------------------------ public API

    @torch.inference_mode()
    def generate(self, root: Path, *, stage1_only: bool = False, reuse_stage1: bool = False) -> dict[str, object]:
        """Stage 1 -> SAM3 entity masks -> Stage 2, all written under ``root``.

        With ``reuse_stage1``, the saved Stage-1 result in ``root`` is kept and only the masks and Stage 2 are
        recomputed, e.g. after editing ``sam_points`` in the scene.
        """
        if reuse_stage1:
            if not (root / "stage1_latents.pt").is_file() or not (root / "stage1_manifest.json").is_file():
                raise FileNotFoundError(f"no saved Stage-1 result to reuse in {root}")
            stage1: dict[str, object] = {"stage1_latents": str(root / "stage1_latents.pt")}
        else:
            stage1 = self.run_stage1(root)
        if stage1_only or not self.config.stage2.enabled:
            return stage1
        token_masks, mask_path = self.run_masks(root)
        stage2 = self.run_stage2(root, token_masks=token_masks)
        return {**stage1, "sam_masks": str(mask_path), **stage2}

    @torch.inference_mode()
    def run_masks(self, root: Path) -> tuple[dict[int, list[float]], Path]:
        """Run SAM3 on the frames exported by :meth:`run_stage1` (LTX models are already released)."""
        manifest = json.loads((root / "stage1_manifest.json").read_text(encoding="utf-8"))
        return run_sam3(
            self.config,
            run_root=root,
            frames_dir=root / str(manifest["frames_directory"]),
            frame_indices=[int(value) for value in manifest["frame_indices"]],
            token_shape=tuple(int(value) for value in manifest["stage2_token_shape"]),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ schedules and carriers

    def _sigmas(self) -> torch.Tensor:
        sigmas = (
            DISTILLED_SIGMAS
            if self.config.generation.schedule == "distilled"
            else self.scheduler.execute(steps=self.config.generation.steps)
        )
        return sigmas.to(device=self.device, dtype=torch.float32)

    def _sampler_kwargs(self, seed: int) -> dict[str, object]:
        if self.config.generation.schedule != "distilled":
            return {}
        return {
            "stepper": EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE),
            "loop": partial(
                euler_ancestral_denoising_loop,
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
                model_dtype=self.dtype,
            ),
        }

    def _load_carriers(self) -> tuple[dict[str, CarrierBank], CarrierBank]:
        activation: dict[str, CarrierBank] = {}
        for group, path in self.config.carriers.activation.items():
            bank = CarrierBank.load(path)
            if bank.role != "activation":
                raise ValueError(f"carrier {path!r} is {bank.role}/{bank.group}, expected an activation carrier")
            activation[group] = bank
        suppression = CarrierBank.load(self.config.carriers.suppression)
        if suppression.role != "suppression":
            raise ValueError(f"carrier {self.config.carriers.suppression!r} is not a suppression carrier")
        expected_steps = set(range(self.config.generation.steps))
        for bank in (*activation.values(), suppression):
            recorded_steps = {step for step, _block in bank.records}
            if recorded_steps != expected_steps:
                raise ValueError(
                    f"carrier {bank.group!r} covers steps {sorted(recorded_steps)}, "
                    f"but this run requires 0-{self.config.generation.steps - 1}"
                )
        return activation, suppression

    def _load_stage2_suppression(self, steps: int) -> CarrierBank | None:
        if not self.config.stage2.suppression_blend_enabled:
            return None
        bank = CarrierBank.load(self.config.stage2.suppression_carrier)
        if bank.role != "suppression":
            raise ValueError(f"Stage-2 carrier {self.config.stage2.suppression_carrier!r} is not a suppression carrier")
        recorded_steps = {step for step, _block in bank.records}
        if recorded_steps != set(range(steps)):
            raise ValueError(
                f"Stage-2 suppression carrier covers steps {sorted(recorded_steps)}, "
                f"but this refinement requires 0-{steps - 1}"
            )
        return bank

    def _encode_generation_prompts(
        self,
        *,
        include_scene_lane: bool,
    ) -> tuple[list[str], list[str], EmbeddingsProcessorOutput, EmbeddingsProcessorOutput, torch.Tensor, torch.Tensor]:
        positives = [self.config.effective_positive(stem) for stem in self.config.stems]
        negatives = [self.config.effective_negative(stem) for stem in self.config.stems]
        prompts = [self.config.visual_positive, self.config.visual_negative, *positives, *negatives]
        if include_scene_lane:
            prompts.extend([self.config.scene_positive(), self.config.scene_negative()])
        encoded = self.prompt_encoder(prompts)
        visual_positive, visual_negative = encoded[:2]
        stem_count = len(self.config.stems)
        positive_audio = torch.cat([item.audio_encoding for item in encoded[2 : 2 + stem_count]], dim=0)
        negative_start = 2 + stem_count
        negative_audio = torch.cat(
            [item.audio_encoding for item in encoded[negative_start : negative_start + stem_count]],
            dim=0,
        )
        if include_scene_lane:
            scene_positive, scene_negative = encoded[-2:]
            positive_audio = torch.cat([positive_audio, scene_positive.audio_encoding], dim=0)
            negative_audio = torch.cat([negative_audio, scene_negative.audio_encoding], dim=0)
        return positives, negatives, visual_positive, visual_negative, positive_audio, negative_audio

    # ------------------------------------------------------------------ Stage 1

    @torch.inference_mode()
    def run_stage1(self, root: Path) -> dict[str, object]:  # noqa: PLR0915
        """Generate one shared video and N controlled audio stems (+ the scene lane) at half resolution."""
        config = self.config
        generation = config.generation
        assert_resolution(height=generation.height, width=generation.width, is_two_stage=config.stage2.enabled)
        scene_lane = config.scene_coupling.uses_scene_lane
        positives, negatives, visual_positive, visual_negative, positive_audio, negative_audio = (
            self._encode_generation_prompts(include_scene_lane=scene_lane)
        )
        stem_count = len(config.stems)
        audio_lane_count = stem_count + int(scene_lane)
        activation, suppression = self._load_carriers()
        hooks = MultiStemHookController(
            num_stems=stem_count,
            duration_seconds=generation.duration_seconds,
            stem_groups=[stem.reference_group for stem in config.stems],
            stem_windows=[stem.windows for stem in config.stems],
            stem_scene_context=[stem.scene_context_enabled for stem in config.stems],
            blend=config.carriers.blend,
            scene_coupling=config.scene_coupling,
            stem_blend_strengths=[stem.blend_strength for stem in config.stems],
            stem_blend_schedules=[stem.blend_schedule for stem in config.stems],
            activation_banks=activation,
            suppression_bank=suppression,
        )
        denoiser = MultiStemDenoiser(
            video_positive=visual_positive.video_encoding,
            video_negative=visual_negative.video_encoding,
            shared_audio_positive=visual_positive.audio_encoding,
            shared_audio_negative=visual_negative.audio_encoding,
            stem_audio_positive=positive_audio,
            stem_audio_negative=negative_audio,
            video_cfg=config.guidance.video_cfg,
            audio_cfg=config.guidance.audio_cfg,
            video_rescale=config.guidance.video_rescale,
            audio_rescale=config.guidance.audio_rescale,
            hooks=hooks,
        )
        generator = torch.Generator(device=self.device).manual_seed(generation.seed)
        noiser = MultiStemGaussianNoiser(generator=generator, num_stems=audio_lane_count)
        root.mkdir(parents=True, exist_ok=True)
        self._save_run_metadata(root, positives=positives, negatives=negatives)
        stage1_width = generation.width // 2 if config.stage2.enabled else generation.width
        stage1_height = generation.height // 2 if config.stage2.enabled else generation.height
        try:
            video_state, audio_state = self.stage(
                denoiser=denoiser,
                sigmas=self._sigmas(),
                noiser=noiser,
                width=stage1_width,
                height=stage1_height,
                frames=generation.frames,
                fps=generation.frame_rate,
                video=ModalitySpec(context=visual_positive.video_encoding),
                audio=ModalitySpec(context=positive_audio[:1]),
                max_batch_size=audio_lane_count,
                **self._sampler_kwargs(generation.seed),
            )
        finally:
            hooks.close()
        if video_state is None or audio_state is None:
            raise RuntimeError("Stage-1 generation did not return both modalities")

        result: dict[str, object] = {"output_directory": str(root)}
        stage1_video = video_state.latent[:1].detach()
        stage1_audio_all = audio_state.latent.detach()
        stage1_audio = stage1_audio_all[:stem_count]
        scene_audio = stage1_audio_all[stem_count:] if scene_lane else None
        latent_path = root / "stage1_latents.pt"
        torch.save(
            {
                "video": stage1_video.cpu(),
                "audio": stage1_audio.cpu(),
                "stem_ids": [stem.id for stem in config.stems],
                **({"scene_audio": scene_audio.cpu()} if scene_audio is not None else {}),
            },
            latent_path,
        )
        result["stage1_latents"] = str(latent_path)
        _decoded_audio, mix, audio_artifacts = self._decode_audio(root, stage1_audio_all, scene_lane=scene_lane)
        result.update({f"stage1_{key}": value for key, value in audio_artifacts.items()})
        if not config.stage2.enabled:
            result["stage1_video"] = self._decode_video(
                root / "stage1_mix.mp4", stage1_video, mix, generator, width=generation.width, height=generation.height
            )
            return result

        frame_indices = sample_indices(generation.frames, config.sam3.frame_stride, config.sam3.max_frames)
        frames_dir = root / "sam_outputs" / "stage1_frames"
        result["stage1_video"] = self._decode_video(
            root / "stage1_mix.mp4",
            stage1_video,
            mix,
            generator,
            width=stage1_width,
            height=stage1_height,
            frame_export=(frames_dir, frame_indices),
        )
        target_shape = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(
                batch=1,
                frames=generation.frames,
                height=generation.height,
                width=generation.width,
                fps=generation.frame_rate,
            )
        )
        manifest = {
            "checkpoint": "stage1_latents.pt",
            "frames_directory": str(frames_dir.relative_to(root)),
            "frame_indices": frame_indices,
            "stage2_token_shape": [target_shape.frames, target_shape.height, target_shape.width],
            "stem_ids": [stem.id for stem in config.stems],
            "scene_render": scene_lane,
        }
        (root / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result["stage1_manifest"] = str(root / "stage1_manifest.json")
        del video_state, audio_state, stage1_video, stage1_audio, stage1_audio_all, scene_audio
        cleanup_memory()
        return result

    # ------------------------------------------------------------------ Stage 2

    @torch.inference_mode()
    def run_stage2(  # noqa: PLR0915
        self,
        root: Path,
        *,
        token_masks: dict[int, list[float]] | None = None,
        audio_latents: torch.Tensor | None = None,
        output_root: Path | None = None,
    ) -> dict[str, object]:
        """Upscale the Stage-1 video and refine it with SAM-routed A2V/V2A and scene broadcast.

        With ``stage2.freeze_audio`` (stem editing), ``audio_latents`` (default: the Stage-1 audio) stay clean and
        fixed while only the video is refined; they are also the saved and decoded audio. Inputs are read from
        ``root`` and outputs written to ``output_root`` (default: ``root``).
        """
        config = self.config
        if not config.stage2.enabled:
            raise RuntimeError("Stage 2 is disabled in this configuration")
        if self.upsampler is None or self.stage_2 is None:
            raise RuntimeError("Stage 2 is enabled but its models were not constructed")
        freeze_audio = config.stage2.freeze_audio
        destination = output_root or root
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(root / "stage1_latents.pt", map_location="cpu", weights_only=True)
        stage1_video_cpu = checkpoint["video"]
        stage1_audio_cpu = checkpoint["audio"] if audio_latents is None else audio_latents
        stage1_scene_audio_cpu = checkpoint.get("scene_audio")
        if token_masks is None:
            token_masks = read_token_masks(config, root / "sam_outputs" / "stage2_a2v_mask.json")

        generation = config.generation
        scene_coupling = config.stage2.scene_coupling
        scene_lane = scene_coupling.uses_scene_lane
        positives, _negatives, visual_positive, _visual_negative, positive_audio, negative_audio = (
            self._encode_generation_prompts(include_scene_lane=scene_lane)
        )
        stem_count = len(config.stems)
        audio_lane_count = stem_count + int(scene_lane)
        generator = torch.Generator(device=self.device).manual_seed(generation.seed)
        upscaled = self.upsampler(stage1_video_cpu.to(device=self.device, dtype=self.dtype))
        repeated_video = upscaled.expand(audio_lane_count, *upscaled.shape[1:]).contiguous()
        stage2_audio = stage1_audio_cpu.to(device=self.device, dtype=self.dtype)
        if scene_lane:
            if stage1_scene_audio_cpu is None:
                raise ValueError("Stage-2 scene coupling requires scene_audio in the Stage-1 checkpoint")
            if stage1_scene_audio_cpu.shape[0] != 1 or stage1_scene_audio_cpu.shape[1:] != stage1_audio_cpu.shape[1:]:
                raise ValueError("Stage-1 scene_audio must contain exactly one lane shaped like the real stems")
            stage2_audio = torch.cat(
                [stage2_audio, stage1_scene_audio_cpu.to(device=self.device, dtype=self.dtype)],
                dim=0,
            )
        stage2_sigmas = densified_stage2_sigmas(config.stage2.start_sigma).to(device=self.device, dtype=torch.float32)
        hooks = Stage2RoutingHookController(
            num_stems=stem_count,
            token_masks=token_masks,
            stem_windows=[stem.windows for stem in config.stems],
            stem_scene_context=[stem.scene_context_enabled for stem in config.stems],
            duration_seconds=generation.duration_seconds,
            feather_seconds=config.carriers.blend.feather_seconds,
            a2v_hard_route=config.stage2.a2v_hard_route,
            v2a_hard_route=config.stage2.v2a_hard_route,
            a2v_active_gain=config.stage2.a2v_active_gain,
            threshold=config.stage2.mask_threshold,
            blend=config.carriers.blend,
            scene_coupling=scene_coupling,
            suppression_bank=self._load_stage2_suppression(len(stage2_sigmas) - 1),
        )
        denoiser = Stage2MultiStemDenoiser(
            video_context=visual_positive.video_encoding,
            audio_positive_context=positive_audio,
            audio_negative_context=negative_audio,
            audio_cfg=config.stage2.audio_cfg,
            audio_rescale=config.guidance.audio_rescale,
            negative_audio_cfg=config.stage2.negative_audio_cfg,
            hooks=hooks,
        )
        try:
            refined_video, refined_audio = self.stage_2(
                denoiser=denoiser,
                sigmas=stage2_sigmas,
                noiser=MultiStemGaussianNoiser(generator=generator, num_stems=audio_lane_count),
                width=generation.width,
                height=generation.height,
                frames=generation.frames,
                fps=generation.frame_rate,
                video=ModalitySpec(
                    context=visual_positive.video_encoding,
                    noise_scale=stage2_sigmas[0].item(),
                    initial_latent=repeated_video,
                ),
                audio=ModalitySpec(
                    context=positive_audio,
                    noise_scale=0.0 if freeze_audio else stage2_sigmas[0].item(),
                    initial_latent=stage2_audio,
                    frozen=freeze_audio,
                ),
                batch_size=audio_lane_count,
                max_batch_size=audio_lane_count,
            )
        finally:
            hooks.close()
        if refined_video is None or refined_audio is None:
            raise RuntimeError("Stage-2 refinement did not return both modalities")
        refined = refined_video.latent[:1]
        if freeze_audio:
            output_audio = stage1_audio_cpu
            refined_real_audio = stage1_audio_cpu[:stem_count]
        else:
            output_audio = refined_audio.latent
            refined_real_audio = refined_audio.latent[:stem_count]
        refined_scene_audio = refined_audio.latent[stem_count : stem_count + 1] if scene_lane else None
        result: dict[str, object] = {
            "stage2_steps": len(stage2_sigmas) - 1,
            "stage2_sigmas": [round(float(value), 6) for value in stage2_sigmas],
        }
        video_path = destination / "stage2_video_latent.pt"
        torch.save({"video": refined.detach().cpu(), "sigmas": stage2_sigmas.detach().cpu()}, video_path)
        audio_path = destination / "stage2_audio_latent.pt"
        torch.save(
            {
                "audio": refined_real_audio.detach().cpu(),
                **({"scene_audio": refined_scene_audio.detach().cpu()} if refined_scene_audio is not None else {}),
                "stem_ids": [stem.id for stem in config.stems],
                "sigmas": stage2_sigmas.detach().cpu(),
            },
            audio_path,
        )
        result.update({"stage2_video_latent": str(video_path), "stage2_audio_latent": str(audio_path)})
        _decoded_audio, mix, audio_artifacts = self._decode_audio(
            destination / "stage2_audio",
            output_audio,
            scene_lane=scene_lane,
        )
        result.update({f"stage2_{key}": value for key, value in audio_artifacts.items()})
        result["video"] = self._decode_video(
            destination / "stage2_mix.mp4",
            refined,
            mix,
            generator,
            width=generation.width,
            height=generation.height,
        )
        result["effective_prompts"] = positives
        return result

    # ------------------------------------------------------------------ carrier recording

    @torch.inference_mode()
    def record_carrier(self, root: Path) -> dict[str, object]:
        """Generate one ordinary AV sample and store one audio token per denoising step and block."""
        carrier = self.config.carrier
        if carrier is None:
            raise RuntimeError("carrier configuration is missing")
        generation = self.config.generation
        assert_resolution(height=generation.height, width=generation.width, is_two_stage=False)
        positive, negative = self.prompt_encoder([carrier.positive, carrier.negative])
        bank = CarrierBank(
            role=carrier.role,
            group=carrier.group,
            quantile=carrier.quantile,
            metadata={
                "config_id": self.config.id,
                "transformer": Path(self.config.model.transformer).name,
                "steps": generation.steps,
                "seed": generation.seed,
                "a2v_enabled": carrier.a2v_enabled,
                "v2a_enabled": carrier.v2a_enabled,
                "audio_postprocess": "per_stem_vocoder_with_bwe",
            },
        )
        hooks = MultiStemHookController(
            num_stems=1,
            duration_seconds=generation.duration_seconds,
            stem_groups=[carrier.group],
            stem_windows=[()],
            blend=self.config.carriers.blend,
            capture_bank=bank,
        )
        denoiser = CarrierCaptureDenoiser(
            video_positive=positive.video_encoding,
            video_negative=negative.video_encoding,
            audio_positive=positive.audio_encoding,
            audio_negative=negative.audio_encoding,
            video_cfg=self.config.guidance.video_cfg,
            audio_cfg=self.config.guidance.audio_cfg,
            video_rescale=self.config.guidance.video_rescale,
            audio_rescale=self.config.guidance.audio_rescale,
            a2v_enabled=carrier.a2v_enabled,
            v2a_enabled=carrier.v2a_enabled,
            hooks=hooks,
        )
        generator = torch.Generator(device=self.device).manual_seed(generation.seed)
        noiser = MultiStemGaussianNoiser(generator=generator, num_stems=1)
        try:
            video_state, audio_state = self.stage(
                denoiser=denoiser,
                sigmas=self._sigmas(),
                noiser=noiser,
                width=generation.width,
                height=generation.height,
                frames=generation.frames,
                fps=generation.frame_rate,
                video=ModalitySpec(context=positive.video_encoding),
                audio=ModalitySpec(context=positive.audio_encoding),
                max_batch_size=1,
                **self._sampler_kwargs(generation.seed),
            )
        finally:
            hooks.close()
        if video_state is None or audio_state is None:
            raise RuntimeError("carrier recording did not return both modalities")

        root.mkdir(parents=True, exist_ok=True)
        _decoded_audio, mix, audio_artifacts = self._decode_audio(root, audio_state.latent)
        preview = self._decode_video(
            root / "carrier_preview.mp4",
            video_state.latent,
            mix,
            generator,
            width=generation.width,
            height=generation.height,
        )
        # Written last: carrier.json marks a complete cache entry.
        bank.save(root)
        return {"carrier": str(root / "carrier.json"), "preview": preview, **audio_artifacts}

    # ------------------------------------------------------------------ decoding

    def _decode_audio(
        self,
        root: Path,
        audio_latent: torch.Tensor,
        *,
        scene_lane: bool = False,
    ) -> tuple[Audio, Audio, dict[str, object]]:
        """Decode every lane independently (with and without BWE), then mix the real stems."""
        pre_bwe_audio, decoded_audio = self.audio_decoder.decode_variants(
            audio_latent.to(device=self.device, dtype=self.dtype)
        )
        waveforms = decoded_audio.waveform.detach().float().cpu()
        pre_bwe_waveforms = pre_bwe_audio.waveform.detach().float().cpu()
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(0)
        if pre_bwe_waveforms.ndim == 2:
            pre_bwe_waveforms = pre_bwe_waveforms.unsqueeze(0)
        if waveforms.ndim != 3 or pre_bwe_waveforms.ndim != 3 or pre_bwe_waveforms.shape[:2] != waveforms.shape[:2]:
            raise ValueError(
                f"decoded audio must have [stem,channel,sample] shape, got {tuple(waveforms.shape)} "
                f"and pre-BWE {tuple(pre_bwe_waveforms.shape)}"
            )
        if self.config.task == "generate":
            ids = [stem.id for stem in self.config.stems]
            volumes = [stem.volume for stem in self.config.stems]
            if scene_lane:
                ids.append("scene_render")
        else:
            ids = ["carrier_preview"]
            volumes = [1.0]
        real_stem_count = len(volumes)
        if waveforms.shape[0] != len(ids):
            raise ValueError(f"decoded audio has {waveforms.shape[0]} rows but {len(ids)} output ids were configured")
        pre_bwe_root = root / "pre_bwe"
        pre_bwe_root.mkdir(parents=True, exist_ok=True)
        stem_paths: list[str] = []
        pre_bwe_stem_paths: list[str] = []
        for index, stem_id in enumerate(ids):
            path = root / f"{stem_id}.wav"
            encode_audio(Audio(waveform=waveforms[index], sampling_rate=decoded_audio.sampling_rate), str(path))
            stem_paths.append(str(path))
            pre_bwe_path = pre_bwe_root / f"{stem_id}.wav"
            encode_audio(
                Audio(waveform=pre_bwe_waveforms[index], sampling_rate=pre_bwe_audio.sampling_rate),
                str(pre_bwe_path),
            )
            pre_bwe_stem_paths.append(str(pre_bwe_path))

        mixed = Audio(
            waveform=mix_waveforms(waveforms[:real_stem_count], volumes),
            sampling_rate=decoded_audio.sampling_rate,
        )
        mix_path = root / "mix.wav"
        encode_audio(mixed, str(mix_path))
        pre_bwe_mix = Audio(
            waveform=mix_waveforms(pre_bwe_waveforms[:real_stem_count], volumes),
            sampling_rate=pre_bwe_audio.sampling_rate,
        )
        pre_bwe_mix_path = pre_bwe_root / "mix.wav"
        encode_audio(pre_bwe_mix, str(pre_bwe_mix_path))
        artifacts: dict[str, object] = {
            "mix": str(mix_path),
            "stems": stem_paths[:real_stem_count],
            "pre_bwe_mix": str(pre_bwe_mix_path),
            "pre_bwe_stems": pre_bwe_stem_paths[:real_stem_count],
        }
        if scene_lane:
            artifacts["scene_render"] = stem_paths[-1]
        return decoded_audio, mixed, artifacts

    def _decode_video(
        self,
        output_path: Path,
        video_latent: torch.Tensor,
        audio: Audio | None,
        generator: torch.Generator,
        *,
        width: int,
        height: int,
        frame_export: tuple[Path, list[int]] | None = None,
    ) -> str:
        generation = self.config.generation
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path),
            video_shape=VideoPixelShape(
                batch=1,
                frames=generation.frames,
                height=height,
                width=width,
                fps=generation.frame_rate,
            ),
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )
        video = self.video_decoder(video_latent, tiling, generator=generator)
        if frame_export is not None:
            video = export_sampled_frames(video, frames_dir=frame_export[0], indices=frame_export[1])
        encode_video(
            video=video,
            fps=int(generation.frame_rate),
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(generation.frames, tiling),
        )
        return str(output_path)

    def _save_run_metadata(self, root: Path, *, positives: list[str], negatives: list[str]) -> None:
        payload = self.config.summary()
        payload["effective_prompts"] = {
            stem.id: {"positive": positive, "negative": negative}
            for stem, positive, negative in zip(self.config.stems, positives, negatives, strict=True)
        }
        (root / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
