"""Ovi-native multi-stem generation engine.

Wraps upstream ``OviFusionEngine`` with two entry points that both drive
``FusionModel.forward_multistem`` (attached by :mod:`soundwich_ovi.fusion`):

- :meth:`OviMultiStemEngine.generate_carrier` records one reference carrier
  bank (a mean audio token per denoising step and transformer block).
- :meth:`OviMultiStemEngine.generate_scene` generates one shared video and
  one independently controlled audio stem per requested source, blending in
  cached carrier banks to keep each stem inside its requested timeline.

Both entry points live on one engine so a single process loads the model
once and can build any missing carriers before generating a scene.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.processing_utils import snap_hw_to_multiple_of_32

from .blending import build_timeline_gates
from .carrier_bank import CarrierBank, CarrierRecorder
from .scene import SceneSpec, build_prompts


def build_carrier_prompts(*, visual_prompt: str, audio_prompt: str) -> tuple[str, str]:
    """Use Ovi's native shared positive prompt for both modalities."""
    shared_context = (
        f"{visual_prompt.strip()} Audio: {audio_prompt.strip().rstrip('.')}."
    ).strip()
    return shared_context, shared_context


class OviMultiStemEngine(OviFusionEngine):
    """Generate carrier banks and multi-stem scenes from one loaded model."""

    @torch.inference_mode()
    def generate_carrier(
        self,
        *,
        carrier_id: str,
        role: str,
        seed: int,
        quantile: float,
        visual_prompt: str,
        audio_prompt: str,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        video_frame_height_width: tuple[int, int] | list[int],
        solver_name: str,
        sample_steps: int,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        a2v_enabled: bool,
        v2a_enabled: bool,
    ) -> dict[str, Any]:
        """Record one quantile-mean carrier token per step and block.

        Only the carrier bank is produced; the reference video/audio is not
        decoded since it is not needed to blend the carrier into later runs.
        """
        video_positive_text, audio_positive_text = build_carrier_prompts(
            visual_prompt=visual_prompt,
            audio_prompt=audio_prompt,
        )
        video_positive_prompt = self.text_formatter(video_positive_text)
        audio_positive_prompt = self.text_formatter(audio_positive_text)
        video_negative_prompt = self.text_formatter(video_negative_prompt)
        audio_negative_prompt = self.text_formatter(audio_negative_prompt)
        if self.cpu_offload:
            self.text_model.model = self.text_model.model.to(self.device)
        embeddings = self.text_model(
            [
                video_positive_prompt,
                audio_positive_prompt,
                video_negative_prompt,
                audio_negative_prompt,
            ],
            self.text_model.device,
        )
        video_positive, audio_positive, video_negative, audio_negative = [
            value.to(device=self.device, dtype=self.target_dtype)
            for value in embeddings
        ]
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )
        requested_h, requested_w = map(int, video_frame_height_width)
        video_h, video_w = snap_hw_to_multiple_of_32(
            requested_h, requested_w, area=self.target_area
        )
        video_latent_h, video_latent_w = video_h // 16, video_w // 16
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_noise = torch.randn(
            (
                self.video_latent_channel,
                self.video_latent_length,
                video_latent_h,
                video_latent_w,
            ),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )
        audio_noise = torch.randn(
            (self.audio_latent_length, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )
        patch_h, patch_w = self.model.video_model.patch_size[1:]
        video_seq_len = (
            video_noise.shape[1]
            * video_noise.shape[2]
            * video_noise.shape[3]
            // (patch_h * patch_w)
        )
        recorder = CarrierRecorder(
            role=role,
            quantile=quantile,
            step_count=sample_steps,
            block_count=self.model.num_blocks,
        )
        common_options = {
            "real_stem_count": 1,
            "mean_blending": False,
            "a2v_enabled": bool(a2v_enabled),
            "v2a_enabled": bool(v2a_enabled),
        }

        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_video.model)
            self.offload_to_cpu(self.vae_model_audio)
            self.model = self.model.to(self.device)
        with torch.amp.autocast(
            "cuda",
            enabled=self.target_dtype != torch.float32,
            dtype=self.target_dtype,
        ):
            for step_index, (t_video, t_audio) in enumerate(
                tqdm(
                    zip(timesteps_video, timesteps_audio),
                    total=len(timesteps_video),
                    desc=f"Ovi carrier: {carrier_id}",
                )
            ):
                recorder.begin_step(step_index)
                positive_options = {
                    **common_options,
                    "carrier_recorder": recorder,
                }
                pred_video_positive, pred_audio_positive = (
                    self.model.forward_multistem(
                        vid=[video_noise],
                        audio=[audio_noise],
                        vid_t=t_video.reshape(1),
                        audio_t=t_audio.reshape(1),
                        vid_context=[video_positive],
                        audio_context=[audio_positive],
                        vid_seq_len=video_seq_len,
                        audio_seq_len=self.audio_latent_length,
                        multistem_options=positive_options,
                    )
                )
                pred_video_negative, pred_audio_negative = (
                    self.model.forward_multistem(
                        vid=[video_noise],
                        audio=[audio_noise],
                        vid_t=t_video.reshape(1),
                        audio_t=t_audio.reshape(1),
                        vid_context=[video_negative],
                        audio_context=[audio_negative],
                        vid_seq_len=video_seq_len,
                        audio_seq_len=self.audio_latent_length,
                        multistem_options=common_options,
                        slg_layer=slg_layer,
                    )
                )
                video_prediction = pred_video_negative[0] + video_guidance_scale * (
                    pred_video_positive[0] - pred_video_negative[0]
                )
                audio_prediction = pred_audio_negative[0] + audio_guidance_scale * (
                    pred_audio_positive[0] - pred_audio_negative[0]
                )
                video_noise = scheduler_video.step(
                    video_prediction.unsqueeze(0),
                    t_video,
                    video_noise.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)
                audio_noise = scheduler_audio.step(
                    audio_prediction.unsqueeze(0),
                    t_audio,
                    audio_noise.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)

        if self.cpu_offload:
            self.offload_to_cpu(self.model)

        timestep_values = tuple(
            float(value) for value in timesteps_audio.detach().cpu().tolist()
        )
        bank = CarrierBank(
            carrier_id=carrier_id,
            role=role,
            tokens=recorder.tokens(),
            model_name=self.model_name,
            solver_name=solver_name,
            shift=shift,
            timesteps=timestep_values,
            quantile=quantile,
            prompt=audio_prompt,
            visual_prompt=visual_prompt,
            a2v_enabled=a2v_enabled,
            v2a_enabled=v2a_enabled,
        )
        return {
            "bank": bank,
            "metadata": {
                "carrier_id": carrier_id,
                "role": role,
                "model_name": self.model_name,
                "sample_steps": sample_steps,
                "solver_name": solver_name,
                "shift": shift,
                "seed": seed,
                "quantile": quantile,
                "a2v_enabled": a2v_enabled,
                "v2a_enabled": v2a_enabled,
                "carrier_shape": list(bank.tokens.shape),
            },
        }

    @torch.inference_mode()
    def generate_scene(
        self,
        scene: SceneSpec,
        *,
        video_frame_height_width: tuple[int, int] | list[int],
        solver_name: str = "unipc",
        sample_steps: int = 50,
        shift: float = 5.0,
        video_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 3.0,
        slg_layer: int = 11,
        activation_carrier_paths: list[str] | tuple[str, ...],
        suppression_carrier_path: str,
        activation_quantile: float = 0.70,
        suppression_quantile: float = 0.30,
        inside_strength: float = 0.15,
        outside_strength: float = 0.50,
        outside_suppression: float = 0.25,
        activation_value_scale: float = 0.80,
        feather_seconds: float = 0.20,
    ) -> dict[str, Any]:
        """Generate one shared video and one audio stem per scene source."""
        if self.model.use_sp:
            raise NotImplementedError(
                "multi-stem Ovi requires sp_size=1; the audio batch is not "
                "sequence-parallelized"
            )
        stem_count = len(scene.stems)
        if stem_count < 1:
            raise ValueError("scene must contain at least one stem")

        prompts = build_prompts(scene)
        activation_carrier_paths = list(activation_carrier_paths)
        if len(activation_carrier_paths) != stem_count:
            raise ValueError("activation carrier paths must match the scene stems")
        formatted = {
            "video_positive": self.text_formatter(prompts["video_positive"]),
            "video_negative": self.text_formatter(prompts["video_negative"]),
            "audio_positive": [
                self.text_formatter(prompt) for prompt in prompts["audio_positive"]
            ],
            "audio_negative": [
                self.text_formatter(prompt) for prompt in prompts["audio_negative"]
            ],
        }

        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )

        requested_h, requested_w = map(int, video_frame_height_width)
        video_h, video_w = snap_hw_to_multiple_of_32(
            requested_h, requested_w, area=self.target_area
        )
        video_latent_h, video_latent_w = video_h // 16, video_w // 16

        all_text = [formatted["video_positive"], formatted["video_negative"]]
        all_text.extend(formatted["audio_positive"])
        all_text.extend(formatted["audio_negative"])
        if self.cpu_offload:
            self.text_model.model = self.text_model.model.to(self.device)
        embeddings = self.text_model(all_text, self.text_model.device)
        embeddings = [
            embedding.to(dtype=self.target_dtype, device=self.device)
            for embedding in embeddings
        ]
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        cursor = 0
        video_positive = embeddings[cursor]
        cursor += 1
        video_negative = embeddings[cursor]
        cursor += 1
        audio_positive = embeddings[cursor : cursor + stem_count]
        cursor += stem_count
        audio_negative = embeddings[cursor : cursor + stem_count]

        generator = torch.Generator(device=self.device).manual_seed(scene.seed)
        video_noise = torch.randn(
            (
                self.video_latent_channel,
                self.video_latent_length,
                video_latent_h,
                video_latent_w,
            ),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )
        audio_noise = torch.randn(
            (stem_count, self.audio_latent_length, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )

        patch_h, patch_w = self.model.video_model.patch_size[1:]
        max_seq_len_video = (
            video_noise.shape[1]
            * video_noise.shape[2]
            * video_noise.shape[3]
            // (patch_h * patch_w)
        )
        max_seq_len_audio = self.audio_latent_length
        inside_gate, outside_gate = build_timeline_gates(
            [stem.windows for stem in scene.stems],
            max_seq_len_audio,
            duration_seconds=scene.duration_seconds,
            feather_seconds=feather_seconds,
            device=self.device,
            dtype=self.target_dtype,
        )
        multistem_options = {
            "real_stem_count": stem_count,
            "mean_blending": True,
            "inside_gate": inside_gate,
            "outside_gate": outside_gate,
            "activation_quantile": activation_quantile,
            "suppression_quantile": suppression_quantile,
            "inside_strength": inside_strength,
            "outside_strength": outside_strength,
            "outside_suppression": outside_suppression,
            "activation_value_scale": activation_value_scale,
        }
        activation_banks = [
            CarrierBank.load(path) for path in activation_carrier_paths
        ]
        suppression_bank = CarrierBank.load(suppression_carrier_path)
        timestep_values = tuple(
            float(value) for value in timesteps_audio.detach().cpu().tolist()
        )
        validation = {
            "model_name": self.model_name,
            "solver_name": solver_name,
            "shift": shift,
            "timesteps": timestep_values,
            "block_count": self.model.num_blocks,
            "hidden_dim": self.model.audio_model.dim,
        }
        for activation_bank in activation_banks:
            activation_bank.validate_for_run(role="activation", **validation)
        suppression_bank.validate_for_run(role="suppression", **validation)
        activation_tokens = torch.stack(
            [bank.tokens for bank in activation_banks], dim=2
        ).to(device=self.device, dtype=self.target_dtype)
        suppression_tokens = suppression_bank.tokens.to(
            device=self.device, dtype=self.target_dtype
        )

        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_video.model)
            self.offload_to_cpu(self.vae_model_audio)
            self.model = self.model.to(self.device)

        logging.info(
            "Multi-stem Ovi: scene=%s stems=%d video=%dx%d steps=%d",
            scene.id,
            stem_count,
            video_h,
            video_w,
            sample_steps,
        )
        with torch.amp.autocast(
            "cuda",
            enabled=self.target_dtype != torch.float32,
            dtype=self.target_dtype,
        ):
            for step_index, (t_video, t_audio) in enumerate(tqdm(
                zip(timesteps_video, timesteps_audio),
                total=len(timesteps_video),
                desc=f"Ovi multi-stem: {scene.id}",
            )):
                vid_t = t_video.reshape(1)
                real_audio_t = t_audio.reshape(1).expand(stem_count)

                real_audio_latents = [
                    audio_noise[index] for index in range(stem_count)
                ]
                positive_options = {
                    **multistem_options,
                    "cached_activation_tokens": activation_tokens[step_index],
                    "cached_suppression_tokens": suppression_tokens[step_index],
                }

                # Audio CFG: real stems only. Cached carriers are replayed in
                # the positive pass and disabled in the negative pass.
                _, pred_audio_positive = self.model.forward_multistem(
                    vid=[video_noise],
                    audio=real_audio_latents,
                    vid_t=vid_t,
                    audio_t=real_audio_t,
                    vid_context=[video_positive],
                    audio_context=list(audio_positive),
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    multistem_options=positive_options,
                )

                disabled_options = {
                    **multistem_options,
                    "mean_blending": False,
                }
                _, pred_audio_negative = self.model.forward_multistem(
                    vid=[video_noise],
                    audio=real_audio_latents,
                    vid_t=vid_t,
                    audio_t=real_audio_t,
                    vid_context=[video_positive],
                    audio_context=list(audio_negative),
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    multistem_options=disabled_options,
                    slg_layer=slg_layer,
                )

                # Video CFG is isolated into two matched shared-video calls.
                # Both use the same real positive audio stems with carrier
                # blending disabled; only the visual conditioning changes.
                pred_video_positive, _ = self.model.forward_multistem(
                    vid=[video_noise],
                    audio=real_audio_latents,
                    vid_t=vid_t,
                    audio_t=real_audio_t,
                    vid_context=[video_positive],
                    audio_context=list(audio_positive),
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    multistem_options=disabled_options,
                )
                pred_video_negative, _ = self.model.forward_multistem(
                    vid=[video_noise],
                    audio=real_audio_latents,
                    vid_t=vid_t,
                    audio_t=real_audio_t,
                    vid_context=[video_negative],
                    audio_context=list(audio_positive),
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    multistem_options=disabled_options,
                )

                audio_positive_tensor = torch.stack(
                    pred_audio_positive[:stem_count], dim=0
                )
                audio_negative_tensor = torch.stack(pred_audio_negative, dim=0)
                pred_audio_guided = audio_negative_tensor + audio_guidance_scale * (
                    audio_positive_tensor - audio_negative_tensor
                )
                pred_video_guided = pred_video_negative[0] + video_guidance_scale * (
                    pred_video_positive[0] - pred_video_negative[0]
                )

                video_noise = scheduler_video.step(
                    pred_video_guided.unsqueeze(0),
                    t_video,
                    video_noise.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)
                audio_noise = scheduler_audio.step(
                    pred_audio_guided,
                    t_audio,
                    audio_noise,
                    return_dict=False,
                )[0]

        if self.cpu_offload:
            self.offload_to_cpu(self.model)

        clean_video_latent = video_noise.detach().cpu()
        clean_audio_latents = audio_noise.detach().cpu()
        del video_noise, audio_noise

        generated_audio: list[np.ndarray] = []
        if self.cpu_offload:
            self.vae_model_audio = self.vae_model_audio.to(self.device)
        for latent in clean_audio_latents:
            audio_for_vae = latent.to(self.device).unsqueeze(0).transpose(1, 2)
            waveform = self.vae_model_audio.wrapped_decode(audio_for_vae)
            generated_audio.append(waveform.squeeze().cpu().float().numpy())
            del audio_for_vae, waveform
        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_audio)

        if self.cpu_offload:
            self.vae_model_video.model = self.vae_model_video.model.to(self.device)
        video_for_vae = clean_video_latent.to(self.device).unsqueeze(0)
        generated_video = self.vae_model_video.wrapped_decode(video_for_vae)
        generated_video = generated_video.squeeze(0).cpu().float().numpy()
        del video_for_vae
        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_video.model)

        return {
            "video": generated_video,
            "stems": generated_audio,
            "prompts": formatted,
            "resolution": [video_h, video_w],
            "method": {
                "activation_carrier_ids": [
                    bank.carrier_id for bank in activation_banks
                ],
                "suppression_carrier_id": suppression_bank.carrier_id,
                "activation_quantile": float(activation_quantile),
                "suppression_quantile": float(suppression_quantile),
                "inside_strength": float(inside_strength),
                "outside_strength": float(outside_strength),
                "outside_suppression": float(outside_suppression),
                "activation_value_scale": float(activation_value_scale),
                "feather_seconds": float(feather_seconds),
            },
        }
