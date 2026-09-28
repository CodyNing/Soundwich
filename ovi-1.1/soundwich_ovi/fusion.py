"""Multi-stem forward pass for Ovi's ``FusionModel``.

Ovi's upstream ``FusionModel`` runs one video lane against one audio lane.
These functions add a second forward path that shares one video lane across
an expanded batch of audio lanes (one per requested sound source) and blend
cached mean carrier tokens into the audio self-attention output. They are
attached to ``ovi.modules.fusion.FusionModel`` as methods on import so the
upstream class gains this path without modifying the upstream file.
"""

from __future__ import annotations

import torch

from ovi.modules.attention import flash_attention
from ovi.modules.fusion import FusionModel
from ovi.modules.model import rope_apply

from .blending import blend_cached_carriers, blend_mean_carriers


def _multistem_cross_attention_forward(
    self,
    cross_attn_block,
    src_seq,
    src_grid_sizes,
    src_freqs,
    target_seq,
    target_seq_lens,
    target_grid_sizes,
    target_freqs,
    context,
    context_lens,
    *,
    target_layout,
    target_active_prefix=None,
):
    """Cross attention for one shared video and an expanded audio batch.

    ``broadcast`` maps one video sequence to every audio branch. ``concat``
    presents all real audio sequences as one key/value sequence to the single
    shared video branch.
    """
    if self.use_sp:
        raise NotImplementedError(
            "multi-stem Ovi inference currently requires sp_size=1"
        )
    if hasattr(cross_attn_block, "k_img"):
        raise NotImplementedError(
            "multi-stem Ovi inference currently supports T2V, not I2V"
        )

    batch, heads, head_dim = (
        src_seq.size(0),
        cross_attn_block.num_heads,
        cross_attn_block.head_dim,
    )
    query, text_key, text_value = cross_attn_block.qkv_fn(src_seq, context)
    text_out = flash_attention(
        query,
        text_key,
        text_value,
        k_lens=context_lens,
    )

    target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
    target_key = cross_attn_block.norm_k_fusion(
        cross_attn_block.k_fusion(target_seq)
    ).view(target_seq.size(0), -1, heads, head_dim)
    target_value = cross_attn_block.v_fusion(target_seq).view(
        target_seq.size(0), -1, heads, head_dim
    )
    query = rope_apply(query, src_grid_sizes, src_freqs)
    target_key = rope_apply(target_key, target_grid_sizes, target_freqs)

    if target_layout == "broadcast":
        if target_key.size(0) != 1:
            raise ValueError("broadcast target must have batch size one")
        target_key = target_key.expand(batch, -1, -1, -1)
        target_value = target_value.expand(batch, -1, -1, -1)
        if target_seq_lens is not None:
            target_seq_lens = target_seq_lens[:1].expand(batch)
    elif target_layout == "concat":
        if batch != 1:
            raise ValueError("concat source must have batch size one")
        target_key = target_key.reshape(
            1, -1, target_key.size(2), target_key.size(3)
        )
        target_value = target_value.reshape(
            1, -1, target_value.size(2), target_value.size(3)
        )
        if target_seq_lens is not None:
            target_seq_lens = target_seq_lens.sum().reshape(1)
    else:
        raise ValueError(f"unknown target layout: {target_layout}")

    if target_active_prefix is not None:
        active_count = int(target_active_prefix)
        if active_count < 0 or active_count > batch:
            raise ValueError("target_active_prefix outside source batch")
        target_out = torch.zeros_like(query)
        if active_count > 0:
            active_lens = (
                target_seq_lens[:active_count]
                if target_seq_lens is not None
                else None
            )
            # Real audio branches are always a contiguous prefix. Slicing
            # preserves the zero-stride broadcast view; CUDA advanced
            # indexing of this large view fails on Blackwell.
            target_out[:active_count] = flash_attention(
                query[:active_count],
                target_key[:active_count],
                target_value[:active_count],
                k_lens=active_lens,
            )
    else:
        target_out = flash_attention(
            query,
            target_key,
            target_value,
            k_lens=target_seq_lens,
        )

    out = (text_out + target_out).flatten(2)
    return cross_attn_block.o(out)


def _multistem_cross_attention_ffn_forward(
    self,
    attn_block,
    src_seq,
    src_grid_sizes,
    src_freqs,
    target_seq,
    target_seq_lens,
    target_grid_sizes,
    target_freqs,
    context,
    context_lens,
    src_e,
    *,
    target_layout,
    target_active_prefix=None,
):
    src_seq = src_seq + self._multistem_cross_attention_forward(
        attn_block.cross_attn,
        attn_block.norm3(src_seq),
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        target_layout=target_layout,
        target_active_prefix=target_active_prefix,
    )
    y = attn_block.ffn(
        attn_block.norm2(src_seq).bfloat16()
        * (1 + src_e[4].squeeze(2))
        + src_e[3].squeeze(2)
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        src_seq = src_seq + y * src_e[5].squeeze(2)
    return src_seq


def single_multistem_fusion_block_forward(
    self,
    block_index,
    vid_block,
    audio_block,
    vid,
    audio,
    vid_e,
    vid_seq_lens,
    vid_grid_sizes,
    vid_freqs,
    vid_context,
    vid_context_lens,
    audio_e,
    audio_seq_lens,
    audio_grid_sizes,
    audio_freqs,
    audio_context,
    audio_context_lens,
    multistem_options,
):
    """One fusion block with one video lane and N real audio lanes."""
    real_count = int(multistem_options["real_stem_count"])
    if real_count <= 0 or real_count > audio.size(0):
        raise ValueError("invalid real_stem_count")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)
    audio_y = audio_block.self_attn(
        audio_block.norm1(audio).bfloat16()
        * (1 + audio_e[1].squeeze(2))
        + audio_e[0].squeeze(2),
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
    )
    carrier_recorder = multistem_options.get("carrier_recorder")
    if carrier_recorder is not None:
        carrier_recorder.capture(block_index, audio_y[0])
    if bool(multistem_options.get("mean_blending", False)):
        # Carrier control modifies the audio self-attention output, before
        # modulation and the residual add. Applying it to the post-residual
        # hidden state would compound outside suppression against the entire
        # model state at every block.
        activation_tokens = multistem_options.get("cached_activation_tokens")
        suppression_tokens = multistem_options.get("cached_suppression_tokens")
        blend_options = {
            "inside_gate": multistem_options["inside_gate"],
            "outside_gate": multistem_options["outside_gate"],
            "inside_strength": float(
                multistem_options.get("inside_strength", 0.10)
            ),
            "outside_strength": float(
                multistem_options.get("outside_strength", 0.50)
            ),
            "outside_suppression": float(
                multistem_options.get("outside_suppression", 0.25)
            ),
            "activation_value_scale": float(
                multistem_options.get("activation_value_scale", 0.80)
            ),
        }
        if activation_tokens is not None and suppression_tokens is not None:
            audio_y = audio_y.clone()
            activation = activation_tokens[block_index]
            suppression = suppression_tokens[block_index]
            if activation.ndim == 1:
                activation = activation.reshape(1, -1)
            if suppression.ndim == 1:
                suppression = suppression.reshape(1, -1)
            audio_y[:real_count] = blend_cached_carriers(
                audio_y[:real_count],
                activation=activation,
                suppression=suppression,
                **blend_options,
            )
        else:
            audio_y = blend_mean_carriers(
                audio_y,
                real_stem_count=real_count,
                activation_index=int(multistem_options["activation_index"]),
                suppression_index=int(multistem_options["suppression_index"]),
                activation_quantile=float(
                    multistem_options.get("activation_quantile", 0.70)
                ),
                suppression_quantile=float(
                    multistem_options.get("suppression_quantile", 0.30)
                ),
                **blend_options,
            )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        audio = audio + audio_y * audio_e[2].squeeze(2)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)
    vid_y = vid_block.self_attn(
        vid_block.norm1(vid).bfloat16()
        * (1 + vid_e[1].squeeze(2))
        + vid_e[0].squeeze(2),
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        vid = vid + vid_y * vid_e[2].squeeze(2)

    # Match the original Ovi ordering: video attends to the audio state from
    # before V2A, while real audio branches receive the shared video state.
    audio_for_video = audio[:real_count]
    audio = self._multistem_cross_attention_ffn_forward(
        audio_block,
        audio,
        audio_grid_sizes,
        audio_freqs,
        vid,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        audio_context,
        audio_context_lens,
        audio_e,
        target_layout="broadcast",
        target_active_prefix=(
            real_count
            if bool(multistem_options.get("v2a_enabled", True))
            else 0
        ),
    )
    vid = self._multistem_cross_attention_ffn_forward(
        vid_block,
        vid,
        vid_grid_sizes,
        vid_freqs,
        audio_for_video,
        audio_seq_lens[:real_count],
        audio_grid_sizes[:real_count],
        audio_freqs,
        vid_context,
        vid_context_lens,
        vid_e,
        target_layout="concat",
        target_active_prefix=(
            None
            if bool(multistem_options.get("a2v_enabled", True))
            else 0
        ),
    )
    return vid, audio


def forward_multistem(
    self,
    vid,
    audio,
    vid_t,
    audio_t,
    vid_context,
    audio_context,
    vid_seq_len,
    audio_seq_len,
    multistem_options,
    first_frame_is_clean=False,
    slg_layer=False,
):
    """Run one shared video stream with an expanded multi-stem audio batch."""
    if self.use_sp:
        raise NotImplementedError(
            "multi-stem Ovi inference currently requires sp_size=1"
        )
    if len(vid) != 1:
        raise ValueError("multi-stem inference requires exactly one video latent")
    if len(audio) != len(audio_context):
        raise ValueError("audio latent and context branch counts differ")

    vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
        x=vid,
        t=vid_t,
        context=vid_context,
        seq_len=vid_seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=first_frame_is_clean,
    )
    audio, audio_e, audio_kwargs = (
        self.audio_model.prepare_transformer_block_kwargs(
            x=audio,
            t=audio_t,
            context=audio_context,
            seq_len=audio_seq_len,
            clip_fea=None,
            y=None,
            first_frame_is_clean=False,
        )
    )
    kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)
    for index in range(self.num_blocks):
        if slg_layer > 0 and index == slg_layer:
            continue
        vid, audio = self.single_multistem_fusion_block_forward(
            block_index=index,
            vid_block=self.video_model.blocks[index],
            audio_block=self.audio_model.blocks[index],
            vid=vid,
            audio=audio,
            multistem_options=multistem_options,
            **kwargs,
        )
    vid = self.video_model.post_transformer_block_out(
        vid, vid_kwargs["grid_sizes"], vid_e
    )
    audio = self.audio_model.post_transformer_block_out(
        audio, audio_kwargs["grid_sizes"], audio_e
    )
    return vid, audio


FusionModel._multistem_cross_attention_forward = _multistem_cross_attention_forward
FusionModel._multistem_cross_attention_ffn_forward = (
    _multistem_cross_attention_ffn_forward
)
FusionModel.single_multistem_fusion_block_forward = (
    single_multistem_fusion_block_forward
)
FusionModel.forward_multistem = forward_multistem
