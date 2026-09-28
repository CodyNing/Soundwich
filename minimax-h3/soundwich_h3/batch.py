# Attention processors adapted from diffusers' MiniMaxH3AttnProcessor
# (Copyright 2025 The MiniMax Team and The HuggingFace Team, Apache License 2.0).
"""N audio stems plus one dedicated video row on H3's native batch axis.

Row zero supplies the shared video; rows 1..N supply the audio stems. Before
each block, every audio row receives row zero's video tokens. At joint
attention, row zero replaces its placeholder audio K/V with all stems' audio
K/V (same RoPE clock); its audio queries/outputs are unused. Audio rows keep
native attention over their own text/audio and the copied shared video.
Outside a controlled stem's windows, its audio queries read only its own audio.
With entity masks (Scene Integration), `EntityRoutedAttention` replaces the
joint-attention processor.
"""
from dataclasses import dataclass

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

AUDIO_TOKENS_PER_SECOND = 40


def pad_contexts(contexts):
    length = max(t.shape[1] for t in contexts)
    padded = contexts[0].new_zeros((len(contexts), length, contexts[0].shape[-1]))
    valid = torch.zeros(len(contexts), length, dtype=torch.bool, device=padded.device)
    for i, text in enumerate(contexts):
        padded[i, -text.shape[1]:] = text[0]
        valid[i, -text.shape[1]:] = True
    return padded, valid


@dataclass
class BatchLayout:
    positions: torch.Tensor
    tags: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def create(cls, text_valid, *, latent_frames, latent_height, latent_width, audio_frames, patch_size):
        device = text_valid.device
        p, t, v, a, x, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(text_valid.shape[1], dtype=torch.long), latent_frames, latent_height,
            latent_width, audio_frames, patch_size, 2, 2, 0)
        valid = torch.ones(text_valid.shape[0], len(t), device=device, dtype=torch.bool)
        valid[:, :text_valid.shape[1]] = text_valid
        return cls(p.to(device), t.to(device), v.to(device), a.to(device), x.to(device), valid)


def _qkv(attn, hidden_states, rotary_emb):
    q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
    k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
    v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
    if rotary_emb is not None:
        q = _apply_rotary_emb(q, *rotary_emb)
        k = _apply_rotary_emb(k, *rotary_emb)
    return q, k, v


class PaddedBatchAttention:
    """Native per-row attention over real tokens. Padding is never a key or query.

    QKV projections remain batched. Each row uses unmasked attention after
    removing its padding, avoiding a dense [B,H,S,S] attention mask.
    """
    def __init__(self, valid):
        self.indices = [torch.where(row)[0] for row in valid]

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None or attn.fused_projections:
            raise ValueError('Expected the original unfused H3 projections and no external mask.')
        q, k, v = _qkv(attn, hidden_states, rotary_emb)
        attended = torch.zeros_like(q)
        for b, index in enumerate(self.indices):
            value = dispatch_attention_fn(q[b:b+1].index_select(1, index),
                k[b:b+1].index_select(1, index), v[b:b+1].index_select(1, index),
                dropout_p=0.0, is_causal=False)
            attended[b:b+1].index_copy_(1, index, value)
        return attn.to_out[1](attn.to_out[0](attended.flatten(2, 3).type_as(q)))


def outside_audio_mask(stems, audio_frames, device):
    """[stem, audio token] mask of channel-major stereo tokens outside the stem's windows.

    Stems without windows are uncontrolled and never restricted.
    """
    times = ((torch.arange(audio_frames, device=device).float() + 0.5) / AUDIO_TOKENS_PER_SECOND).repeat(2)
    masks = []
    for stem in stems:
        windows = stem.get('windows', [])
        inside = torch.zeros_like(times, dtype=torch.bool)
        for window in windows:
            inside |= (times >= float(window['start'])) & (times < float(window['end']))
        masks.append(~inside if windows else torch.zeros_like(inside))
    return torch.stack(masks)


class ExpandedAttention(PaddedBatchAttention):
    """Stem Formation joint attention: shared video row reads all stems' audio."""
    def __init__(self, layout, audio_only_mask):
        super().__init__(layout.valid)
        self.layout = layout
        if audio_only_mask.dtype != torch.bool or audio_only_mask.shape != (len(self.indices)-1, len(layout.audio)):
            raise ValueError('Expected boolean [stems, audio tokens] routing mask')
        audio_only_mask = audio_only_mask.to(layout.audio.device)
        self.routing = []
        for row, index in enumerate(self.indices[1:]):
            restricted = layout.audio[audio_only_mask[row]]
            allowed = torch.ones(len(layout.tags), dtype=torch.bool, device=index.device)
            allowed[restricted] = False
            self.routing.append((index[allowed[index]], restricted))

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None or attn.fused_projections:
            raise ValueError('Expected unfused H3 attention without an external mask.')
        l = self.layout
        q, k, v = _qkv(attn, hidden_states, rotary_emb)
        out = torch.zeros_like(q)
        # Replace the video row's audio sequence by all N stem audio sequences.
        visual_text = l.text[l.valid[0, l.text]]
        qi = torch.cat([visual_text, l.video])
        k0 = torch.cat([k[0:1, visual_text], k[1:, l.audio].reshape(1, -1, attn.heads, k.shape[-1]), k[0:1, l.video]], dim=1)
        v0 = torch.cat([v[0:1, visual_text], v[1:, l.audio].reshape(1, -1, attn.heads, v.shape[-1]), v[0:1, l.video]], dim=1)
        attended = dispatch_attention_fn(q[0:1, qi], k0, v0, dropout_p=0.0, is_causal=False)
        out[0:1].index_copy_(1, qi, attended)
        for row, (normal, restricted) in enumerate(self.routing, 1):
            index = self.indices[row]
            if normal.numel():
                attended = dispatch_attention_fn(q[row:row+1, normal], k[row:row+1, index], v[row:row+1, index],
                                                 dropout_p=0.0, is_causal=False)
                out[row:row+1].index_copy_(1, normal, attended)
            if restricted.numel():
                attended = dispatch_attention_fn(q[row:row+1, restricted], k[row:row+1, l.audio], v[row:row+1, l.audio],
                                                 dropout_p=0.0, is_causal=False)
                out[row:row+1].index_copy_(1, restricted, attended)
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))


class BatchExpansion:
    """Native N+1 forward with explicit video sharing and audio concatenation.

    `entity_masks` ([stem, video token] booleans) switches joint attention to
    Scene Integration entity routing.
    """
    def __init__(self, model, layout, stems, audio_hook=None, entity_masks=None):
        if layout.valid.shape[0] < 2:
            raise ValueError('Need a video row and at least one audio row.')
        self.model, self.layout, self.audio_hook = model, layout, audio_hook
        self.step = 0
        self.original = []
        self.handles = []
        audio_only = outside_audio_mask(stems, len(layout.audio)//2, layout.audio.device)
        if entity_masks is not None:
            from .routing import EntityRoutedAttention
            processor = EntityRoutedAttention(layout, entity_masks, stems, audio_only)
        else:
            processor = ExpandedAttention(layout, audio_only)
        for block in model.token_refiner.refiner_blocks:
            self.original.append((block.attn, block.attn.processor))
            block.attn.set_processor(PaddedBatchAttention(layout.valid[:, :len(layout.text)]))
        for index, block in enumerate(model.transformer_blocks):
            self.original.append((block.attn, block.attn.processor))
            block.attn.set_processor(processor)
            self.handles.append(block.register_forward_pre_hook(self._copy_video))
            if audio_hook is not None:
                self.handles.append(block.attn.register_forward_hook(self._hook(index)))

    def _copy_video(self, module, args):
        hidden = args[0].clone()
        hidden[1:, self.layout.video] = hidden[0:1, self.layout.video]
        return (hidden, *args[1:])

    def _hook(self, index):
        def apply(module, args, output):
            audio = output[1:].index_select(1, self.layout.audio)
            changed = self.audio_hook(self.step, index, audio)
            if changed is None:
                return output
            result = output.clone()
            result[1:, self.layout.audio] = changed
            return result
        return apply

    def close(self):
        for module, processor in self.original:
            module.set_processor(processor)
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def __call__(self, video, audio, text, video_t, audio_t, step):
        l = self.layout
        self.step = step
        batch = l.valid.shape[0]
        if video.shape[0] != 1 or audio.shape[0] != batch-1 or text.shape[0] != batch:
            raise ValueError('Expected one shared video, N audio states, and N+1 text contexts.')
        all_audio = torch.cat([torch.zeros_like(audio[:1]), audio])
        video_pred, audio_pred = native_forward(self.model, l, video.expand(batch, -1, -1), all_audio, text, video_t, audio_t)
        return video_pred[:1], audio_pred[1:]


def native_forward(model, layout, video, audio, text, video_t, audio_t):
    """Build native modality timesteps and call the unmodified H3 forward."""
    l = layout
    rows = torch.full((len(l.tags),), float(video_t), device=video.device, dtype=torch.float32)
    rows[l.audio] = float(audio_t)
    timestep, indices = torch.unique(rows, sorted=True, return_inverse=True)
    return model(hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=text,
                 timestep=timestep, timestep_indices=indices, token_tags=l.tags,
                 position_ids=l.positions, video_indices=l.video, audio_indices=l.audio,
                 text_indices=l.text, return_dict=False)


class NativeCapture:
    """Unmodified single-row H3 forward with read-only audio capture hooks (carrier references)."""
    def __init__(self, model, layout, audio_hook):
        if layout.valid.shape[0] != 1 or not layout.valid.all():
            raise ValueError('Native capture needs one unpadded reference row')
        self.model, self.layout, self.audio_hook = model, layout, audio_hook
        self.step = 0
        self.handles = [block.attn.register_forward_hook(self._hook(i))
                        for i, block in enumerate(model.transformer_blocks)]

    def _hook(self, block):
        def capture(module, args, output):
            self.audio_hook(self.step, block, output.index_select(1, self.layout.audio))
            return output
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def __call__(self, video, audio, text, video_t, audio_t, step):
        self.step = step
        return native_forward(self.model, self.layout, video, audio, text, video_t, audio_t)
