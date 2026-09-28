# Attention processor adapted from diffusers' MiniMaxH3AttnProcessor
# (Copyright 2025 The MiniMax Team and The HuggingFace Team, Apache License 2.0).
"""Scene Integration entity routing inside H3's packed N+1 joint attention.

Audio queries of a stem see only its owner's video tokens (SAM masks); owner
video queries see the stem's full audio K/V. Background video tokens read no
stem audio directly. Text queries and shared video-to-video attention remain
native, so routing constrains direct paths only; it does not guarantee
information isolation across blocks. Query groups with identical allowed keys
are dispatched separately; no dense S-by-S mask is built.
"""
import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

from .batch import _qkv


def _dispatch(q, k, v, row, queries, keys):
    return dispatch_attention_fn(q[row:row+1].index_select(1, queries),
        k[row:row+1].index_select(1, keys),
        v[row:row+1].index_select(1, keys), dropout_p=0., is_causal=False)


class EntityRoutedAttention:
    def __init__(self, layout, owners, stems, audio_only):
        n = len(stems)
        if not 2 <= n <= 10 or layout.valid.shape[0] != n+1:
            raise ValueError('Expected 2-10 audio stems plus one shared-video row')
        if owners.dtype != torch.bool or owners.shape != (n, len(layout.video)):
            raise ValueError('Expected boolean SAM [stem, video token] masks')
        self.layout = layout
        self.owners = owners.to(layout.video.device)
        self.audio_only = audio_only.to(layout.audio.device)
        if self.audio_only.shape != (n, len(layout.audio)):
            raise ValueError('Expected boolean outside-window [stem, audio token] mask')
        self.background_only = [s.get('video_routing') == 'background_only' for s in stems]
        self.n = n
        bits = torch.zeros(len(layout.video), dtype=torch.long, device=layout.video.device)
        for i in range(n):
            bits |= self.owners[i].long() << i
        self.groups = [(code, layout.video[bits == code]) for code in torch.unique(bits).tolist()]

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None or attn.fused_projections:
            raise ValueError('Expected unfused H3 attention without dense mask')
        l = self.layout
        q, k, v = _qkv(attn, hidden_states, rotary_emb)
        out = torch.zeros_like(q)
        visual_text = l.text[l.valid[0, l.text]]
        base_k = torch.cat((k[0:1, visual_text], k[0:1, l.video]), dim=1)
        base_v = torch.cat((v[0:1, visual_text], v[0:1, l.video]), dim=1)
        if visual_text.numel():
            keys = torch.cat((base_k, *(k[i+1:i+2, l.audio] for i in range(self.n))), dim=1)
            values = torch.cat((base_v, *(v[i+1:i+2, l.audio] for i in range(self.n))), dim=1)
            result = dispatch_attention_fn(q[0:1, visual_text], keys, values, dropout_p=0., is_causal=False)
            out[0:1].index_copy_(1, visual_text, result)
        for code, queries in self.groups:
            key_chunks, value_chunks = [base_k], [base_v]
            for i in range(self.n):
                if code & (1 << i):
                    # Owner video queries read the stem's full audio at all times.
                    key_chunks.append(k[i+1:i+2, l.audio])
                    value_chunks.append(v[i+1:i+2, l.audio])
            result = dispatch_attention_fn(q[0:1, queries], torch.cat(key_chunks, dim=1),
                torch.cat(value_chunks, dim=1), dropout_p=0., is_causal=False)
            out[0:1].index_copy_(1, queries, result)
        for i in range(self.n):
            row = i+1
            own_text = l.text[l.valid[row, l.text]]
            if own_text.numel():
                result = _dispatch(q, k, v, row, own_text,
                    torch.cat((own_text, l.audio, l.video[self.owners[i]] if self.background_only[i] else l.video)))
                out[row:row+1].index_copy_(1, own_text, result)
            inside = l.audio[~self.audio_only[i]]
            outside = l.audio[self.audio_only[i]]
            if inside.numel():
                result = _dispatch(q, k, v, row, inside, torch.cat((own_text, l.audio, l.video[self.owners[i]])))
                out[row:row+1].index_copy_(1, inside, result)
            if outside.numel():
                result = _dispatch(q, k, v, row, outside, l.audio)
                out[row:row+1].index_copy_(1, outside, result)
            # Audio-row video-query outputs are overwritten by the shared video before the next block.
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))
