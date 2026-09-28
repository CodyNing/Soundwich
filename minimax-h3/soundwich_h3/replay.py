"""Timeline control with cached carriers at H3's joint-attention audio outputs.

Capture: a native single-row reference generation records one activation or
suppression carrier per (step, block). Replay: inside a stem's windows the
activation carrier is added; outside, the stem's features are attenuated and
blended toward the quiet (suppression) carrier. H3 packs stereo audio
channel-major ([left times, right times]) at 40 tokens per second.
"""
from dataclasses import asdict, dataclass

import torch

from .batch import AUDIO_TOKENS_PER_SECOND
from .carriers import CarrierBank, CarrierRecord

HOOK = 'joint_attention_output_audio_tokens'
AUDIO_LAYOUT = 'channel_major_stereo_40hz'


@dataclass(frozen=True)
class Window:
    start: float
    end: float


@dataclass(frozen=True)
class BlendConfig:
    strength: float = 0.10
    value_scale: float = 0.80
    outside_suppression: float = 0.25   # outside-window attenuation
    outside_silence_blend: float = 0.60  # quiet-reference blending weight
    feather_seconds: float = 0.20
    rms_scale_limit: float = 4.0

    def activation_strength(self, step, total_steps):
        # Brief initial boost over the first 10% of denoising, then `strength`.
        return 0.50 if step / max(total_steps - 1, 1) <= 0.10 else self.strength


def stereo_positions(audio_frames, duration, device):
    """Normalized token-center times for H3's [left, right] stereo packing."""
    return ((torch.arange(audio_frames, device=device).float() + 0.5) / AUDIO_TOKENS_PER_SECOND / duration).repeat(2)


def stem_windows(scene):
    duration = scene['num_frames'] / 24
    result = []
    for stem in scene['stems']:
        windows = tuple(Window(float(w['start']), float(w['end'])) for w in stem.get('windows', []))
        if any(w.start < 0 or w.end <= w.start or w.end > duration for w in windows):
            raise ValueError(f'Window of stem {stem["id"]!r} must lie within the clip')
        result.append(windows)
    return result


def timeline_gate(windows, positions, *, duration_seconds, feather_seconds):
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


def _check_positions(positions, token_count, device):
    if positions is None or positions.shape != (token_count,):
        raise ValueError('H3 requires explicit channel-major stereo positions')
    return positions.to(device=device, dtype=torch.float32)


def _suppression_token(record: CarrierRecord, hidden):
    return record.restore(device=hidden.device, dtype=hidden.dtype, own_rms=True).view(1, hidden.shape[-1])


def replay_carriers(hidden, *, activation, suppression, windows, positions, duration_seconds, blend,
                    activation_strength):
    """Apply activation inside each controlled window and suppression outside it."""
    if hidden.ndim != 3 or hidden.shape[0] != len(windows) or len(activation) != len(windows):
        raise ValueError('hidden, activation carriers, and windows must have one row per stem')
    stems, token_count, dim = hidden.shape
    token_positions = _check_positions(positions, token_count, hidden.device)
    output = hidden.clone()
    suppression_token = _suppression_token(suppression, hidden)
    for stem_index in range(stems):
        if not windows[stem_index]:
            continue
        gate = timeline_gate(windows[stem_index], token_positions, duration_seconds=duration_seconds,
                             feather_seconds=blend.feather_seconds).to(dtype=hidden.dtype).view(token_count, 1)
        outside = 1 - gate
        carrier = activation[stem_index].restore(device=hidden.device, dtype=hidden.dtype, own_rms=False).view(1, dim)
        moved = carrier.expand(token_count, dim) * gate
        # Match the in-window RMS of the current features (bounded by rms_scale_limit).
        denominator = (gate.float().sum() * dim).clamp_min(1.0)
        target_rms = ((hidden[stem_index].float().square() * gate.float()).sum() / denominator).sqrt().clamp_min(1e-6)
        guide_rms = ((moved.float().square() * gate.float()).sum() / denominator).sqrt().clamp_min(1e-6)
        rms_scale = (target_rms / guide_rms).clamp(0, blend.rms_scale_limit).to(hidden.dtype)
        guide = moved * rms_scale * blend.value_scale

        base = hidden[stem_index] * (1 - blend.outside_suppression * outside)
        silence_mix = blend.outside_silence_blend * outside
        base = base * (1 - silence_mix) + suppression_token * silence_mix
        output[stem_index] = base + float(activation_strength) * gate * guide
    return output


def replay_suppression(hidden, *, suppression, windows, positions, duration_seconds, blend):
    """Keep native features in-window and blend the quiet carrier outside."""
    if hidden.ndim != 3 or hidden.shape[0] != len(windows):
        raise ValueError('hidden and windows must have one row per stem')
    _stems, token_count, dim = hidden.shape
    token_positions = _check_positions(positions, token_count, hidden.device)
    suppression_token = _suppression_token(suppression, hidden)
    output = hidden.clone()
    for stem_index, windows_i in enumerate(windows):
        if not windows_i:
            continue
        gate = timeline_gate(windows_i, token_positions, duration_seconds=duration_seconds,
                             feather_seconds=blend.feather_seconds).to(dtype=hidden.dtype).view(token_count, 1)
        outside = 1 - gate
        base = hidden[stem_index] * (1 - blend.outside_suppression * outside)
        silence_mix = blend.outside_silence_blend * outside
        output[stem_index] = base * (1 - silence_mix) + suppression_token * silence_mix
    return output


def _check_bank(bank, role, identity, schedule, audio_frames, num_blocks):
    if bank.role != role:
        raise ValueError(f'Carrier role mismatch for {bank.group!r}')
    expected = dict(identity, schedule=schedule, audio_frames=audio_frames, num_blocks=num_blocks,
                    hook=HOOK, audio_layout=AUDIO_LAYOUT)
    for key, value in expected.items():
        if bank.metadata.get(key) != value:
            raise ValueError(f'Carrier {bank.group!r} identity/geometry/schedule mismatch: {key}')
    steps = len(schedule['video'])
    if set(bank.records) != {(s, b) for s in range(steps) for b in range(num_blocks)}:
        raise ValueError(f'Carrier {bank.group!r} is missing step/block records')
    if any(r.token.shape != (identity['hidden_size'],) or not torch.isfinite(r.token).all()
           for r in bank.records.values()):
        raise ValueError(f'Carrier {bank.group!r} hidden width or values are invalid')


class CaptureCarrier:
    mode = 'capture'

    def __init__(self, role, group, metadata, quantile):
        if role not in ('activation', 'suppression') or not 0 <= quantile < 1:
            raise ValueError('Invalid carrier role or energy quantile')
        self.bank = CarrierBank(role, group, quantile, metadata=dict(metadata))

    def validate(self, schedule, audio_frames, num_blocks):
        self.bank.metadata.update(schedule=schedule, audio_frames=audio_frames, num_blocks=num_blocks,
                                  hook=HOOK, audio_layout=AUDIO_LAYOUT)

    def __call__(self, step, block, hidden):
        self.bank.capture(step=step, block=block, hidden=hidden)
        return None

    def summary(self):
        return dict(mode=self.mode, role=self.bank.role, group=self.bank.group,
                    quantile=self.bank.quantile, records=len(self.bank.records))


class ReplayCarriers:
    """Stem Formation: activation inside windows, quiet suppression outside."""
    mode = 'replay'

    def __init__(self, scene, activation, suppression, identity, blend):
        self.activation = activation
        self.suppression = suppression
        self.identity = identity
        self.duration = scene['num_frames'] / 24
        self.blend = blend
        self.windows = stem_windows(scene)
        if len(activation) != len(self.windows):
            raise ValueError('Activation mapping must have one entry per stem')
        self._positions = None

    def validate(self, schedule, audio_frames, num_blocks):
        self.total_steps = len(schedule['video'])
        if len(schedule['audio']) != self.total_steps:
            raise ValueError('Video/audio schedules must have equal lengths')
        self.audio_frames = audio_frames
        _check_bank(self.suppression, 'suppression', self.identity, schedule, audio_frames, num_blocks)
        for windows, bank in zip(self.windows, self.activation):
            if windows:
                if bank is None:
                    raise ValueError('Missing activation carrier for a controlled stem')
                _check_bank(bank, 'activation', self.identity, schedule, audio_frames, num_blocks)

    def __call__(self, step, block, hidden):
        if hidden.shape[:2] != (len(self.windows), self.audio_frames * 2):
            raise ValueError('Replay expects N audio rows and channel-major stereo tokens')
        if self._positions is None or self._positions.device != hidden.device:
            self._positions = stereo_positions(self.audio_frames, self.duration, hidden.device)
        # An uncontrolled stem's placeholder record is never used.
        records = [(bank or self.suppression).record(step, block) for bank in self.activation]
        return replay_carriers(hidden, activation=records, suppression=self.suppression.record(step, block),
                               windows=self.windows, positions=self._positions, duration_seconds=self.duration,
                               blend=self.blend,
                               activation_strength=self.blend.activation_strength(step, self.total_steps))

    def summary(self):
        return dict(mode=self.mode, blend=asdict(self.blend),
                    activation_schedule=[{'end_progress': 0.1, 'strength': 0.5},
                                         {'end_progress': 1.0, 'strength': self.blend.strength}],
                    activation_groups=[bank.group if bank else None for bank in self.activation],
                    suppression_group=self.suppression.group, audio_layout=AUDIO_LAYOUT)


class QuietReplay:
    """Scene Integration: outside-window quiet suppression only, no activation."""
    mode = 'quiet_suppression_only'

    def __init__(self, scene, bank, identity, blend):
        self.bank = bank
        self.identity = identity
        self.duration = scene['num_frames'] / 24
        self.windows = stem_windows(scene)
        self.blend = blend
        self.positions = None

    def validate(self, schedule, audio_frames, num_blocks):
        if len(schedule['video']) != len(schedule['audio']):
            raise ValueError('Video/audio schedules must have equal lengths')
        _check_bank(self.bank, 'suppression', self.identity, schedule, audio_frames, num_blocks)
        self.audio_frames = audio_frames

    def __call__(self, step, block, hidden):
        if hidden.shape[:2] != (len(self.windows), 2 * self.audio_frames):
            raise ValueError('Quiet replay requires one native stereo row per stem')
        if self.positions is None or self.positions.device != hidden.device:
            self.positions = stereo_positions(self.audio_frames, self.duration, hidden.device)
        return replay_suppression(hidden, suppression=self.bank.record(step, block), windows=self.windows,
                                  positions=self.positions, duration_seconds=self.duration, blend=self.blend)

    def summary(self):
        return dict(mode=self.mode, suppression_group=self.bank.group,
                    outside_suppression=self.blend.outside_suppression,
                    outside_silence_blend=self.blend.outside_silence_blend,
                    feather_seconds=self.blend.feather_seconds, activation_injection=False)
