"""Cached carrier storage: one normalized feature vector per denoising step and block."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

CARRIER_FORMAT = 'soundwich_h3_carrier_v1'
CarrierRole = Literal['activation', 'suppression']


def quantile_mean_token(hidden: torch.Tensor, quantile: float, *, largest: bool) -> torch.Tensor:
    """Select a top/bottom RMS-energy quantile of tokens and average it to one token per batch row."""
    if hidden.ndim != 3:
        raise ValueError(f'hidden must have shape [B,T,D], got {tuple(hidden.shape)}')
    token_count = hidden.shape[1]
    keep_fraction = max(1.0 - quantile, 1.0 / max(token_count, 1))
    keep = max(1, min(token_count, math.ceil(token_count * keep_fraction)))
    energy = hidden.detach().float().square().mean(dim=-1).sqrt()
    indices = energy.topk(keep, dim=1, largest=largest).indices
    selected = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    return selected.mean(dim=1)


def normalize_token(token: torch.Tensor) -> tuple[torch.Tensor, float]:
    value = token.detach().float().cpu().reshape(-1)
    rms = float(value.square().mean().sqrt().item())
    if rms > 1e-8:
        value = value / rms
    return value.to(torch.float16).contiguous(), rms


@dataclass(frozen=True)
class CarrierRecord:
    token: torch.Tensor
    rms: float

    def restore(self, *, device: torch.device, dtype: torch.dtype, own_rms: bool) -> torch.Tensor:
        value = self.token.to(device=device, dtype=dtype)
        return value * self.rms if own_rms else value


@dataclass
class CarrierBank:
    role: CarrierRole
    group: str
    quantile: float
    records: dict[tuple[int, int], CarrierRecord] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def capture(self, *, step: int, block: int, hidden: torch.Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError('Capture one independent reference row at a time')
        token = quantile_mean_token(hidden, self.quantile, largest=self.role == 'activation')
        normalized, rms = normalize_token(token)
        self.records[(step, block)] = CarrierRecord(token=normalized, rms=rms)

    def record(self, step: int, block: int) -> CarrierRecord:
        try:
            return self.records[(step, block)]
        except KeyError as exc:
            raise KeyError(f'carrier {self.group!r} has no step={step}, block={block} token') from exc

    def save(self, directory: str | Path) -> Path:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        specs: dict[str, dict[str, object]] = {}
        for (step, block), record in sorted(self.records.items()):
            key = f'step_{step:04d}_block_{block:02d}'
            tensors[key] = record.token
            specs[key] = {'step': step, 'block': block, 'rms': record.rms}
        save_file(tensors, str(root / 'carrier.safetensors'), metadata={'format': CARRIER_FORMAT})
        manifest = {
            'format': CARRIER_FORMAT,
            'role': self.role,
            'group': self.group,
            'quantile': self.quantile,
            'metadata': self.metadata,
            'records': specs,
        }
        (root / 'carrier.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return root

    @classmethod
    def load(cls, path: str | Path) -> 'CarrierBank':
        root = Path(path).expanduser()
        manifest_path = root / 'carrier.json' if root.is_dir() else root
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('format') != CARRIER_FORMAT:
            raise ValueError(f'unsupported carrier format: {manifest.get("format")}')
        tensors = load_file(str(manifest_path.with_name('carrier.safetensors')), device='cpu')
        records = {
            (int(spec['step']), int(spec['block'])): CarrierRecord(token=tensors[key], rms=float(spec['rms']))
            for key, spec in manifest['records'].items()
        }
        return cls(role=manifest['role'], group=str(manifest['group']), quantile=float(manifest['quantile']),
                   records=records, metadata=dict(manifest.get('metadata', {})))
