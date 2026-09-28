"""Small reusable mean-carrier artifacts for Ovi multi-stem inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .blending import select_mean_token


class CarrierRecorder:
    """Collect one quantile-mean audio token per denoising step and block."""

    def __init__(
        self,
        *,
        role: str,
        quantile: float,
        step_count: int,
        block_count: int,
    ) -> None:
        if role not in {"activation", "suppression"}:
            raise ValueError("carrier role must be activation or suppression")
        self.role = role
        self.quantile = float(quantile)
        self._current_step = -1
        self._values: list[list[torch.Tensor | None]] = [
            [None] * int(block_count) for _ in range(int(step_count))
        ]

    def begin_step(self, step_index: int) -> None:
        self._current_step = int(step_index)

    def capture(self, block_index: int, hidden: torch.Tensor) -> None:
        mode = "top" if self.role == "activation" else "bottom"
        token = select_mean_token(
            hidden, quantile=self.quantile, mode=mode
        ).squeeze(0)
        self._values[self._current_step][int(block_index)] = token.detach().to(
            device="cpu", dtype=torch.bfloat16
        )

    def tokens(self) -> torch.Tensor:
        return torch.stack(
            [torch.stack(step, dim=0) for step in self._values], dim=0
        )


@dataclass(frozen=True)
class CarrierBank:
    carrier_id: str
    role: str
    tokens: torch.Tensor
    model_name: str
    solver_name: str
    shift: float
    timesteps: tuple[float, ...]
    quantile: float
    prompt: str
    visual_prompt: str
    a2v_enabled: bool
    v2a_enabled: bool

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "carrier_id": self.carrier_id,
                "role": self.role,
                "tokens": self.tokens.detach().to(device="cpu", dtype=torch.bfloat16),
                "model_name": self.model_name,
                "solver_name": self.solver_name,
                "shift": float(self.shift),
                "timesteps": list(self.timesteps),
                "quantile": float(self.quantile),
                "prompt": self.prompt,
                "visual_prompt": self.visual_prompt,
                "a2v_enabled": bool(self.a2v_enabled),
                "v2a_enabled": bool(self.v2a_enabled),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CarrierBank":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            carrier_id=str(data["carrier_id"]),
            role=str(data["role"]),
            tokens=data["tokens"],
            model_name=str(data["model_name"]),
            solver_name=str(data["solver_name"]),
            shift=float(data["shift"]),
            timesteps=tuple(float(value) for value in data["timesteps"]),
            quantile=float(data["quantile"]),
            prompt=str(data["prompt"]),
            visual_prompt=str(data["visual_prompt"]),
            a2v_enabled=bool(data["a2v_enabled"]),
            v2a_enabled=bool(data["v2a_enabled"]),
        )

    def validate_for_run(
        self,
        *,
        role: str,
        model_name: str,
        solver_name: str,
        shift: float,
        timesteps: tuple[float, ...],
        block_count: int,
        hidden_dim: int,
    ) -> None:
        expected_shape = (len(timesteps), int(block_count), int(hidden_dim))
        if self.role != role:
            raise ValueError(f"expected {role} carrier, got {self.role}")
        if self.model_name != model_name:
            raise ValueError(
                f"carrier model {self.model_name} does not match {model_name}"
            )
        if self.solver_name != solver_name or self.shift != float(shift):
            raise ValueError("carrier scheduler does not match target run")
        if tuple(self.tokens.shape) != expected_shape:
            raise ValueError(
                f"carrier tokens have shape {tuple(self.tokens.shape)}, "
                f"expected {expected_shape}"
            )
        expected_times = torch.tensor(timesteps, dtype=torch.float32)
        saved_times = torch.tensor(self.timesteps, dtype=torch.float32)
        if not torch.allclose(saved_times, expected_times, atol=1e-4, rtol=0.0):
            raise ValueError("carrier timesteps do not match target run")

    def token(
        self,
        step_index: int,
        block_index: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.tokens[step_index, block_index].reshape(1, -1).to(
            device=device, dtype=dtype
        )
