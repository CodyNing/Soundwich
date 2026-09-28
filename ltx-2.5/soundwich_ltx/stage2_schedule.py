"""Densified distilled schedule for the Stage-2 refinement."""

from __future__ import annotations

import torch

from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS


def densified_stage2_sigmas(start_sigma: float | None = None) -> torch.Tensor:
    """Split the Stage-2 intervals into eight steps with an optional higher start."""
    anchors = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32).clone()
    if start_sigma is not None:
        official_start = float(anchors[0])
        if not official_start <= start_sigma <= 1.0:
            raise ValueError(f"Stage-2 start sigma must be in [{official_start}, 1.0]")
        anchors[0] = start_sigma
    return torch.cat(
        [
            torch.linspace(anchors[0], anchors[1], 3)[:-1],
            torch.linspace(anchors[1], anchors[2], 4)[:-1],
            torch.linspace(anchors[2], anchors[3], 4),
        ]
    )
