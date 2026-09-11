"""Optimizer construction helpers."""

from __future__ import annotations

from typing import Any

import torch


def build_optimizer_scheduler(model, official_config: Any):
    """Build the optimizer and scheduler through the official model API."""
    return model.init_optimizer_scheduler(
        official_config.optimizer, official_config.scheduler
    )


def build_grad_scaler() -> torch.amp.GradScaler:
    """BF16 does not use FP16 gradient scaling."""
    return torch.amp.GradScaler("cuda", enabled=False)

