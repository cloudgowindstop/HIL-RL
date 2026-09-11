#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GradScaler checkpoint utilities aligned with Imaginaire ``Checkpointer.save``."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.constants import GRAD_SCALER_STATE
from lerobot.datasets.utils import write_json
from lerobot.utils.io_utils import deserialize_json_into_object

GradScaler = torch.amp.GradScaler


@dataclass
class GradScalerConfig:
    """Build kwargs for ``torch.amp.GradScaler`` (matches ``trainer.grad_scaler_args`` in Cosmos)."""

    enabled: bool = False
    init_scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    extra_args: dict[str, Any] = field(default_factory=dict)

    def build(self, device: str = "cuda") -> GradScaler:
        kwargs = {
            "enabled": self.enabled,
            "init_scale": self.init_scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            **self.extra_args,
        }
        return build_grad_scaler(device=device, **kwargs)


def build_grad_scaler(device: str = "cuda", **grad_scaler_args: Any) -> GradScaler:
    """Create a GradScaler the same way as Imaginaire trainer / Checkpointer consumers."""
    return torch.amp.GradScaler(device, **grad_scaler_args)


def save_grad_scaler_state(grad_scaler: GradScaler, save_dir: Path) -> None:
    """Save GradScaler state via ``state_dict()`` (same as ``Checkpointer.save``)."""
    state_dict = grad_scaler.state_dict()
    write_json(state_dict, save_dir / GRAD_SCALER_STATE)


def load_grad_scaler_state(grad_scaler: GradScaler, save_dir: Path) -> GradScaler:
    """Restore GradScaler from checkpoint via ``load_state_dict()``."""
    state_path = save_dir / GRAD_SCALER_STATE
    if not state_path.is_file():
        return grad_scaler

    state_dict = deserialize_json_into_object(state_path, grad_scaler.state_dict())
    grad_scaler.load_state_dict(state_dict)
    return grad_scaler
