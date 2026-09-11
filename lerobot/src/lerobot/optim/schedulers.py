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
import abc
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import draccus
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.constants import SCHEDULER_STATE
from lerobot.datasets.utils import write_json
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0"""

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        del num_training_steps

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, self.num_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.num_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < self.num_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def _make_state_json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.dim() == 0 else value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_make_state_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_make_state_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_state_json_safe(v) for k, v in value.items()}
    return value


def _restore_state_tensors(value: Any, template: Any) -> Any:
    if isinstance(template, torch.Tensor):
        return (
            torch.tensor(value, dtype=template.dtype, device=template.device)
            if not isinstance(value, list)
            else torch.tensor(value, dtype=template.dtype, device=template.device)
        )
    if isinstance(template, dict):
        return {k: _restore_state_tensors(value[k], template[k]) for k in template}
    if isinstance(template, list):
        return [_restore_state_tensors(v, t) for v, t in zip(value, template, strict=False)]
    if isinstance(template, tuple):
        return tuple(_restore_state_tensors(v, t) for v, t in zip(value, template, strict=False))
    return value


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(_make_state_json_safe(state_dict), save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    current_state_dict = scheduler.state_dict()
    state_dict = deserialize_json_into_object(
        save_dir / SCHEDULER_STATE,
        _make_state_json_safe(current_state_dict),
    )
    state_dict = _restore_state_tensors(state_dict, current_state_dict)
    scheduler.load_state_dict(state_dict)
    return scheduler
