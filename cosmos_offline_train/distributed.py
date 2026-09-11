"""DDP initialization, device ownership, barriers, and metric reductions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize() -> DistributedContext:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not torch.cuda.is_available():
        raise RuntimeError("Cosmos 2B training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return DistributedContext(rank, local_rank, world_size, torch.device("cuda", local_rank))


def reduce_metrics(
    metrics: dict[str, torch.Tensor],
    context: DistributedContext,
    *,
    sum_names: frozenset[str] = frozenset(),
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, value in metrics.items():
        tensor = value.detach().float().to(context.device)
        if context.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if name not in sum_names:
                tensor /= context.world_size
        output[name] = tensor.item()
    return output


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
