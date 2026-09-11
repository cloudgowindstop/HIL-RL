"""DataLoader and DistributedSampler construction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from .dataset import CosmosParquetDataset


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class LoaderBundle:
    loader: DataLoader
    sampler: object

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class EpisodeBalancedSampler(Sampler[int]):
    """Select balanced rows, then group them by episode to avoid Parquet cache thrashing."""

    def __init__(
        self,
        dataset,
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        max_samples: int | None = None,
    ):
        base = getattr(dataset, "dataset", dataset)
        if not hasattr(base, "records") or not hasattr(base, "_offsets"):
            raise TypeError("episode-balanced sampling requires CosmosParquetDataset metadata")
        self.rows = [int(record.rows) for record in base.records]
        self.offsets = [int(value) for value in base._offsets[:-1]]
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not self.rows or any(rows < 1 for rows in self.rows):
            raise ValueError("episode-balanced sampling requires non-empty episodes")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank/world_size")

        rng = np.random.default_rng(seed)
        self.episode_order = rng.permutation(len(self.rows)).tolist()
        self.starts: list[int] = []
        self.strides: list[int] = []
        for rows in self.rows:
            self.starts.append(int(rng.integers(rows)))
            stride = int(rng.integers(1, rows + 1))
            while math.gcd(stride, rows) != 1:
                stride = stride % rows + 1
            self.strides.append(stride)

        target = sum(self.rows) if max_samples is None else min(int(max_samples), sum(self.rows))
        if target < 1:
            raise ValueError("episode-balanced max_samples must be positive")
        self.samples_per_episode = [0 for _ in self.rows]
        allocated = 0
        while allocated < target:
            for episode_index in self.episode_order:
                if allocated >= target:
                    break
                if self.samples_per_episode[episode_index] < self.rows[episode_index]:
                    self.samples_per_episode[episode_index] += 1
                    allocated += 1
        self.selected_total = allocated

    def __iter__(self):
        global_position = 0
        for episode_index in self.episode_order:
            rows = self.rows[episode_index]
            for round_index in range(self.samples_per_episode[episode_index]):
                row = (
                    self.starts[episode_index] + round_index * self.strides[episode_index]
                ) % rows
                if global_position % self.world_size == self.rank:
                    yield self.offsets[episode_index] + row
                global_position += 1

    def __len__(self) -> int:
        return max(
            0,
            (self.selected_total - self.rank + self.world_size - 1) // self.world_size,
        )


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices without padding or duplication."""

    def __init__(self, dataset, *, rank: int, world_size: int):
        self.size = len(dataset)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank/world_size")

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.size - self.rank + self.world_size - 1) // self.world_size)


def build_loader(
    dataset: CosmosParquetDataset,
    *,
    batch_size: int,
    workers: int,
    train: bool,
    seed: int,
    distributed: bool,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    episode_balanced: bool = False,
    episode_balanced_max_samples: int | None = None,
    distributed_eval_no_padding: bool = False,
) -> LoaderBundle:
    if episode_balanced:
        if train:
            raise ValueError("episode_balanced sampler is validation-only")
        sampler = EpisodeBalancedSampler(
            dataset,
            seed=seed,
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
            max_samples=episode_balanced_max_samples,
        )
    elif distributed and distributed_eval_no_padding:
        if train:
            raise ValueError("distributed_eval_no_padding is validation-only")
        sampler = DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
    elif distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=train, seed=seed, drop_last=train
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator) if train else SequentialSampler(dataset)
    worker_options = {}
    if workers > 0:
        worker_options["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and workers > 0,
        worker_init_fn=seed_worker,
        drop_last=train,
        **worker_options,
    )
    return LoaderBundle(loader=loader, sampler=sampler)
