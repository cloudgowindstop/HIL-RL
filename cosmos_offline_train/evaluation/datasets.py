"""Evaluation-only dataset views."""

from __future__ import annotations

from torch.utils.data import Dataset

from ..dataset import CosmosParquetDataset
from ..masks import FUTURE_INDICES


class InverseFutureShuffleDataset(Dataset):
    """Attach future latents from a deterministic different episode.

    Current observation and action target stay unchanged. Pairing by a half-list
    episode offset avoids adjacent frames from the same trajectory.
    """

    def __init__(self, dataset: CosmosParquetDataset):
        if len(dataset.records) < 2:
            raise ValueError("future shuffle requires at least two episodes")
        self.dataset = dataset
        self.records = dataset.records
        self._episode_offset = max(1, len(self.records) // 2)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        episode_index, row_index = self.dataset._locate(index)
        partner_episode = (episode_index + self._episode_offset) % len(self.records)
        if partner_episode == episode_index:
            partner_episode = (partner_episode + 1) % len(self.records)
        partner_record = self.records[partner_episode]
        partner_index = int(self.dataset._offsets[partner_episode]) + row_index % partner_record.rows
        partner = self.dataset[partner_index]
        future = partner["video"][:, list(FUTURE_INDICES)].clone()
        sample["shuffled_future_video"] = future
        sample["shuffled_future_sample_id"] = partner["sample_id"]
        return sample
