"""Lazy indexed dataset for pre-encoded Cosmos episode Parquet files."""

from __future__ import annotations

import json
import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeRecord:
    path: str
    dataset_root: str
    group: str
    task: str
    outcome: str
    rows: int
    source_path: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class DatasetSplits:
    train: list[EpisodeRecord]
    validation: list[EpisodeRecord]
    test: list[EpisodeRecord]


def discover_dataset_roots(root: Path) -> list[Path]:
    if (root / "cosmos_dataset_metadata.json").is_file():
        return [root]
    return sorted(path.parent for path in root.rglob("cosmos_dataset_metadata.json"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_episode_index(root: str | Path) -> list[EpisodeRecord]:
    root = Path(root).expanduser().resolve()
    records: list[EpisodeRecord] = []
    for dataset_root in discover_dataset_roots(root):
        metadata = _load_json(dataset_root / "cosmos_dataset_metadata.json")
        task = str(metadata.get("task_description", dataset_root.name))
        outcome = str(metadata.get("episode_labeling", {}).get("outcome", "unknown"))
        group = metadata.get("collection_group")
        if not group:
            match = re.search(r"^(.+?_\d{8}_(?:am|pm))", dataset_root.name)
            group = match.group(1) if match else dataset_root.name
        source_paths: list[str] = []
        conversion_manifest = dataset_root / "meta" / "conversion_manifest.json"
        if conversion_manifest.is_file():
            payload = _load_json(conversion_manifest)
            source_paths = [str(item.get("path", "")) for item in payload.get("sources", [])]
        for path in sorted((dataset_root / "data").rglob("episode_*.parquet")):
            match = re.search(r"episode_(\d+)\.parquet$", path.name)
            episode_index = int(match.group(1)) if match else -1
            source_path = source_paths[episode_index] if 0 <= episode_index < len(source_paths) else ""
            source_id = ""
            if source_path:
                source = Path(source_path)
                source_id = source.parent.parent.name if source.parent.name == "data" else source.parent.name
            records.append(
                EpisodeRecord(
                    path=str(path.resolve()),
                    dataset_root=str(dataset_root.resolve()),
                    group=group,
                    task=task,
                    outcome=outcome,
                    rows=pq.ParquetFile(path).metadata.num_rows,
                    source_path=source_path,
                    source_id=source_id,
                )
            )
    return records


def build_pre_split_index(
    train_root: str | Path,
    validation_root: str | Path,
    test_root: str | Path | None = None,
) -> DatasetSplits:
    """Read existing directory splits without reshuffling episodes."""
    train = build_episode_index(train_root)
    validation = build_episode_index(validation_root)
    test = build_episode_index(test_root) if test_root is not None else []
    return DatasetSplits(train, validation, test)


def save_manifest(
    records: Iterable[EpisodeRecord], path: str | Path, *, overwrite: bool = False
) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"manifest 已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.__dict__ for record in records]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def load_manifest(path: str | Path) -> list[EpisodeRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records")
    if not isinstance(payload, list):
        raise ValueError(f"manifest 必须包含 episode record 列表: {path}")
    return [EpisodeRecord(**item) for item in payload]


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_episode_subset(
    records: list[EpisodeRecord], max_episodes: int, seed: int
) -> list[EpisodeRecord]:
    """Select a deterministic task-stratified subset of complete episodes."""
    if max_episodes <= 0 or max_episodes >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)
    for values in by_task.values():
        rng.shuffle(values)

    selected: list[EpisodeRecord] = []
    tasks = sorted(by_task)
    while len(selected) < max_episodes:
        progress = False
        for task in tasks:
            if by_task[task] and len(selected) < max_episodes:
                selected.append(by_task[task].pop())
                progress = True
        if not progress:
            break
    return selected


def split_episode_index(
    records: list[EpisodeRecord], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> tuple[list[EpisodeRecord], list[EpisodeRecord], list[EpisodeRecord]]:
    """Stratify by task, then split complete episodes within each task."""
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")

    rng = np.random.default_rng(seed)
    by_task: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)

    splits: list[list[EpisodeRecord]] = [[], [], []]
    for task in sorted(by_task):
        task_records = list(by_task[task])
        rng.shuffle(task_records)
        total = len(task_records)

        # Tiny tasks cannot populate all three splits. Preserve training first,
        # validation second; for larger tasks give every enabled split one item.
        counts = np.zeros(3, dtype=np.int64)
        enabled_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]
        for split_index in enabled_splits[:total]:
            counts[split_index] = 1
        targets = np.asarray(ratios) * total
        while int(counts.sum()) < total:
            split_index = int(np.argmax(targets - counts))
            counts[split_index] += 1

        train_end = int(counts[0])
        val_end = train_end + int(counts[1])
        splits[0].extend(task_records[:train_end])
        splits[1].extend(task_records[train_end:val_end])
        splits[2].extend(task_records[val_end:])

    return tuple(splits)  # type: ignore[return-value]


_ROW_COLUMNS = (
    "video",
    "action",
    "proprio",
    "future_proprio",
    "value_function_return",
    "next.reward",
    "next.done",
    "clean_restore_latent",
)


class CosmosParquetDataset(Dataset):
    """Map-style samples; cache recent row groups instead of whole episode tables."""

    def __init__(
        self,
        records: list[EpisodeRecord],
        cache_episodes: int = 1,
        action_dimension: int = 20,
    ):
        self.records = records
        self.cache_episodes = max(1, int(cache_episodes))
        self.action_dimension = int(action_dimension)
        if self.action_dimension < 1:
            raise ValueError("action_dimension must be positive")
        self._offsets = np.cumsum([0] + [record.rows for record in records], dtype=np.int64)
        self._files: dict[str, pq.ParquetFile] = {}
        self._row_group_starts: dict[str, np.ndarray] = {}
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return episode, int(index - self._offsets[episode])

    def _parquet(self, path: str) -> pq.ParquetFile:
        handle = self._files.get(path)
        if handle is None:
            handle = pq.ParquetFile(path)
            starts = []
            offset = 0
            for group_index in range(handle.num_row_groups):
                starts.append(offset)
                offset += handle.metadata.row_group(group_index).num_rows
            self._files[path] = handle
            self._row_group_starts[path] = np.asarray(starts, dtype=np.int64)
        return handle

    def _row_group(self, path: str, row_index: int):
        handle = self._parquet(path)
        starts = self._row_group_starts[path]
        group_index = int(np.searchsorted(starts, row_index, side="right") - 1)
        if group_index < 0 or group_index >= len(starts):
            raise IndexError(f"{path}: row {row_index} is outside parquet row groups")
        local_index = int(row_index - starts[group_index])
        key = (path, group_index)
        table = self._cache.pop(key, None)
        if table is None:
            names = set(handle.schema_arrow.names)
            columns = [name for name in _ROW_COLUMNS if name in names]
            table = handle.read_row_group(group_index, columns=columns)
        self._cache[key] = table
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return table, local_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, row_index = self._locate(index)
        record = self.records[episode_index]
        table, local_index = self._row_group(record.path, row_index)
        row = table.slice(local_index, 1).to_pylist()[0]
        video = torch.as_tensor(np.asarray(row["video"], dtype=np.float32))
        action = torch.as_tensor(np.asarray(row["action"], dtype=np.float32))
        proprio = torch.as_tensor(np.asarray(row["proprio"], dtype=np.float32))
        future_proprio = torch.as_tensor(np.asarray(row["future_proprio"], dtype=np.float32))
        value = torch.tensor(float(row["value_function_return"]), dtype=torch.float32)
        sample = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "future_proprio": future_proprio,
            "value": value,
            "reward": torch.tensor(float(row["next.reward"]), dtype=torch.float32),
            "done": torch.tensor(bool(row["next.done"])),
            "sample_id": f"{record.path}:{row_index}",
            "task": record.task,
            "outcome": record.outcome,
            "episode_path": record.path,
            "row_index": row_index,
        }
        if "clean_restore_latent" in row and row["clean_restore_latent"] is not None:
            clean_restore = torch.as_tensor(
                np.asarray(row["clean_restore_latent"], dtype=np.float16)
            )
            if clean_restore.shape != (16, 4, 28, 28):
                raise ValueError(
                    f"{record.path}:{row_index} clean_restore_latent shape="
                    f"{tuple(clean_restore.shape)}, expected (16,4,28,28)"
                )
            if not torch.isfinite(clean_restore).all():
                raise ValueError(
                    f"{record.path}:{row_index} clean_restore_latent contains NaN/Inf"
                )
            sample["clean_restore_latent"] = clean_restore
        if video.shape != (16, 9, 28, 28):
            raise ValueError(f"{record.path}:{row_index} video shape={tuple(video.shape)}, expected (16,9,28,28)")
        expected_action_shape = (self.action_dimension,)
        if action.shape != expected_action_shape:
            raise ValueError(
                f"{record.path}:{row_index} action shape={tuple(action.shape)}, "
                f"expected {expected_action_shape}"
            )
        for name in ("video", "action", "proprio", "future_proprio", "value"):
            if not torch.isfinite(sample[name]).all():
                raise ValueError(f"{record.path}:{row_index} {name} contains NaN/Inf")
        return sample
