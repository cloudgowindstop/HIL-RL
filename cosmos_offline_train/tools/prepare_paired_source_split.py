#!/usr/bin/env python3
"""Build one representation-independent train/validation split from raw HDF5 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use COLLECTION=ROOT")
    collection, root = value.split("=", 1)
    path = Path(root).expanduser().resolve()
    if not collection or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid source: {value}")
    return collection, path


def _discover(
    sources: list[tuple[str, Path]],
    *,
    task: str,
    outcome: str,
    excluded_ids: set[str],
) -> list[dict]:
    records = []
    seen_ids: set[str] = set()
    for collection, root in sources:
        paths = sorted(path.resolve() for path in root.rglob("trajectory.hdf5") if path.is_file())
        for path in paths:
            episode_id = path.parent.parent.name if path.parent.name == "data" else path.parent.name
            source_id = f"{collection}/{episode_id}"
            if episode_id in excluded_ids or source_id in excluded_ids:
                continue
            if source_id in seen_ids:
                raise ValueError(f"duplicate source_id: {source_id}")
            seen_ids.add(source_id)
            stat = path.stat()
            records.append(
                {
                    "collection": collection,
                    "episode_id": episode_id,
                    "outcome": outcome,
                    "path": str(path),
                    "size": stat.st_size,
                    "source_id": source_id,
                    "task": task,
                }
            )
    if len(records) < 2:
        raise ValueError(f"at least two usable episodes are required, found {len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a shared raw-episode split for Euler/rotation-6D conversion"
    )
    parser.add_argument("--source", action="append", required=True, type=_source)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--outcome", default="success", choices=("success", "failure"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--exclude-source-id", action="append", default=[])
    parser.add_argument("--visualization-episodes", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be in (0, 1)")

    output = args.output_dir.expanduser().resolve()
    records = _discover(
        args.source,
        task=args.task,
        outcome=args.outcome,
        excluded_ids=set(args.exclude_source_id),
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records)).tolist()
    train_count = int(round(len(records) * args.train_ratio))
    train_count = min(max(train_count, 1), len(records) - 1)
    train = [records[index] for index in order[:train_count]]
    validation = [records[index] for index in order[train_count:]]
    visual_count = min(max(args.visualization_episodes, 0), len(validation))
    visualization = validation[:visual_count]

    outputs = {
        "all.json": records,
        "train.json": train,
        "val.json": validation,
        "visualization_val.json": visualization,
    }
    for collection, _ in args.source:
        collection_records = [
            record for record in records if record["collection"] == collection
        ]
        outputs[f"all_{collection}.json"] = collection_records
        outputs[f"smoke_{collection}.json"] = collection_records[:1]
        outputs[f"visualization_val_{collection}.json"] = [
            record for record in visualization if record["collection"] == collection
        ]
    for name, payload in outputs.items():
        _atomic_json(output / name, payload)

    summary = {
        "collections": {
            collection: sum(record["collection"] == collection for record in records)
            for collection, _ in args.source
        },
        "excluded_source_ids": sorted(set(args.exclude_source_id)),
        "seed": args.seed,
        "split_unit": "complete_raw_episode",
        "split_policy": "task-level random split; collection date is not a grouping constraint",
        "task": args.task,
        "total_episodes": len(records),
        "train_episodes": len(train),
        "train_ratio": args.train_ratio,
        "val_episodes": len(validation),
        "visualization_episodes": len(visualization),
    }
    _atomic_json(output / "summary.json", summary)
    summary["manifest_sha256"] = {
        name: _sha256(output / name) for name in outputs
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
