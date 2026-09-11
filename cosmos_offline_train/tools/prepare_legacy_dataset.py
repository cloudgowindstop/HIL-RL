"""Validate a pre-split legacy dataset and add Cosmos training sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..action_spec import get_action_spec
from ..config import resolve_pre_split_directories
from ..dataset import build_episode_index, save_manifest


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_statistics(path: Path, dimension: int) -> str:
    statistics = _json(path)
    for key in ("actions_min", "actions_max"):
        values = statistics.get(key)
        if not isinstance(values, list) or len(values) != dimension:
            raise ValueError(f"{path}: {key} must contain {dimension} values")
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError(f"{path}: {key} contains NaN/Inf")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_split(root: Path, dimension: int) -> tuple[dict, list[Path]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    info = _json(info_path)
    expected_shapes = {
        "video": [16, 9, 28, 28],
        "action": [dimension],
        "proprio": [16],
        "future_proprio": [16],
    }
    features = info.get("features", {})
    for name, expected in expected_shapes.items():
        actual = features.get(name, {}).get("shape")
        if actual != expected:
            raise ValueError(f"{info_path}: {name} shape={actual!r}, expected {expected}")
    paths = sorted((root / "data").rglob("episode_*.parquet"))
    if len(paths) != int(info.get("total_episodes", -1)):
        raise ValueError(
            f"{root}: parquet={len(paths)} != total_episodes={info.get('total_episodes')}"
        )
    for path in paths:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows < 1:
            raise ValueError(f"empty episode: {path}")
        required = {
            "video", "action", "proprio", "future_proprio",
            "value_function_return", "next.reward", "next.done",
        }
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        action = parquet.read_row_group(0, columns=["action"]).slice(0, 1).to_pylist()[0]["action"]
        if np.asarray(action).shape != (dimension,):
            raise ValueError(f"{path}: action shape={np.asarray(action).shape}")
    return info, paths


def _metadata(
    *,
    task: str,
    outcome: str,
    group: str,
    action_source: str,
    spec,
    chunk_size: int,
    statistics_path: Path,
    statistics_sha256: str,
    rotation_scale: float,
) -> dict:
    return {
        "format_version": 1,
        "task_description": task,
        "collection_group": group,
        "episode_labeling": {"outcome": outcome, "source": "legacy_adapter_cli"},
        "action": {
            "training_field": "action",
            "encoding": spec.encoding,
            "source": action_source,
            "dimension": spec.dimension,
            "order": spec.order(number_of_arms=2),
            "pose_semantics": "local_delta_inv_current_times_target",
            "rotation_representation": spec.rotation_representation,
            "rotation_scale": rotation_scale if spec.encoding == "legacy_euler" else None,
            "chunk_size": chunk_size,
            "action_includes_gripper": True,
            "injected_into_video_latent": True,
            "latent_index": 4,
        },
        "proprio": {"dimension": 16},
        "statistics": {
            "source_path": str(statistics_path),
            "sha256": statistics_sha256,
            "copied_into_dataset": False,
        },
        "adapter": {
            "source_format": "lerobot_pre_split",
            "parquet_modified": False,
        },
    }


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    statistics_path = Path(args.statistics_path).expanduser().resolve()
    spec = get_action_spec(args.action_encoding)
    if spec.encoding == "legacy_euler" and args.rotation_scale <= 0:
        raise ValueError("--rotation-scale must be positive for legacy_euler")
    statistics_sha256 = _validate_statistics(statistics_path, spec.dimension)
    data = {
        "train_directory": args.train_directory,
        "val_directory": args.val_directory,
        "test_directory": args.test_directory,
    }
    train, validation, test = resolve_pre_split_directories(data, root)
    split_roots = [("train", train), ("val", validation)]
    if test is not None:
        split_roots.append(("test", test))

    metadata = _metadata(
        task=args.task,
        outcome=args.outcome,
        group=args.group or root.name,
        action_source=args.action_source,
        spec=spec,
        chunk_size=args.chunk_size,
        statistics_path=statistics_path,
        statistics_sha256=statistics_sha256,
        rotation_scale=args.rotation_scale,
    )
    manifests = root / "manifests"
    prepared: list[tuple[str, Path, list[Path], Path, Path]] = []
    for name, split_root in split_roots:
        _, paths = _validate_split(split_root, spec.dimension)
        metadata_path = split_root / "cosmos_dataset_metadata.json"
        manifest_path = manifests / f"{name}.json"
        prepared.append((name, split_root, paths, metadata_path, manifest_path))

    if not args.overwrite:
        existing = [
            path
            for _, _, _, metadata_path, manifest_path in prepared
            for path in (metadata_path, manifest_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing sidecars: {existing}")

    for name, split_root, paths, metadata_path, manifest_path in prepared:
        if not args.dry_run:
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            records = build_episode_index(split_root)
            save_manifest(records, manifest_path, overwrite=args.overwrite)
        print(
            f"[PREPARE] split={name} episodes={len(paths)} "
            f"metadata={metadata_path} manifest={manifest_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--statistics-path", required=True)
    parser.add_argument(
        "--action-encoding",
        choices=("legacy_euler", "cosmos_rotation_6d"),
        required=True,
    )
    parser.add_argument("--action-source", default="puppet_next_frame")
    parser.add_argument("--outcome", default="success")
    parser.add_argument("--group")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--rotation-scale", type=float, default=0.06)
    parser.add_argument("--train-directory", default="train")
    parser.add_argument("--val-directory", default="eval")
    parser.add_argument("--test-directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
