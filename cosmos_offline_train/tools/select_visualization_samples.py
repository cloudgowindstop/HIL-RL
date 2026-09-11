"""Freeze visualization frames from a converted Cosmos episode tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..dataset import EpisodeRecord
from .paired_index import index_converted_records, load_json, parse_collection_name_map


def selected_row_indices(rows: int, fractions: list[float]) -> list[tuple[float, int]]:
    if rows < 1:
        raise ValueError("episode must contain at least one row")
    chosen: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"row fraction must be in [0, 1], got {fraction}")
        index = int(fraction * (rows - 1))
        if index not in seen:
            seen.add(index)
            chosen.append((fraction, index))
    return chosen


def _paired_source_id(record: EpisodeRecord, fallback: str) -> str:
    if "/" in fallback:
        return fallback
    collection = fallback
    return f"{collection}/{record.source_id}" if collection else record.source_id


def _stored_restore_dtype(table) -> str:
    field = table.schema.field("clean_restore_latent")
    metadata = field.metadata or {}
    extension = metadata.get(b"ARROW:extension:metadata", b"").decode("utf-8")
    if "float16" in extension or "halffloat" in str(field.type):
        return "float16"
    raise ValueError(
        f"clean_restore_latent stored type is not float16: type={field.type} "
        f"metadata={extension or None}"
    )


def inspect_restore(path: str, row_index: int) -> dict[str, object]:
    table = pq.read_table(path, columns=["clean_restore_latent"])
    if "clean_restore_latent" not in table.column_names:
        raise ValueError(f"{path} has no clean_restore_latent")
    stored_dtype = _stored_restore_dtype(table)
    row = table.slice(row_index, 1).to_pylist()[0]
    payload = row.get("clean_restore_latent")
    if payload is None:
        raise ValueError(f"{path}:{row_index} clean_restore_latent is null")
    # Parquet stores float16; Arrow to_pylist() promotes to Python float/float64.
    # Training/eval recast the same way as dataset.py.
    array = np.asarray(payload, dtype=np.float16)
    if array.shape != (16, 4, 28, 28):
        raise ValueError(
            f"{path}:{row_index} clean_restore_latent shape={array.shape}, "
            "expected (16, 4, 28, 28)"
        )
    if not np.isfinite(array.astype(np.float32)).all():
        raise ValueError(f"{path}:{row_index} clean_restore_latent contains NaN/Inf")
    return {
        "shape": list(array.shape),
        "dtype": stored_dtype,
        "finite": True,
    }


def build_samples(
    *,
    dataset_root: Path,
    source_ids: list[str],
    fractions: list[float],
    encoding: str,
    require_clean_restore: bool,
    collection_name_map: dict[str, str] | None = None,
) -> dict[str, object]:
    by_id = index_converted_records(
        dataset_root,
        collection_name_map=collection_name_map,
        outcomes={"success"},
    )
    missing = [source_id for source_id in source_ids if source_id not in by_id]
    if missing:
        raise ValueError(f"visualization episodes missing from {dataset_root}: {missing}")

    samples: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    for source_id in source_ids:
        record = by_id[source_id]
        rows = int(record.rows)
        selected = selected_row_indices(rows, fractions)
        row_indices = [index for _, index in selected]
        episodes.append(
            {
                "source_id": source_id,
                "episode_id": record.source_id,
                "episode_path": record.path,
                "rows": rows,
                "row_indices": row_indices,
            }
        )
        for fraction, row_index in selected:
            sample = {
                "sample_id": f"{record.path}:{row_index}",
                "source_id": source_id,
                "episode_path": record.path,
                "row_index": row_index,
                "row_fraction": fraction,
                "rows": rows,
                "encoding": encoding,
                "task": record.task,
                "outcome": record.outcome,
            }
            if require_clean_restore:
                sample["clean_restore_latent"] = inspect_restore(record.path, row_index)
            samples.append(sample)
    return {
        "encoding": encoding,
        "dataset_root": str(dataset_root),
        "row_fractions": fractions,
        "require_clean_restore": require_clean_restore,
        "episode_count": len(episodes),
        "sample_count": len(samples),
        "episodes": episodes,
        "samples": samples,
    }


def compare_sample_lists(left: dict[str, object], right: dict[str, object]) -> list[str]:
    errors: list[str] = []
    left_samples = {item["source_id"]: item for item in left["episodes"]}
    right_samples = {item["source_id"]: item for item in right["episodes"]}
    if set(left_samples) != set(right_samples):
        errors.append(
            "visualization source_id sets differ: "
            f"only_left={sorted(set(left_samples) - set(right_samples))[:5]} "
            f"only_right={sorted(set(right_samples) - set(left_samples))[:5]}"
        )
        return errors
    for source_id, left_episode in left_samples.items():
        right_episode = right_samples[source_id]
        if left_episode["rows"] != right_episode["rows"]:
            errors.append(
                f"{source_id}: rows {left_episode['rows']} != {right_episode['rows']}"
            )
        if left_episode["row_indices"] != right_episode["row_indices"]:
            errors.append(
                f"{source_id}: row_indices {left_episode['row_indices']} != "
                f"{right_episode['row_indices']}"
            )
    return errors


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select frozen visualization rows from converted Cosmos episodes"
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-split", required=True, type=Path)
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.4, 0.7])
    parser.add_argument("--collection-name-map", action="append", default=[])
    parser.add_argument("--require-clean-restore", action="store_true")
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args()

    source_ids = [
        str(record["source_id"]) for record in load_json(args.source_split.expanduser())
    ]
    payload = build_samples(
        dataset_root=args.dataset_root.expanduser().resolve(),
        source_ids=source_ids,
        fractions=list(args.fractions),
        encoding=str(args.encoding),
        require_clean_restore=bool(args.require_clean_restore),
        collection_name_map=parse_collection_name_map(args.collection_name_map),
    )
    output = args.output.expanduser().resolve()
    _write(output, payload)
    print(
        f"[VIZ-SAMPLES] encoding={args.encoding} episodes={payload['episode_count']} "
        f"samples={payload['sample_count']} output={output}"
    )
    if args.compare_with:
        other = load_json(args.compare_with.expanduser())
        errors = compare_sample_lists(payload, other)
        if errors:
            raise SystemExit("visualization sample mismatch:\n" + "\n".join(errors))
        print(f"[VIZ-SAMPLES] matched {args.compare_with}")


if __name__ == "__main__":
    main()
