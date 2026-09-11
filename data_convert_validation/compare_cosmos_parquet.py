#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_FIELDS = (
    "action",
    "proprio",
    "future_proprio",
    "value_function_return",
    "next.reward",
    "next.done",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def dataset_root(parquet_path: Path) -> Path:
    """Resolve .../data/chunk-NNN/episode.parquet to its dataset root."""
    if parquet_path.parent.parent.name != "data":
        raise ValueError(f"unexpected LeRobot parquet layout: {parquet_path}")
    return parquet_path.parents[2]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def as_numpy(table, field: str) -> np.ndarray:
    return np.asarray(table[field].to_pylist())


def compare_regular_fields(candidate: Path, reference: Path) -> dict[str, dict[str, Any]]:
    candidate_schema = pq.read_schema(candidate)
    reference_schema = pq.read_schema(reference)
    common = [
        field
        for field in DEFAULT_FIELDS
        if field in candidate_schema.names and field in reference_schema.names
    ]
    left = pq.read_table(candidate, columns=common)
    right = pq.read_table(reference, columns=common)
    result: dict[str, dict[str, Any]] = {}
    for field in common:
        lhs = as_numpy(left, field)
        rhs = as_numpy(right, field)
        same_shape = lhs.shape == rhs.shape
        exact = same_shape and np.array_equal(lhs, rhs)
        item: dict[str, Any] = {
            "candidate_shape": list(lhs.shape),
            "reference_shape": list(rhs.shape),
            "exact": bool(exact),
        }
        if same_shape:
            if lhs.dtype == bool or rhs.dtype == bool:
                item["different_elements"] = int(np.count_nonzero(lhs != rhs))
            else:
                difference = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
                item["max_abs_difference"] = float(difference.max(initial=0.0))
                item["mean_abs_difference"] = float(difference.mean())
                item["different_elements"] = int(np.count_nonzero(difference))
        result[field] = item
    return result


def action_chunk(actions: np.ndarray, frame: int, chunk_size: int) -> np.ndarray:
    end = min(frame + chunk_size, len(actions))
    chunk = np.empty((chunk_size, actions.shape[1]), dtype=np.float32)
    available = end - frame
    chunk[:available] = actions[frame:end]
    chunk[available:] = actions[-1]
    return chunk


def expected_action_latent(
    actions: np.ndarray,
    frame: int,
    chunk_size: int,
    latent_elements: int,
) -> np.ndarray:
    flat = action_chunk(actions, frame, chunk_size).reshape(-1)
    repeats = (latent_elements + len(flat) - 1) // len(flat)
    return np.tile(flat, repeats)[:latent_elements]


def inspect_video_stream(
    candidate: Path,
    reference: Path,
    candidate_actions: np.ndarray,
    reference_actions: np.ndarray,
    candidate_metadata: dict[str, Any] | None,
    reference_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_batches = pq.ParquetFile(candidate).iter_batches(batch_size=1, columns=["video"])
    reference_batches = pq.ParquetFile(reference).iter_batches(batch_size=1, columns=["video"])

    temporal_size: int | None = None
    difference_sum: np.ndarray | None = None
    difference_count: np.ndarray | None = None
    difference_max: np.ndarray | None = None
    different_elements: np.ndarray | None = None
    injection = {
        "candidate": {"max_abs_difference": 0.0, "different_elements": 0},
        "reference": {"max_abs_difference": 0.0, "different_elements": 0},
    }
    candidate_action_meta = (candidate_metadata or {}).get("action", {})
    reference_action_meta = (reference_metadata or {}).get("action", {})
    candidate_action_index = int(candidate_action_meta.get("latent_index", 4))
    reference_action_index = int(reference_action_meta.get("latent_index", 4))
    candidate_chunk_size = int(candidate_action_meta.get("chunk_size", 16))
    reference_chunk_size = int(reference_action_meta.get("chunk_size", 16))

    rows = 0
    for candidate_batch, reference_batch in zip(candidate_batches, reference_batches):
        candidate_video = np.asarray(candidate_batch.column(0)[0].as_py(), dtype=np.float32)
        reference_video = np.asarray(reference_batch.column(0)[0].as_py(), dtype=np.float32)
        if candidate_video.shape != reference_video.shape:
            raise ValueError(
                f"video shape mismatch at row {rows}: "
                f"{candidate_video.shape} != {reference_video.shape}"
            )
        if temporal_size is None:
            temporal_size = candidate_video.shape[1]
            difference_sum = np.zeros(temporal_size, dtype=np.float64)
            difference_count = np.zeros(temporal_size, dtype=np.int64)
            difference_max = np.zeros(temporal_size, dtype=np.float64)
            different_elements = np.zeros(temporal_size, dtype=np.int64)

        difference = np.abs(candidate_video - reference_video)
        for latent_t in range(temporal_size):
            current = difference[:, latent_t]
            difference_sum[latent_t] += float(current.sum())
            difference_count[latent_t] += current.size
            difference_max[latent_t] = max(difference_max[latent_t], float(current.max()))
            different_elements[latent_t] += int(np.count_nonzero(current))

        for label, video, actions, action_index, chunk_size in (
            (
                "candidate",
                candidate_video,
                candidate_actions,
                candidate_action_index,
                candidate_chunk_size,
            ),
            (
                "reference",
                reference_video,
                reference_actions,
                reference_action_index,
                reference_chunk_size,
            ),
        ):
            target = video[:, action_index].reshape(-1)
            expected = expected_action_latent(actions, rows, chunk_size, target.size)
            current = np.abs(target - expected)
            injection[label]["max_abs_difference"] = max(
                injection[label]["max_abs_difference"], float(current.max())
            )
            injection[label]["different_elements"] += int(np.count_nonzero(current))
        rows += 1

    if rows != len(candidate_actions) or rows != len(reference_actions):
        raise ValueError(
            f"video/action row mismatch: video={rows}, "
            f"candidate_action={len(candidate_actions)}, reference_action={len(reference_actions)}"
        )
    assert difference_sum is not None
    assert difference_count is not None
    assert difference_max is not None
    assert different_elements is not None
    return {
        "rows": rows,
        "action_injection": injection,
        "latent_time_difference": [
            {
                "latent_t": index,
                "mean_abs_difference": float(difference_sum[index] / difference_count[index]),
                "max_abs_difference": float(difference_max[index]),
                "different_elements": int(different_elements[index]),
            }
            for index in range(len(difference_sum))
        ],
    }


def action_normalization_roundtrip(
    candidate_actions: np.ndarray,
    reference_actions: np.ndarray,
    candidate_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    if not candidate_stats:
        return {"available": False, "reason": "candidate statistics file is missing"}
    if "actions_min" not in candidate_stats or "actions_max" not in candidate_stats:
        return {"available": False, "reason": "candidate has no actions_min/actions_max"}
    minimum = np.asarray(candidate_stats["actions_min"], dtype=np.float32)
    maximum = np.asarray(candidate_stats["actions_max"], dtype=np.float32)
    if candidate_actions.shape != reference_actions.shape:
        return {"available": False, "reason": "action shapes differ"}
    restored = 0.5 * (candidate_actions + 1.0) * (maximum - minimum) + minimum
    difference = np.abs(restored - reference_actions)
    return {
        "available": True,
        "formula": "0.5 * (candidate + 1) * (actions_max - actions_min) + actions_min",
        "max_abs_difference": float(difference.max(initial=0.0)),
        "mean_abs_difference": float(difference.mean()),
        "allclose_atol_2e-6": bool(np.allclose(restored, reference_actions, atol=2e-6, rtol=0)),
    }


def compare(candidate: Path, reference: Path) -> dict[str, Any]:
    candidate = candidate.resolve()
    reference = reference.resolve()
    candidate_root = dataset_root(candidate)
    reference_root = dataset_root(reference)
    candidate_metadata = load_json(candidate_root / "cosmos_dataset_metadata.json")
    reference_metadata = load_json(reference_root / "cosmos_dataset_metadata.json")
    candidate_stats = load_json(candidate_root / "dataset_statistics.json")

    candidate_table = pq.read_table(candidate, columns=["action"])
    reference_table = pq.read_table(reference, columns=["action"])
    candidate_actions = np.asarray(candidate_table["action"].to_pylist(), dtype=np.float32)
    reference_actions = np.asarray(reference_table["action"].to_pylist(), dtype=np.float32)

    return {
        "candidate": str(candidate),
        "reference": str(reference),
        "candidate_dataset_root": str(candidate_root),
        "reference_dataset_root": str(reference_root),
        "schema_equal": pq.read_schema(candidate).equals(pq.read_schema(reference)),
        "candidate_metadata": candidate_metadata,
        "reference_metadata": reference_metadata,
        "field_comparison": compare_regular_fields(candidate, reference),
        "action_normalization_roundtrip": action_normalization_roundtrip(
            candidate_actions,
            reference_actions,
            candidate_stats,
        ),
        "video": inspect_video_stream(
            candidate,
            reference,
            candidate_actions,
            reference_actions,
            candidate_metadata,
            reference_metadata,
        ),
    }


def text_report(report: dict[str, Any]) -> str:
    lines = [
        "Cosmos Parquet comparison",
        "=========================",
        f"candidate: {report['candidate']}",
        f"reference: {report['reference']}",
        f"schema_equal: {report['schema_equal']}",
        "",
        "Regular fields",
        "--------------",
    ]
    for field, item in report["field_comparison"].items():
        lines.append(
            f"{field}: exact={item['exact']} "
            f"max_abs={item.get('max_abs_difference', 'n/a')} "
            f"different={item.get('different_elements', 'n/a')}"
        )
    lines.extend(["", "Action normalization round-trip", "------------------------------"])
    for key, value in report["action_normalization_roundtrip"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "Latent action injection", "-----------------------"])
    for label, item in report["video"]["action_injection"].items():
        passed = item["max_abs_difference"] == 0.0 and item["different_elements"] == 0
        lines.append(
            f"{label}: passed={passed} max_abs={item['max_abs_difference']} "
            f"different={item['different_elements']}"
        )
    lines.extend(["", "Latent time comparison", "----------------------"])
    for item in report["video"]["latent_time_difference"]:
        lines.append(
            f"t={item['latent_t']}: mean_abs={item['mean_abs_difference']:.12g} "
            f"max_abs={item['max_abs_difference']:.12g} "
            f"different={item['different_elements']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Cosmos LeRobot episode Parquets and verify latent injection"
    )
    parser.add_argument("--candidate", required=True, type=Path, help="new/candidate parquet")
    parser.add_argument("--reference", required=True, type=Path, help="reference parquet")
    parser.add_argument("--output", type=Path, help="optional text report path")
    parser.add_argument("--json-output", type=Path, help="optional full JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args.candidate, args.reference)
    rendered = text_report(report)
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

