#!/usr/bin/env python3
"""逐episode、逐列对比两个转换后的LeRobot数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def episode_files(dataset: Path) -> list[Path]:
    """按相对路径稳定排序全部episode Parquet。"""
    return sorted(dataset.glob("data/chunk-*/episode_*.parquet"))


def compare_datasets(
    baseline: Path,
    candidate: Path,
    *,
    rtol: float,
    atol: float,
) -> None:
    """浮点列按容差比较，任务索引等非浮点列要求完全一致。"""
    baseline_files = episode_files(baseline)
    candidate_files = episode_files(candidate)
    baseline_rel = [path.relative_to(baseline) for path in baseline_files]
    candidate_rel = [path.relative_to(candidate) for path in candidate_files]
    if baseline_rel != candidate_rel:
        raise AssertionError(
            f"episode file sets differ: baseline={baseline_rel}, candidate={candidate_rel}"
        )
    if not baseline_files:
        raise AssertionError("no episode parquet files found")

    for relative_path in baseline_rel:
        baseline_table = pq.read_table(baseline / relative_path)
        candidate_table = pq.read_table(candidate / relative_path)
        if baseline_table.column_names != candidate_table.column_names:
            raise AssertionError(f"column names differ in {relative_path}")
        if baseline_table.num_rows != candidate_table.num_rows:
            raise AssertionError(f"row count differs in {relative_path}")

        for column_name in baseline_table.column_names:
            expected_values = baseline_table[column_name].combine_chunks().to_pylist()
            actual_values = candidate_table[column_name].combine_chunks().to_pylist()
            try:
                expected = np.asarray(expected_values)
                actual = np.asarray(actual_values)
            except ValueError:
                expected = actual = None

            if (
                expected is not None
                and actual is not None
                and expected.shape == actual.shape
                and expected.dtype.kind in "fc"
                and actual.dtype.kind in "fc"
            ):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=f"floating column differs: {relative_path}:{column_name}",
                )
            elif actual_values != expected_values:
                raise AssertionError(
                    f"non-floating column differs: {relative_path}:{column_name}"
                )


def main() -> None:
    """解析两个数据集路径和浮点容差并执行完整对比。"""
    parser = argparse.ArgumentParser(
        description="Compare a single-GPU baseline with a multi-GPU conversion"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    compare_datasets(
        args.baseline.expanduser().resolve(),
        args.candidate.expanduser().resolve(),
        rtol=args.rtol,
        atol=args.atol,
    )
    print("Conversion outputs match within tolerance.")


if __name__ == "__main__":
    main()
