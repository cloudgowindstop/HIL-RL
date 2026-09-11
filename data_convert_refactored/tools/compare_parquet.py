#!/usr/bin/env python3
"""按批次比较两个 Parquet 文件，适用于检查 Cosmos VAE latent。"""

from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _value_type(data_type: pa.DataType) -> pa.DataType:
    """取出 list/fixed_size_list 最内层元素类型。"""
    while (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        data_type = data_type.value_type
    return data_type


def _format_sample(value: object, *, full: bool = False) -> str:
    """紧凑显示大数组，避免把整块 VAE latent 打到终端。"""
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(value)
    threshold = array.size if full else 16
    content = np.array2string(
        array, threshold=threshold, edgeitems=2, max_line_width=120
    )
    return f"shape={array.shape} values={content}"


def _center_sample(value: object, size: int) -> tuple[np.ndarray, str]:
    """截取二维VAE特征图中心区域，仅用于终端展示。"""
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"head VAE sample must be 2D, got {array.shape}")
    if size <= 0 or size > min(array.shape):
        raise ValueError(f"display_center_size must be in [1,{min(array.shape)}]")
    row_start = (array.shape[0] - size) // 2
    column_start = (array.shape[1] - size) // 2
    row_end = row_start + size
    column_end = column_start + size
    return (
        array[row_start:row_end, column_start:column_end],
        f"center[{row_start}:{row_end},{column_start}:{column_end}]",
    )


def compare_parquet(
    baseline: Path,
    candidate: Path,
    *,
    columns: list[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    batch_size: int = 8,
    sample_rows: int = 1,
    seed: int = 0,
    head_channel_only: bool = False,
    display_center_size: int = 6,
) -> None:
    """逐 batch、逐列比较；发现首个差异时抛出 AssertionError。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sample_rows < 0:
        raise ValueError("sample_rows must not be negative")
    if display_center_size < 0:
        raise ValueError("display_center_size must not be negative")
    if not baseline.is_file() or not candidate.is_file():
        raise FileNotFoundError("baseline or candidate parquet does not exist")

    expected_file = pq.ParquetFile(baseline)
    actual_file = pq.ParquetFile(candidate)
    expected_schema = expected_file.schema_arrow
    actual_schema = actual_file.schema_arrow

    if expected_file.metadata.num_rows != actual_file.metadata.num_rows:
        raise AssertionError(
            f"row count differs: {expected_file.metadata.num_rows} != "
            f"{actual_file.metadata.num_rows}"
        )
    if expected_schema.names != actual_schema.names:
        raise AssertionError("column names or order differ")

    selected = columns or expected_schema.names
    missing = [name for name in selected if name not in expected_schema.names]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    for name in selected:
        if expected_schema.field(name).type != actual_schema.field(name).type:
            raise AssertionError(f"column type differs: {name}")

    expected_batches = expected_file.iter_batches(batch_size=batch_size, columns=selected)
    actual_batches = actual_file.iter_batches(batch_size=batch_size, columns=selected)
    row_offset = 0
    max_errors = {name: 0.0 for name in selected}
    row_count = expected_file.metadata.num_rows
    sample_indices = set(
        np.random.default_rng(seed)
        .choice(row_count, size=min(sample_rows, row_count), replace=False)
        .tolist()
    )
    samples: dict[int, dict[str, tuple[object, object]]] = {
        row: {} for row in sample_indices
    }
    first_difference: str | None = None

    for expected_batch, actual_batch in zip_longest(expected_batches, actual_batches):
        if expected_batch is None or actual_batch is None:
            raise AssertionError("batch count differs")
        if expected_batch.num_rows != actual_batch.num_rows:
            raise AssertionError(f"batch row count differs at row {row_offset}")

        for index, name in enumerate(selected):
            expected_array = expected_batch.column(index)
            actual_array = actual_batch.column(index)
            value_type = _value_type(expected_schema.field(name).type)

            if pa.types.is_floating(value_type):
                expected = np.asarray(expected_array.to_pylist())
                actual = np.asarray(actual_array.to_pylist())
                if name == "video" and head_channel_only:
                    # video 每行布局为 [VAE通道=16, latent时间=9, 高=28, 宽=28]。
                    # latent时间索引3保存当前主相机（双臂数据中为head相机），这里只检查VAE通道0。
                    expected = expected[:, 0, 3, :, :]
                    actual = actual[:, 0, 3, :, :]
                if expected.shape != actual.shape:
                    if first_difference is None:
                        first_difference = (
                            f"shape differs: column={name} row={row_offset} "
                            f"{expected.shape} != {actual.shape}"
                        )
                    continue
                close = np.isclose(
                    actual, expected, rtol=rtol, atol=atol, equal_nan=True
                )
                if not np.all(close) and first_difference is None:
                    location = tuple(np.argwhere(~close)[0])
                    global_row = row_offset + location[0]
                    value_index = location[1:]
                    first_difference = (
                        f"value differs: column={name} row={global_row} "
                        f"index={value_index} baseline={expected[location]!r} "
                        f"candidate={actual[location]!r}"
                    )
                finite = np.isfinite(expected) & np.isfinite(actual)
                if np.any(finite):
                    max_errors[name] = max(
                        max_errors[name], float(np.max(np.abs(actual[finite] - expected[finite])))
                    )
                sample_expected = expected
                sample_actual = actual
            elif not expected_array.equals(actual_array) and first_difference is None:
                first_difference = f"value differs: column={name} near row={row_offset}"
                sample_expected = expected_array.to_pylist()
                sample_actual = actual_array.to_pylist()
            else:
                sample_expected = expected_array.to_pylist()
                sample_actual = actual_array.to_pylist()

            for global_row in sample_indices:
                local_row = global_row - row_offset
                if 0 <= local_row < expected_batch.num_rows:
                    samples[global_row][name] = (
                        sample_expected[local_row],
                        sample_actual[local_row],
                    )

        row_offset += expected_batch.num_rows

    for row in sorted(samples):
        print(f"[SAMPLE] row={row}")
        for name in selected:
            expected, actual = samples[row][name]
            label = name
            display_full = False
            if name == "video" and head_channel_only and display_center_size:
                expected, center_label = _center_sample(expected, display_center_size)
                actual, _ = _center_sample(actual, display_center_size)
                label = f"{name}.{center_label}"
                display_full = True
            print(f"  {label}.baseline: {_format_sample(expected, full=display_full)}")
            print(f"  {label}.candidate: {_format_sample(actual, full=display_full)}")
    if first_difference is not None:
        sys.stdout.flush()
        raise AssertionError(first_difference)

    print(
        f"[MATCH] rows={row_offset} columns={len(selected)} "
        f"rtol={rtol} atol={atol}"
    )
    for name in selected:
        if pa.types.is_floating(_value_type(expected_schema.field(name).type)):
            print(f"  {name}: max_abs_error={max_errors[name]:.9g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Parquet files")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--head-channel-only",
        action="store_true",
        help="For video, compare only current head latent at video[0, 3, :, :]",
    )
    parser.add_argument(
        "--display-center-size",
        type=int,
        default=6,
        help="Display only this center size for head VAE samples; 0 displays full 28x28",
    )
    args = parser.parse_args()

    try:
        compare_parquet(
            args.baseline.expanduser().resolve(),
            args.candidate.expanduser().resolve(),
            columns=args.columns,
            rtol=args.rtol,
            atol=args.atol,
            batch_size=args.batch_size,
            sample_rows=args.sample_rows,
            seed=args.seed,
            head_channel_only=args.head_channel_only,
            display_center_size=args.display_center_size,
        )
    except AssertionError as error:
        print(f"[DIFFERENT] {error}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
