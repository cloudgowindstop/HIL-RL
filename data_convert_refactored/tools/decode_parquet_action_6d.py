#!/usr/bin/env python3
"""把单臂 rotation-6D Parquet action反解为真机单位，并可与参考NPY比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# 支持直接执行本文件，同时复用重构目录中的rotation-6D实现。
HIL_RL_ROOT = Path(__file__).resolve().parents[2]
if str(HIL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIL_RL_ROOT))

from data_convert_refactored.preparation.rotation import rotation_6d_to_matrix


def find_dataset_root(parquet_path: Path) -> Path:
    """向上查找包含Cosmos转换元数据的数据集根目录。"""
    for parent in parquet_path.parents:
        if (parent / "cosmos_dataset_metadata.json").is_file():
            return parent
    raise FileNotFoundError(
        f"cannot find cosmos_dataset_metadata.json above {parquet_path}"
    )


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_serialized_actions(parquet_path: Path) -> np.ndarray:
    """读取Parquet最终序列化的(T,10) action。"""
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(
        table.column("action").combine_chunks().to_pylist(), dtype=np.float64
    )
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected single-arm action shape (T,10), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Parquet action contains NaN or Inf")
    return actions


def undo_dataset_normalization(
    serialized_actions: np.ndarray,
    metadata: dict,
    stats: dict,
) -> np.ndarray:
    """撤销Cosmos dataset min/max归一化，恢复机器人scale处理后的10D action。"""
    action_metadata = metadata["action"]
    if not action_metadata.get("inference_unnormalize_actions", False):
        return serialized_actions.copy()

    minimum = np.asarray(stats.get("actions_min"), dtype=np.float64)
    maximum = np.asarray(stats.get("actions_max"), dtype=np.float64)
    if minimum.shape != (10,) or maximum.shape != (10,):
        raise ValueError(
            f"expected actions_min/actions_max shape (10,), got {minimum.shape}/{maximum.shape}"
        )
    span = maximum - minimum
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("action stats contain NaN or Inf")
    if np.any(span <= 0):
        raise ValueError("action stats contain constant or reversed channels")
    return (serialized_actions + 1.0) * 0.5 * span + minimum


def decode_metric_actions(
    actions_10d: np.ndarray,
    translation_scale: float,
    gripper_scale: float,
) -> tuple[np.ndarray, dict]:
    """把10D控制action反解为米、Euler XYZ弧度和绝对夹爪组成的7D action。"""
    if translation_scale <= 0 or gripper_scale <= 0:
        raise ValueError("translation_scale and gripper_scale must be positive")

    actions_7d = np.empty((len(actions_10d), 7), dtype=np.float64)
    determinants = []
    orthogonality_errors = []
    for index, action in enumerate(actions_10d):
        matrix = rotation_6d_to_matrix(action[3:9])
        actions_7d[index, :3] = action[:3] * translation_scale
        actions_7d[index, 3:6] = Rotation.from_matrix(matrix).as_euler("xyz")
        actions_7d[index, 6] = action[9] * gripper_scale
        determinants.append(np.linalg.det(matrix))
        orthogonality_errors.append(
            np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro")
        )

    validation = {
        "max_rotation_orthogonality_error": float(
            max(orthogonality_errors, default=0.0)
        ),
        "min_rotation_determinant": float(min(determinants, default=1.0)),
        "max_rotation_determinant": float(max(determinants, default=1.0)),
    }
    return actions_7d.astype(np.float32), validation


def compare_array(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    rtol: float,
    atol: float,
) -> bool:
    """比较一个数组并打印最大、平均误差及首个差异位置。"""
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        print(
            f"[DIFFERENT] {name}: shape {candidate.shape} != {reference.shape}",
            file=sys.stderr,
        )
        return False

    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    close = np.isclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=True)
    print(
        f"[{('MATCH' if np.all(close) else 'DIFFERENT')}] {name}: "
        f"shape={candidate.shape} max_abs_error={float(difference.max(initial=0.0)):.9g} "
        f"mean_abs_error={float(difference.mean()) if difference.size else 0.0:.9g}"
    )
    if not np.all(close):
        location = tuple(int(value) for value in np.argwhere(~close)[0])
        print(
            f"  first mismatch index={location} "
            f"reference={reference[location]!r} candidate={candidate[location]!r}",
            file=sys.stderr,
        )
        return False
    return True


def load_reference(path: Path) -> dict:
    """加载旧验证工具生成的标量dict NPY。"""
    value = np.load(path, allow_pickle=True)
    if value.shape != () or not isinstance(value.item(), dict):
        raise ValueError(f"reference NPY must contain one dictionary: {path}")
    return value.item()


def print_action_samples(
    payload: dict,
    reference: dict | None,
    sample_rows: int,
    seed: int,
) -> None:
    """在报告末尾随机显示若干帧的10D控制action和7D真机action。"""
    if sample_rows <= 0:
        return
    row_count = len(payload["metric_action_7d"])
    indices = np.random.default_rng(seed).choice(
        row_count, size=min(sample_rows, row_count), replace=False
    )
    for row in sorted(int(value) for value in indices):
        print(f"[SAMPLE] row={row}")
        for name in ("cosmos_action_10d", "metric_action_7d"):
            candidate = np.array2string(
                payload[name][row], precision=8, separator=", ", max_line_width=160
            )
            print(f"  {name}.candidate: {candidate}")
            if reference is not None:
                expected = np.array2string(
                    np.asarray(reference[name])[row],
                    precision=8,
                    separator=", ",
                    max_line_width=160,
                )
                print(f"  {name}.reference: {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode single-arm Cosmos rotation-6D Parquet actions"
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        parquet_path = args.parquet.expanduser().resolve()
        dataset_root = find_dataset_root(parquet_path)
        metadata_path = dataset_root / "cosmos_dataset_metadata.json"
        stats_path = dataset_root / "dataset_statistics.json"
        metadata = load_json(metadata_path)
        action_metadata = metadata.get("action", {})
        if action_metadata.get("encoding") != "cosmos_rotation_6d":
            raise ValueError("dataset action encoding is not cosmos_rotation_6d")
        if action_metadata.get("dimension") != 10:
            raise ValueError("only single-arm 10D rotation-6D action is supported")

        stats = load_json(stats_path)
        expected_hash = metadata.get("statistics", {}).get("source_sha256")
        actual_hash = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(
                f"dataset stats SHA256 mismatch: {actual_hash} != {expected_hash}"
            )

        serialized_actions = load_serialized_actions(parquet_path)
        control_actions = undo_dataset_normalization(
            serialized_actions, metadata, stats
        )
        last_action_dropped = bool(action_metadata.get("last_action_is_padding", False))
        if last_action_dropped:
            serialized_actions = serialized_actions[:-1]
            control_actions = control_actions[:-1]

        translation_scale = float(action_metadata["translation_scale"])
        gripper_scale = float(action_metadata["gripper_scale"])
        metric_actions, validation = decode_metric_actions(
            control_actions, translation_scale, gripper_scale
        )
        payload = {
            "format_version": 1,
            "action_units": "dx_dy_dz_meters__droll_dpitch_dyaw_radians__gripper_absolute",
            "source_parquet": str(parquet_path),
            "source_dataset": str(dataset_root),
            "statistics_path": str(stats_path),
            "statistics_sha256": actual_hash,
            "translation_scale_used": np.float32(translation_scale),
            "gripper_scale_used": np.float32(gripper_scale),
            "last_action_dropped": last_action_dropped,
            "serialized_action_10d": serialized_actions.astype(np.float32),
            "cosmos_action_10d": control_actions.astype(np.float32),
            "metric_action_7d": metric_actions,
            "validation": validation,
        }
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, payload, allow_pickle=True)
        print(f"saved: {output_path}")
        print(f"serialized_action_10d: {payload['serialized_action_10d'].shape}")
        print(f"cosmos_action_10d: {payload['cosmos_action_10d'].shape}")
        print(f"metric_action_7d: {payload['metric_action_7d'].shape}")

        if args.reference is None:
            print_action_samples(payload, None, args.sample_rows, args.seed)
            print("[CONCLUSION] DECODED; no reference NPY was provided.")
            return 0
        reference = load_reference(args.reference.expanduser().resolve())
        required = ("cosmos_action_10d", "metric_action_7d")
        missing = [name for name in required if name not in reference]
        if missing:
            raise ValueError(f"reference NPY missing keys: {missing}")
        print_action_samples(payload, reference, args.sample_rows, args.seed)
        matches = [
            compare_array(name, payload[name], reference[name], args.rtol, args.atol)
            for name in required
        ]
        matched = all(matches)
        print(
            "[CONCLUSION] MATCH; cosmos_action_10d and metric_action_7d "
            "match within tolerance."
            if matched
            else "[CONCLUSION] DIFFERENT; decoded actions do not match reference."
        )
        return 0 if matched else 1
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
