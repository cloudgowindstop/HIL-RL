"""Audit Euler 14D and rotation-6D 20D trees against a frozen raw split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..action_spec import get_action_spec
from .paired_index import (
    index_converted_records,
    load_json,
    load_source_ids,
    parse_collection_name_map,
    physical_action_channels,
)

METADATA_NAME = "cosmos_dataset_metadata.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that Euler and rotation-6D conversions share one frozen split"
    )
    parser.add_argument("--euler-root", required=True, type=Path)
    parser.add_argument("--rotation6d-root", required=True, type=Path)
    parser.add_argument("--source-train", required=True, type=Path)
    parser.add_argument("--source-val", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collection-name-map", action="append", default=[])
    parser.add_argument("--translation-atol", type=float, default=1e-5)
    parser.add_argument("--gripper-atol", type=float, default=1e-5)
    return parser.parse_args()


def _metadata_errors(root: Path, encoding: str) -> list[str]:
    spec = get_action_spec(encoding)
    errors: list[str] = []
    for metadata_path in sorted(root.rglob(METADATA_NAME)):
        dataset_root = metadata_path.parent
        outcome = str(
            load_json(metadata_path).get("episode_labeling", {}).get("outcome", "unknown")
        )
        if outcome.lower() != "success":
            continue
        metadata = load_json(metadata_path)
        action = metadata.get("action", {})
        prefix = dataset_root.name
        checks = {
            "encoding": (action.get("encoding"), spec.encoding),
            "dimension": (action.get("dimension"), spec.dimension),
            "chunk_size": (action.get("chunk_size"), 16),
            "source": (action.get("source"), "puppet_next_frame"),
            "number_of_arms": (2, 2),
        }
        for name, (actual, expected) in checks.items():
            if name == "number_of_arms":
                continue
            if actual != expected:
                errors.append(f"{prefix}: action.{name}={actual!r}, expected {expected!r}")
        stats_path = dataset_root / "dataset_statistics.json"
        if not stats_path.is_file():
            errors.append(f"{prefix}: missing dataset_statistics.json")
            continue
        stats = load_json(stats_path)
        minimum = stats.get("actions_min", stats.get("action_min"))
        maximum = stats.get("actions_max", stats.get("action_max"))
        if not isinstance(minimum, list) or len(minimum) != spec.dimension:
            errors.append(f"{prefix}: actions_min length is not {spec.dimension}")
        if not isinstance(maximum, list) or len(maximum) != spec.dimension:
            errors.append(f"{prefix}: actions_max length is not {spec.dimension}")
    return errors


def _first_success_stats(root: Path) -> dict[str, list[float]]:
    for stats_path in sorted(root.rglob("dataset_statistics.json")):
        metadata_path = stats_path.parent / METADATA_NAME
        if not metadata_path.is_file():
            continue
        outcome = str(
            load_json(metadata_path).get("episode_labeling", {}).get("outcome", "unknown")
        )
        if outcome.lower() != "success":
            continue
        return load_json(stats_path)
    raise FileNotFoundError(f"no success dataset_statistics.json under {root}")


def _channel_values(stats: dict[str, list[float]], indices: tuple[int, ...], key: str) -> list[float]:
    values = stats.get(key)
    if not isinstance(values, list):
        raise ValueError(f"statistics missing {key}")
    return [float(values[index]) for index in indices]


def _compare_physical_stats(
    euler_stats: dict[str, list[float]],
    rotation_stats: dict[str, list[float]],
    *,
    translation_atol: float,
    gripper_atol: float,
) -> list[str]:
    euler_channels = physical_action_channels("legacy_euler")
    rotation_channels = physical_action_channels("cosmos_rotation_6d")
    errors: list[str] = []
    for name in ("left_translation", "right_translation", "left_gripper", "right_gripper"):
        atol = gripper_atol if "gripper" in name else translation_atol
        for stat_key in ("actions_min", "actions_max"):
            euler_values = _channel_values(euler_stats, euler_channels[name], stat_key)
            rotation_values = _channel_values(rotation_stats, rotation_channels[name], stat_key)
            deltas = [
                abs(left - right) for left, right in zip(euler_values, rotation_values)
            ]
            if any(delta > atol for delta in deltas):
                errors.append(
                    f"{stat_key}.{name} differs beyond atol={atol}: "
                    f"euler={euler_values} rotation6d={rotation_values}"
                )
    return errors


def _split_errors(
    name: str,
    by_id: dict[str, object],
    requested: list[str],
) -> list[str]:
    missing = sorted(set(requested) - set(by_id))
    extra = sorted(set(by_id) - set(requested))
    errors: list[str] = []
    if missing:
        errors.append(f"{name} missing {len(missing)} source ids, e.g. {missing[:5]}")
    if extra:
        errors.append(f"{name} has {len(extra)} unexpected source ids, e.g. {extra[:5]}")
    return errors


def audit(
    *,
    euler_root: Path,
    rotation6d_root: Path,
    source_train: Path,
    source_val: Path,
    collection_name_map: dict[str, str] | None = None,
    translation_atol: float = 1e-5,
    gripper_atol: float = 1e-5,
) -> dict[str, object]:
    train_ids = load_source_ids(source_train)
    val_ids = load_source_ids(source_val)
    requested = train_ids + val_ids
    overlap = sorted(set(train_ids) & set(val_ids))
    errors: list[str] = []
    warnings: list[str] = []
    if overlap:
        errors.append(f"train/val source overlap: {overlap[:5]}")

    euler = index_converted_records(
        euler_root, collection_name_map=collection_name_map, outcomes={"success"}
    )
    rotation = index_converted_records(
        rotation6d_root, collection_name_map=collection_name_map, outcomes={"success"}
    )
    errors.extend(_metadata_errors(euler_root, "legacy_euler"))
    errors.extend(_metadata_errors(rotation6d_root, "cosmos_rotation_6d"))
    errors.extend(_split_errors("euler", euler, requested))
    errors.extend(_split_errors("rotation6d", rotation, requested))

    tasks = {record.task for record in list(euler.values()) + list(rotation.values())}
    if len(tasks) > 1:
        errors.append(f"task strings are not unique: {sorted(tasks)}")

    row_mismatches: list[str] = []
    for source_id in requested:
        left = euler.get(source_id)
        right = rotation.get(source_id)
        if left is None or right is None:
            continue
        if left.rows != right.rows:
            row_mismatches.append(f"{source_id}: euler_rows={left.rows} rotation6d_rows={right.rows}")
    if row_mismatches:
        errors.append(
            f"{len(row_mismatches)} episode length mismatches, e.g. {row_mismatches[:5]}"
        )

    try:
        errors.extend(
            _compare_physical_stats(
                _first_success_stats(euler_root),
                _first_success_stats(rotation6d_root),
                translation_atol=translation_atol,
                gripper_atol=gripper_atol,
            )
        )
    except (FileNotFoundError, ValueError, IndexError) as error:
        errors.append(str(error))

    report = {
        "passed": not errors,
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "euler_success_episodes": len(euler),
        "rotation6d_success_episodes": len(rotation),
        "shared_requested_episodes": len(set(requested) & set(euler) & set(rotation)),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> None:
    args = _parse_args()
    report = audit(
        euler_root=args.euler_root.expanduser().resolve(),
        rotation6d_root=args.rotation6d_root.expanduser().resolve(),
        source_train=args.source_train.expanduser().resolve(),
        source_val=args.source_val.expanduser().resolve(),
        collection_name_map=parse_collection_name_map(args.collection_name_map),
        translation_atol=args.translation_atol,
        gripper_atol=args.gripper_atol,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
