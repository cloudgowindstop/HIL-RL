#!/usr/bin/env python3
"""Summarize an organized HDF5 dataset without reading image payloads.

The original scan manifest still contains pre-move paths. This tool joins it
with the move plan, reconstructs every current HDF5 path, validates filesystem
coverage and file size, then emits portable JSON/JSONL/CSV/Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


OUTCOME_RULE = "source_batch name contains 'fail' => failure; otherwise success"
QUALITY_BUCKETS = ("normal", "review", "quarantine", "unreadable")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            result.append(value)
    return result


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in columns})
    os.replace(temporary, path)


def classify_outcome(source_batch: str) -> str:
    return "failure" if "fail" in source_batch.lower() else "success"


def quality_bucket(status: str) -> str:
    if status in {"passed", "warning"}:
        return "normal"
    if status in {"review", "quarantine", "unreadable"}:
        return status
    return "unknown"


def build_source_index(move_plan: list[dict[str, Any]]) -> dict[Path, dict[str, Any]]:
    result: dict[Path, dict[str, Any]] = {}
    for move in move_plan:
        source = Path(move["source"])
        if source in result:
            raise ValueError(f"duplicate move source: {source}")
        result[source] = move
    return result


def find_covering_move(old_path: Path, index: dict[Path, dict[str, Any]]) -> dict[str, Any] | None:
    current = old_path.parent
    while current != current.parent:
        if current in index:
            return index[current]
        current = current.parent
    return None


def episode_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["station_id"]),
        str(record["task_id"]),
        str(record["source_batch"]),
        str(record["episode"]),
    )


def build_inventory(
    *,
    input_root: Path,
    manifest: list[dict[str, Any]],
    move_plan: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = build_source_index(move_plan)
    schemas = catalog.get("schemas", {})
    inventory: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_new_paths: set[Path] = set()
    resolved_root = input_root.resolve()

    for source_record in manifest:
        old_path = Path(source_record["path"])
        move = find_covering_move(old_path, index)
        if move is None:
            errors.append({"type": "missing_covering_move", "old_path": str(old_path)})
            continue
        move_source = Path(move["source"])
        try:
            relative = old_path.relative_to(move_source)
        except ValueError:
            errors.append(
                {
                    "type": "path_not_beneath_move_source",
                    "old_path": str(old_path),
                    "move_source": str(move_source),
                }
            )
            continue
        new_path = Path(move["destination"]) / relative
        try:
            new_path.resolve().relative_to(resolved_root)
        except ValueError:
            errors.append({"type": "destination_outside_input", "path": str(new_path)})
            continue
        if new_path in seen_new_paths:
            errors.append({"type": "duplicate_destination_path", "path": str(new_path)})
            continue
        seen_new_paths.add(new_path)

        schema_id = source_record.get("schema_id")
        schema = schemas.get(schema_id)
        if schema is None:
            errors.append({"type": "schema_missing_from_catalog", "schema_id": schema_id, "path": str(new_path)})
            continue
        capabilities = schema.get("capabilities", {})
        status = str(source_record.get("quality_status", move.get("quality_status", "unknown")))
        bucket = quality_bucket(status)
        actual_exists = new_path.is_file()
        actual_bytes = new_path.stat().st_size if actual_exists else None
        expected_bytes = int(source_record.get("file_bytes", 0))
        if not actual_exists:
            errors.append({"type": "missing_file", "path": str(new_path)})
        elif actual_bytes != expected_bytes:
            errors.append(
                {
                    "type": "file_size_mismatch",
                    "path": str(new_path),
                    "expected": expected_bytes,
                    "actual": actual_bytes,
                }
            )
        inventory.append(
            {
                "path": str(new_path),
                "old_path": str(old_path),
                "family": schema["structure_family"],
                "schema_id": schema_id,
                "quality_status": status,
                "quality_bucket": bucket,
                "station_id": source_record["station_id"],
                "task_id": source_record["task_id"],
                "source_batch": source_record["source_batch"],
                "episode": source_record["episode"],
                "outcome": classify_outcome(str(source_record["source_batch"])),
                "trajectory_length": source_record.get("trajectory_length"),
                "file_bytes": expected_bytes,
                "actual_exists": actual_exists,
                "actual_bytes": actual_bytes,
                "issues": source_record.get("issues", []),
                "arm_mode": capabilities.get("arm_mode"),
                "cameras": capabilities.get("cameras", []),
                "has_puppet_pose": bool(capabilities.get("has_puppet_pose")),
                "has_master_pose": bool(capabilities.get("has_master_pose")),
                "has_puppet_joints": bool(capabilities.get("has_puppet_joints")),
                "has_master_joints": bool(capabilities.get("has_master_joints")),
                "has_force": bool(capabilities.get("has_bilateral_force")),
            }
        )
    return sorted(inventory, key=lambda record: record["path"]), errors


def aggregate(records: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in group_keys)].append(record)
    result: list[dict[str, Any]] = []
    for values, members in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        known_lengths = [int(record["trajectory_length"]) for record in members if record["trajectory_length"] is not None]
        quality_counts = Counter(record["quality_bucket"] for record in members)
        outcome_counts = Counter(record["outcome"] for record in members)
        row: dict[str, Any] = dict(zip(group_keys, values))
        row.update(
            {
                "hdf5_count": len(members),
                "episode_count": len({episode_key(record) for record in members}),
                "total_frames_known": sum(known_lengths),
                "trajectory_length_missing_count": len(members) - len(known_lengths),
                "trajectory_length_min": min(known_lengths) if known_lengths else None,
                "trajectory_length_max": max(known_lengths) if known_lengths else None,
                "trajectory_length_mean": (sum(known_lengths) / len(known_lengths)) if known_lengths else None,
                "total_bytes": sum(int(record["file_bytes"]) for record in members),
                "success_count": outcome_counts.get("success", 0),
                "failure_count": outcome_counts.get("failure", 0),
                "normal_count": quality_counts.get("normal", 0),
                "review_count": quality_counts.get("review", 0),
                "quarantine_count": quality_counts.get("quarantine", 0),
                "unreadable_count": quality_counts.get("unreadable", 0),
                "stations": sorted({str(record["station_id"]) for record in members}),
                "tasks": sorted({str(record["task_id"]) for record in members}),
                "families": sorted({str(record["family"]) for record in members}),
                "schema_ids": sorted({str(record["schema_id"]) for record in members}),
                "cameras": sorted({camera for record in members for camera in record["cameras"]}),
            }
        )
        result.append(row)
    return result


def build_summary(
    inventory: list[dict[str, Any]],
    actual_paths: set[Path],
    move_plan: list[dict[str, Any]],
    catalog: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    known_lengths = [int(record["trajectory_length"]) for record in inventory if record["trajectory_length"] is not None]
    quality_counts = Counter(record["quality_bucket"] for record in inventory)
    outcome_counts = Counter(record["outcome"] for record in inventory)
    inventory_paths = {Path(record["path"]) for record in inventory}
    untracked = sorted(str(path) for path in actual_paths - inventory_paths)
    missing_from_filesystem = sorted(str(path) for path in inventory_paths - actual_paths)
    if untracked:
        errors.extend({"type": "untracked_hdf5", "path": path} for path in untracked)
    if missing_from_filesystem:
        known_missing = {error.get("path") for error in errors if error.get("type") == "missing_file"}
        errors.extend(
            {"type": "inventory_file_missing_from_filesystem", "path": path}
            for path in missing_from_filesystem
            if path not in known_missing
        )
    return {
        "hdf5_count": len(inventory),
        "filesystem_hdf5_count": len(actual_paths),
        "episode_count": len({episode_key(record) for record in inventory}),
        "total_bytes": sum(int(record["file_bytes"]) for record in inventory),
        "total_frames_known": sum(known_lengths),
        "trajectory_length_known_count": len(known_lengths),
        "trajectory_length_missing_count": len(inventory) - len(known_lengths),
        "station_count": len({record["station_id"] for record in inventory}),
        "task_count": len({record["task_id"] for record in inventory}),
        "schema_count": len({record["schema_id"] for record in inventory}),
        "catalog_schema_count": int(catalog.get("schema_count", 0)),
        "family_count": len({record["family"] for record in inventory}),
        "move_count": len(move_plan),
        "quality_counts": {bucket: quality_counts.get(bucket, 0) for bucket in QUALITY_BUCKETS},
        "outcome_counts": {outcome: outcome_counts.get(outcome, 0) for outcome in ("success", "failure")},
        "error_count": len(errors),
        "untracked_hdf5_count": len(untracked),
        "missing_hdf5_count": len(missing_from_filesystem),
    }


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0 or unit == "PiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def collapse_dataset_path(path: str) -> str:
    """Collapse storage-level variants into one human-readable signal name."""
    parts = path.split("/")
    if parts[-1] in {"data", "timestamp", "is_intervene"}:
        parts.pop()
    if parts:
        for suffix in ("_align", "_raw"):
            if parts[-1].endswith(suffix):
                parts[-1] = parts[-1][: -len(suffix)]
                break
    return "/".join(parts)


def schema_comparison_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    schemas = catalog.get("schemas", {})
    baselines: dict[str, str] = {}
    by_family: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for schema_id, entry in schemas.items():
        by_family[entry["structure_family"]].append((schema_id, entry))
    for family, members in by_family.items():
        baselines[family] = max(
            members,
            key=lambda item: (int(item[1].get("file_count", 0)), item[0]),
        )[0]

    semantic_paths = {
        schema_id: {
            collapse_dataset_path(dataset["path"])
            for dataset in entry["descriptor"]["datasets"]
        }
        for schema_id, entry in schemas.items()
    }
    rows: list[dict[str, Any]] = []
    for schema_id, entry in sorted(
        schemas.items(),
        key=lambda item: (item[1]["structure_family"], -int(item[1].get("file_count", 0)), item[0]),
    ):
        family = entry["structure_family"]
        baseline_id = baselines[family]
        paths = {dataset["path"] for dataset in entry["descriptor"]["datasets"]}
        capabilities = entry.get("capabilities", {})
        cameras = sorted(capabilities.get("cameras", []))
        if "camera_head" in cameras:
            camera_mode = "head"
        elif "camera_top" in cameras:
            camera_mode = "top"
        elif cameras:
            camera_mode = "other"
        else:
            camera_mode = "missing"
        has_puppet_waist = any(path.startswith("puppet/waist_") for path in paths)
        has_master_waist = any(path.startswith("master/waist_") for path in paths)
        waist_state = (
            "puppet+master"
            if has_puppet_waist and has_master_waist
            else "puppet" if has_puppet_waist else "master" if has_master_waist else "none"
        )
        quality_counts = entry.get("quality_counts", {})
        rows.append(
            {
                "schema_id": schema_id,
                "family": family,
                "file_count": int(entry.get("file_count", 0)),
                "dataset_count": len(entry["descriptor"]["datasets"]),
                "camera_mode": camera_mode,
                "cameras": cameras,
                "puppet_pose": bool(capabilities.get("has_puppet_pose")),
                "master_pose": bool(capabilities.get("has_master_pose")),
                "puppet_joints": bool(capabilities.get("has_puppet_joints")),
                "master_joints": bool(capabilities.get("has_master_joints")),
                "force": bool(capabilities.get("has_bilateral_force")),
                "puppet_head_state": any(path.startswith("puppet/head_") for path in paths),
                "waist_state": waist_state,
                "trajectory_length_key": "metadata/trajectory_length" in paths,
                "family_baseline": baseline_id,
                "is_family_baseline": schema_id == baseline_id,
                "added_vs_baseline": sorted(semantic_paths[schema_id] - semantic_paths[baseline_id]),
                "missing_vs_baseline": sorted(semantic_paths[baseline_id] - semantic_paths[schema_id]),
                "normal_count": int(quality_counts.get("passed", 0)) + int(quality_counts.get("warning", 0)),
                "quarantine_count": int(quality_counts.get("quarantine", 0)),
            }
        )
    return rows


def yes_no(value: bool) -> str:
    return "Y" if value else "-"


def write_schema_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HDF5 Schema Semantic Comparison",
        "",
        "本报告将 `align/raw × data/timestamp/is_intervene` 折叠为一个语义信号。",
        "Group是容器；能力判断主要依据实际存储数据的Dataset。",
        "",
        "## Capability matrix",
        "",
        "| Schema | Family | Files | Datasets | Camera | Puppet pose | Force | Head state | Waist | T key | Quarantine |",
        "|---|---|---:|---:|---|:---:|:---:|:---:|---|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['schema_id']}` | {row['family']} | {row['file_count']} | "
            f"{row['dataset_count']} | {row['camera_mode']} | {yes_no(row['puppet_pose'])} | "
            f"{yes_no(row['force'])} | {yes_no(row['puppet_head_state'])} | "
            f"{row['waist_state']} | {yes_no(row['trajectory_length_key'])} | "
            f"{row['quarantine_count']} |"
        )
    lines.extend(
        [
            "",
            "## Differences from the most common Schema in each family",
            "",
            "每个Family以文件数最多的Schema作为固定展示基准；这里只列折叠后的语义差异。",
            "",
        ]
    )
    for row in rows:
        lines.extend([f"### {row['schema_id']}", ""])
        if row["is_family_baseline"]:
            lines.extend([f"Family基准（{row['family']}）。", ""])
            continue
        lines.append(f"基准：`{row['family_baseline']}`")
        lines.append("")
        added = row["added_vs_baseline"]
        missing = row["missing_vs_baseline"]
        lines.append("新增：" + (", ".join(f"`{item}`" for item in added) if added else "无"))
        lines.append("")
        lines.append("缺少：" + (", ".join(f"`{item}`" for item in missing) if missing else "无"))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_schema_keys_report(
    path: Path,
    catalog: dict[str, Any],
    inventory: list[dict[str, Any]],
    catalog_path: Path,
) -> None:
    """Write the complete schema key reference beside the dataset it describes."""
    representatives: dict[str, str] = {}
    for record in inventory:
        representatives.setdefault(record["schema_id"], record["path"])
    lines = [
        "# HDF5 Schema完整Key清单",
        "",
        f"- 来源：`{catalog_path.resolve()}`",
        f"- Logical schema数量：{catalog.get('schema_count', 0)}",
        f"- Storage schema数量：{catalog.get('storage_schema_count', 0)}",
        "- Shape中的 `T` 表示与 `metadata/trajectory_length` 对齐的时间长度。",
        "- Shape中的 `N` 表示非对齐原始采样长度。",
        "",
    ]
    schemas = catalog.get("schemas", {})
    for schema_id, entry in sorted(
        schemas.items(), key=lambda item: (-int(item[1].get("file_count", 0)), item[0])
    ):
        descriptor = entry["descriptor"]
        capabilities = entry.get("capabilities", {})
        lines.extend(
            [
                f"## {schema_id}",
                "",
                f"- Structure family：{entry['structure_family']}",
                f"- 文件数：{entry.get('file_count', 0)}",
                f"- Group数：{len(descriptor.get('groups', []))}",
                f"- Dataset数：{len(descriptor.get('datasets', []))}",
                f"- 工站：{', '.join(entry.get('stations', []))}",
                f"- 任务：{', '.join(entry.get('tasks', []))}",
                f"- Arm mode：{capabilities.get('arm_mode')}",
                f"- Puppet pose：{capabilities.get('has_puppet_pose')}",
                f"- Master pose：{capabilities.get('has_master_pose')}",
                f"- Puppet joints：{capabilities.get('has_puppet_joints')}",
                f"- Master joints：{capabilities.get('has_master_joints')}",
                f"- Cameras：{', '.join(capabilities.get('cameras', []))}",
                f"- 代表文件：`{representatives.get(schema_id, 'missing')}`",
                "",
                f"### Group keys（{len(descriptor.get('groups', []))}）",
                "",
                "```text",
            ]
        )
        lines.extend(group["path"] for group in descriptor.get("groups", []))
        lines.extend(
            [
                "```",
                "",
                f"### Dataset keys（{len(descriptor.get('datasets', []))}）",
                "",
                "```text",
            ]
        )
        for dataset in descriptor.get("datasets", []):
            lines.append(
                f"{dataset['path']}  shape={dataset.get('shape')}  dtype={dataset.get('dtype')}"
            )
        lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata_readme(path: Path) -> None:
    text = """# Dataset Metadata

本目录保存 `raw_data_0831download_by_schema` 的处理状态与统计信息。

- `download_state/`：下载计划、结果、未解析信息和日志。
- `schema_state/`：Schema catalog、文件映射、校验报告和完整Key清单。
- `move_state/`：移动计划、journal和dry-run日志。
- `statistics/`：最终统计、文件清单和Schema语义对比。

HDF5数据位于本目录的上一级各结构分类目录；`_quarantine/`保存隔离数据。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    totals = summary["totals"]
    lines = [
        "# Organized HDF5 Dataset Statistics",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Dataset: `{summary['input_root']}`",
        f"Outcome rule: `{summary['outcome_rule']}`",
        "",
        "## Overview",
        "",
        f"- HDF5 files: {totals['hdf5_count']}",
        f"- Episodes: {totals['episode_count']}",
        f"- Total size: {human_bytes(totals['total_bytes'])}",
        f"- Known frames: {totals['total_frames_known']}",
        f"- Missing trajectory length: {totals['trajectory_length_missing_count']}",
        f"- Stations: {totals['station_count']}",
        f"- Tasks: {totals['task_count']}",
        f"- Logical schemas: {totals['schema_count']}",
        f"- Quality: {totals['quality_counts']}",
        f"- Outcomes: {totals['outcome_counts']}",
        f"- Consistency errors: {totals['error_count']}",
        "",
        "## By structure family",
        "",
        "| Family | HDF5 | Episodes | Frames | Size | Normal | Quarantine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["by_family"]:
        lines.append(
            f"| {row['family']} | {row['hdf5_count']} | {row['episode_count']} | "
            f"{row['total_frames_known']} | {human_bytes(row['total_bytes'])} | "
            f"{row['normal_count']} | {row['quarantine_count']} |"
        )
    lines.extend(
        [
            "",
            "## By task",
            "",
            "| Task | HDF5 | Episodes | Frames | Success | Failure | Schemas |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tables["by_task"]:
        lines.append(
            f"| {row['task_id']} | {row['hdf5_count']} | {row['episode_count']} | "
            f"{row['total_frames_known']} | {row['success_count']} | {row['failure_count']} | "
            f"{len(row['schema_ids'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_statistics(args: argparse.Namespace) -> int:
    input_root = args.input.resolve()
    metadata_root = input_root / "_metadata"
    catalog_path = (args.catalog or metadata_root / "schema_state" / "schema_catalog.json").resolve()
    manifest_path = (args.manifest or metadata_root / "schema_state" / "hdf5_manifest.jsonl").resolve()
    move_plan_path = (args.move_plan or metadata_root / "move_state" / "move_plan.jsonl").resolve()
    output_path = (args.output or metadata_root / "statistics").resolve()
    canonical_layout = all(
        value is None for value in (args.catalog, args.manifest, args.move_plan, args.output)
    )
    manifest = load_jsonl(manifest_path)
    move_plan = load_jsonl(move_plan_path)
    catalog = load_json(catalog_path)
    inventory, errors = build_inventory(
        input_root=input_root,
        manifest=manifest,
        move_plan=move_plan,
        catalog=catalog,
    )
    actual_paths = set(input_root.rglob("trajectory.hdf5"))
    totals = build_summary(inventory, actual_paths, move_plan, catalog, errors)
    tables = {
        "by_family": aggregate(inventory, ("family",)),
        "by_schema": aggregate(inventory, ("family", "schema_id")),
        "by_station": aggregate(inventory, ("station_id",)),
        "by_task": aggregate(inventory, ("task_id",)),
        "by_station_task": aggregate(inventory, ("station_id", "task_id")),
        "by_quality": aggregate(inventory, ("quality_bucket",)),
        "by_outcome": aggregate(inventory, ("outcome",)),
    }
    comparison_rows = schema_comparison_rows(catalog)
    summary = {
        "generated_at": utc_now(),
        "input_root": str(input_root),
        "outcome_rule": OUTCOME_RULE,
        "sources": {
            "manifest": str(manifest_path),
            "catalog": str(catalog_path),
            "move_plan": str(move_plan_path),
        },
        "totals": totals,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path / "summary.json", summary)
    write_jsonl_atomic(output_path / "file_inventory.jsonl", inventory)
    write_jsonl_atomic(output_path / "statistics_errors.jsonl", errors)
    for name, rows in tables.items():
        write_csv(output_path / f"{name}.csv", rows)
    write_csv(output_path / "schema_comparison.csv", comparison_rows)
    write_schema_comparison(output_path / "schema_comparison.md", comparison_rows)
    write_markdown(output_path / "report.md", summary, tables)
    if canonical_layout:
        write_schema_keys_report(
            metadata_root / "schema_state" / "HDF5_SCHEMA_KEYS.md",
            catalog,
            inventory,
            catalog_path,
        )
        write_metadata_readme(metadata_root / "README.md")

    print(f"hdf5 files: {totals['hdf5_count']}")
    print(f"filesystem hdf5 files: {totals['filesystem_hdf5_count']}")
    print(f"episodes: {totals['episode_count']}")
    print(f"total size: {human_bytes(totals['total_bytes'])}")
    print(f"known frames: {totals['total_frames_known']}")
    print(f"quality: {totals['quality_counts']}")
    print(f"outcomes: {totals['outcome_counts']}")
    print(f"errors: {totals['error_count']}")
    print(f"report: {output_path / 'report.md'}")
    return 1 if errors else 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        old = root / "old" / "station" / "task" / "batch" / "episode" / "data"
        new = root / "organized" / "A" / "schema_test" / "station" / "task" / "batch" / "episode" / "data"
        old_path = old / "trajectory.hdf5"
        new.mkdir(parents=True)
        new_path = new / "trajectory.hdf5"
        new_path.write_bytes(b"test")
        manifest = [
            {
                "path": str(old_path),
                "schema_id": "schema_test",
                "quality_status": "warning",
                "station_id": "station",
                "task_id": "task",
                "source_batch": "batch_fail",
                "episode": "episode",
                "trajectory_length": 10,
                "file_bytes": 4,
                "issues": ["test_warning"],
            }
        ]
        plan = [
            {
                "source": str(old.parent.parent),
                "destination": str(new.parent.parent),
                "quality_status": "warning",
            }
        ]
        catalog = {
            "schema_count": 1,
            "schemas": {
                "schema_test": {
                    "structure_family": "A",
                    "file_count": 1,
                    "descriptor": {
                        "datasets": [
                            {"path": "metadata/trajectory_length"},
                            {"path": "camera_observations/color_images/camera_head"},
                            {"path": "puppet/arm_single_position_align/data"},
                        ]
                    },
                    "capabilities": {"arm_mode": "single", "cameras": ["camera_head"]},
                }
            },
        }
        inventory, errors = build_inventory(
            input_root=root / "organized",
            manifest=manifest,
            move_plan=plan,
            catalog=catalog,
        )
        assert not errors and len(inventory) == 1
        assert inventory[0]["path"] == str(new_path)
        assert inventory[0]["outcome"] == "failure"
        totals = build_summary(inventory, {new_path}, plan, catalog, errors)
        assert totals["hdf5_count"] == 1 and totals["total_frames_known"] == 10
        assert totals["quality_counts"]["normal"] == 1
        comparison = schema_comparison_rows(catalog)
        assert len(comparison) == 1 and comparison[0]["is_family_baseline"]
    print("summarize_organized_data self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--catalog", type=Path)
    summarize.add_argument("--manifest", type=Path)
    summarize.add_argument("--move-plan", type=Path)
    summarize.add_argument("--output", type=Path)
    summarize.set_defaults(handler=generate_statistics)
    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(handler=lambda _: run_self_test())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
