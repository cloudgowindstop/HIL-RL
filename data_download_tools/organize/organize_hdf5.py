#!/usr/bin/env python3
"""Discover HDF5 schemas, report differences, and transactionally organize data.

Logical schema identity excludes trajectory length and storage filters. Length
problems are validation issues, while missing datasets or dtype/shape-tail
changes create a different schema. Movement uses os.rename and rejects
cross-filesystem operations and destination conflicts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


FINGERPRINT_VERSION = 1
UNKNOWN_STATION = "unknown_station"
UNKNOWN_TASK = "unknown_task"
QUALITY_STATUSES = ("passed", "warning", "review", "quarantine", "unreadable")
CAMERA_TIMESTAMP_STALL_MIN_STEPS = 5

STRUCTURE_FAMILIES: dict[str, dict[str, str]] = {
    "A_head_pose_force": {
        "label": "Head图像 + Puppet pose + 力传感器",
        "conversion_status": "ready",
    },
    "B_head_pose_no_force": {
        "label": "Head图像 + Puppet pose，无力传感器",
        "conversion_status": "ready",
    },
    "C_top_joint_only": {
        "label": "Top图像 + 无Puppet pose（关节/FK分支）",
        "conversion_status": "adapter_required",
    },
    "D_head_joint_only": {
        "label": "Head图像 + 无Puppet pose（关节/FK分支）",
        "conversion_status": "adapter_required",
    },
    "E_missing_images": {
        "label": "缺少Cosmos所需RGB图像",
        "conversion_status": "quarantine",
    },
}


@dataclass(frozen=True)
class HDF5Record:
    path: str
    batch_root: str
    episode_root: str
    station_id: str
    task_id: str
    source_batch: str
    episode: str
    schema_id: str | None
    schema_hash: str | None
    storage_id: str | None
    storage_hash: str | None
    trajectory_length: int | None
    readable: bool
    quality_status: str
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata_language_instruction: str | None = None
    file_bytes: int = 0


@dataclass(frozen=True)
class MoveRecord:
    move_id: str
    source: str
    destination: str
    family: str
    schema_id: str
    station_id: str
    task_id: str
    source_batch: str
    quality_status: str
    quality_bucket: str
    unit: str
    hdf5_count: int
    total_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return str(value)


def describe_attributes(obj: h5py.Group | h5py.Dataset) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in sorted(obj.attrs.keys()):
        value = np.asarray(obj.attrs[name])
        result.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
    return result


def is_align_timeseries(path: str) -> bool:
    return "/camera_observations/" in f"/{path}" or "_align/" in path


def is_raw_timeseries(path: str) -> bool:
    return "_raw/" in path


def normalize_shape(path: str, shape: tuple[Any, ...]) -> list[Any]:
    normalized: list[int | str] = list(shape)
    if normalized and is_align_timeseries(path):
        normalized[0] = "T"
    elif normalized and is_raw_timeseries(path):
        normalized[0] = "N"
    return normalized


def dtype_descriptor(dtype: np.dtype[Any]) -> str:
    variable = h5py.check_dtype(vlen=dtype)
    if variable is bytes:
        return "vlen_bytes"
    if variable is str:
        return "vlen_string"
    if variable is not None:
        try:
            return f"vlen_{np.dtype(variable)}"
        except TypeError:
            return f"vlen_{variable}"
    return str(dtype)


def build_logical_schema(h5_file: h5py.File) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []

    def visitor(path: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            groups.append({"path": path, "attributes": describe_attributes(obj)})
        elif isinstance(obj, h5py.Dataset):
            datasets.append(
                {
                    "path": path,
                    "rank": obj.ndim,
                    "shape": normalize_shape(path, obj.shape),
                    "dtype": dtype_descriptor(obj.dtype),
                    "attributes": describe_attributes(obj),
                }
            )

    h5_file.visititems(visitor)
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "groups": sorted(groups, key=lambda item: item["path"]),
        "datasets": sorted(datasets, key=lambda item: item["path"]),
    }


def build_storage_schema(h5_file: h5py.File) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []

    def visitor(path: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        datasets.append(
            {
                "path": path,
                "chunks": list(obj.chunks) if obj.chunks is not None else None,
                "compression": obj.compression,
                "compression_opts": json_safe(obj.compression_opts),
                "shuffle": bool(obj.shuffle),
                "fletcher32": bool(obj.fletcher32),
                "maxshape": normalize_shape(
                    path,
                    tuple(value if value is not None else "unlimited" for value in obj.maxshape),
                ),
            }
        )

    h5_file.visititems(visitor)
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "datasets": sorted(datasets, key=lambda item: item["path"]),
    }


def canonical_hash(payload: dict[str, Any], prefix: str) -> tuple[str, str]:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}", digest


def scalar_text(h5_file: h5py.File, path: str) -> str | None:
    if path not in h5_file:
        return None
    try:
        value = h5_file[path][()]
    except (OSError, ValueError, TypeError):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def trajectory_length(h5_file: h5py.File) -> int | None:
    if "metadata/trajectory_length" not in h5_file:
        return None
    try:
        return int(h5_file["metadata/trajectory_length"][()])
    except (OSError, ValueError, TypeError, OverflowError):
        return None


def validate_hdf5(h5_file: h5py.File, expected_language: str | None = None) -> list[str]:
    issues: list[str] = []
    datasets: dict[str, h5py.Dataset] = {}
    length = trajectory_length(h5_file)
    if length is None or length <= 0:
        issues.append("invalid_or_missing_trajectory_length")

    def visitor(path: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        datasets[path] = obj
        if obj.ndim > 0 and obj.shape[0] == 0:
            issues.append(f"empty_dataset:{path}")
        if length is not None and obj.ndim > 0 and is_align_timeseries(path) and obj.shape[0] != length:
            issues.append(f"length_mismatch:{path}:expected={length}:actual={obj.shape[0]}")

    h5_file.visititems(visitor)

    declared = set(h5_file["camera_color_channel"].keys()) if "camera_color_channel" in h5_file else set()
    colors = set(h5_file["camera_observations/color_images"].keys()) if "camera_observations/color_images" in h5_file else set()
    depths = set(h5_file["camera_observations/depth_images"].keys()) if "camera_observations/depth_images" in h5_file else set()
    for name in sorted(declared - colors):
        issues.append(f"declared_camera_missing_color:{name}")
    for name in sorted(colors - declared):
        issues.append(f"color_camera_not_declared:{name}")
    for name in sorted(declared - depths):
        issues.append(f"declared_camera_missing_depth:{name}")
    for name in sorted(depths - declared):
        issues.append(f"depth_camera_not_declared:{name}")

    rgb_lengths = {
        f"camera_observations/color_images/{name}": datasets[f"camera_observations/color_images/{name}"].shape[0]
        for name in sorted(colors)
    }
    inferred_length: int | None = None
    if rgb_lengths:
        inferred_length = Counter(rgb_lengths.values()).most_common(1)[0][0]
        for path, actual in rgb_lengths.items():
            if actual != inferred_length:
                issues.append(f"rgb_length_mismatch:{path}:expected={inferred_length}:actual={actual}")

    # RGB frame count is the best fallback reference when metadata T is absent.
    # This catches partially finalized episodes instead of treating missing T as
    # the only problem while silently accepting mismatched robot streams.
    if length is None and inferred_length is not None:
        for path, dataset in sorted(datasets.items()):
            if dataset.ndim > 0 and is_align_timeseries(path) and dataset.shape[0] != inferred_length:
                issues.append(
                    f"inferred_length_mismatch:{path}:expected={inferred_length}:actual={dataset.shape[0]}"
                )

    # Some interrupted recordings keep appending robot states while the camera
    # timestamp and compressed images remain frozen. Inspect only the usable RGB
    # prefix so a trailing bookkeeping timestamp cannot hide that condition.
    timestamp_path = "camera_observations/timestamp"
    if inferred_length and timestamp_path in datasets:
        timestamp_dataset = datasets[timestamp_path]
        usable = min(inferred_length, timestamp_dataset.shape[0])
        if usable > CAMERA_TIMESTAMP_STALL_MIN_STEPS:
            try:
                timestamps = np.asarray(timestamp_dataset[:usable]).reshape(usable, -1)[:, 0]
                stalled_steps = 0
                for index in range(usable - 1, 0, -1):
                    if not np.isfinite(timestamps[index]) or timestamps[index] <= timestamps[index - 1]:
                        stalled_steps += 1
                    else:
                        break
                if stalled_steps >= CAMERA_TIMESTAMP_STALL_MIN_STEPS:
                    issues.append(f"camera_timestamp_terminal_stall:steps={stalled_steps}")
            except OSError as error:
                issues.append(f"unreadable_camera_timestamp:{type(error).__name__}:{error}")
            except (ValueError, TypeError):
                issues.append("invalid_camera_timestamp")

    paths: set[str] = set()
    h5_file.visititems(lambda path, obj: paths.add(path) if isinstance(obj, h5py.Dataset) else None)
    for path in sorted(paths):
        if "_left_" in path:
            peer = path.replace("_left_", "_right_")
            if peer not in paths:
                issues.append(f"missing_right_peer:{peer}")
        elif "_right_" in path:
            peer = path.replace("_right_", "_left_")
            if peer not in paths:
                issues.append(f"missing_left_peer:{peer}")

    language = scalar_text(h5_file, "metadata/language_instruction")
    if not language:
        issues.append("missing_language_instruction")
    elif expected_language and language != expected_language:
        issues.append(f"language_instruction_mismatch:actual={language!r}")
    return sorted(set(issues))


def issue_severity(issue: str) -> str:
    """Map one validation issue to a non-blocking or actionable quality level."""
    warning_prefixes = (
        "language_instruction_mismatch:",
        "missing_language_instruction",
        "missing_left_peer:",
        "missing_right_peer:",
        "color_camera_not_declared:",
        "depth_camera_not_declared:",
    )
    if issue.startswith(warning_prefixes):
        return "warning"
    if issue.startswith(("invalid_or_missing_trajectory_length", "declared_camera_missing_depth:")):
        return "review"
    if issue.startswith("length_mismatch:"):
        path = issue.split(":", 2)[1]
        return "review" if "/depth_images/" in f"/{path}" else "quarantine"
    if issue.startswith("inferred_length_mismatch:"):
        path = issue.split(":", 2)[1]
        return "review" if "/depth_images/" in f"/{path}" else "quarantine"
    if issue.startswith(
        (
            "declared_camera_missing_color:",
            "empty_dataset:",
            "rgb_length_mismatch:",
            "camera_timestamp_terminal_stall:",
            "unreadable_camera_timestamp:",
            "invalid_camera_timestamp",
        )
    ):
        return "quarantine"
    return "review"


def quality_status_for_issues(issues: Iterable[str]) -> str:
    """Return the highest-severity status among all issues for one HDF5 file."""
    rank = {status: index for index, status in enumerate(QUALITY_STATUSES)}
    severities = [issue_severity(issue) for issue in issues]
    return max(severities, key=rank.__getitem__) if severities else "passed"


def expected_language_by_task(mapping_path: Path | None) -> dict[str, str]:
    if mapping_path is None:
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for entry in payload.get("tasks", {}).values():
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("task_id")
        description = entry.get("task_description_en")
        if task_id and description and entry.get("reviewed"):
            result[str(task_id)] = str(description)
    return result


def path_metadata(root: Path, hdf5_path: Path) -> tuple[str, str, str, str, Path, Path]:
    relative = hdf5_path.relative_to(root)
    parts = relative.parts
    incremental = len(parts) >= 5 and parts[0] in {"new", "repaired"}
    structured = len(parts) >= 4 and (parts[0].startswith("tienyi_") or parts[0] == UNKNOWN_STATION)
    if incremental:
        # Incremental downloads deliberately omit station inference. Their
        # layout is <new|repaired>/<task>/<batch>/..., so retain the reviewed
        # task while representing the unavailable station explicitly.
        station_id, task_id, source_batch = UNKNOWN_STATION, parts[1], parts[2]
        batch_root = root / parts[0] / task_id / source_batch
        if hdf5_path.parent.name == "data" and hdf5_path.parent.parent != batch_root:
            episode_root = hdf5_path.parent.parent
            episode = episode_root.name
        else:
            episode = source_batch
            episode_root = batch_root
    elif structured:
        station_id, task_id, source_batch = parts[0], parts[1], parts[2]
        batch_root = root / station_id / task_id / source_batch
        if hdf5_path.parent.name == "data" and hdf5_path.parent.parent != batch_root:
            episode_root = hdf5_path.parent.parent
            episode = episode_root.name
        else:
            episode = source_batch
            episode_root = batch_root
    else:
        station_id, task_id = UNKNOWN_STATION, UNKNOWN_TASK
        source_batch = parts[0] if parts else "unknown_batch"
        batch_root = root / source_batch
        if hdf5_path.parent.name == "data" and hdf5_path.parent.parent != batch_root:
            episode_root = hdf5_path.parent.parent
            episode = episode_root.name
        else:
            episode = source_batch
            episode_root = batch_root
    return station_id, task_id, source_batch, episode, batch_root, episode_root


def inspect_hdf5(path_text: str, root_text: str, language_map: dict[str, str]) -> tuple[HDF5Record, dict[str, Any] | None, dict[str, Any] | None]:
    path = Path(path_text)
    root = Path(root_text)
    station_id, task_id, source_batch, episode, batch_root, episode_root = path_metadata(root, path)
    try:
        file_bytes = path.stat().st_size
        with h5py.File(path, "r") as h5_file:
            logical = build_logical_schema(h5_file)
            storage = build_storage_schema(h5_file)
            schema_id, schema_hash = canonical_hash(logical, "schema")
            storage_id, storage_hash = canonical_hash(storage, "storage")
            issues = validate_hdf5(h5_file, language_map.get(task_id))
            length = trajectory_length(h5_file)
            language = scalar_text(h5_file, "metadata/language_instruction")
        record = HDF5Record(
            path=str(path),
            batch_root=str(batch_root),
            episode_root=str(episode_root),
            station_id=station_id,
            task_id=task_id,
            source_batch=source_batch,
            episode=episode,
            schema_id=schema_id,
            schema_hash=schema_hash,
            storage_id=storage_id,
            storage_hash=storage_hash,
            trajectory_length=length,
            readable=True,
            quality_status=quality_status_for_issues(issues),
            issues=tuple(issues),
            metadata_language_instruction=language,
            file_bytes=file_bytes,
        )
        return record, logical, storage
    except Exception as error:  # malformed HDF5 libraries can raise several subclasses
        record = HDF5Record(
            path=str(path),
            batch_root=str(batch_root),
            episode_root=str(episode_root),
            station_id=station_id,
            task_id=task_id,
            source_batch=source_batch,
            episode=episode,
            schema_id=None,
            schema_hash=None,
            storage_id=None,
            storage_hash=None,
            trajectory_length=None,
            readable=False,
            quality_status="unreadable",
            issues=(f"unreadable_hdf5:{type(error).__name__}:{error}",),
            file_bytes=path.stat().st_size if path.exists() else 0,
        )
        return record, None, None


def inspect_worker(arguments: tuple[str, str, dict[str, str]]) -> tuple[HDF5Record, dict[str, Any] | None, dict[str, Any] | None]:
    return inspect_hdf5(*arguments)


def scan_all(root: Path, workers: int, language_map: dict[str, str]) -> tuple[list[HDF5Record], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = sorted(
        path
        for path in root.rglob("trajectory.hdf5")
        if path.is_file() and not any(part.endswith(".partial") for part in path.relative_to(root).parts)
    )
    arguments = [(str(path), str(root), language_map) for path in paths]
    if workers == 1:
        results = map(inspect_worker, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(inspect_worker, arguments, chunksize=8)
    records: list[HDF5Record] = []
    logical_schemas: dict[str, dict[str, Any]] = {}
    storage_schemas: dict[str, dict[str, Any]] = {}
    try:
        for record, logical, storage in results:
            records.append(record)
            if record.schema_id and logical is not None:
                logical_schemas.setdefault(record.schema_id, logical)
            if record.storage_id and storage is not None:
                storage_schemas.setdefault(record.storage_id, storage)
    finally:
        if workers != 1:
            executor.shutdown()
    return records, logical_schemas, storage_schemas


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value for key, value in record.items()}
            )


def read_hdf5_records(path: Path) -> list[HDF5Record]:
    result: list[HDF5Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                payload["issues"] = tuple(payload.get("issues", []))
                # Read manifests written before quality levels replaced passed/failed.
                legacy_status = payload.pop("validation_status", None)
                if "quality_status" not in payload:
                    payload["quality_status"] = (
                        "unreadable"
                        if legacy_status == "unreadable"
                        else quality_status_for_issues(payload["issues"])
                    )
                result.append(HDF5Record(**payload))
    return result


def classify_structure_family(dataset_paths: set[str]) -> tuple[str, dict[str, Any]]:
    """Classify one exact schema into the five human-readable structure families."""
    cameras = {
        path.rsplit("/", 1)[-1]
        for path in dataset_paths
        if path.startswith("camera_observations/color_images/")
    }
    head_rgb = {"camera_head", "camera_left", "camera_right"}.issubset(cameras)
    top_rgb = {"camera_top", "camera_left", "camera_right"}.issubset(cameras)
    puppet_pose = {
        "puppet/end_effector_left_pose_align/data",
        "puppet/end_effector_right_pose_align/data",
    }.issubset(dataset_paths)
    force = {
        "force_observations/force_left_end_effector_align/data",
        "force_observations/force_right_end_effector_align/data",
    }.issubset(dataset_paths)

    facts = {
        "cameras": sorted(cameras),
        "has_required_rgb": head_rgb or top_rgb,
        "has_head_rgb": head_rgb,
        "has_top_rgb": top_rgb,
        "has_complete_puppet_pose": puppet_pose,
        "has_bilateral_force": force,
    }
    # Missing/incomplete RGB wins over pose/force so corrupted image schemas are quarantined.
    if not facts["has_required_rgb"]:
        return "E_missing_images", facts
    if head_rgb and puppet_pose and force:
        return "A_head_pose_force", facts
    if head_rgb and puppet_pose and not force:
        return "B_head_pose_no_force", facts
    if top_rgb and not puppet_pose:
        return "C_top_joint_only", facts
    if head_rgb and not puppet_pose:
        return "D_head_joint_only", facts
    raise ValueError(f"schema does not match A-E structure families: {facts}")


def build_schema_catalog(
    records: list[HDF5Record], logical_schemas: dict[str, dict[str, Any]], storage_schemas: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[HDF5Record]] = defaultdict(list)
    for record in records:
        if record.schema_id:
            grouped[record.schema_id].append(record)
    schemas: dict[str, Any] = {}
    for schema_id, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        descriptor = logical_schemas[schema_id]
        dataset_paths = {item["path"]: item for item in descriptor["datasets"]}
        colors = sorted(
            path.rsplit("/", 1)[-1]
            for path in dataset_paths
            if path.startswith("camera_observations/color_images/")
        )
        arm_mode = "dual" if any("_left_" in path or "/arm_left_" in path for path in dataset_paths) else "single" if any("_single_" in path for path in dataset_paths) else "unknown"
        capabilities = {
            "arm_mode": arm_mode,
            "has_puppet_pose": any(path.startswith("puppet/end_effector") and "_pose_align/data" in path for path in dataset_paths),
            "has_master_pose": any(path.startswith("master/end_effector") and "_pose_align/data" in path for path in dataset_paths),
            "has_puppet_joints": any(path.startswith("puppet/arm") and path.endswith("_position_align/data") for path in dataset_paths),
            "has_master_joints": any(path.startswith("master/arm") and path.endswith("_position_align/data") for path in dataset_paths),
            "cameras": colors,
        }
        family, family_facts = classify_structure_family(set(dataset_paths))
        capabilities.update(family_facts)
        quality_counts = Counter(record.quality_status for record in members)
        schemas[schema_id] = {
            "schema_hash": members[0].schema_hash,
            "file_count": len(members),
            "group_count": len(descriptor["groups"]),
            "dataset_count": len(descriptor["datasets"]),
            "stations": sorted({record.station_id for record in members}),
            "tasks": sorted({record.task_id for record in members}),
            "storage_ids": sorted({record.storage_id for record in members if record.storage_id}),
            "quality_counts": {
                status: quality_counts.get(status, 0)
                for status in QUALITY_STATUSES
            },
            "representative_file": members[0].path,
            "capabilities": capabilities,
            "structure_family": family,
            "descriptor": descriptor,
        }
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "schema_count": len(schemas),
        "storage_schema_count": len(storage_schemas),
        "schemas": schemas,
        "storage_schemas": storage_schemas,
    }


def schema_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_map = {item["path"]: item for item in left["datasets"]}
    right_map = {item["path"]: item for item in right["datasets"]}
    common = sorted(set(left_map) & set(right_map))
    changed = [
        {"path": path, "left": left_map[path], "right": right_map[path]}
        for path in common
        if left_map[path] != right_map[path]
    ]
    return {
        "only_left": sorted(set(left_map) - set(right_map)),
        "only_right": sorted(set(right_map) - set(left_map)),
        "changed": changed,
    }


def build_schema_classification(catalog: dict[str, Any]) -> dict[str, Any]:
    families = {
        family: {
            **metadata,
            "schema_ids": [],
            "file_count": 0,
        }
        for family, metadata in STRUCTURE_FAMILIES.items()
    }
    schemas: dict[str, Any] = {}
    for schema_id, entry in catalog["schemas"].items():
        family = entry["structure_family"]
        if family not in families:
            raise ValueError(f"unknown structure family for {schema_id}: {family}")
        families[family]["schema_ids"].append(schema_id)
        families[family]["file_count"] += entry["file_count"]
        capabilities = entry["capabilities"]
        schemas[schema_id] = {
            "family": family,
            "file_count": entry["file_count"],
            "conversion_status": families[family]["conversion_status"],
            "primary_camera": (
                "camera_head"
                if capabilities["has_head_rgb"]
                else "camera_top" if capabilities["has_top_rgb"] else None
            ),
            "has_puppet_pose": capabilities["has_complete_puppet_pose"],
            "has_force": capabilities["has_bilateral_force"],
            "has_required_rgb": capabilities["has_required_rgb"],
        }
    return {
        "classification_version": 1,
        "generated_at": utc_now(),
        "families": families,
        "schemas": schemas,
    }


def write_schema_reports(state_dir: Path, records: list[HDF5Record], catalog: dict[str, Any]) -> None:
    write_jsonl_atomic(state_dir / "hdf5_manifest.jsonl", (asdict(record) for record in records))
    write_csv(
        state_dir / "file_to_schema.csv",
        [
            {
                "hdf5_path": record.path,
                "schema_id": record.schema_id,
                "structure_family": (
                    catalog["schemas"][record.schema_id]["structure_family"]
                    if record.schema_id in catalog["schemas"]
                    else None
                ),
                "storage_id": record.storage_id,
                "station_id": record.station_id,
                "task_id": record.task_id,
                "readable": record.readable,
                "quality_status": record.quality_status,
                "issues": list(record.issues),
            }
            for record in records
        ],
    )
    summary_rows: list[dict[str, Any]] = []
    for schema_id, entry in catalog["schemas"].items():
        summary_rows.append(
            {
                "schema_id": schema_id,
                "structure_family": entry["structure_family"],
                "file_count": entry["file_count"],
                "group_count": entry["group_count"],
                "dataset_count": entry["dataset_count"],
                "stations": entry["stations"],
                "tasks": entry["tasks"],
                "arm_mode": entry["capabilities"]["arm_mode"],
                "has_puppet_pose": entry["capabilities"]["has_puppet_pose"],
                "has_master_pose": entry["capabilities"]["has_master_pose"],
                "has_puppet_joints": entry["capabilities"]["has_puppet_joints"],
                "has_master_joints": entry["capabilities"]["has_master_joints"],
                "cameras": entry["capabilities"]["cameras"],
                "has_required_rgb": entry["capabilities"]["has_required_rgb"],
                "has_bilateral_force": entry["capabilities"]["has_bilateral_force"],
                **{
                    f"{status}_count": entry["quality_counts"][status]
                    for status in QUALITY_STATUSES
                },
                "representative_file": entry["representative_file"],
            }
        )
    write_csv(state_dir / "schema_summary.csv", summary_rows)
    write_json_atomic(state_dir / "schema_catalog.json", catalog)
    write_json_atomic(
        state_dir / "schema_classification.json",
        build_schema_classification(catalog),
    )
    write_jsonl_atomic(
        state_dir / "validation_issues.jsonl",
        (asdict(record) for record in records if record.quality_status != "passed"),
    )
    for status in QUALITY_STATUSES[1:]:
        write_jsonl_atomic(
            state_dir / f"validation_{status}.jsonl",
            (asdict(record) for record in records if record.quality_status == status),
        )
    # Avoid leaving a stale, misleading binary-failure report from older scans.
    (state_dir / "validation_errors.jsonl").unlink(missing_ok=True)
    quality_counts = Counter(record.quality_status for record in records)
    scan_summary = {
        "hdf5_count": len(records),
        "readable_count": sum(record.readable for record in records),
        "unreadable_count": sum(not record.readable for record in records),
        "quality_counts": {
            status: quality_counts.get(status, 0)
            for status in QUALITY_STATUSES
        },
        "schema_count": catalog["schema_count"],
        "storage_schema_count": catalog["storage_schema_count"],
        "generated_at": utc_now(),
    }
    write_json_atomic(state_dir / "scan_summary.json", scan_summary)

    report = ["# HDF5 Structure Report", "", f"Files: {len(records)}", f"Logical schemas: {catalog['schema_count']}", f"Storage schemas: {catalog['storage_schema_count']}", ""]
    for schema_id, entry in catalog["schemas"].items():
        cap = entry["capabilities"]
        report.extend(
            [
                f"## {schema_id}",
                "",
                f"- Structure family: {entry['structure_family']}",
                f"- Files: {entry['file_count']}",
                f"- Groups: {entry['group_count']}",
                f"- Datasets: {entry['dataset_count']}",
                f"- Stations: {', '.join(entry['stations'])}",
                f"- Tasks: {', '.join(entry['tasks'])}",
                f"- Arm mode: {cap['arm_mode']}",
                f"- Puppet pose: {cap['has_puppet_pose']}",
                f"- Master pose: {cap['has_master_pose']}",
                f"- Puppet joints: {cap['has_puppet_joints']}",
                f"- Master joints: {cap['has_master_joints']}",
                f"- Cameras: {', '.join(cap['cameras'])}",
                "- Quality: " + ", ".join(
                    f"{status}={entry['quality_counts'][status]}"
                    for status in QUALITY_STATUSES
                ),
                f"- Representative: `{entry['representative_file']}`",
                "",
                "### Datasets",
                "",
            ]
        )
        for dataset in entry["descriptor"]["datasets"]:
            report.append(f"- `{dataset['path']}` shape={dataset['shape']} dtype={dataset['dtype']}")
        report.append("")
    (state_dir / "schema_report.md").write_text("\n".join(report), encoding="utf-8")

    schema_items = list(catalog["schemas"].items())
    pairs: list[tuple[tuple[str, Any], tuple[str, Any]]] = []
    if len(schema_items) <= 20:
        for index, left in enumerate(schema_items):
            for right in schema_items[index + 1 :]:
                pairs.append((left, right))
    elif schema_items:
        pairs = [(schema_items[0], item) for item in schema_items[1:]]
    differences = ["# HDF5 Schema Differences", ""]
    for (left_id, left), (right_id, right) in pairs:
        difference = schema_difference(left["descriptor"], right["descriptor"])
        differences.extend([f"## {left_id} vs {right_id}", ""])
        differences.append(f"Only {left_id}: {difference['only_left']}")
        differences.append(f"Only {right_id}: {difference['only_right']}")
        differences.append(f"Changed: {[item['path'] for item in difference['changed']]}")
        differences.append("")
    (state_dir / "schema_differences.md").write_text("\n".join(differences), encoding="utf-8")


def ensure_catalog_structure_families(catalog: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an older catalog in memory; source JSON stays untouched."""
    for entry in catalog.get("schemas", {}).values():
        paths = {item["path"] for item in entry["descriptor"]["datasets"]}
        family, facts = classify_structure_family(paths)
        entry["structure_family"] = family
        entry.setdefault("capabilities", {}).update(facts)
    return catalog


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return ensure_catalog_structure_families(catalog)


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing parent for {path}")
        current = current.parent
    return current


def move_id(source: Path, destination: Path, schema_id: str) -> str:
    digest = hashlib.sha256(f"{source}\0{destination}\0{schema_id}".encode("utf-8")).hexdigest()[:16]
    return f"move_{digest}"


def quality_bucket(status: str) -> str:
    """Map file-level quality to a destination namespace."""
    if status in {"passed", "warning"}:
        return "normal"
    if status in {"review", "quarantine"}:
        return status
    if status == "unreadable":
        return "skip"
    raise ValueError(f"unknown quality status: {status}")


def quality_output_root(output_root: Path, bucket: str) -> Path | None:
    if bucket == "normal":
        return output_root
    if bucket in {"review", "quarantine"}:
        return output_root / f"_{bucket}"
    if bucket == "skip":
        return None
    raise ValueError(f"unknown quality bucket: {bucket}")


def aggregate_quality_status(records: Iterable[HDF5Record]) -> str:
    """Preserve the highest file-level status represented by one move unit."""
    rank = {status: index for index, status in enumerate(QUALITY_STATUSES)}
    statuses = [record.quality_status for record in records]
    return max(statuses, key=rank.__getitem__)


def build_move_plan(
    records: list[HDF5Record],
    output_root: Path,
    schema_families: dict[str, str],
) -> tuple[list[MoveRecord], list[dict[str, Any]]]:
    by_batch: dict[str, list[HDF5Record]] = defaultdict(list)
    for record in records:
        by_batch[record.batch_root].append(record)
    plan: list[MoveRecord] = []
    conflicts: list[dict[str, Any]] = []
    for batch_text, members in sorted(by_batch.items()):
        movable = [
            record
            for record in members
            if record.readable
            and record.schema_id
            and quality_bucket(record.quality_status) != "skip"
        ]
        if not movable:
            conflicts.append({"source": batch_text, "reason": "no_readable_hdf5"})
            continue
        schema_ids = {record.schema_id for record in movable}
        quality_buckets = {quality_bucket(record.quality_status) for record in movable}
        all_movable = len(movable) == len(members)
        if len(schema_ids) == 1 and len(quality_buckets) == 1 and all_movable:
            first = movable[0]
            family = schema_families.get(first.schema_id)
            if family not in STRUCTURE_FAMILIES:
                conflicts.append(
                    {
                        "source": batch_text,
                        "reason": "missing_or_unknown_structure_family",
                        "schema_id": first.schema_id,
                        "family": family,
                    }
                )
                continue
            bucket = quality_bucket(first.quality_status)
            routed_root = quality_output_root(output_root, bucket)
            if routed_root is None:
                conflicts.append({"source": batch_text, "reason": "quality_status_not_movable"})
                continue
            source = Path(batch_text)
            destination = (
                routed_root
                / family
                / first.schema_id
                / first.station_id
                / first.task_id
                / first.source_batch
            )
            plan.append(
                MoveRecord(
                    move_id=move_id(source, destination, first.schema_id),
                    source=str(source),
                    destination=str(destination),
                    family=family,
                    schema_id=first.schema_id,
                    station_id=first.station_id,
                    task_id=first.task_id,
                    source_batch=first.source_batch,
                    quality_status=aggregate_quality_status(movable),
                    quality_bucket=bucket,
                    unit="batch",
                    hdf5_count=len(movable),
                    total_bytes=sum(record.file_bytes for record in movable),
                )
            )
            continue
        by_episode: dict[str, list[HDF5Record]] = defaultdict(list)
        for record in members:
            hdf5_path = Path(record.path)
            # Older manifests recorded success_episodes/failure_episodes as one
            # episode_root. Derive the real <episode>/data/trajectory.hdf5 root
            # from the file path so mixed-schema batches can split safely.
            episode_root = (
                hdf5_path.parent.parent
                if hdf5_path.parent.name == "data"
                else Path(record.episode_root)
            )
            by_episode[str(episode_root)].append(record)
        for episode_text, episode_records in sorted(by_episode.items()):
            episode_movable = [
                record
                for record in episode_records
                if record.readable
                and record.schema_id
                and quality_bucket(record.quality_status) != "skip"
            ]
            if len(episode_movable) != len(episode_records):
                conflicts.append(
                    {
                        "source": episode_text,
                        "reason": "episode_contains_unreadable_or_unclassified_hdf5",
                    }
                )
                continue
            episode_schema_ids = {record.schema_id for record in episode_movable}
            if len(episode_schema_ids) != 1:
                conflicts.append({"source": episode_text, "reason": "multiple_schemas_in_episode", "schema_ids": sorted(episode_schema_ids)})
                continue
            episode_buckets = {quality_bucket(record.quality_status) for record in episode_movable}
            if len(episode_buckets) != 1:
                conflicts.append(
                    {
                        "source": episode_text,
                        "reason": "multiple_quality_buckets_in_episode",
                        "quality_buckets": sorted(episode_buckets),
                    }
                )
                continue
            first = episode_movable[0]
            family = schema_families.get(first.schema_id)
            if family not in STRUCTURE_FAMILIES:
                conflicts.append(
                    {
                        "source": episode_text,
                        "reason": "missing_or_unknown_structure_family",
                        "schema_id": first.schema_id,
                        "family": family,
                    }
                )
                continue
            bucket = next(iter(episode_buckets))
            routed_root = quality_output_root(output_root, bucket)
            if routed_root is None:
                conflicts.append({"source": episode_text, "reason": "quality_status_not_movable"})
                continue
            source = Path(episode_text)
            try:
                relative_episode = source.relative_to(Path(first.batch_root))
            except ValueError:
                conflicts.append(
                    {
                        "source": episode_text,
                        "reason": "episode_outside_batch_root",
                        "batch_root": first.batch_root,
                    }
                )
                continue
            destination = (
                routed_root
                / family
                / first.schema_id
                / first.station_id
                / first.task_id
                / first.source_batch
                / relative_episode
            )
            plan.append(
                MoveRecord(
                    move_id=move_id(source, destination, first.schema_id),
                    source=str(source),
                    destination=str(destination),
                    family=family,
                    schema_id=first.schema_id,
                    station_id=first.station_id,
                    task_id=first.task_id,
                    source_batch=first.source_batch,
                    quality_status=aggregate_quality_status(episode_movable),
                    quality_bucket=bucket,
                    unit="episode",
                    hdf5_count=len(episode_movable),
                    total_bytes=sum(record.file_bytes for record in episode_movable),
                )
            )
    source_counts = Counter(record.source for record in plan)
    destination_counts = Counter(record.destination for record in plan)
    for value, count in source_counts.items():
        if count > 1:
            conflicts.append({"source": value, "reason": "duplicate_source_in_plan", "count": count})
    for value, count in destination_counts.items():
        if count > 1:
            conflicts.append({"destination": value, "reason": "duplicate_destination_in_plan", "count": count})
    return plan, conflicts


def read_move_plan(path: Path) -> list[MoveRecord]:
    result: list[MoveRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if "family" not in payload:
                    raise ValueError(
                        f"legacy move plan has no family field; rerun plan-move: {path}"
                    )
                payload.setdefault("quality_status", "warning")
                payload.setdefault("quality_bucket", "normal")
                result.append(MoveRecord(**payload))
    return result


def append_journal(path: Path, record: MoveRecord, status: str, error: str | None = None) -> None:
    payload = {**asdict(record), "status": status, "error": error, "timestamp": utc_now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_journal(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                result[payload["move_id"]] = payload
    return result


def verify_destination(record: MoveRecord) -> list[str]:
    destination = Path(record.destination)
    errors: list[str] = []
    parts = destination.parts
    if record.family not in parts:
        errors.append(f"destination missing family component: {record.family}")
    if record.schema_id not in parts:
        errors.append(f"destination missing schema component: {record.schema_id}")
    files = sorted(destination.rglob("trajectory.hdf5")) if destination.is_dir() else []
    if len(files) != record.hdf5_count:
        errors.append(f"hdf5_count expected={record.hdf5_count} actual={len(files)}")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes != record.total_bytes:
        errors.append(f"total_bytes expected={record.total_bytes} actual={total_bytes}")
    for path in files:
        try:
            with h5py.File(path, "r") as h5_file:
                logical = build_logical_schema(h5_file)
            schema_id, _ = canonical_hash(logical, "schema")
            if schema_id != record.schema_id:
                errors.append(f"schema mismatch: {path}: {schema_id}")
        except Exception as error:
            errors.append(f"unreadable: {path}: {error}")
    return errors


def execute_move_plan(plan: list[MoveRecord], journal: Path, *, dry_run: bool) -> int:
    latest = latest_journal(journal)
    failures = 0
    lock_path = journal.with_suffix(journal.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, record in enumerate(plan, start=1):
            source = Path(record.source)
            destination = Path(record.destination)
            print(f"[{index}/{len(plan)}] {source} -> {destination}")
            if latest.get(record.move_id, {}).get("status") == "verified":
                print("  skipped_verified")
                continue
            if dry_run:
                if not source.exists():
                    print("  conflict: source missing")
                    failures += 1
                elif destination.exists():
                    print("  conflict: destination exists")
                    failures += 1
                else:
                    existing_parent = nearest_existing_parent(destination.parent)
                    if source.stat().st_dev != existing_parent.stat().st_dev:
                        print("  conflict: cross-filesystem move")
                        failures += 1
                    else:
                        print("  planned")
                continue
            try:
                if not source.exists():
                    raise FileNotFoundError(f"source missing: {source}")
                if destination.exists():
                    raise FileExistsError(f"destination exists: {destination}")
                existing_parent = nearest_existing_parent(destination.parent)
                if source.stat().st_dev != existing_parent.stat().st_dev:
                    raise OSError("cross-filesystem move refused")
                append_journal(journal, record, "moving")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, destination)
                append_journal(journal, record, "moved")
                errors = verify_destination(record)
                if errors:
                    raise RuntimeError("; ".join(errors))
                append_journal(journal, record, "verified")
                print("  verified")
            except Exception as error:
                append_journal(journal, record, "failed", str(error))
                print(f"  failed: {error}", file=sys.stderr)
                failures += 1
    return 1 if failures else 0


def recover_plan(plan: list[MoveRecord], journal: Path, *, dry_run: bool) -> int:
    latest = latest_journal(journal)
    pending: list[MoveRecord] = []
    failures = 0
    for record in plan:
        source = Path(record.source)
        destination = Path(record.destination)
        status = latest.get(record.move_id, {}).get("status")
        if status == "verified":
            continue
        if source.exists() and not destination.exists():
            print(f"pending: {source}")
            pending.append(record)
        elif not source.exists() and destination.exists():
            errors = verify_destination(record)
            if errors:
                print(f"failed verification: {destination}: {'; '.join(errors)}", file=sys.stderr)
                if not dry_run:
                    append_journal(journal, record, "failed", "; ".join(errors))
                failures += 1
            else:
                print(f"recover verified: {destination}")
                if not dry_run:
                    append_journal(journal, record, "verified")
        elif source.exists() and destination.exists():
            print(f"conflict: source and destination both exist: {source}", file=sys.stderr)
            failures += 1
        else:
            print(f"missing: source and destination both absent: {source}", file=sys.stderr)
            failures += 1
    if pending:
        result = execute_move_plan(pending, journal, dry_run=dry_run)
        failures += int(result != 0)
    return 1 if failures else 0


def rollback_plan(plan: list[MoveRecord], journal: Path, *, dry_run: bool) -> int:
    latest = latest_journal(journal)
    failures = 0
    for record in reversed(plan):
        status = latest.get(record.move_id, {}).get("status")
        source = Path(record.source)
        destination = Path(record.destination)
        if status not in {"moved", "verified", "failed"} or not destination.exists():
            continue
        print(f"rollback {destination} -> {source}")
        if dry_run:
            continue
        try:
            if source.exists():
                raise FileExistsError(f"rollback source exists: {source}")
            existing_parent = nearest_existing_parent(source.parent)
            if destination.stat().st_dev != existing_parent.stat().st_dev:
                raise OSError("cross-filesystem rollback refused")
            source.parent.mkdir(parents=True, exist_ok=True)
            os.rename(destination, source)
            append_journal(journal, record, "rolled_back")
        except Exception as error:
            append_journal(journal, record, "rollback_failed", str(error))
            print(f"  failed: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def run_scan(args: argparse.Namespace) -> int:
    language_map = expected_language_by_task(args.mapping)
    records, logical, storage = scan_all(args.input, args.workers, language_map)
    catalog = build_schema_catalog(records, logical, storage)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_schema_reports(args.state_dir, records, catalog)
    print(f"hdf5 files: {len(records)}")
    print(f"readable: {sum(record.readable for record in records)}")
    print(f"unreadable: {sum(not record.readable for record in records)}")
    print(f"logical schemas: {catalog['schema_count']}")
    print(f"storage schemas: {catalog['storage_schema_count']}")
    for schema_id, entry in catalog["schemas"].items():
        quality = " ".join(
            f"{status}={entry['quality_counts'][status]}"
            for status in QUALITY_STATUSES
        )
        print(
            f"  {entry['structure_family']}/{schema_id}: files={entry['file_count']} "
            f"{quality}"
        )
    print(f"report: {args.state_dir / 'schema_report.md'}")
    return 1 if any(not record.readable for record in records) else 0


def run_list_schemas(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(f"logical schemas: {catalog['schema_count']}")
    for schema_id, entry in catalog["schemas"].items():
        print(
            f"{entry['structure_family']}/{schema_id} files={entry['file_count']} "
            f"arms={entry['capabilities']['arm_mode']} "
            f"stations={entry['stations']} tasks={entry['tasks']}"
        )
    return 0


def run_show_schema(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    entry = catalog["schemas"].get(args.schema_id)
    if entry is None:
        raise KeyError(f"unknown schema: {args.schema_id}")
    print(json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_compare_schema(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    left = catalog["schemas"][args.left]["descriptor"]
    right = catalog["schemas"][args.right]["descriptor"]
    print(json.dumps(schema_difference(left, right), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_plan_move(args: argparse.Namespace) -> int:
    records = read_hdf5_records(args.manifest)
    catalog = load_catalog(args.catalog)
    schema_families = {
        schema_id: entry["structure_family"]
        for schema_id, entry in catalog["schemas"].items()
    }
    plan, conflicts = build_move_plan(records, args.output, schema_families)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.state_dir / "move_plan.jsonl", (asdict(record) for record in plan))
    write_csv(args.state_dir / "move_plan.csv", [asdict(record) for record in plan])
    write_csv(args.state_dir / "move_conflicts.csv", conflicts)
    bucket_counts = Counter(record.quality_bucket for record in plan)
    print(f"moves: {len(plan)}")
    print(
        "quality buckets: "
        + " ".join(
            f"{bucket}={bucket_counts.get(bucket, 0)}"
            for bucket in ("normal", "review", "quarantine")
        )
    )
    print(f"conflicts: {len(conflicts)}")
    print(f"plan: {args.state_dir / 'move_plan.jsonl'}")
    return 1 if conflicts else 0


def run_export_schema_metadata(args: argparse.Namespace) -> int:
    """Write one family.json and schema.json per existing organized directory."""
    catalog = load_catalog(args.catalog)
    classification = build_schema_classification(catalog)
    missing: list[str] = []
    for family, family_entry in classification["families"].items():
        if not family_entry["schema_ids"]:
            continue
        family_dir = args.output / family
        if not family_dir.is_dir():
            missing.append(str(family_dir))
            continue
        write_json_atomic(
            family_dir / "family.json",
            {
                "classification_version": classification["classification_version"],
                "family": family,
                **family_entry,
            },
        )
        for schema_id in family_entry["schema_ids"]:
            schema_dir = family_dir / schema_id
            if not schema_dir.is_dir():
                missing.append(str(schema_dir))
                continue
            write_json_atomic(
                schema_dir / "schema.json",
                {
                    "fingerprint_version": catalog["fingerprint_version"],
                    "schema_id": schema_id,
                    "structure_family": family,
                    **catalog["schemas"][schema_id],
                },
            )
    if missing:
        for path in missing:
            print(f"missing organized directory: {path}", file=sys.stderr)
        return 1
    metadata_dir = args.output / "_metadata"
    write_json_atomic(metadata_dir / "schema_catalog.json", catalog)
    write_json_atomic(metadata_dir / "schema_classification.json", classification)
    print(f"schema metadata exported: {args.output}")
    return 0


def run_move(args: argparse.Namespace) -> int:
    plan = read_move_plan(args.plan)
    return execute_move_plan(plan, args.journal, dry_run=not args.execute)


def run_verify(args: argparse.Namespace) -> int:
    plan = read_move_plan(args.plan)
    failures = 0
    for record in plan:
        destination = Path(record.destination)
        if not destination.exists():
            print(f"missing: {destination}")
            failures += 1
            continue
        errors = verify_destination(record)
        if errors:
            print(f"failed: {destination}: {'; '.join(errors)}")
            failures += 1
        else:
            print(f"verified: {destination}")
    return 1 if failures else 0


def create_synthetic(path: Path, *, length: int, camera_length: int | None = None, include_pose: bool = True, compression: str | None = None, attribute_value: int = 1) -> None:
    camera_length = length if camera_length is None else camera_length
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("metadata/trajectory_length", data=length)
        h5_file.create_dataset("metadata/language_instruction", data="test")
        h5_file.create_dataset("camera_color_channel/camera_head", data="rgb")
        h5_file.create_dataset(
            "camera_observations/color_images/camera_head",
            data=np.arange(camera_length, dtype=np.uint8),
            compression=compression,
        )
        h5_file.create_dataset(
            "camera_observations/depth_images/camera_head",
            data=np.arange(length, dtype=np.uint16),
            compression=compression,
        )
        h5_file.create_dataset(
            "camera_observations/timestamp",
            data=np.arange(camera_length, dtype=np.float64) / 30.0,
        )
        joints = h5_file.create_dataset("puppet/arm_single_position_align/data", data=np.zeros((length, 6), dtype=np.float32))
        joints.attrs["version"] = attribute_value
        if include_pose:
            h5_file.create_dataset("puppet/end_effector_single_pose_align/data", data=np.zeros((length, 7), dtype=np.float32))


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        head_rgb = {
            "camera_observations/color_images/camera_head",
            "camera_observations/color_images/camera_left",
            "camera_observations/color_images/camera_right",
        }
        top_rgb = {
            "camera_observations/color_images/camera_top",
            "camera_observations/color_images/camera_left",
            "camera_observations/color_images/camera_right",
        }
        puppet_pose = {
            "puppet/end_effector_left_pose_align/data",
            "puppet/end_effector_right_pose_align/data",
        }
        force = {
            "force_observations/force_left_end_effector_align/data",
            "force_observations/force_right_end_effector_align/data",
        }
        assert classify_structure_family(head_rgb | puppet_pose | force)[0] == "A_head_pose_force"
        assert classify_structure_family(head_rgb | puppet_pose)[0] == "B_head_pose_no_force"
        assert classify_structure_family(top_rgb)[0] == "C_top_joint_only"
        assert classify_structure_family(head_rgb)[0] == "D_head_joint_only"
        assert classify_structure_family(puppet_pose | force)[0] == "E_missing_images"

        paths = {
            name: root / f"{name}.hdf5"
            for name in ["a", "b", "short", "missing", "missing_t", "stalled", "compressed", "attr"]
        }
        create_synthetic(paths["a"], length=3)
        create_synthetic(paths["b"], length=5)
        create_synthetic(paths["short"], length=5, camera_length=4)
        create_synthetic(paths["missing"], length=3, include_pose=False)
        create_synthetic(paths["missing_t"], length=8, camera_length=7)
        with h5py.File(paths["missing_t"], "a") as h5_file:
            del h5_file["metadata/trajectory_length"]
        create_synthetic(paths["stalled"], length=8)
        with h5py.File(paths["stalled"], "a") as h5_file:
            h5_file["camera_observations/timestamp"][-6:] = 1.0
        create_synthetic(paths["compressed"], length=3, compression="gzip")
        create_synthetic(paths["attr"], length=3, attribute_value=99)
        inspected = {name: inspect_hdf5(str(path), str(root), {})[0] for name, path in paths.items()}
        assert inspected["a"].schema_id == inspected["b"].schema_id
        assert inspected["a"].schema_id == inspected["short"].schema_id
        assert inspected["a"].quality_status == "passed"
        assert inspected["short"].quality_status == "quarantine"
        assert any(issue.startswith("length_mismatch:") for issue in inspected["short"].issues)
        assert inspected["missing_t"].quality_status == "quarantine"
        assert any(issue.startswith("inferred_length_mismatch:") for issue in inspected["missing_t"].issues)
        assert inspected["stalled"].quality_status == "quarantine"
        assert "camera_timestamp_terminal_stall:steps=5" in inspected["stalled"].issues
        assert inspected["a"].schema_id != inspected["missing"].schema_id
        assert inspected["a"].schema_id == inspected["compressed"].schema_id
        assert inspected["a"].storage_id != inspected["compressed"].storage_id
        assert inspected["a"].schema_id == inspected["attr"].schema_id

        incremental_path = root / "incremental" / "new" / "insert_hose" / "batch_1" / "episode_1" / "data"
        incremental_path.mkdir(parents=True)
        incremental_hdf5 = incremental_path / "trajectory.hdf5"
        create_synthetic(incremental_hdf5, length=3)
        station, task, batch, episode, batch_root, episode_root = path_metadata(
            root / "incremental", incremental_hdf5
        )
        assert station == UNKNOWN_STATION
        assert task == "insert_hose" and batch == "batch_1" and episode == "episode_1"
        assert batch_root == root / "incremental" / "new" / "insert_hose" / "batch_1"
        assert episode_root == incremental_path.parent

        staging = root / "staging" / "tienyi_9" / "plug_cables" / "batch" / "episode" / "data"
        staging.mkdir(parents=True)
        create_synthetic(staging / "trajectory.hdf5", length=3)
        records, logical, storage = scan_all(root / "staging", 1, {})
        catalog = build_schema_catalog(records, logical, storage)
        report_dir = root / "reports"
        report_dir.mkdir()
        write_schema_reports(report_dir, records, catalog)
        classification = json.loads(
            (report_dir / "schema_classification.json").read_text(encoding="utf-8")
        )
        assert classification["families"]["E_missing_images"]["file_count"] == 1
        mapping_text = (report_dir / "file_to_schema.csv").read_text(encoding="utf-8-sig")
        assert "structure_family" in mapping_text and "E_missing_images" in mapping_text
        schema_families = {
            schema_id: entry["structure_family"]
            for schema_id, entry in catalog["schemas"].items()
        }
        plan, conflicts = build_move_plan(records, root / "final", schema_families)
        assert len(plan) == 1 and not conflicts
        assert plan[0].family == "E_missing_images"
        assert f"/{plan[0].family}/{plan[0].schema_id}/" in plan[0].destination
        journal = root / "journal.jsonl"
        assert execute_move_plan(plan, journal, dry_run=True) == 0
        assert Path(plan[0].source).exists() and not Path(plan[0].destination).exists()
        assert execute_move_plan(plan, journal, dry_run=False) == 0
        assert not Path(plan[0].source).exists() and Path(plan[0].destination).exists()
        catalog_path = root / "catalog.json"
        write_json_atomic(catalog_path, catalog)
        assert run_export_schema_metadata(
            argparse.Namespace(catalog=catalog_path, output=root / "final")
        ) == 0
        assert (root / "final" / plan[0].family / "family.json").is_file()
        assert (
            root / "final" / plan[0].family / plan[0].schema_id / "schema.json"
        ).is_file()
        assert rollback_plan(plan, journal, dry_run=False) == 0
        assert Path(plan[0].source).exists() and not Path(plan[0].destination).exists()

        # A batch containing normal and quarantined episodes must never move as
        # one directory; split it and route only the bad episode to quarantine.
        mixed_root = root / "mixed"
        mixed_batch = mixed_root / "tienyi_9" / "plug_cables" / "mixed_batch" / "success_episodes"
        normal_data = mixed_batch / "episode_normal" / "data"
        bad_data = mixed_batch / "episode_bad" / "data"
        normal_data.mkdir(parents=True)
        bad_data.mkdir(parents=True)
        create_synthetic(normal_data / "trajectory.hdf5", length=8)
        create_synthetic(bad_data / "trajectory.hdf5", length=8)
        with h5py.File(bad_data / "trajectory.hdf5", "a") as h5_file:
            h5_file["camera_observations/timestamp"][-6:] = 1.0
        mixed_records, mixed_logical, mixed_storage = scan_all(mixed_root, 1, {})
        mixed_catalog = build_schema_catalog(mixed_records, mixed_logical, mixed_storage)
        mixed_families = {
            schema_id: entry["structure_family"]
            for schema_id, entry in mixed_catalog["schemas"].items()
        }
        mixed_plan, mixed_conflicts = build_move_plan(
            mixed_records, root / "mixed_final", mixed_families
        )
        assert len(mixed_plan) == 2 and not mixed_conflicts
        assert {record.unit for record in mixed_plan} == {"episode"}
        assert {record.quality_bucket for record in mixed_plan} == {"normal", "quarantine"}
        quarantined = next(record for record in mixed_plan if record.quality_bucket == "quarantine")
        assert "/_quarantine/" in quarantined.destination
    print("organize_hdf5 self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--input", type=Path, required=True)
    scan.add_argument("--state-dir", type=Path, required=True)
    scan.add_argument("--mapping", type=Path)
    scan.add_argument("--workers", type=int, default=4)
    scan.set_defaults(handler=run_scan)

    listing = subparsers.add_parser("list-schemas")
    listing.add_argument("--catalog", type=Path, required=True)
    listing.set_defaults(handler=run_list_schemas)

    show = subparsers.add_parser("show-schema")
    show.add_argument("--catalog", type=Path, required=True)
    show.add_argument("--schema-id", required=True)
    show.set_defaults(handler=run_show_schema)

    compare = subparsers.add_parser("compare-schema")
    compare.add_argument("--catalog", type=Path, required=True)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.set_defaults(handler=run_compare_schema)

    plan_move = subparsers.add_parser("plan-move")
    plan_move.add_argument("--manifest", type=Path, required=True)
    plan_move.add_argument("--catalog", type=Path, required=True)
    plan_move.add_argument("--output", type=Path, required=True)
    plan_move.add_argument("--state-dir", type=Path, required=True)
    plan_move.set_defaults(handler=run_plan_move)

    move = subparsers.add_parser("move")
    move.add_argument("--plan", type=Path, required=True)
    move.add_argument("--journal", type=Path, required=True)
    move.add_argument("--execute", action="store_true")
    move.set_defaults(handler=run_move)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.set_defaults(handler=run_verify)

    export_metadata = subparsers.add_parser("export-schema-metadata")
    export_metadata.add_argument("--catalog", type=Path, required=True)
    export_metadata.add_argument("--output", type=Path, required=True)
    export_metadata.set_defaults(handler=run_export_schema_metadata)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--plan", type=Path, required=True)
    recover.add_argument("--journal", type=Path, required=True)
    recover.add_argument("--execute", action="store_true")
    recover.set_defaults(
        handler=lambda args: recover_plan(
            read_move_plan(args.plan), args.journal, dry_run=not args.execute
        )
    )

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--plan", type=Path, required=True)
    rollback.add_argument("--journal", type=Path, required=True)
    rollback.add_argument("--execute", action="store_true")
    rollback.set_defaults(handler=lambda args: rollback_plan(read_move_plan(args.plan), args.journal, dry_run=not args.execute))

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(handler=lambda _: run_self_test())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be >= 1")
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
