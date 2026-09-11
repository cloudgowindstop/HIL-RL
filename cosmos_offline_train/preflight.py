"""Fast dataset audit without loading Cosmos model or episode tensors."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_spec import action_spec_from_data
from .config import (
    ConfigError,
    resolve_pre_split_directories,
    validate_preflight_config,
)


METADATA_NAME = "cosmos_dataset_metadata.json"


@dataclass
class PreflightReport:
    root: Path
    datasets: int = 0
    episodes: int = 0
    raw_episodes: int = 0
    parquet_files: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.datasets > 0 and not self.errors


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"无法读取 JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"JSON 根节点必须是 object: {path}")
    return value


def _dataset_roots(root: Path) -> list[Path]:
    if (root / METADATA_NAME).is_file():
        return [root]
    return sorted(path.parent for path in root.rglob(METADATA_NAME))


def run_preflight(config: dict[str, Any]) -> PreflightReport:
    root = validate_preflight_config(config)
    report = PreflightReport(root=root)
    data = config["data"]
    spec = action_spec_from_data(data)
    if data["layout"] == "pre_split":
        train, validation, test = resolve_pre_split_directories(data, root)
        expected_roots = [path for path in (train, validation, test) if path is not None]
        roots = [path for path in expected_roots if (path / METADATA_NAME).is_file()]
        for path in expected_roots:
            if not (path / METADATA_NAME).is_file():
                report.errors.append(f"缺少 {path / METADATA_NAME}")
    else:
        roots = _dataset_roots(root)
    if not roots:
        raw_count = sum(1 for _ in root.rglob("trajectory.hdf5"))
        report.raw_episodes = raw_count
        if raw_count:
            report.errors.append(
                f"发现 {raw_count} 个 raw trajectory.hdf5，但没有 {METADATA_NAME}；"
                "请先完成Cosmos转换或运行旧数据适配工具。"
            )
        else:
            report.errors.append(f"没有找到 {METADATA_NAME}: {root}")
        return report

    expected_order = spec.order(number_of_arms=2)
    statistics_hashes: set[str] = set()
    configured_stats = data.get("statistics_path")
    configured_path = (
        Path(str(configured_stats)).expanduser().resolve() if configured_stats else None
    )
    for dataset_root in roots:
        report.datasets += 1
        metadata = _read_json(dataset_root / METADATA_NAME)
        action = metadata.get("action", {})
        relative = dataset_root.relative_to(root) if dataset_root != root else Path(".")
        prefix = str(relative)

        checks = {
            "action.encoding": (action.get("encoding"), spec.encoding),
            "action.dimension": (action.get("dimension"), spec.dimension),
            "action.rotation_representation": (
                action.get("rotation_representation"), spec.rotation_representation
            ),
            "action.pose_semantics": (
                action.get("pose_semantics"), "local_delta_inv_current_times_target"
            ),
            "action.chunk_size": (action.get("chunk_size"), 16),
            "action.order": (action.get("order"), expected_order),
        }
        if spec.encoding == "legacy_euler":
            checks["action.rotation_scale"] = (
                action.get("rotation_scale"), float(data["rotation_scale"])
            )
        for name, (actual, expected) in checks.items():
            if actual != expected:
                report.errors.append(f"{prefix}: {name}={actual!r}，期望 {expected!r}")

        info_path = dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            report.errors.append(f"{prefix}: 缺少 meta/info.json")
            continue
        info = _read_json(info_path)
        episodes = int(info.get("total_episodes", 0))
        parquet_count = sum(1 for _ in (dataset_root / "data").rglob("episode_*.parquet"))
        report.episodes += episodes
        report.parquet_files += parquet_count
        if episodes <= 0:
            report.errors.append(f"{prefix}: total_episodes={episodes}")
        if parquet_count != episodes:
            report.errors.append(
                f"{prefix}: parquet={parquet_count} 与 total_episodes={episodes} 不一致"
            )
        info_action = info.get("features", {}).get("action", {}).get("shape")
        if info_action != [spec.dimension]:
            report.errors.append(
                f"{prefix}: meta/info.json action shape={info_action!r}，期望 [{spec.dimension}]"
            )

        local_stats = dataset_root / "dataset_statistics.json"
        stats_path = local_stats if local_stats.is_file() else configured_path
        if stats_path is None or not stats_path.is_file():
            report.errors.append(f"{prefix}: 缺少可用的dataset statistics")
        else:
            stats = _read_json(stats_path)
            statistics_hashes.add(hashlib.sha256(stats_path.read_bytes()).hexdigest())
            action_min = stats.get("actions_min", stats.get("action_min"))
            action_max = stats.get("actions_max", stats.get("action_max"))
            if not isinstance(action_min, list) or len(action_min) != spec.dimension:
                report.errors.append(
                    f"{prefix}: action min 统计长度不是 {spec.dimension}"
                )
            if not isinstance(action_max, list) or len(action_max) != spec.dimension:
                report.errors.append(
                    f"{prefix}: action max 统计长度不是 {spec.dimension}"
                )

    if len(statistics_hashes) > 1:
        report.errors.append(
            f"发现 {len(statistics_hashes)} 份不同 action statistics；所有 success/failure/group 必须共享 train-only statistics"
        )

    if configured_path is not None:
        if not configured_path.is_file():
            report.errors.append(f"配置的 statistics_path 不存在: {configured_path}")
        else:
            configured_hash = hashlib.sha256(configured_path.read_bytes()).hexdigest()
            if statistics_hashes and configured_hash not in statistics_hashes:
                report.errors.append(
                    "data.statistics_path 与数据集内 dataset_statistics.json 不一致；"
                    "训练归一化与数据转换必须使用同一份 train-only statistics"
                )

    return report


def print_report(report: PreflightReport) -> None:
    print(f"[PREFLIGHT] root={report.root}")
    print(f"[PREFLIGHT] datasets={report.datasets}")
    print(f"[PREFLIGHT] raw_episodes={report.raw_episodes}")
    print(f"[PREFLIGHT] episodes={report.episodes}")
    print(f"[PREFLIGHT] parquet_files={report.parquet_files}")
    for warning in report.warnings:
        print(f"[WARN] {warning}")
    for error in report.errors:
        print(f"[ERROR] {error}")
    print(f"[PREFLIGHT] status={'PASS' if report.passed else 'FAIL'}")
