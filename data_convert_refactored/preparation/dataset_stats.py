"""加载、适配、生成并校验action/proprio共享dataset statistics。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ActionEncoding, ActionScale, ActionSource, StatsMode
from .actions import EncodedEpisode
from .episode import Episode
from .proprio import raw_proprio_for_episode


@dataclass(frozen=True)
class PreparedDatasetStats:
    """同时保留磁盘原始stats和可能经过维度适配的运行时stats。"""

    source_path: Path
    source: dict
    effective: dict
    source_sha256: str


def load_dataset_stats_official(path: Path) -> dict:
    """复用collect_data_cosmos.py使用的官方stats loader。"""
    from cosmos_policy.experiments.robot.cosmos_utils import load_dataset_stats

    return load_dataset_stats(str(path))


def compute_action_stats(encoded_episodes: list[EncodedEpisode]) -> dict[str, np.ndarray]:
    """在全部训练episode的第一层action上计算逐通道min/max。"""
    if not encoded_episodes:
        raise ValueError("cannot compute action statistics without episodes")
    action_dim = encoded_episodes[0].action_dim
    for item in encoded_episodes:
        if item.action_dim != action_dim:
            raise ValueError(
                f"mixed action dimensions: expected {action_dim}, got {item.action_dim}"
            )
    actions = np.concatenate([item.actions for item in encoded_episodes], axis=0).astype(
        np.float32
    )
    actions_min = actions.min(axis=0)
    actions_max = actions.max(axis=0)
    constant_mask = np.abs(actions_max - actions_min) < 1e-8
    if np.any(constant_mask):
        indices = np.flatnonzero(constant_mask).tolist()
        raise ValueError(
            "collect_data_cosmos min/max action normalization is undefined for constant "
            f"channels {indices}; actions_max equals actions_min"
        )
    return {
        "actions_min": actions_min,
        "actions_max": actions_max,
        "actions_constant_mask": constant_mask,
        "num_action_samples": np.asarray(len(actions), dtype=np.int64),
    }


def compute_shared_stats(
    episodes: list[Episode], encoded_episodes: list[EncodedEpisode]
) -> dict[str, np.ndarray]:
    """一次性计算训练集合共享的action/proprio stats，供成功和失败数据共同使用。"""
    if not episodes or len(episodes) != len(encoded_episodes):
        raise ValueError("episodes and encoded episodes must be non-empty and aligned")
    actions = np.concatenate([item.actions for item in encoded_episodes], axis=0).astype(np.float32)
    proprio = np.concatenate([raw_proprio_for_episode(item) for item in episodes], axis=0)
    result = {
        "actions_min": actions.min(axis=0),
        "actions_max": actions.max(axis=0),
        "proprio_min": proprio.min(axis=0),
        "proprio_max": proprio.max(axis=0),
        "num_action_samples": np.asarray(len(actions), dtype=np.int64),
        "num_proprio_samples": np.asarray(len(proprio), dtype=np.int64),
    }
    for prefix in ("actions", "proprio"):
        span = result[f"{prefix}_max"] - result[f"{prefix}_min"]
        constant = np.flatnonzero(np.abs(span) < 1e-8)
        if len(constant):
            raise ValueError(
                f"official Cosmos min/max normalization is undefined for constant {prefix} "
                f"channels {constant.tolist()}"
            )
        result[f"{prefix}_constant_mask"] = np.zeros_like(span, dtype=bool)
    return result


def stats_sha256(path: Path) -> str:
    """计算stats文件摘要，用于阻止误用不同版本的归一化参数。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset_stats(
    path: Path,
    mode: StatsMode,
    action_encoding: ActionEncoding,
    action_dimension: int,
) -> PreparedDatasetStats:
    """只加载一次stats，并根据旋转编码生成实际参与归一化的effective版本。"""
    source = load_dataset_stats_official(path)
    effective = (
        effective_official_stats(source, action_encoding, action_dimension)
        if mode is StatsMode.OFFICIAL
        else source
    )
    return PreparedDatasetStats(
        source_path=path,
        source=source,
        effective=effective,
        source_sha256=stats_sha256(path),
    )


def inspect_stats_ranges(
    stats: dict, episodes: list[Episode], encoded_episodes: list[EncodedEpisode]
) -> dict:
    """统计每条episode超出训练stats范围的通道和值数量，不修改输入。"""
    tolerance = 1e-6
    report = {"out_of_range_policy": "reject", "episodes": []}
    for episode, encoded in zip(episodes, encoded_episodes, strict=True):
        episode_report = {"episode": str(episode.path)}
        for name, values in (
            ("actions", encoded.actions),
            ("proprio", raw_proprio_for_episode(episode)),
        ):
            lower = np.asarray(stats[f"{name}_min"], dtype=np.float32)
            upper = np.asarray(stats[f"{name}_max"], dtype=np.float32)
            below = np.flatnonzero(np.any(values < lower - tolerance, axis=0))
            above = np.flatnonzero(np.any(values > upper + tolerance, axis=0))
            below_mask = values < lower - tolerance
            above_mask = values > upper + tolerance
            episode_report[name] = {
                "below_channels": below.tolist(),
                "above_channels": above.tolist(),
                "below_values": int(np.count_nonzero(below_mask)),
                "above_values": int(np.count_nonzero(above_mask)),
            }
        report["episodes"].append(episode_report)
    return report


def validate_or_warn_stats_ranges(
    stats: dict,
    episodes: list[Episode],
    encoded_episodes: list[EncodedEpisode],
    mode: StatsMode,
) -> dict:
    """generated模式拒绝越界；official模式与官方公式一致，不裁剪但明确警告。"""
    report = inspect_stats_ranges(stats, episodes, encoded_episodes)
    violations = []
    for episode in report["episodes"]:
        for name in ("actions", "proprio"):
            item = episode[name]
            if item["below_values"] or item["above_values"]:
                violations.append(
                    f"{episode['episode']}: {name} below={item['below_channels']} "
                    f"above={item['above_channels']}"
                )
    if mode is StatsMode.GENERATED and violations:
        raise ValueError("values exceed generated stats:\n" + "\n".join(violations))
    report["out_of_range_policy"] = "allow_and_warn" if mode is StatsMode.OFFICIAL else "reject"
    if violations:
        print("[WARNING] official stats range exceeded; normalization is not clipped:")
        print("\n".join(f"  {item}" for item in violations))
    return report


def validate_official_stats(stats: dict, action_dimension: int, proprio_dimension: int) -> None:
    """只校验collect_data_cosmos.py归一化实际需要的字段和维度。"""
    required = {
        "actions_min": action_dimension,
        "actions_max": action_dimension,
        "proprio_min": proprio_dimension,
        "proprio_max": proprio_dimension,
    }
    for key, dimension in required.items():
        if key not in stats:
            raise ValueError(f"official stats missing {key}")
        value = np.asarray(stats[key], dtype=np.float32)
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError(f"official stats {key} must be finite shape ({dimension},)")
        stats[key] = value
    for prefix in ("actions", "proprio"):
        if np.any(stats[f"{prefix}_max"] - stats[f"{prefix}_min"] <= 0):
            raise ValueError(f"official stats contain constant or reversed {prefix} channels")


def effective_official_stats(
    stats: dict,
    action_encoding: ActionEncoding,
    action_dimension: int,
) -> dict:
    """在内存中把兼容的双臂14D Euler stats适配为20D rotation-6D stats。

    平移和夹爪沿用原stats；rotation-6D六个通道使用几何范围[-1,1]。源JSON不改写，
    `source`仍保留原14D内容，`effective`用于当前转换。
    """
    action_min = np.asarray(stats.get("actions_min"), dtype=np.float32)
    action_max = np.asarray(stats.get("actions_max"), dtype=np.float32)
    if action_min.shape == (action_dimension,) and action_max.shape == (action_dimension,):
        return stats
    compatible = (
        action_encoding is ActionEncoding.ROTATION_6D
        and action_dimension == 20
        and action_min.shape == (14,)
        and action_max.shape == (14,)
    )
    if not compatible:
        return stats

    def expand(values: np.ndarray, rotation_value: float) -> np.ndarray:
        return np.concatenate(
            (
                values[0:3], np.full(6, rotation_value), values[6:7],
                values[7:10], np.full(6, rotation_value), values[13:14],
            )
        ).astype(np.float32)

    output = dict(stats)
    output["actions_min"] = expand(action_min, -1.0)
    output["actions_max"] = expand(action_max, 1.0)
    output["rotation_normalization"] = "identity_fixed_minus1_plus1"
    output["action_stats_runtime_adapter"] = "dual_arm_legacy_euler_14d_to_rotation6d_20d"
    return output


def write_shared_stats(
    output: Path,
    stats: dict[str, np.ndarray],
    *,
    inputs: list[Path],
    robot_type: str,
    action_source: ActionSource,
    action_encoding: ActionEncoding,
    action_scale: ActionScale,
    action_dimension: int,
    proprio_dimension: int,
) -> tuple[Path, Path]:
    """写共享stats及语义sidecar；sidecar固定action来源、编码、scale和维度。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stats.items()
    }
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    sidecar = output.with_suffix(".metadata.json")
    metadata = {
        "format_version": 1,
        "statistics_sha256": stats_sha256(output),
        "input_directories": [str(path.resolve()) for path in inputs],
        "robot_type": robot_type,
        "action_source": action_source.value,
        "action_encoding": action_encoding.value,
        "translation_scale": action_scale.translation_m,
        "rotation_scale": (
            action_scale.rotation_rad if action_encoding is ActionEncoding.LEGACY_EULER else None
        ),
        "gripper_scale": action_scale.gripper,
        "last_action_padding_strategy": (
            "repeat_last_valid" if action_source is ActionSource.PUPPET_NEXT_FRAME else "none"
        ),
        "action_dimension": action_dimension,
        "proprio_dimension": proprio_dimension,
    }
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output, sidecar


def validate_external_stats(
    stats: dict,
    stats_path: Path,
    *,
    robot_type: str,
    action_source: ActionSource,
    action_encoding: ActionEncoding,
    action_scale: ActionScale,
    action_dimension: int,
    proprio_dimension: int,
) -> dict:
    """校验generated stats及sidecar与当前转换语义完全一致。"""
    required = {
        "actions_min": action_dimension,
        "actions_max": action_dimension,
        "proprio_min": proprio_dimension,
        "proprio_max": proprio_dimension,
    }
    for key, dimension in required.items():
        if key not in stats:
            raise ValueError(f"external stats missing {key}")
        value = np.asarray(stats[key], dtype=np.float32)
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError(f"external stats {key} must be finite shape ({dimension},)")
        stats[key] = value
    for prefix in ("actions", "proprio"):
        if np.any(stats[f"{prefix}_max"] - stats[f"{prefix}_min"] <= 1e-8):
            raise ValueError(f"external stats contain constant or reversed {prefix} channels")

    sidecar = stats_path.with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"external stats semantic metadata is missing: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "statistics_sha256": stats_sha256(stats_path),
        "robot_type": robot_type,
        "action_source": action_source.value,
        "action_encoding": action_encoding.value,
        "translation_scale": action_scale.translation_m,
        "rotation_scale": (
            action_scale.rotation_rad if action_encoding is ActionEncoding.LEGACY_EULER else None
        ),
        "gripper_scale": action_scale.gripper,
        "last_action_padding_strategy": (
            "repeat_last_valid" if action_source is ActionSource.PUPPET_NEXT_FRAME else "none"
        ),
        "action_dimension": action_dimension,
        "proprio_dimension": proprio_dimension,
    }
    mismatches = [
        f"{key}: stats={metadata.get(key)!r}, conversion={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError("external stats semantics mismatch:\n" + "\n".join(mismatches))
    return metadata
