"""单数据集断点续转：输入清单、完整性检查、事务记录和追加writer。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ConversionConfig


MANIFEST_NAME = "conversion_manifest.json"
CHECKPOINT_NAME = "conversion_checkpoint.json"
TRANSACTION_NAME = "conversion_transaction.json"
_PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")


@dataclass(frozen=True)
class ResumeState:
    """已安全提交的数据集尾部。"""

    start_episode: int
    total_frames: int
    source_episode_count: int
    resumed: bool


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_cosmos_config(config: ConversionConfig) -> Path:
    path = config.cosmos_config_path.expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_manifest(config: ConversionConfig, source_paths: Iterable[Path]) -> dict[str, Any]:
    """只包含影响数据语义的配置；GPU、batch和监控参数允许恢复时改变。"""
    semantic_config = {
        "task": config.task,
        "episode_outcome": config.episode_outcome.value,
        "stats_mode": config.stats_mode.value,
        "stats_path": str(config.stats_path.resolve()),
        "stats_sha256": _sha256(config.stats_path),
        "action_encoding": config.action_encoding.value,
        "action_source": config.action_source.value,
        "action_scale": asdict(config.action_scale),
        "kinematics_config": (
            str(config.kinematics_config.resolve()) if config.kinematics_config else None
        ),
        "kinematics_sha256": _sha256(config.kinematics_config),
        "t5_embeddings": str(config.t5_embeddings.resolve()) if config.t5_embeddings else None,
        "t5_sha256": _sha256(config.t5_embeddings),
        "skip_t5": config.skip_t5,
        "cosmos_config": str(_resolved_cosmos_config(config)),
        "cosmos_config_sha256": _sha256(_resolved_cosmos_config(config)),
        "image_size": config.image_size,
        "use_jpeg_compression": config.use_jpeg_compression,
        "trained_with_image_aug": config.trained_with_image_aug,
        "camera_state": config.camera_state.value,
        "wrist_crop_mode": config.wrist_crop_mode,
        "wrist_crop_fraction": config.wrist_crop_fraction,
        "wrist_left_center_offset_x": config.wrist_left_center_offset_x,
        "wrist_right_center_offset_x": config.wrist_right_center_offset_x,
        "head_crop_top_pixels": config.head_crop_top_pixels,
    }
    payload = {
        "version": 1,
        "semantic_config": semantic_config,
        "sources": [_source_record(path) for path in source_paths],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _parquet_files(output_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in output_dir.glob("data/chunk-*/episode_*.parquet"):
        match = _PARQUET_RE.search(path.name)
        if match:
            index = int(match.group(1))
            if index in result:
                raise RuntimeError(f"duplicate Parquet episode index {index}: {path}")
            result[index] = path
    return result


def _valid_parquet_container(path: Path) -> bool:
    if path.stat().st_size < 8:
        return False
    with path.open("rb") as file:
        head = file.read(4)
        file.seek(-4, os.SEEK_END)
        tail = file.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def _metadata_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _truncate(path: Path, size: int) -> None:
    if not path.exists() and size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.truncate(size)
        file.flush()
        os.fsync(file.fileno())


def _transaction_parquet_path(output_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return output_dir / template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _recover_interrupted_transaction(output_dir: Path) -> None:
    """回滚未完成save_episode；完整落盘但未更新checkpoint时保留结果。"""
    transaction_path = output_dir / "meta" / TRANSACTION_NAME
    if not transaction_path.is_file():
        return
    with transaction_path.open(encoding="utf-8") as file:
        transaction = json.load(file)
    if transaction.get("status") != "writing" or "metadata_before_save" not in transaction:
        return

    before = transaction["metadata_before_save"]
    prior_count = int(before["info"]["total_episodes"])
    episode_index = int(transaction["episode_index"])
    if episode_index != prior_count:
        raise RuntimeError(
            f"transaction episode {episode_index} does not match prior count {prior_count}"
        )

    current_info_path = output_dir / "meta/info.json"
    current_info = json.loads(current_info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonlines(output_dir / "meta/episodes.jsonl")
    episode_stats = _read_jsonlines(output_dir / "meta/episodes_stats.jsonl")
    parquet_path = _transaction_parquet_path(output_dir, before["info"], episode_index)
    fully_committed = (
        int(current_info.get("total_episodes", 0)) == prior_count + 1
        and len(episodes) == prior_count + 1
        and len(episode_stats) == prior_count + 1
        and parquet_path.is_file()
        and _valid_parquet_container(parquet_path)
    )
    if fully_committed:
        transaction["status"] = "committed_recovered"
        _atomic_json(transaction_path, transaction)
        return

    if parquet_path.exists():
        parquet_path.unlink()
    _atomic_json(current_info_path, before["info"])
    for name, size in before["jsonl_sizes"].items():
        _truncate(output_dir / "meta" / name, int(size))
    transaction["status"] = "rolled_back"
    transaction["recovery_reason"] = "save_episode was not fully committed"
    _atomic_json(transaction_path, transaction)


def inspect_committed_output(output_dir: Path) -> tuple[int, int]:
    """严格检查LeRobot metadata和Parquet一一对应，返回episode数和总帧数。"""
    info_path = output_dir / "meta/info.json"
    if not info_path.is_file():
        orphaned = list(output_dir.glob("data/chunk-*/episode_*.parquet"))
        if orphaned:
            raise RuntimeError(f"Parquet files exist without meta/info.json: {orphaned}")
        return 0, 0
    _recover_interrupted_transaction(output_dir)
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    episodes = _read_jsonlines(output_dir / "meta/episodes.jsonl")
    episode_stats = _read_jsonlines(output_dir / "meta/episodes_stats.jsonl")
    parquet = _parquet_files(output_dir)

    # Parquet先于metadata落盘。仅自动清理确定未提交且为0字节的下一条文件。
    next_file = parquet.get(total_episodes)
    if next_file is not None and next_file.stat().st_size == 0:
        next_file.unlink()
        parquet.pop(total_episodes)

    counts = {
        "info.total_episodes": total_episodes,
        "episodes.jsonl": len(episodes),
        "episodes_stats.jsonl": len(episode_stats),
        "parquet": len(parquet),
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"inconsistent resume tail: {counts}")
    expected = list(range(total_episodes))
    actual = sorted(parquet)
    if actual != expected:
        raise RuntimeError(f"non-contiguous Parquet indexes: expected={expected}, actual={actual}")
    episode_indexes = [int(record.get("episode_index", -1)) for record in episodes]
    if episode_indexes != expected:
        raise RuntimeError(
            f"non-contiguous episodes.jsonl indexes: expected={expected}, actual={episode_indexes}"
        )
    bad = [str(parquet[index]) for index in expected if not _valid_parquet_container(parquet[index])]
    if bad:
        raise RuntimeError(f"empty or incomplete Parquet files: {bad}")
    lengths = [int(record.get("length", -1)) for record in episodes]
    if any(length < 0 for length in lengths) or sum(lengths) != total_frames:
        raise RuntimeError(
            f"frame count mismatch: info.total_frames={total_frames}, episode lengths={sum(lengths)}"
        )
    return total_episodes, total_frames


def prepare_resume(
    config: ConversionConfig,
    source_paths: list[Path],
    *,
    enabled: bool,
) -> ResumeState:
    """创建/核对manifest，确定需要跳过的已提交episode数量。"""
    manifest = build_manifest(config, source_paths)
    manifest_path = config.output_dir / "meta" / MANIFEST_NAME
    info_exists = (config.output_dir / "meta/info.json").is_file()
    if info_exists and not enabled:
        raise FileExistsError(
            f"output dataset already exists: {config.output_dir}; use --resume after inspection"
        )

    committed, total_frames = inspect_committed_output(config.output_dir)
    if committed > len(source_paths):
        raise RuntimeError(
            f"output has {committed} episodes but input manifest has only {len(source_paths)}"
        )
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("fingerprint_sha256") != manifest["fingerprint_sha256"]:
            raise RuntimeError(
                "resume manifest mismatch: input files or semantic conversion config changed"
            )
    else:
        # 兼容监控功能加入前已经产生、但尾部完整的转换结果。
        manifest["bootstrapped_from_existing_episodes"] = committed
        _atomic_json(manifest_path, manifest)

    _atomic_json(
        config.output_dir / "meta" / CHECKPOINT_NAME,
        {
            "status": "ready",
            "completed_episodes": committed,
            "total_frames": total_frames,
            "source_episode_count": len(source_paths),
        },
    )
    return ResumeState(committed, total_frames, len(source_paths), committed > 0)


def begin_episode_transaction(output_dir: Path, episode_index: int, source_path: Path) -> None:
    info_path = output_dir / "meta/info.json"
    if not info_path.is_file():
        raise RuntimeError(f"cannot start episode transaction without {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if int(info.get("total_episodes", -1)) != episode_index:
        raise RuntimeError(
            f"transaction index {episode_index} != metadata total_episodes "
            f"{info.get('total_episodes')}"
        )
    _atomic_json(
        output_dir / "meta" / TRANSACTION_NAME,
        {
            "status": "writing",
            "episode_index": episode_index,
            "source_path": str(source_path.resolve()),
            "pid": os.getpid(),
            "metadata_before_save": {
                "info": info,
                "jsonl_sizes": {
                    name: _metadata_size(output_dir / "meta" / name)
                    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")
                },
            },
        },
    )


def commit_episode_transaction(
    output_dir: Path,
    episode_index: int,
    source_path: Path,
    written: int,
    total_frames: int,
    source_episode_count: int,
) -> None:
    value = {
        "status": "committed",
        "episode_index": episode_index,
        "source_path": str(source_path.resolve()),
        "frames_written": written,
        "completed_episodes": episode_index + 1,
        "total_frames": total_frames,
        "source_episode_count": source_episode_count,
        "pid": os.getpid(),
    }
    _atomic_json(output_dir / "meta" / TRANSACTION_NAME, value)
    _atomic_json(output_dir / "meta" / CHECKPOINT_NAME, value)


def open_dataset_for_append(output_dir: Path, repo_id: str):
    """加载metadata并创建空内存writer；不读取历史Parquet。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import get_safe_default_codec

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id, root=output_dir)
    dataset.repo_id = dataset.meta.repo_id
    dataset.root = dataset.meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.batch_encoding_size = 1
    dataset.episodes_since_last_encoding = 0
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = get_safe_default_codec()
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset
