"""发现HDF5 episode，并读取一次低维机器人数据供后续阶段复用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class ArmTrajectory:
    """单臂对齐轨迹；pose固定为(T,7)，其余可选字段首维必须同为T。"""

    puppet_pose_xyzw: np.ndarray
    puppet_gripper: np.ndarray
    puppet_joints: np.ndarray | None = None
    master_pose_xyzw: np.ndarray | None = None
    master_gripper: np.ndarray | None = None
    master_joints: np.ndarray | None = None


@dataclass(frozen=True)
class Episode:
    """单条轨迹的低维内存表示；图像不在此处加载，避免长期占用CPU内存。"""

    path: Path
    length: int
    camera_names: tuple[str, ...]
    left: ArmTrajectory
    right: ArmTrajectory | None = None
    source_fps: int = 0

    @property
    def is_dual_arm(self) -> bool:
        return self.right is not None


def _optional(h5_file: h5py.File, path: str) -> np.ndarray | None:
    return np.asarray(h5_file[path][:]) if path in h5_file else None


def _required(h5_file: h5py.File, path: str) -> np.ndarray:
    if path not in h5_file:
        raise ValueError(f"required HDF5 field is missing: {path}")
    return np.asarray(h5_file[path][:])


def _arm(h5_file: h5py.File, name: str, dual: bool) -> ArmTrajectory:
    """把单臂或双臂命名差异统一映射为ArmTrajectory。"""
    suffix = f"_{name}" if dual else "_single"
    return ArmTrajectory(
        puppet_pose_xyzw=_required(h5_file, f"puppet/end_effector{suffix}_pose_align/data"),
        puppet_gripper=_required(h5_file, f"puppet/end_effector{suffix}_position_align/data").reshape(-1),
        puppet_joints=_optional(h5_file, f"puppet/arm{suffix}_position_align/data"),
        master_pose_xyzw=_optional(h5_file, f"master/end_effector{suffix}_pose_align/data"),
        master_gripper=(
            value.reshape(-1)
            if (value := _optional(h5_file, f"master/end_effector{suffix}_position_align/data")) is not None
            else None
        ),
        master_joints=_optional(h5_file, f"master/arm{suffix}_position_align/data"),
    )


def _validate_arm(arm: ArmTrajectory, length: int, label: str) -> None:
    """拒绝长度不齐、非有限值或非(T,7)的puppet末端位姿。"""
    fields = {
        "puppet_pose_xyzw": arm.puppet_pose_xyzw,
        "puppet_gripper": arm.puppet_gripper,
        "puppet_joints": arm.puppet_joints,
        "master_pose_xyzw": arm.master_pose_xyzw,
        "master_gripper": arm.master_gripper,
        "master_joints": arm.master_joints,
    }
    for name, value in fields.items():
        if value is None:
            continue
        if len(value) != length:
            raise ValueError(f"{label}.{name} length {len(value)} != {length}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label}.{name} contains NaN or Inf")
    if arm.puppet_pose_xyzw.shape != (length, 7):
        raise ValueError(f"{label}.puppet_pose_xyzw must be ({length},7)")


def load_episode(path: Path) -> Episode:
    """读取一条trajectory.hdf5的低维字段、相机名和采集FPS。

    episode长度以puppet末端pose为基准。所有已存在的master/puppet字段必须严格
    对齐该长度；相机图像由transition阶段按需再次打开HDF5读取。
    """
    path = path.resolve()
    with h5py.File(path, "r") as h5_file:
        puppet = h5_file.get("puppet")
        if puppet is None:
            raise ValueError("HDF5 has no puppet group")
        dual = (
            "end_effector_left_pose_align" in puppet
            and "end_effector_right_pose_align" in puppet
        )
        left = _arm(h5_file, "left" if dual else "single", dual)
        right = _arm(h5_file, "right", True) if dual else None
        length = len(left.puppet_pose_xyzw)
        cameras = tuple(sorted(h5_file["camera_observations/color_images"].keys()))
        timestamps = np.asarray(h5_file["camera_observations/timestamp"][:], dtype=np.float64)

    _validate_arm(left, length, "left")
    if right is not None:
        _validate_arm(right, length, "right")
    if len(cameras) < 2:
        raise ValueError(f"at least two cameras are required, got {cameras}")
    if len(timestamps) < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"camera timestamps must contain at least two finite values: {path}")
    deltas = np.diff(timestamps)
    if np.any(deltas < 0):
        raise ValueError(f"camera timestamps must be non-decreasing: {path}")
    positive_deltas = deltas[deltas > 0]
    if not len(positive_deltas):
        raise ValueError(f"camera timestamps contain no positive interval: {path}")
    # 实际数据含重复timestamp；它不代表0 Hz，因此只用正时间差估计物理采集FPS。
    source_fps = int(round(1.0 / float(np.median(positive_deltas))))
    return Episode(
        path=path,
        length=length,
        camera_names=cameras,
        left=left,
        right=right,
        source_fps=source_fps,
    )


def discover_episodes(
    input_dir: Path, limit: int = 0, episode_manifest: Path | None = None
) -> list[Path]:
    """稳定发现episode；manifest可冻结跨动作表示完全一致的源数据顺序。"""
    if episode_manifest is None:
        paths = sorted(path for path in input_dir.rglob("trajectory.hdf5") if path.is_file())
    else:
        import json

        payload = json.loads(episode_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records")
        if not isinstance(payload, list):
            raise ValueError(f"episode manifest must contain a record list: {episode_manifest}")
        paths = []
        input_root = input_dir.resolve()
        for index, record in enumerate(payload):
            value = record.get("path") if isinstance(record, dict) else record
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid episode manifest record {index}: {record!r}")
            path = Path(value).expanduser().resolve()
            if not path.is_relative_to(input_root):
                raise ValueError(
                    f"episode manifest path is outside --input: {path} not under {input_root}"
                )
            if path.name != "trajectory.hdf5" or not path.is_file():
                raise FileNotFoundError(f"invalid episode path in manifest: {path}")
            paths.append(path)
        if len(paths) != len(set(paths)):
            raise ValueError(f"episode manifest contains duplicate paths: {episode_manifest}")
    return paths[:limit] if limit > 0 else paths


def action_proprio_source_arrays(
    episode: Episode, action_source: str
) -> dict[str, np.ndarray]:
    """返回当前action分支和proprio实际使用的HDF5 align字段。

    双臂数组统一按left、right拼接。字段只用于Parquet审查，不参与模型归一化或
    latent注入。
    """
    arms = [episode.left] + ([episode.right] if episode.right is not None else [])

    def concatenate(name: str) -> np.ndarray:
        values = [getattr(arm, name) for arm in arms]
        if any(value is None for value in values):
            raise ValueError(f"action source {action_source} requires source field {name}")
        arrays = [
            np.asarray(value)[:, None] if np.asarray(value).ndim == 1 else np.asarray(value)
            for value in values
        ]
        output = np.concatenate(arrays, axis=1).astype(np.float32)
        if output.shape[0] != episode.length or not np.all(np.isfinite(output)):
            raise ValueError(f"invalid source field {name}: {output.shape}")
        return output

    output = {
        "source.puppet.pose_xyzw": concatenate("puppet_pose_xyzw"),
        "source.puppet.gripper": concatenate("puppet_gripper"),
    }
    if action_source == "master_same_frame":
        output.update(
            {
                "source.master.pose_xyzw": concatenate("master_pose_xyzw"),
                "source.master.gripper": concatenate("master_gripper"),
            }
        )
    elif action_source == "master_joint_fk_same_frame":
        output.update(
            {
                "source.puppet.joints": concatenate("puppet_joints"),
                "source.master.joints": concatenate("master_joints"),
                "source.master.gripper": concatenate("master_gripper"),
            }
        )
    elif action_source != "puppet_next_frame":
        raise ValueError(f"unsupported action source for source fields: {action_source}")
    return output
