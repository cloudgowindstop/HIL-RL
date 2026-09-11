"""模型加载前的数据布局、维度、FPS和可选FK一致性检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .preparation.actions import pose_to_matrix
from .config import ActionSource, ConversionConfig
from .preparation.episode import Episode
from .preparation.kinematics import DualArmKinematics


@dataclass(frozen=True)
class ArmFKReport:
    """记录FK末端相对HDF5末端pose的位置和旋转误差。"""

    max_position_error_m: float
    mean_position_error_m: float
    max_rotation_error_deg: float
    mean_rotation_error_deg: float


@dataclass(frozen=True)
class PreflightReport:
    """后续Policy schema、stats校验和metadata写入共享的预检结果。"""

    episode_count: int
    robot_type: str
    action_dimension: int
    proprio_dimension: int
    cameras: tuple[str, ...]
    action_source: str
    action_encoding: str
    episode_outcome: str
    source_fps: int
    cosmos_conditioning_fps: int
    fk: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_arm_fk(
    joints: np.ndarray,
    recorded_poses: np.ndarray,
    forward,
    max_samples: int = 200,
) -> ArmFKReport:
    """抽样比较FK和记录pose；首帧只用于消除两者固定base坐标系差异。"""
    indices = np.unique(
        np.linspace(0, len(joints) - 1, min(len(joints), max_samples), dtype=np.int64)
    )
    first_fk = forward(joints[indices[0]])
    first_recorded = pose_to_matrix(recorded_poses[indices[0]])
    base_alignment = first_recorded @ np.linalg.inv(first_fk)
    position_errors, rotation_errors = [], []
    for index in indices:
        predicted = base_alignment @ forward(joints[index])
        recorded = pose_to_matrix(recorded_poses[index])
        error = np.linalg.inv(recorded) @ predicted
        position_errors.append(float(np.linalg.norm(error[:3, 3])))
        rotation_errors.append(
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec())))
        )
    return ArmFKReport(
        max(position_errors),
        float(np.mean(position_errors)),
        max(rotation_errors),
        float(np.mean(rotation_errors)),
    )


def run_preflight(
    config: ConversionConfig,
    episodes: list[Episode],
    kinematics: DualArmKinematics | None,
) -> PreflightReport:
    """保证同批episode机器人类型、相机布局和采集FPS一致，并确定D/P维度。"""
    if not episodes:
        raise ValueError("no HDF5 episodes found")
    dual = episodes[0].is_dual_arm
    for episode in episodes:
        if episode.is_dual_arm != dual:
            raise ValueError(f"mixed single/dual-arm dataset: {episode.path}")
        if episode.camera_names != episodes[0].camera_names:
            raise ValueError(f"camera layout differs in {episode.path}: {episode.camera_names}")
        if episode.source_fps != episodes[0].source_fps:
            raise ValueError(
                f"mixed source FPS: expected {episodes[0].source_fps}, "
                f"got {episode.source_fps} in {episode.path}"
            )

    fk_report = None
    if config.action_source is ActionSource.MASTER_JOINT_FK_SAME_FRAME:
        if not dual or kinematics is None or episodes[0].right is None:
            raise ValueError("master joint FK requires dual-arm episodes and kinematics")
        episode = episodes[0]
        left = _validate_arm_fk(
            episode.left.puppet_joints,
            episode.left.puppet_pose_xyzw,
            kinematics.forward_left,
        )
        right = _validate_arm_fk(
            episode.right.puppet_joints,
            episode.right.puppet_pose_xyzw,
            kinematics.forward_right,
        )
        for name, result in (("left", left), ("right", right)):
            if (
                result.max_position_error_m > config.fk_max_position_error_m
                or result.max_rotation_error_deg > config.fk_max_rotation_error_deg
            ):
                raise ValueError(
                    f"{name} FK validation failed: {asdict(result)}; limits are "
                    f"{config.fk_max_position_error_m} m and "
                    f"{config.fk_max_rotation_error_deg} deg"
                )
        fk_report = {"left": asdict(left), "right": asdict(right), "passed": True}

    # 每臂Euler为7D，rotation-6D为10D；双臂按左、右顺序拼接。
    arm_action_dim = 10 if config.action_encoding.value == "cosmos_rotation_6d" else 7
    return PreflightReport(
        episode_count=len(episodes),
        robot_type="dual" if dual else "single_absolute",
        action_dimension=arm_action_dim * (2 if dual else 1),
        proprio_dimension=16 if dual else 8,
        cameras=episodes[0].camera_names,
        action_source=config.action_source.value,
        action_encoding=config.action_encoding.value,
        episode_outcome=config.episode_outcome.value,
        source_fps=episodes[0].source_fps,
        cosmos_conditioning_fps=16,
        fk=fk_report,
    )
