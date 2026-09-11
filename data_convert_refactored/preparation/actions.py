"""机器人action计算、时间对齐及第二层dataset归一化。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ActionEncoding, ActionScale, ActionSource
from .episode import ArmTrajectory, Episode
from .rotation import matrix_to_rotation_6d, pose_xyzw_to_transform as pose_to_matrix


AUXILIARY_ACTION_FIELDS = (
    "action.target_rotation_6d",
    "action.delta_rotation_6d",
    "action.target_ee_pose_xyzw",
    "action.target_joint_position",
    "action.delta_ee_xyz_euler",
    "action.delta_joint_position",
    "action.target_gripper",
)
AUXILIARY_OBSERVATION_FIELDS = (
    "observation.rotation_6d",
    "observation.ee_pose_xyzw",
    "observation.joint_position",
    "observation.gripper",
)
AUXILIARY_FIELDS = AUXILIARY_ACTION_FIELDS + AUXILIARY_OBSERVATION_FIELDS


class DualArmFK(Protocol):
    """双臂FK最小接口：输入单臂关节角，输出base到末端的4×4变换。"""

    def forward_left(self, joints: np.ndarray) -> np.ndarray: ...
    def forward_right(self, joints: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EncodedEpisode:
    """绑定训练action、辅助观测/动作与原episode。"""

    episode: Episode
    actions: np.ndarray
    action_source: ActionSource
    action_encoding: ActionEncoding
    last_action_is_padding: bool
    euler_actions: np.ndarray | None = None
    rotation_6d_actions: np.ndarray | None = None
    auxiliary_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (T,D), got {actions.shape}")
        if len(actions) != self.episode.length:
            raise ValueError(
                f"action length {len(actions)} != episode length {self.episode.length}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or Inf")
        object.__setattr__(self, "actions", actions)
        expected_dims = {"euler_actions": 7, "rotation_6d_actions": 10}
        arm_count = 2 if self.episode.is_dual_arm else 1
        for name, per_arm_dim in expected_dims.items():
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            expected_shape = (self.episode.length, per_arm_dim * arm_count)
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)
        auxiliary = {}
        for name, value in self.auxiliary_fields.items():
            if name not in AUXILIARY_FIELDS:
                raise ValueError(f"unsupported auxiliary field: {name}")
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 2 or value.shape[0] != self.episode.length:
                raise ValueError(
                    f"{name} must have shape (episode_length,D), got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
            auxiliary[name] = value
        object.__setattr__(self, "auxiliary_fields", auxiliary)

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class ActionRepresentations:
    """同一目标编码出的训练action与未归一化审查表示。"""

    euler_control: np.ndarray
    rotation_6d_control: np.ndarray
    auxiliary_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def primary(self, encoding: ActionEncoding) -> np.ndarray:
        return (
            self.rotation_6d_control
            if encoding is ActionEncoding.ROTATION_6D
            else self.euler_control
        )


def encode_transform(
    delta: np.ndarray,
    target_gripper: float,
    encoding: ActionEncoding,
    scale: ActionScale,
) -> np.ndarray:
    """把末端局部相对变换编码成单臂action。

    `delta = inv(T_current) @ T_target`，因此平移和旋转都表达在当前末端坐标系。
    Euler输出7维，rotation-6D输出10维；夹爪始终使用目标帧绝对位置。
    """
    translation = np.clip(delta[:3, 3] / scale.translation_m, -1.0, 1.0)
    gripper = np.clip(target_gripper / scale.gripper, 0.0, 1.0)
    if encoding is ActionEncoding.ROTATION_6D:
        rotation = matrix_to_rotation_6d(delta[:3, :3])
    else:
        rotation = np.clip(
            Rotation.from_matrix(delta[:3, :3]).as_euler("xyz") / scale.rotation_rad,
            -1.0,
            1.0,
        )
    return np.concatenate((translation, rotation, [gripper])).astype(np.float32)


class ActionEncoder(Protocol):
    """所有action分支共享的接口；输出必须是与episode等长的(T,D)。"""

    source: ActionSource
    last_action_is_padding: bool

    def encode_episode(self, episode: Episode) -> np.ndarray: ...
    def encode_episode_all(self, episode: Episode) -> ActionRepresentations: ...


def _encode_both(
    delta: np.ndarray, target_gripper: float, scale: ActionScale
) -> tuple[np.ndarray, np.ndarray]:
    """同一delta分别编码两种旋转表示，避免重复计算相对位姿。"""
    return (
        encode_transform(delta, target_gripper, ActionEncoding.LEGACY_EULER, scale),
        encode_transform(delta, target_gripper, ActionEncoding.ROTATION_6D, scale),
    )


def _pose_from_transform(transform: np.ndarray) -> np.ndarray:
    """把4×4变换转成[x,y,z,qx,qy,qz,qw]。"""
    return np.concatenate(
        (transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat())
    ).astype(np.float32)


def _auxiliary_for_arm(
    current_transform: np.ndarray,
    target_transform: np.ndarray,
    current_joints: np.ndarray | None,
    target_joints: np.ndarray | None,
    current_gripper: float,
    target_gripper: float,
    *,
    current_pose_xyzw: np.ndarray | None = None,
    target_pose_xyzw: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """生成单臂当前观测和目标/增量动作；不做任何归一化。"""
    delta = np.linalg.inv(current_transform) @ target_transform
    result = {
        "observation.rotation_6d": matrix_to_rotation_6d(current_transform[:3, :3]),
        "observation.ee_pose_xyzw": (
            np.asarray(current_pose_xyzw, dtype=np.float32)
            if current_pose_xyzw is not None
            else _pose_from_transform(current_transform)
        ),
        "observation.gripper": np.asarray([current_gripper], dtype=np.float32),
        "action.target_rotation_6d": matrix_to_rotation_6d(target_transform[:3, :3]),
        "action.delta_rotation_6d": matrix_to_rotation_6d(delta[:3, :3]),
        "action.target_ee_pose_xyzw": (
            np.asarray(target_pose_xyzw, dtype=np.float32)
            if target_pose_xyzw is not None
            else _pose_from_transform(target_transform)
        ),
        "action.delta_ee_xyz_euler": np.concatenate(
            (delta[:3, 3], Rotation.from_matrix(delta[:3, :3]).as_euler("xyz"))
        ).astype(np.float32),
        "action.target_gripper": np.asarray([target_gripper], dtype=np.float32),
    }
    if current_joints is not None and target_joints is not None:
        current = np.asarray(current_joints, dtype=np.float32).reshape(-1)
        target = np.asarray(target_joints, dtype=np.float32).reshape(-1)
        if current.shape != target.shape:
            raise ValueError(
                f"current/target joint shape mismatch: {current.shape} != {target.shape}"
            )
        result["observation.joint_position"] = current.copy()
        result["action.target_joint_position"] = target.copy()
        result["action.delta_joint_position"] = target - current
    return result


def _concatenate_arm_auxiliary(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """按left、right顺序拼接同一帧的单臂审查字段。"""
    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    return {name: np.concatenate([part[name] for part in parts]) for name in sorted(common)}


def _stack_auxiliary(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """把逐帧审查字段转成(T,D)，并拒绝帧间schema变化。"""
    names = set(frames[0])
    if any(set(frame) != names for frame in frames[1:]):
        raise ValueError("auxiliary field schema changed within one episode")
    return {name: np.stack([frame[name] for frame in frames]) for name in sorted(names)}


@dataclass
class PuppetNextFrameEncoder:
    """以puppet[t+1]作为puppet[t]目标，表示示教轨迹的逐帧局部增量。"""

    encoding: ActionEncoding
    scale: ActionScale
    source: ActionSource = ActionSource.PUPPET_NEXT_FRAME
    last_action_is_padding: bool = True

    def _arm(
        self, arm: ArmTrajectory, frame: int, target: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        current_transform = pose_to_matrix(arm.puppet_pose_xyzw[frame])
        target_transform = pose_to_matrix(arm.puppet_pose_xyzw[target])
        delta = np.linalg.inv(current_transform) @ target_transform
        euler, rotation_6d = _encode_both(
            delta, float(arm.puppet_gripper[target]), self.scale
        )
        auxiliary = _auxiliary_for_arm(
            current_transform,
            target_transform,
            None if arm.puppet_joints is None else arm.puppet_joints[frame],
            None if arm.puppet_joints is None else arm.puppet_joints[target],
            float(arm.puppet_gripper[frame]),
            float(arm.puppet_gripper[target]),
            current_pose_xyzw=arm.puppet_pose_xyzw[frame],
            target_pose_xyzw=arm.puppet_pose_xyzw[target],
        )
        return euler, rotation_6d, auxiliary

    def encode_episode_all(self, episode: Episode) -> ActionRepresentations:
        """计算相邻帧action；末帧复制最后一个有效action并标记为padding。"""
        euler_actions, rotation_6d_actions, auxiliary_frames = [], [], []
        frame_targets = [(0, 0)] if episode.length == 1 else [
            (frame, frame + 1) for frame in range(episode.length - 1)
        ]
        for frame, target in frame_targets:
            pairs = [self._arm(episode.left, frame, target)]
            if episode.right is not None:
                pairs.append(self._arm(episode.right, frame, target))
            euler_actions.append(np.concatenate([pair[0] for pair in pairs]))
            rotation_6d_actions.append(np.concatenate([pair[1] for pair in pairs]))
            auxiliary_frames.append(_concatenate_arm_auxiliary([pair[2] for pair in pairs]))
        if episode.length > 1:
            # 最后一帧没有t+1目标；action复制最后有效值，observation使用真实末帧。
            euler_actions.append(euler_actions[-1].copy())
            rotation_6d_actions.append(rotation_6d_actions[-1].copy())
            final_pairs = [self._arm(episode.left, episode.length - 1, episode.length - 1)]
            if episode.right is not None:
                final_pairs.append(
                    self._arm(episode.right, episode.length - 1, episode.length - 1)
                )
            final_observation = _concatenate_arm_auxiliary(
                [pair[2] for pair in final_pairs]
            )
            auxiliary_frames.append(
                {
                    name: (
                        final_observation[name].copy()
                        if name.startswith("observation.")
                        else value.copy()
                    )
                    for name, value in auxiliary_frames[-1].items()
                }
            )
        return ActionRepresentations(
            np.stack(euler_actions),
            np.stack(rotation_6d_actions),
            _stack_auxiliary(auxiliary_frames),
        )

    def encode_episode(self, episode: Episode) -> np.ndarray:
        return self.encode_episode_all(episode).primary(self.encoding)


@dataclass
class MasterPoseSameFrameEncoder:
    """以同帧master末端位姿作为目标：inv(T_puppet[t]) @ T_master[t]。"""

    encoding: ActionEncoding
    scale: ActionScale
    source: ActionSource = ActionSource.MASTER_POSE_SAME_FRAME
    last_action_is_padding: bool = False

    def _arm(
        self, arm: ArmTrajectory, frame: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if arm.master_pose_xyzw is None or arm.master_gripper is None:
            raise ValueError("master pose action requires aligned master pose and gripper")
        current_transform = pose_to_matrix(arm.puppet_pose_xyzw[frame])
        target_transform = pose_to_matrix(arm.master_pose_xyzw[frame])
        delta = np.linalg.inv(current_transform) @ target_transform
        euler, rotation_6d = _encode_both(
            delta, float(arm.master_gripper[frame]), self.scale
        )
        auxiliary = _auxiliary_for_arm(
            current_transform,
            target_transform,
            None if arm.puppet_joints is None else arm.puppet_joints[frame],
            None if arm.master_joints is None else arm.master_joints[frame],
            float(arm.puppet_gripper[frame]),
            float(arm.master_gripper[frame]),
            current_pose_xyzw=arm.puppet_pose_xyzw[frame],
            target_pose_xyzw=arm.master_pose_xyzw[frame],
        )
        return euler, rotation_6d, auxiliary

    def encode_episode_all(self, episode: Episode) -> ActionRepresentations:
        """逐帧计算master相对puppet的控制目标；不存在末帧padding。"""
        euler_actions, rotation_6d_actions, auxiliary_frames = [], [], []
        for frame in range(episode.length):
            pairs = [self._arm(episode.left, frame)]
            if episode.right is not None:
                pairs.append(self._arm(episode.right, frame))
            euler_actions.append(np.concatenate([pair[0] for pair in pairs]))
            rotation_6d_actions.append(np.concatenate([pair[1] for pair in pairs]))
            auxiliary_frames.append(_concatenate_arm_auxiliary([pair[2] for pair in pairs]))
        return ActionRepresentations(
            np.stack(euler_actions),
            np.stack(rotation_6d_actions),
            _stack_auxiliary(auxiliary_frames),
        )

    def encode_episode(self, episode: Episode) -> np.ndarray:
        return self.encode_episode_all(episode).primary(self.encoding)


@dataclass
class MasterJointFKSameFrameEncoder:
    """分别对同帧puppet/master关节做FK，再计算两者末端相对变换。"""

    kinematics: DualArmFK
    encoding: ActionEncoding
    scale: ActionScale
    source: ActionSource = ActionSource.MASTER_JOINT_FK_SAME_FRAME
    last_action_is_padding: bool = False

    def encode_episode_all(self, episode: Episode) -> ActionRepresentations:
        """计算天翼双臂官方对齐分支；要求双臂关节、master夹爪和准确运动学配置。"""
        if not episode.is_dual_arm or episode.right is None:
            raise ValueError("master joint FK action requires a dual-arm episode")
        for arm in (episode.left, episode.right):
            if arm.puppet_joints is None or arm.master_joints is None or arm.master_gripper is None:
                raise ValueError("master joint FK action requires puppet/master joints and master gripper")

        euler_actions, rotation_6d_actions, auxiliary_frames = [], [], []
        for frame in range(episode.length):
            left_current = self.kinematics.forward_left(episode.left.puppet_joints[frame])
            left_target = self.kinematics.forward_left(episode.left.master_joints[frame])
            right_current = self.kinematics.forward_right(episode.right.puppet_joints[frame])
            right_target = self.kinematics.forward_right(episode.right.master_joints[frame])
            left_delta = np.linalg.inv(left_current) @ left_target
            right_delta = np.linalg.inv(right_current) @ right_target
            left_pair = _encode_both(
                left_delta, float(episode.left.master_gripper[frame]), self.scale
            )
            right_pair = _encode_both(
                right_delta, float(episode.right.master_gripper[frame]), self.scale
            )
            euler_actions.append(np.concatenate((left_pair[0], right_pair[0])))
            rotation_6d_actions.append(np.concatenate((left_pair[1], right_pair[1])))
            auxiliary_frames.append(
                _concatenate_arm_auxiliary(
                    [
                        _auxiliary_for_arm(
                            left_current,
                            left_target,
                            episode.left.puppet_joints[frame],
                            episode.left.master_joints[frame],
                            float(episode.left.puppet_gripper[frame]),
                            float(episode.left.master_gripper[frame]),
                        ),
                        _auxiliary_for_arm(
                            right_current,
                            right_target,
                            episode.right.puppet_joints[frame],
                            episode.right.master_joints[frame],
                            float(episode.right.puppet_gripper[frame]),
                            float(episode.right.master_gripper[frame]),
                        ),
                    ]
                )
            )
        return ActionRepresentations(
            np.stack(euler_actions),
            np.stack(rotation_6d_actions),
            _stack_auxiliary(auxiliary_frames),
        )

    def encode_episode(self, episode: Episode) -> np.ndarray:
        return self.encode_episode_all(episode).primary(self.encoding)


def make_action_encoder(
    source: ActionSource,
    encoding: ActionEncoding,
    scale: ActionScale,
    kinematics: DualArmFK | None = None,
) -> ActionEncoder:
    """根据action source创建唯一对应的encoder，避免主流程内出现分支实现。"""
    if source is ActionSource.PUPPET_NEXT_FRAME:
        return PuppetNextFrameEncoder(encoding, scale)
    if source is ActionSource.MASTER_POSE_SAME_FRAME:
        return MasterPoseSameFrameEncoder(encoding, scale)
    if kinematics is None:
        raise ValueError("master joint FK action requires kinematics")
    return MasterJointFKSameFrameEncoder(kinematics, encoding, scale)


def normalize_encoded_episodes(
    encoded_episodes: list[EncodedEpisode],
    action_stats: dict[str, np.ndarray],
    rescale_fn: Callable,
) -> list[EncodedEpisode]:
    """对第一层action应用Cosmos dataset min/max归一化。

    输入是机器人scale处理后的(T,D)，输出映射到训练使用的[-1,1]尺度。本函数
    复用调用方传入的官方`rescale_action`，不裁剪超出stats范围的值。
    """
    result = []
    for item in encoded_episodes:
        normalized = rescale_fn(
            item.actions,
            action_stats,
            non_negative_only=False,
            scale_multiplier=1.0,
        )
        normalized = np.asarray(normalized, dtype=np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"normalized action contains NaN or Inf: {item.episode.path}")
        result.append(replace(item, actions=normalized))
    return result
