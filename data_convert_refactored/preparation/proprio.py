"""构造并归一化Cosmos conditioning使用的机器人proprio。"""

from __future__ import annotations

import numpy as np

from .episode import Episode


def _continuous_quaternions(poses: np.ndarray) -> np.ndarray:
    """归一化四元数，并消除q与-q等价造成的相邻帧符号跳变。"""
    poses = np.asarray(poses, dtype=np.float64).copy()
    for index in range(len(poses)):
        quaternion = poses[index, 3:7]
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            raise ValueError("proprio quaternion norm is too small")
        quaternion = quaternion / norm
        if index and np.dot(poses[index - 1, 3:7], quaternion) < 0:
            quaternion = -quaternion
        poses[index, 3:7] = quaternion
    return poses


def raw_proprio_for_episode(episode: Episode) -> np.ndarray:
    """构造(T,P) proprio；每臂为xyz+xyzw四元数+绝对夹爪，共8维。"""
    left_pose = _continuous_quaternions(episode.left.puppet_pose_xyzw)
    left = np.concatenate((left_pose, episode.left.puppet_gripper[:, None]), axis=1)
    if episode.right is None:
        return left.astype(np.float32)
    right_pose = _continuous_quaternions(episode.right.puppet_pose_xyzw)
    right = np.concatenate((right_pose, episode.right.puppet_gripper[:, None]), axis=1)
    return np.concatenate((left, right), axis=1).astype(np.float32)


def normalized_proprio_for_episode(episode: Episode, dataset_stats: dict) -> np.ndarray:
    """严格复用官方rescale_proprio，将每个proprio通道映射到训练尺度。"""
    from cosmos_policy.experiments.robot.cosmos_utils import rescale_proprio

    normalized = rescale_proprio(
        raw_proprio_for_episode(episode),
        dataset_stats,
        non_negative_only=False,
    )
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape[0] != episode.length:
        raise ValueError(
            f"normalized proprio length {normalized.shape[0]} != episode length {episode.length}"
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"normalized proprio contains NaN or Inf: {episode.path}")
    return normalized
