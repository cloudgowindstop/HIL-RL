"""Cosmos 3 style continuous 6D rotation and robot-action utilities.

Quaternion convention is ``xyzw``.  A 6D rotation is stored column-major as
``[R[:, 0], R[:, 1]]``.  Robot actions use the local relative transform
``inv(T_current) @ T_target`` and an absolute gripper target.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


LEGACY_EULER = "legacy_euler"
COSMOS_ROTATION_6D = "cosmos_rotation_6d"
ACTION_ENCODINGS = (LEGACY_EULER, COSMOS_ROTATION_6D)


def _as_finite_array(value: np.ndarray | Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def quaternion_xyzw_to_matrix(quaternion: np.ndarray | Sequence[float], eps: float = 1e-8) -> np.ndarray:
    """Convert a normalized, guarded xyzw quaternion to a 3x3 rotation matrix."""
    quat = _as_finite_array(quaternion, (4,), "quaternion")
    norm = np.linalg.norm(quat)
    if norm < eps:
        raise ValueError(f"quaternion norm is too small: {norm}")
    return Rotation.from_quat(quat / norm).as_matrix()


def matrix_to_rotation_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Encode a rotation matrix using its first two columns."""
    matrix = _as_finite_array(rotation_matrix, (3, 3), "rotation_matrix")
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray | Sequence[float], eps: float = 1e-8) -> np.ndarray:
    """Project an arbitrary 6D prediction onto SO(3) with Gram--Schmidt."""
    rotation = _as_finite_array(rotation_6d, (6,), "rotation_6d")
    first, second = rotation[:3], rotation[3:]

    first_norm = np.linalg.norm(first)
    if first_norm < eps:
        raise ValueError("the first rotation-6D vector has near-zero norm")
    basis_1 = first / first_norm

    second_orthogonal = second - np.dot(basis_1, second) * basis_1
    second_norm = np.linalg.norm(second_orthogonal)
    if second_norm < eps:
        raise ValueError("the two rotation-6D vectors are nearly parallel")
    basis_2 = second_orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)
    return np.stack((basis_1, basis_2, basis_3), axis=1)


def pose_xyzw_to_transform(pose_7d: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert ``xyz + quaternion_xyzw`` to a homogeneous transform."""
    pose = _as_finite_array(pose_7d, (7,), "pose_7d")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_xyzw_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def relative_transform(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
    """Return the target pose in the current end-effector coordinate frame."""
    current = pose_xyzw_to_transform(current_pose_7d)
    target = pose_xyzw_to_transform(target_pose_7d)
    return np.linalg.inv(current) @ target


def relative_rotation_6d(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
    """Encode the local relative rotation between two 7D poses."""
    return matrix_to_rotation_6d(relative_transform(current_pose_7d, target_pose_7d)[:3, :3])


def encode_single_arm_action_6d(
    current_pose_7d: np.ndarray,
    target_pose_7d: np.ndarray,
    target_gripper: float,
    translation_scale: float = 0.02,
    gripper_scale: float = 1.0,
) -> np.ndarray:
    """Encode one arm as local translation(3) + rotation-6D(6) + gripper(1)."""
    if translation_scale <= 0 or gripper_scale <= 0:
        raise ValueError("translation_scale and gripper_scale must be positive")
    if not np.isfinite(target_gripper):
        raise ValueError("target_gripper contains NaN or Inf")

    delta = relative_transform(current_pose_7d, target_pose_7d)
    translation = np.clip(delta[:3, 3] / translation_scale, -1.0, 1.0)
    rotation = matrix_to_rotation_6d(delta[:3, :3])
    gripper = np.clip(float(target_gripper) / gripper_scale, 0.0, 1.0)
    return np.concatenate((translation, rotation, [gripper])).astype(np.float32)


def encode_dual_arm_action_6d(
    current_pose_14d: np.ndarray,
    target_pose_14d: np.ndarray,
    target_gripper_2d: np.ndarray,
    translation_scale: float = 0.02,
    gripper_scale: float = 1.0,
) -> np.ndarray:
    """Encode left and right arms independently, returning a 20D action."""
    current = _as_finite_array(current_pose_14d, (14,), "current_pose_14d")
    target = _as_finite_array(target_pose_14d, (14,), "target_pose_14d")
    gripper = _as_finite_array(target_gripper_2d, (2,), "target_gripper_2d")
    left = encode_single_arm_action_6d(
        current[:7], target[:7], gripper[0], translation_scale, gripper_scale
    )
    right = encode_single_arm_action_6d(
        current[7:], target[7:], gripper[1], translation_scale, gripper_scale
    )
    return np.concatenate((left, right)).astype(np.float32)


def decode_single_arm_action_6d(
    current_pose_7d: np.ndarray,
    action_10d: np.ndarray,
    translation_scale: float = 0.02,
    gripper_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Decode a 10D action into a target homogeneous pose and gripper target."""
    action = _as_finite_array(action_10d, (10,), "action_10d")
    if translation_scale <= 0 or gripper_scale <= 0:
        raise ValueError("translation_scale and gripper_scale must be positive")
    delta = np.eye(4, dtype=np.float64)
    delta[:3, 3] = action[:3] * translation_scale
    delta[:3, :3] = rotation_6d_to_matrix(action[3:9])
    target_transform = pose_xyzw_to_transform(current_pose_7d) @ delta
    target_gripper = float(np.clip(action[9], 0.0, 1.0) * gripper_scale)
    return target_transform, target_gripper


def decode_dual_arm_action_6d(
    current_pose_14d: np.ndarray,
    action_20d: np.ndarray,
    translation_scale: float = 0.02,
    gripper_scale: float = 1.0,
) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
    """Decode a 20D action into left and right target transforms/grippers."""
    current = _as_finite_array(current_pose_14d, (14,), "current_pose_14d")
    action = _as_finite_array(action_20d, (20,), "action_20d")
    return (
        decode_single_arm_action_6d(current[:7], action[:10], translation_scale, gripper_scale),
        decode_single_arm_action_6d(current[7:], action[10:], translation_scale, gripper_scale),
    )


def get_action_dim(robot_type: str, action_encoding: str) -> int:
    """Return the action dimension for the selected robot and representation."""
    if action_encoding not in ACTION_ENCODINGS:
        raise ValueError(f"unsupported action encoding: {action_encoding}")
    is_dual = robot_type == "dual"
    if action_encoding == LEGACY_EULER:
        return 14 if is_dual else 7
    return 20 if is_dual else 10
