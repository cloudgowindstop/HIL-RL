"""Policy and inverse-dynamics action metrics in physical space."""

from __future__ import annotations

import torch

from ..action_spec import get_action_spec
from ..rotation_6d import geodesic_error_degrees, split_dual_arm_action


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert xyz Euler angles to rotation matrices using Rz @ Ry @ Rx."""
    x, y, z = euler.unbind(dim=-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    return torch.stack(
        (
            cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz,
            cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz,
            -sy, sx * cy, cx * cy,
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))


def _matrix_geodesic_degrees(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def _rotation_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    encoding: str,
    euler_rotation_scale: float,
) -> torch.Tensor:
    if encoding == "cosmos_rotation_6d":
        return geodesic_error_degrees(prediction, target)
    return _matrix_geodesic_degrees(
        _euler_xyz_to_matrix(prediction * euler_rotation_scale),
        _euler_xyz_to_matrix(target * euler_rotation_scale),
    )


def action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gripper_thresholds: tuple[float, float] = (0.5, 0.5),
    action_encoding: str = "cosmos_rotation_6d",
    euler_rotation_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return physical dual-arm metrics for Euler-14D or rotation-6D-20D actions."""
    spec = get_action_spec(action_encoding)
    if prediction.shape != target.shape or prediction.shape[-1] != spec.dimension:
        raise ValueError(
            f"action metric shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}, expected last dim {spec.dimension}"
        )
    if prediction.shape[0] == 0:
        zero = torch.zeros((), device=prediction.device, dtype=torch.float32)
        output = {"action_physical_mae": zero, "action_physical_mse": zero}
        for arm in ("left", "right"):
            for suffix in (
                "translation_mae", "rotation_geodesic_deg",
                "gripper_mae", "gripper_accuracy",
            ):
                output[f"{arm}_{suffix}"] = zero
            for horizon in (1, 4, 8, 16):
                for suffix in ("translation_mae", "rotation_geodesic_deg", "gripper_mae"):
                    output[f"{arm}_horizon_{horizon:02d}_{suffix}"] = zero
        return output
    output: dict[str, torch.Tensor] = {
        "action_physical_mae": (prediction - target).abs().mean(),
        "action_physical_mse": (prediction - target).square().mean(),
    }
    if action_encoding == "cosmos_rotation_6d":
        left_prediction, right_prediction = split_dual_arm_action(prediction)
        left_target, right_target = split_dual_arm_action(target)
    else:
        left_prediction, right_prediction = prediction.split(spec.per_arm_dimension, dim=-1)
        left_target, right_target = target.split(spec.per_arm_dimension, dim=-1)
    for arm, predicted, expected, threshold in (
        ("left", left_prediction, left_target, gripper_thresholds[0]),
        ("right", right_prediction, right_target, gripper_thresholds[1]),
    ):
        output[f"{arm}_translation_mae"] = (predicted[..., :3] - expected[..., :3]).abs().mean()
        output[f"{arm}_rotation_geodesic_deg"] = _rotation_error(
            predicted[..., spec.rotation_slice],
            expected[..., spec.rotation_slice],
            action_encoding,
            euler_rotation_scale,
        ).mean()
        output[f"{arm}_gripper_mae"] = (
            predicted[..., spec.gripper_index] - expected[..., spec.gripper_index]
        ).abs().mean()
        output[f"{arm}_gripper_accuracy"] = (
            (predicted[..., spec.gripper_index] >= threshold)
            == (expected[..., spec.gripper_index] >= threshold)
        ).float().mean()
        for horizon in (1, 4, 8, 16):
            index = horizon - 1
            prefix = f"{arm}_horizon_{horizon:02d}"
            output[f"{prefix}_translation_mae"] = (
                predicted[:, index, :3] - expected[:, index, :3]
            ).abs().mean()
            output[f"{prefix}_rotation_geodesic_deg"] = _rotation_error(
                predicted[:, index, spec.rotation_slice],
                expected[:, index, spec.rotation_slice],
                action_encoding,
                euler_rotation_scale,
            ).mean()
            output[f"{prefix}_gripper_mae"] = (
                predicted[:, index, spec.gripper_index]
                - expected[:, index, spec.gripper_index]
            ).abs().mean()
    return output
