"""Torch rotation-6D projection and physical-space metrics."""

from __future__ import annotations

import torch


def rotation_6d_to_matrix(rotation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rotation.shape[-1] != 6:
        raise ValueError(f"rotation last dimension must be 6, got {rotation.shape}")
    first, second = rotation[..., :3], rotation[..., 3:]
    basis_1 = torch.nn.functional.normalize(first, dim=-1, eps=eps)
    second_orthogonal = second - (basis_1 * second).sum(-1, keepdim=True) * basis_1
    basis_2 = torch.nn.functional.normalize(second_orthogonal, dim=-1, eps=eps)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end with (3,3), got {matrix.shape}")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def geodesic_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_matrix = rotation_6d_to_matrix(prediction)
    target_matrix = rotation_6d_to_matrix(target)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def split_dual_arm_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if action.shape[-1] != 20:
        raise ValueError(f"dual-arm rotation-6D action must be 20D, got {action.shape[-1]}")
    return action[..., :10], action[..., 10:]
