"""Policy/world/value/inverse-dynamics masks for nine latent positions."""

from __future__ import annotations

import torch

POLICY = 0
WORLD = 1
VALUE = 2
INVERSE_DYNAMICS = 3
OBJECTIVE_NAMES = ("policy", "world", "value", "inverse_dynamics")
ACTION_INDEX = 4
FUTURE_INDICES = (5, 6, 7)
VALUE_INDEX = 8


def sample_types(batch_size: int, probabilities: tuple[float, ...], device=None) -> torch.Tensor:
    probs = torch.tensor(probabilities, dtype=torch.float32, device=device)
    return torch.multinomial(probs, batch_size, replacement=True)


INVERSE_CONDITION_MODES = ("normal", "current_only", "future_only", "future_shuffle")


def temporal_masks(
    types: torch.Tensor,
    time: int = 9,
    *,
    inverse_condition_mode: str = "normal",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return condition mask and target-loss mask, shape ``[B,T]``."""
    batch = types.shape[0]
    condition = torch.zeros((batch, time), dtype=torch.bool, device=types.device)
    condition[:, :4] = True
    condition[types == WORLD, ACTION_INDEX] = True
    value_rows = types == VALUE
    condition[value_rows, :VALUE_INDEX] = True
    loss = ~condition
    inverse_rows = types == INVERSE_DYNAMICS
    # Inverse dynamics sees current and future observations, then predicts only
    # the intervening action. Value is neither condition nor target.
    condition[inverse_rows, 5:VALUE_INDEX] = True
    if inverse_condition_mode not in INVERSE_CONDITION_MODES:
        raise ValueError(f"unknown inverse condition mode: {inverse_condition_mode}")
    if inverse_condition_mode == "current_only":
        condition[inverse_rows, 5:VALUE_INDEX] = False
    elif inverse_condition_mode == "future_only":
        condition[inverse_rows, :4] = False
    loss[inverse_rows] = False
    loss[inverse_rows, ACTION_INDEX] = True
    # Nothing after value index is a target in current 9-position layout.
    if time > 9:
        loss[:, 9:] = False
    return condition, loss


def apply_failed_action_mask(loss_mask: torch.Tensor, types: torch.Tensor, failed: torch.Tensor) -> torch.Tensor:
    output = loss_mask.clone()
    output[(types == POLICY) & failed, ACTION_INDEX] = False
    return output
