"""Masked EDM reduction and per-region diagnostics."""

from __future__ import annotations

import torch

from .masks import OBJECTIVE_NAMES

REGIONS = {
    "action": 4,
    "future_proprio": 5,
    "future_wrist_image": 6,
    "future_image": 7,
    "value": 8,
}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


def edm_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma_weights: torch.Tensor,
    temporal_loss_mask: torch.Tensor,
    sample_types: torch.Tensor | None = None,
    reduction: str = "masked_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    squared = (prediction - target).square()
    if sigma_weights.ndim != 2:
        raise ValueError(f"sigma_weights must be [B,T], got {sigma_weights.shape}")
    weights = sigma_weights[:, None, :, None, None]
    weighted = squared * weights
    # Latent layout is [B,C,T,H,W].
    temporal = temporal_loss_mask[:, None, :, None, None]
    masked_total = masked_mean(weighted, temporal)
    # learner_copy_dist.py zeros non-target positions, then averages over the
    # complete [B,C,T,H,W] tensor. Keep both reductions explicit because their
    # scales differ by the target-mask ratio.
    learner_total = (weighted * temporal.to(weighted.dtype)).mean()
    if reduction == "masked_mean":
        total = masked_total
    elif reduction == "learner_actor_full_tensor_mean":
        total = learner_total
    else:
        raise ValueError(f"unknown EDM loss reduction: {reduction}")
    metrics: dict[str, torch.Tensor] = {
        "total_edm_loss": total.detach(),
        "policy_masked_edm_loss": masked_total.detach(),
        "loss_actor_learner": learner_total.detach(),
    }
    if sample_types is not None:
        for index, name in enumerate(OBJECTIVE_NAMES):
            sample_mask = temporal_loss_mask & (sample_types == index)[:, None]
            sample_mask_5d = sample_mask[:, None, :, None, None]
            # Keep sample-objective metrics separate from temporal-region metrics.
            # In particular, "value_edm_loss" used to be overwritten below by
            # the value-region metric, hiding the value-sample result in joint modes.
            metrics[f"sample_{name}_edm_loss"] = masked_mean(
                weighted, sample_mask_5d
            ).detach()
    for name, index in REGIONS.items():
        valid = temporal_loss_mask[:, index]
        metrics[f"{name}_edm_loss"] = masked_mean(weighted[:, :, index], valid).detach()
        metrics[f"{name}_mse"] = masked_mean(squared[:, :, index], valid).detach()
        metrics[f"{name}_l1"] = masked_mean((prediction-target).abs()[:, :, index], valid).detach()
    metrics["effective_mask_ratio"] = temporal_loss_mask.float().mean().detach()
    metrics["effective_temporal_positions"] = temporal_loss_mask.sum().detach()
    return total, metrics
