"""Scalar value regression, classification, ranking, and calibration."""

from __future__ import annotations

import numpy as np
import torch

from .statistics import (
    average_precision,
    binary_auroc,
    expected_calibration_error,
    mean_absolute_error,
    mean_squared_error,
    spearman_correlation,
)


def extract_value_scalar(latent: torch.Tensor, index: int = 8) -> torch.Tensor:
    return latent[:, :, index].float().flatten(1).mean(1)


def pop_value_rank_pairs(
    metrics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Remove per-sample ranking tensors so they are not mean-reduced."""
    pred = metrics.pop("value_rank_pred", None)
    target = metrics.pop("value_rank_target", None)
    valid = metrics.pop("value_rank_valid", None)
    if pred is None or target is None:
        return None
    if valid is None:
        valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        valid = valid.bool()
    if not valid.any():
        empty = pred.new_zeros((0,))
        return empty, empty
    return pred[valid].detach().float().cpu(), target[valid].detach().float().cpu()


def pooled_value_spearman(prediction, target) -> tuple[float, int]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return spearman_correlation(prediction, target), int(prediction.size)


def value_metrics(prediction, target, *, success_threshold: float = 0.5) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = (target >= success_threshold).astype(np.int64)
    probabilities = np.clip(prediction, 0.0, 1.0)
    return {
        "value_mae": mean_absolute_error(prediction, target),
        "value_mse": mean_squared_error(prediction, target),
        "value_auroc": binary_auroc(probabilities, labels),
        "value_auprc": average_precision(probabilities, labels),
        "value_spearman": spearman_correlation(prediction, target),
        "value_ece": expected_calibration_error(probabilities, labels),
    }

