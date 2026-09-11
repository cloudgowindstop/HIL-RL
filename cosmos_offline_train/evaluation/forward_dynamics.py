"""Forward-dynamics latent metrics."""

from __future__ import annotations

import torch


FUTURE_REGIONS = {
    "future_proprio": 5,
    "future_wrist_image": 6,
    "future_image": 7,
}


def future_latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    output = {}
    for name, index in FUTURE_REGIONS.items():
        difference = prediction[:, :, index].float() - target[:, :, index].float()
        output[f"{name}_mae"] = difference.abs().mean()
        output[f"{name}_mse"] = difference.square().mean()
    return output

