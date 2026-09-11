"""Reproducible evaluation protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    objectives: tuple[str, ...] = ("policy", "world", "value", "inverse_dynamics")
    sigma_values: tuple[float, ...] = (0.1, 0.5, 0.9)
    inverse_ablations: tuple[str, ...] = (
        "normal", "current_only", "future_only", "future_shuffle"
    )
    noise_seed: int = 20260820
    max_batches: int | None = 200

