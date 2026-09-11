"""Inverse-dynamics ablation definitions and derived metrics."""

from __future__ import annotations


ABLATIONS = ("normal", "current_only", "future_only", "future_shuffle")


def future_information_gain(normal_error: float, shuffled_error: float) -> float:
    """Positive values indicate that matched future observations reduce error."""
    return float(shuffled_error - normal_error)


def current_information_gain(normal_error: float, future_only_error: float) -> float:
    return float(future_only_error - normal_error)

