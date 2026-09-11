"""Dependency-light scalar metrics and episode-level uncertainty."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def mean_absolute_error(prediction, target) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def mean_squared_error(prediction, target) -> float:
    difference = np.asarray(prediction) - np.asarray(target)
    return float(np.mean(difference * difference))


def binary_auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def average_precision(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float((precision * sorted_labels).sum() / positives)


def spearman_correlation(prediction, target) -> float:
    from scipy.stats import spearmanr

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size != target.size:
        raise ValueError(
            f"spearman prediction/target size mismatch: {prediction.size} vs {target.size}"
        )
    if prediction.size < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
        return float("nan")
    result = spearmanr(prediction, target)
    statistic = float(result.statistic)
    return statistic if np.isfinite(statistic) else float("nan")


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if selected.any():
            total += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(total)


def bootstrap_confidence_interval(
    values,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [tail, 1.0 - tail]))


def aggregate_episode_metrics(rows: Iterable[dict], *, bootstrap_samples: int = 2000) -> dict:
    """Average samples within episodes, then summarize episodes with a CI."""
    by_episode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        episode = str(row["episode_path"])
        for name, value in row["metrics"].items():
            if np.isfinite(float(value)):
                by_episode[episode][name].append(float(value))
    episode_values: dict[str, list[float]] = defaultdict(list)
    for metrics in by_episode.values():
        for name, values in metrics.items():
            episode_values[name].append(float(np.mean(values)))
    output = {}
    for name, values in episode_values.items():
        lower, upper = bootstrap_confidence_interval(values, samples=bootstrap_samples)
        output[name] = {
            "episodes": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
    return output

