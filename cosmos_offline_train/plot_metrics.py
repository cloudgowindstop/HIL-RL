"""Reusable JSONL exports and raw EDM training-result plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np


OBJECTIVE_COUNTS = {
    "sample_policy_edm_loss": ("policy_samples",),
    "sample_world_edm_loss": ("world_samples",),
    "sample_value_edm_loss": ("value_samples",),
    "sample_inverse_dynamics_edm_loss": ("inverse_dynamics_samples",),
}
OBJECTIVE_LABELS = {
    "sample_policy_edm_loss": "Policy",
    "sample_world_edm_loss": "World",
    "sample_value_edm_loss": "Value",
    "sample_inverse_dynamics_edm_loss": "Inverse dynamics",
}
REGION_COUNTS = {
    "action": ("policy_samples", "inverse_dynamics_samples"),
    "future_proprio": ("policy_samples", "world_samples"),
    "future_wrist_image": ("policy_samples", "world_samples"),
    "future_image": ("policy_samples", "world_samples"),
    "value": ("policy_samples", "world_samples", "value_samples"),
}
REGION_LABELS = {
    "action": "Action",
    "future_proprio": "Future proprio",
    "future_wrist_image": "Future wrist image",
    "future_image": "Future image",
    "value": "Value",
}
VALIDATION_METRICS = {
    "eval_loss_actor": "Actor EDM",
    "eval_policy_masked_edm_loss": "Policy masked EDM",
    "eval_action_l1_loss": "Action L1",
    "eval_future_proprio_l1_loss": "Future proprio L1",
    "eval_future_wrist_image_l1_loss": "Future wrist image L1",
    "eval_future_image_l1_loss": "Future image L1",
    "eval_value_l1_loss": "Value L1",
}
OPTIMIZATION_METRICS = {
    "gradient_norm": "Gradient norm",
    "learning_rate": "Learning rate",
    "loss_scale": "Loss scale",
    "samples_per_second": "Samples / second",
    "step_time_seconds": "Seconds / optimizer step",
    "gpu_memory_allocated_gib": "GPU memory allocated (GiB)",
}
OPTIMIZATION_YLABELS = {
    "gradient_norm": "L2 norm",
    "learning_rate": "learning rate",
    "loss_scale": "loss scale",
    "samples_per_second": "samples / second",
    "step_time_seconds": "seconds",
    "gpu_memory_allocated_gib": "GiB",
}
LEGACY_IMAGES = {
    "loss_total.png",
    "loss_objectives.png",
    "loss_regions_edm.png",
    "loss_regions_mse.png",
    "loss_regions_l1.png",
    "action_physical.png",
    "optimization.png",
    "train_eval_loss_actor.png",
}
RESULT_IMAGES = {
    "train_eval_edm_loss.png",
    "objective_edm_loss.png",
    "latent_region_edm_loss.png",
    "validation_metrics.png",
    "optimization_health.png",
    "checkpoint_comparison.png",
}


def deduplicate_records(records: Iterable[dict]) -> list[dict]:
    """Keep latest scalar record for a repeated run/mode/step tuple."""
    output: list[dict] = []
    positions: dict[tuple[str, str, int], int] = {}
    for record in records:
        mode = str(record.get("mode", ""))
        step = record.get("global_step")
        if mode in {"train", "smoke_train", "eval", "validation"} and isinstance(step, int):
            key = (mode, str(record.get("protocol", "")), step)
            if key in positions:
                output[positions[key]] = record
                continue
            positions[key] = len(output)
        output.append(record)
    return output


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return deduplicate_records(records)


def _required_counts(metric: str) -> tuple[str, ...]:
    if metric in OBJECTIVE_COUNTS:
        return OBJECTIVE_COUNTS[metric]
    for region, counts in REGION_COUNTS.items():
        if metric.startswith(region + "_"):
            return counts
    return ()


def metric_series(
    records: Iterable[dict],
    metric: str,
    modes: frozenset[str] = frozenset({"train", "smoke_train"}),
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    required_counts = _required_counts(metric)
    for record in records:
        if record.get("mode") not in modes or metric not in record:
            continue
        if required_counts and sum(float(record.get(name, 0.0)) for name in required_counts) <= 0:
            continue
        value = float(record[metric])
        if np.isfinite(value):
            points.append((float(record["global_step"]), value))
    points.sort(key=lambda item: item[0])
    if not points:
        return np.asarray([]), np.asarray([])
    steps, values = zip(*points)
    return np.asarray(steps), np.asarray(values)


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmos_offline_train_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axis(axis, *, xlabel: str = "optimizer step", ylabel: str = "EDM loss") -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _plot_train_eval(records: list[dict], path: Path) -> bool:
    train_steps, train_values = metric_series(records, "loss_actor_learner")
    eval_steps, eval_values = metric_series(
        records, "eval_loss_actor", modes=frozenset({"eval"})
    )
    if train_values.size == 0 and eval_values.size == 0:
        return False
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(11, 6))
    if train_values.size:
        axis.plot(train_steps, train_values, label="Train actor EDM", linewidth=1.0)
    if eval_values.size:
        axis.plot(eval_steps, eval_values, "o-", label="Validation actor EDM", linewidth=1.8)
    axis.set_title("Train and validation actor EDM loss")
    _style_axis(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _plot_objectives(records: list[dict], path: Path) -> bool:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(11, 6))
    plotted = False
    for metric, label in OBJECTIVE_LABELS.items():
        steps, values = metric_series(records, metric)
        if values.size == 0:
            continue
        axis.plot(steps, values, label=label, linewidth=1.0)
        plotted = True
    if not plotted:
        plt.close(figure)
        return False
    axis.set_title("Training EDM loss by objective")
    _style_axis(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _plot_panels(
    records: list[dict],
    metrics: dict[str, str],
    path: Path,
    *,
    title: str,
    modes: frozenset[str],
    ylabel_for_metric,
) -> bool:
    series = []
    for metric, label in metrics.items():
        steps, values = metric_series(records, metric, modes=modes)
        if values.size:
            series.append((metric, label, steps, values))
    if not series:
        return False
    plt = _matplotlib()
    columns = 2
    rows = (len(series) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.5 * rows), squeeze=False)
    for axis, (metric, label, steps, values) in zip(axes.flat, series):
        marker = "o" if modes == frozenset({"eval"}) else None
        axis.plot(steps, values, linewidth=1.0, marker=marker)
        axis.set_title(label)
        _style_axis(axis, ylabel=ylabel_for_metric(metric))
        if metric == "gpu_memory_allocated_gib":
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.set_ylim(0, max(1.0, float(values.max()) * 1.1))
        elif metric == "loss_scale" and np.ptp(values) == 0:
            axis.set_ylim(0, max(1.0, float(values[0]) * 1.1))
    for axis in list(axes.flat)[len(series):]:
        axis.remove()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _checkpoint_steps(metrics_path: Path, records: list[dict]) -> list[int]:
    available = {
        int(record["global_step"])
        for record in records
        if record.get("mode") == "eval" and isinstance(record.get("global_step"), int)
    }
    selected = {0} if 0 in available else set()
    checkpoint_dir = metrics_path.parent / "checkpoints"
    for path in checkpoint_dir.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if match:
            selected.add(int(match.group(1)))
    return sorted(selected & available)


def _plot_checkpoint_comparison(records: list[dict], metrics_path: Path, path: Path) -> bool:
    steps = _checkpoint_steps(metrics_path, records)
    if len(steps) < 2:
        return False
    by_step = {
        int(record["global_step"]): record
        for record in records
        if record.get("mode") == "eval" and int(record.get("global_step", -1)) in steps
    }
    available_metrics = [
        (metric, label)
        for metric, label in VALIDATION_METRICS.items()
        if all(metric in by_step[step] for step in steps)
    ]
    if not available_metrics:
        return False
    plt = _matplotlib()
    columns = 3
    rows = (len(available_metrics) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows), squeeze=False)
    labels = ["Base" if step == 0 else f"Step {step}" for step in steps]
    for axis, (metric, label) in zip(axes.flat, available_metrics):
        values = [float(by_step[step][metric]) for step in steps]
        axis.bar(labels, values)
        axis.set_title(label)
        axis.set_ylabel("EDM loss" if "edm" in metric or metric == "eval_loss_actor" else "L1")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    for axis in list(axes.flat)[len(available_metrics):]:
        axis.remove()
    figure.suptitle("Validation comparison at saved checkpoints")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _write_csv(records: list[dict], path: Path, modes: frozenset[str]) -> None:
    selected = [record for record in records if record.get("mode") in modes]
    fields = ["mode", "training_mode", "global_step", "samples_seen"]
    extras = sorted({key for record in selected for key in record} - set(fields))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)


def _summary(records: list[dict], metrics: Iterable[str], tail: int = 100) -> dict:
    output: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        _, values = metric_series(records, metric)
        if values.size == 0:
            continue
        width = min(tail, values.size)
        output[metric] = {
            "points": int(values.size),
            "first_mean": float(values[:width].mean()),
            "last_mean": float(values[-width:].mean()),
            "minimum": float(values.min()),
        }
    return output


def _remove_owned_images(output_dir: Path) -> None:
    for name in LEGACY_IMAGES | RESULT_IMAGES:
        (output_dir / name).unlink(missing_ok=True)


def generate(metrics_path: Path, output_dir: Path, window: int | None = None) -> list[Path]:
    """Generate raw, unsmoothed plots. ``window`` remains only for API compatibility."""
    del window
    records = load_records(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_owned_images(output_dir)
    outputs: list[Path] = []

    plotters = [
        ("train_eval_edm_loss.png", lambda path: _plot_train_eval(records, path)),
        ("objective_edm_loss.png", lambda path: _plot_objectives(records, path)),
        (
            "latent_region_edm_loss.png",
            lambda path: _plot_panels(
                records,
                {f"{region}_edm_loss": label for region, label in REGION_LABELS.items()},
                path,
                title="Training EDM loss by latent position",
                modes=frozenset({"train", "smoke_train"}),
                ylabel_for_metric=lambda _metric: "EDM loss",
            ),
        ),
        (
            "validation_metrics.png",
            lambda path: _plot_panels(
                records,
                VALIDATION_METRICS,
                path,
                title="Validation metrics",
                modes=frozenset({"eval"}),
                ylabel_for_metric=lambda metric: (
                    "EDM loss" if "edm" in metric or metric == "eval_loss_actor" else "L1"
                ),
            ),
        ),
        (
            "optimization_health.png",
            lambda path: _plot_panels(
                records,
                OPTIMIZATION_METRICS,
                path,
                title="Optimization stability",
                modes=frozenset({"train", "smoke_train"}),
                ylabel_for_metric=lambda metric: OPTIMIZATION_YLABELS[metric],
            ),
        ),
        (
            "checkpoint_comparison.png",
            lambda path: _plot_checkpoint_comparison(records, metrics_path, path),
        ),
    ]
    for filename, plotter in plotters:
        path = output_dir / filename
        if plotter(path):
            outputs.append(path)

    train_metrics = sorted(
        {key for record in records for key in record if key.endswith(("_loss", "_mse", "_l1"))}
    )
    csv_path = output_dir / "train_metrics.csv"
    _write_csv(records, csv_path, frozenset({"train", "smoke_train"}))
    outputs.append(csv_path)
    eval_csv_path = output_dir / "eval_metrics.csv"
    _write_csv(records, eval_csv_path, frozenset({"eval"}))
    outputs.append(eval_csv_path)
    summary_path = output_dir / "loss_summary.json"
    summary_path.write_text(
        json.dumps(_summary(records, train_metrics), indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs.append(summary_path)
    validation_path = output_dir / "validation_records.json"
    validation_path.write_text(
        json.dumps(
            [record for record in records if record.get("mode") in {"validation", "eval"}],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    outputs.append(validation_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Deprecated compatibility option; plots always use raw values.",
    )
    args = parser.parse_args()
    metrics_path = args.metrics.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else metrics_path.parent / "plots"
    )
    for path in generate(metrics_path, output_dir):
        print(path)


if __name__ == "__main__":
    main()
