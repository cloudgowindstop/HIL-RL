"""Checkpoint summary comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def compare_summaries(named_paths: dict[str, Path], output_dir: Path) -> tuple[Path, Path]:
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in named_paths.items()
    }
    metric_names = sorted(set.intersection(*(set(payload) for payload in payloads.values())))
    rows = []
    for metric in metric_names:
        if all(isinstance(payload[metric], (int, float)) for payload in payloads.values()):
            rows.append({"metric": metric, **{name: payload[metric] for name, payload in payloads.items()}})
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_comparison.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = output_dir / "checkpoint_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", *payloads])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def compare_episode_summaries(
    named_paths: dict[str, Path], output_dir: Path
) -> tuple[Path, Path]:
    """Compare episode-level mean and bootstrap CI across checkpoints."""
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in named_paths.items()
    }
    metric_names = sorted(set.intersection(*(set(payload) for payload in payloads.values())))
    statistics = ("episodes", "mean", "median", "std", "ci95_lower", "ci95_upper")
    rows = []
    for metric in metric_names:
        if not all(isinstance(payload[metric], dict) for payload in payloads.values()):
            continue
        row = {"metric": metric}
        for name, payload in payloads.items():
            row.update(
                {
                    f"{name}_{statistic}": payload[metric].get(statistic)
                    for statistic in statistics
                }
            )
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_episode_comparison.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = output_dir / "checkpoint_episode_comparison.csv"
    fieldnames = [
        "metric",
        *(
            f"{name}_{statistic}"
            for name in payloads
            for statistic in statistics
        ),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
