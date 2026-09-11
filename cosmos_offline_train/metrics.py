"""Small metric accumulator and JSONL logger."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class MeanMetrics:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def update(self, values: dict[str, float], *, weight: float = 1.0) -> None:
        for name, value in values.items():
            self.sums[name] += float(value) * float(weight)
            self.counts[name] += float(weight)

    def compute(self) -> dict[str, float]:
        return {name: total / max(1, self.counts[name]) for name, total in self.sums.items()}


class JsonlLogger:
    def __init__(
        self,
        output_dir: Path,
        exporter: Any | None = None,
        exporters: list[Any] | None = None,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "metrics.jsonl"
        self.exporters = [item for item in ((exporter,) if exporter is not None else ()) ]
        if exporters:
            self.exporters.extend(exporters)

    def log(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        for exporter in self.exporters:
            exporter.update(payload)
