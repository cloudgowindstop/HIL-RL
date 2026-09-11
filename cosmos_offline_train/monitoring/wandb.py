"""Rank-zero Weights & Biases exporter for JSONL training payloads."""

from __future__ import annotations

import math
import os
from typing import Any


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


class WandBExporter:
    """Log rank-zero scalar payloads to WandB. Missing package only errors when enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        mode: str = "online",
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.enabled = False
        self.project = project
        self.name = name
        self.entity = entity
        self.mode = mode
        self.run_url = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "monitoring.wandb_enabled=true 但未安装 wandb；"
                "请在训练环境执行 pip install wandb"
            ) from error
        self._wandb = wandb.init(
            project=project,
            name=name or None,
            entity=entity or None,
            id=run_id or None,
            resume="allow" if run_id else None,
            mode=mode,
            config=config or {},
            dir=output_dir,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        self.enabled = True
        self.run_url = getattr(wandb.run, "url", None)
        print(f"[MONITOR] WandB run={self.name or wandb.run.name} url={self.run_url}", flush=True)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        is_main: bool,
        evaluation_only: bool = False,
    ) -> "WandBExporter":
        monitoring = config.get("monitoring", {})
        configured = bool(monitoring.get("wandb_enabled", False))
        enabled = _environment_bool("COSMOS_WANDB_ENABLE", configured)
        project = str(
            monitoring.get("wandb_project")
            or os.environ.get("WANDB_PROJECT")
            or "cosmos-offline-compare"
        )
        name = str(
            monitoring.get("wandb_name")
            or os.environ.get("WANDB_NAME")
            or config.get("evaluation", {}).get("compare_name")
            or ""
        )
        entity = os.environ.get("WANDB_ENTITY", monitoring.get("wandb_entity") or "")
        mode = os.environ.get(
            "WANDB_MODE", str(monitoring.get("wandb_mode", "online"))
        ).strip().lower()
        run_id = os.environ.get("WANDB_RUN_ID", monitoring.get("wandb_run_id") or "")
        output_dir = config.get("runtime", {}).get("output_dir")
        return cls(
            enabled=enabled and is_main and not evaluation_only and mode != "disabled",
            project=project,
            name=name or None,
            entity=entity or None,
            mode=mode if mode in {"online", "offline", "disabled"} else "online",
            run_id=run_id or None,
            config={
                "action_encoding": config.get("data", {}).get("action_encoding"),
                "action_dimension": config.get("data", {}).get("action_dimension"),
                "training_mode": config.get("objectives", {}).get("training_mode"),
                "max_steps": config.get("training", {}).get("max_steps"),
                "batch_size_per_rank": config.get("training", {}).get("batch_size_per_rank"),
                "gradient_accumulation_steps": config.get("training", {}).get(
                    "gradient_accumulation_steps"
                ),
                "seed": config.get("training", {}).get("seed"),
                "output_dir": output_dir,
            },
            output_dir=str(output_dir) if output_dir else None,
        )

    def update(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._wandb is None:
            return
        mode = str(payload.get("mode", "train"))
        prefix = "eval" if mode in {"eval", "validation"} else "train"
        data: dict[str, float] = {}
        for key, value in payload.items():
            if key in {"mode", "global_step"}:
                continue
            number = _scalar(value)
            if number is not None:
                data[f"{prefix}/{key}"] = number
        if not data:
            return
        step = payload.get("global_step")
        if isinstance(step, (int, float)):
            data["global_step"] = int(step)
        self._wandb.log(data, commit=True)

    def close(self) -> None:
        if self._wandb is None:
            self.enabled = False
            return
        self._wandb.finish()
        self._wandb = None
        self.enabled = False
