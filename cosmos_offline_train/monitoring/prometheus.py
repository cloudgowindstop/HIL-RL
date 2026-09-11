"""Small dependency-free Prometheus exporter for rank-zero training metrics."""

from __future__ import annotations

import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_TRAIN_METRICS = {
    "total_edm_loss": "cosmos_train_total_edm_loss",
    "loss_actor_learner": "cosmos_train_loss_actor",
    "policy_masked_edm_loss": "cosmos_train_policy_masked_edm_loss",
    "gradient_norm": "cosmos_gradient_norm",
    "learning_rate": "cosmos_learning_rate",
    "loss_scale": "cosmos_loss_scale",
    "samples_per_second": "cosmos_samples_per_second",
    "step_time_seconds": "cosmos_step_time_seconds",
    "gpu_memory_allocated_gib": "cosmos_gpu_memory_allocated_gib",
    "policy_samples": "cosmos_train_policy_samples",
    "world_samples": "cosmos_train_world_samples",
    "value_samples": "cosmos_train_value_samples",
    "inverse_dynamics_samples": "cosmos_train_inverse_dynamics_samples",
    "sample_policy_edm_loss": "cosmos_train_sample_policy_edm_loss",
    "sample_world_edm_loss": "cosmos_train_sample_world_edm_loss",
    "sample_value_edm_loss": "cosmos_train_sample_value_edm_loss",
    "sample_inverse_dynamics_edm_loss": "cosmos_train_sample_inverse_dynamics_edm_loss",
    "effective_mask_ratio": "cosmos_train_effective_mask_ratio",
    "left_translation_mae": "cosmos_train_left_translation_mae",
    "right_translation_mae": "cosmos_train_right_translation_mae",
    "left_rotation_geodesic_deg": "cosmos_train_left_rotation_geodesic_deg",
    "right_rotation_geodesic_deg": "cosmos_train_right_rotation_geodesic_deg",
    "left_gripper_mae": "cosmos_train_left_gripper_mae",
    "right_gripper_mae": "cosmos_train_right_gripper_mae",
}

for _region in ("action", "future_proprio", "future_wrist_image", "future_image", "value"):
    for _suffix in ("edm_loss", "mse", "l1"):
        _source = f"{_region}_{_suffix}"
        _TRAIN_METRICS[_source] = f"cosmos_train_{_source}"

_EVAL_METRICS = {
    "total_edm_loss": "cosmos_validation_total_edm_loss",
    "eval_loss_actor": "cosmos_eval_loss_actor",
    "eval_policy_masked_edm_loss": "cosmos_eval_policy_masked_edm_loss",
    "eval_action_l1_loss": "cosmos_eval_action_l1_loss",
    "eval_future_image_l1_loss": "cosmos_eval_future_image_l1_loss",
    "eval_future_wrist_image_l1_loss": "cosmos_eval_future_wrist_image_l1_loss",
    "eval_future_proprio_l1_loss": "cosmos_eval_future_proprio_l1_loss",
    "eval_value_l1_loss": "cosmos_eval_value_l1_loss",
    "validation_samples": "cosmos_validation_samples",
}


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


class _MetricStore:
    def __init__(self) -> None:
        self._values: dict[str, float] = {"cosmos_monitor_up": 1.0}
        self._lock = threading.Lock()

    def update(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        mapping = _EVAL_METRICS if mode in {"eval", "validation"} else _TRAIN_METRICS
        values: dict[str, float] = {}
        for source, target in mapping.items():
            value = payload.get(source)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[target] = float(value)
        for source, target in (
            ("global_step", "cosmos_global_step"),
            ("samples_seen", "cosmos_samples_seen"),
            ("epoch", "cosmos_epoch"),
        ):
            value = payload.get(source)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[target] = float(value)
        with self._lock:
            self._values.update(values)

    def render(self) -> bytes:
        with self._lock:
            values = dict(self._values)
        lines: list[str] = []
        for name, value in sorted(values.items()):
            lines.extend(
                (
                    f"# HELP {name} Cosmos offline training metric.",
                    f"# TYPE {name} gauge",
                    f"{name} {value:.12g}",
                )
            )
        return ("\n".join(lines) + "\n").encode("utf-8")


def _handler(store: _MetricStore, metrics_path: str):
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
            if self.path.split("?", 1)[0] != metrics_path:
                self.send_error(404)
                return
            body = store.render()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return MetricsHandler


class PrometheusExporter:
    """Own one rank-zero HTTP server and publish scalar JSONL payloads."""

    def __init__(
        self,
        *,
        enabled: bool,
        host: str = "0.0.0.0",
        port: int = 8000,
        path: str = "/metrics",
        serve: bool = True,
    ) -> None:
        self.enabled = False
        self.host = host
        self.port = int(port)
        self.path = path
        self._store = _MetricStore()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        if not enabled:
            return
        if not serve:
            self.enabled = True
            return
        try:
            self._server = ThreadingHTTPServer(
                (self.host, self.port), _handler(self._store, self.path)
            )
        except OSError as error:
            print(f"[MONITOR] Prometheus disabled: {error}", flush=True)
            return
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cosmos-prometheus",
            daemon=True,
        )
        self._thread.start()
        self.enabled = True
        print(
            f"[MONITOR] Prometheus listening on http://{self.host}:{self.port}{self.path}",
            flush=True,
        )

    @classmethod
    def from_config(
        cls, config: dict[str, Any], *, is_main: bool, evaluation_only: bool = False
    ) -> "PrometheusExporter":
        monitoring = config.get("monitoring", {})
        configured = bool(monitoring.get("prometheus_enabled", False))
        enabled = _environment_bool("COSMOS_METRICS_ENABLE", configured)
        host = os.environ.get(
            "COSMOS_METRICS_HOST", str(monitoring.get("host", "0.0.0.0"))
        )
        port = int(os.environ.get("COSMOS_METRICS_PORT", monitoring.get("port", 8000)))
        path = os.environ.get(
            "COSMOS_METRICS_PATH", str(monitoring.get("path", "/metrics"))
        )
        return cls(
            enabled=enabled and is_main and not evaluation_only,
            host=host,
            port=port,
            path=path,
        )

    def update(self, payload: dict[str, Any]) -> None:
        if self.enabled:
            self._store.update(payload)

    def render_metrics(self) -> str:
        """Return current exposition text for diagnostics and unit tests."""
        return self._store.render().decode("utf-8")

    def close(self) -> None:
        if self._server is None:
            self.enabled = False
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
        self.enabled = False
