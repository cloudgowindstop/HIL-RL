"""Single-file checkpoint for exact optimizer resume."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _net(model):
    net = model.net
    return net.module if hasattr(net, "module") else net


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    samples_seen: int,
    config: dict[str, Any],
    data_identity: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _net(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "data_identity": dict(data_identity or {}),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": _cpu_byte_rng_states(torch.cuda.get_rng_state_all()),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_byte_rng_states(states: Any) -> list[torch.Tensor]:
    """CUDA RNG restore requires CPU uint8 ByteTensors, not GPU-mapped copies."""
    if states is None:
        return []
    if isinstance(states, torch.Tensor):
        states = [states]
    restored: list[torch.Tensor] = []
    for state in states:
        if not isinstance(state, torch.Tensor):
            continue
        restored.append(state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
    return restored


def _restore_cuda_rng(states: Any) -> None:
    if not torch.cuda.is_available():
        return
    restored = _cpu_byte_rng_states(states)
    if not restored:
        return
    device_count = torch.cuda.device_count()
    if len(restored) > device_count:
        print(
            f"[CKPT] cuda_rng has {len(restored)} states, this process sees "
            f"{device_count} GPU(s); restoring the first {device_count}",
            flush=True,
        )
        restored = restored[:device_count]
    for index, state in enumerate(restored):
        torch.cuda.set_rng_state(state, index)


def _validate_identity(
    saved: dict[str, str], expected: dict[str, str] | None
) -> None:
    if expected is None:
        return
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={old!r}, current={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise RuntimeError(f"checkpoint 数据身份不匹配: {details}")


def load_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    expected_identity: dict[str, str] | None = None,
) -> dict[str, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    _validate_identity(payload.get("data_identity", {}), expected_identity)
    _net(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    _restore_cuda_rng(payload.get("cuda_rng"))
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "samples_seen": int(payload["samples_seen"]),
        "batch_in_epoch": int(payload.get("batch_in_epoch", 0)),
    }


def load_checkpoint_for_evaluation(
    path: Path,
    *,
    model,
    device,
    expected_identity: dict[str, str] | None = None,
) -> dict[str, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    _validate_identity(payload.get("data_identity", {}), expected_identity)
    _net(model).load_state_dict(payload["model"], strict=True)
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "samples_seen": int(payload["samples_seen"]),
        "batch_in_epoch": int(payload.get("batch_in_epoch", 0)),
    }
