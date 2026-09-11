"""可选shape打印和始终启用的shape强制校验。"""

from __future__ import annotations

import os
from typing import Any


def require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    """维度不一致时立即终止；不使用可能被python -O关闭的assert。"""
    actual = tuple(value.shape)
    if actual != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")


def trace_shape(stage: str, **values: Any) -> None:
    """COSMOS_TRACE_SHAPES=1时打印shape/dtype，不计算min/max或触发GPU同步。"""
    if os.environ.get("COSMOS_TRACE_SHAPES") != "1":
        return
    fields = []
    for name, value in values.items():
        shape = tuple(value.shape)
        dtype = getattr(value, "dtype", type(value).__name__)
        fields.append(f"{name}={shape} dtype={dtype}")
    print(f"[SHAPE][{stage}] " + " | ".join(fields), flush=True)
