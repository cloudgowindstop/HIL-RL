#!/usr/bin/env python3
"""
分析 demo_action_l1_spike 目录中相邻两步的损失跳变。

用法:
  python analyze_demo_action_l1_spike.py /path/to/spike_00000790_to_00000791_dl0.2111
  python analyze_demo_action_l1_spike.py   # 使用下面 DEFAULT_SPIKE_DIR

依据 learner_cosmos.py 中 snapshot.pt 结构:
  threshold, loss_delta, prev, curr
  每个 entry: optimization_step, demo_sample_action_l1_loss,
              forward_batch_all_tensors_cpu, wrist_image, primary_image,
              action, output_metrics
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import torch

# 默认示例目录（可改为你的实验路径）
DEFAULT_SPIKE_DIR = (
    "/media/HIL-RL-Project/HIL-RL/experiments/close_trashbin_franka_1028/"
    "logs/demo_action_l1_spike/spike_00000790_to_00000791_dl0.2111"
)

# 用于判断「是否是同一批 rollout / 样本」的常见键（若不一致，跳变可能来自数据而非同一噪声）
_BATCH_IDENTITY_KEYS = (
    "__key__",
    "global_rollout_idx",
    "rollout_data_mask",
    "rollout_data_success_mask",
)


def _walk_tensors(d: dict[str, Any], prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    for k in sorted(d.keys()):
        v = d[k]
        path = f"{prefix}{k}" if prefix else k
        if isinstance(v, torch.Tensor):
            out.append((path, v))
        elif isinstance(v, dict):
            out.extend(_walk_tensors(v, path + "."))
    return out


def _tensor_stats_delta(
    a: torch.Tensor, b: torch.Tensor
) -> dict[str, Any] | None:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
        }
    af = a.detach().float().cpu().contiguous()
    bf = b.detach().float().cpu().contiguous()
    diff = (af - bf).abs()
    return {
        "shape_match": True,
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "rmse": float(torch.sqrt((diff**2).mean())),
        "mean_a": float(af.mean()),
        "mean_b": float(bf.mean()),
    }


def _exact_tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.cpu(), b.cpu()))


def _compare_batch_identity(fb_p: dict, fb_c: dict) -> list[str]:
    notes: list[str] = []
    for key in _BATCH_IDENTITY_KEYS:
        if key not in fb_p or key not in fb_c:
            continue
        vp, vc = fb_p[key], fb_c[key]
        if isinstance(vp, torch.Tensor) and isinstance(vc, torch.Tensor):
            if not _exact_tensor_equal(vp, vc):
                notes.append(
                    f"  [{key}] 不一致: 本步 minibatch 与上一步不同（索引/键变化），"
                    "损失跳变可能部分来自不同轨迹或样本。"
                )
        elif vp != vc:
            notes.append(f"  [{key}] 不一致: {vp!r} vs {vc!r}")
    return notes


def analyze_snapshot(spike_dir: str) -> None:
    # 加载snapshot.pt文件
    snap_path = os.path.join(spike_dir, "snapshot.pt")
    if not os.path.isfile(snap_path):
        print(f"错误: 找不到 {snap_path}", file=sys.stderr)
        sys.exit(1)

    obj = torch.load(snap_path, map_location="cpu", weights_only=False)
    threshold = float(obj.get("threshold", float("nan")))
    loss_delta = float(obj.get("loss_delta", float("nan")))
    prev: dict[str, Any] = obj["prev"]
    curr: dict[str, Any] = obj["curr"]

    step_p = int(prev["optimization_step"])
    step_c = int(curr["optimization_step"])
    l_p = float(prev["demo_sample_action_l1_loss"])
    l_c = float(curr["demo_sample_action_l1_loss"])

    print("=" * 72)
    print("demo_sample_action_l1_loss 相邻步跳变分析")
    print("=" * 72)
    print(f"目录: {spike_dir}")
    print(f"阈值 threshold: {threshold}")
    print(f"prev_step={step_p}  loss={l_p:.6g}")
    print(f"curr_step={step_c}  loss={l_c:.6g}")
    print(f"loss_delta (curr - prev): {loss_delta:.6g}  (记录值应与 {l_c - l_p:.6g} 一致)")
    print()

    fb_p = prev.get("forward_batch_all_tensors_cpu") or {}
    fb_c = curr.get("forward_batch_all_tensors_cpu") or {}

    id_notes = _compare_batch_identity(fb_p, fb_c)
    if id_notes:
        print("[批次一致性]")
        print("\n".join(id_notes))
        print()
    else:
        print(
            "[批次一致性] 已检查键 "
            + ", ".join(_BATCH_IDENTITY_KEYS)
            + " — 未发现上述键的不一致（或未保存）。"
        )
        print()

    om_p = prev.get("output_metrics") or {}
    om_c = curr.get("output_metrics") or {}
    all_metric_keys = sorted(set(om_p.keys()) | set(om_c.keys()))
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for k in all_metric_keys:
        vp = om_p.get(k)
        vc = om_c.get(k)
        if vp is None or vc is None:
            rows.append((k, vp, vc, None))
            continue
        d = float(vc) - float(vp)
        rows.append((k, float(vp), float(vc), d))

    rows.sort(key=lambda r: abs(r[3]) if r[3] is not None else -1.0, reverse=True)
    print("[output_metrics 标量变化 curr - prev]（按 |Δ| 降序）")
    print(f"{'metric':<48} {'prev':>14} {'curr':>14} {'delta':>14}")
    print("-" * 92)
    for k, vp, vc, d in rows:
        if d is None:
            print(f"{k:<48} {str(vp):>14} {str(vc):>14} {'N/A':>14}")
        else:
            print(f"{k:<48} {vp:>14.6g} {vc:>14.6g} {d:>+14.6g}")
    print()

    # forward_batch 张量两两对比（仅同名同路径）
    tensors_p = dict(_walk_tensors(fb_p))
    tensors_c = dict(_walk_tensors(fb_c))
    all_paths = sorted(set(tensors_p.keys()) | set(tensors_c.keys()))
    tensor_rows: list[tuple[str, dict[str, Any]]] = []
    for path in all_paths:
        tp = tensors_p.get(path)
        tc = tensors_c.get(path)
        if tp is None or tc is None:
            tensor_rows.append((path, {"missing": True}))
            continue
        st = _tensor_stats_delta(tp, tc)
        if st is not None:
            tensor_rows.append((path, st))

    # 按 MAE 排序（形状不匹配单独列出）
    def sort_key(item: tuple[str, dict[str, Any]]) -> float:
        pth, st = item
        if st.get("missing"):
            return -1.0
        if not st.get("shape_match", True):
            return 1e9
        return float(st.get("mae", 0.0))

    tensor_rows.sort(key=sort_key, reverse=True)

    print("[forward_batch 张量 PREV vs CURR]")
    print(
        f"{'path':<52} {'MAE':>12} {'max|Δ|':>12} {'mean_prev':>12} {'mean_curr':>12}  note"
    )
    print("-" * 110)
    for path, st in tensor_rows[:40]:
        if st.get("missing"):
            print(f"{path:<52} {'—':>12} {'—':>12} {'—':>12} {'—':>12}  仅一侧存在")
            continue
        if not st.get("shape_match"):
            print(
                f"{path:<52} {'—':>12} {'—':>12} {'—':>12} {'—':>12}  "
                f"shape {st['shape_a']} vs {st['shape_b']}"
            )
            continue
        print(
            f"{path:<52} {st['mae']:>12.6g} {st['max_abs']:>12.6g} "
            f"{st['mean_a']:>12.6g} {st['mean_b']:>12.6g}"
        )
    if len(tensor_rows) > 40:
        print(f"... 其余 {len(tensor_rows) - 40} 个张量路径已省略（可按需增大切片）")
    print()

    comparable_maes = [
        float(st["mae"])
        for _, st in tensor_rows
        if st.get("shape_match") and not st.get("missing")
    ]
    if comparable_maes and max(comparable_maes) < 1e-12:
        print(
            "[结论] forward_batch 内所有可对齐张量与上一步完全一致（MAE≈0）。"
            "本次 demo_sample_action_l1_loss 上升不是换样本导致，"
            "请结合 output_metrics 中的 sigma / edm_loss / 噪声采样等训练随机因素解读。"
        )
        print()

    # actions：逐样本 L1（若形状一致）
    ap = fb_p.get("actions")
    ac = fb_c.get("actions")
    if isinstance(ap, torch.Tensor) and isinstance(ac, torch.Tensor) and ap.shape == ac.shape:
        # [B, T, D] -> per-sample mean L1 over T,D
        d = (ap.float() - ac.float()).abs()
        if d.ndim >= 2:
            per_b = d.reshape(d.shape[0], -1).mean(dim=1).cpu().numpy()
            print("[actions 逐样本平均 L1 |curr-prev|]（排查是否个别样本导致跳变）")
            print(f"  batch 维大小: {per_b.shape[0]}")
            print(f"  全局 MAE: {float(per_b.mean()):.6g}")
            worst = int(np.argmax(per_b))
            print(f"  最大样本索引: {worst}  MAE={float(per_b[worst]):.6g}")
            print()

    print("解读提示:")
    print("  - 若 global_rollout_idx / __key__ 等变化：相邻两步抽到的轨迹不同，")
    print("    action L1 升高可能是新样本更难，而非同一输入下模型退化。")
    print("  - 若批次一致但 demo_sample_action_l1_loss 上升：可看 actions/video 等 MAE")
    print("    与 output_metrics 中其它损失是否同步恶化。")
    print("  - 完整原始张量始终以 snapshot.pt 为准；report.txt 为摘要。")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 demo_action_l1_spike 目录中的 snapshot.pt")
    parser.add_argument(
        "spike_dir",
        nargs="?",
        default=DEFAULT_SPIKE_DIR,
        help="spike 子目录路径（内含 snapshot.pt）",
    )
    args = parser.parse_args()
    analyze_snapshot(os.path.abspath(args.spike_dir))


if __name__ == "__main__":
    main()
