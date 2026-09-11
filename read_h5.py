#!/usr/bin/env python3
"""读取并打印 HDF5 文件结构（适用于 LIBERO demo 等）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _visit(name: str, obj) -> None:
    import h5py

    indent = name.count("/") * 2
    pad = " " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}[D] {name.split('/')[-1] or '/'}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        short = name.split("/")[-1] or "/"
        n_keys = len(obj.keys())
        print(f"{pad}[G] {short}/  ({n_keys} children)")


def print_structure(h5_path: Path) -> None:
    import h5py

    with h5py.File(h5_path, "r") as f:
        print(f"File: {h5_path.resolve()}")
        f.visititems(_visit)


def _print_group_summary(grp, indent: str, head: int) -> None:
    """打印 demo 下全部 Dataset / Group（预览）。"""
    import h5py

    for k in sorted(grp.keys()):
        obj = grp[k]
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{k}: shape={obj.shape} dtype={obj.dtype}")
            if head <= 0 or obj.size == 0:
                continue
            try:
                if obj.ndim == 0:
                    print(f"{indent}  value: {obj[()]}")
                else:
                    n = min(head, int(obj.shape[0]))
                    sl = obj[:n]
                    print(f"{indent}  first rows (up to {head}): {sl}")
            except Exception as e:
                print(f"{indent}  (read preview failed: {e})")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{k}/  [Group, {len(obj.keys())} keys]")
            _print_group_summary(obj, indent + "  ", head)


def _export_actions_only(grp, rel_prefix: str, out_path: Path, written: list[int]) -> None:
    """只查找名为 actions 的 Dataset；每行等宽空格对齐：步号 + 各维浮点。"""
    import h5py
    import numpy as np

    for k in sorted(grp.keys()):
        obj = grp[k]
        rel = f"{rel_prefix}/{k}" if rel_prefix else k
        if isinstance(obj, h5py.Dataset) and k == "actions":
            arr = np.asarray(obj[:], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n_rows = len(arr)
            w_step = max(4, len(str(max(0, n_rows - 1))))
            w_float = 14
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if written[0] == 0 else "a"
            with open(out_path, mode, encoding="utf-8") as fp:
                if written[0] > 0:
                    fp.write(f"\n# --- {rel} ---\n")
                for interaction_step, step_action in enumerate(arr):
                    vec = np.asarray(step_action).reshape(-1)
                    cells = [f"{interaction_step:>{w_step}d}"] + [f"{float(x):>{w_float}.6f}" for x in vec]
                    fp.write(" ".join(cells) + "\n")
            written[0] += 1
        elif isinstance(obj, h5py.Group):
            _export_actions_only(obj, rel, out_path, written)


def print_demo_summary(h5_path: Path, demo_key: str, head: int, actions_out: Path | None) -> None:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            print("No top-level group 'data'.", file=sys.stderr)
            return
        if demo_key not in f["data"]:
            keys = list(f["data"].keys())[:20]
            print(f"No group data/{demo_key}. First keys: {keys}", file=sys.stderr)
            return
        g = f["data"][demo_key]

        if actions_out is not None:
            written = [0]
            _export_actions_only(g, f"data/{demo_key}", actions_out, written)
            if written[0] == 0:
                print(f"未找到名为 actions 的 Dataset（在 data/{demo_key} 下）。", file=sys.stderr)
                return
            print(f"已写入 {written[0]} 个 actions 数据集 -> {actions_out.resolve()}")
            return

        print(f"\n=== data/{demo_key} ===")
        _print_group_summary(g, "  ", head)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 file structure and optional demo slice.")
    parser.add_argument("h5_path", type=Path, help="Path to .h5 / .hdf5 file")
    parser.add_argument(
        "--demo",
        type=str,
        default="",
        help="If set, print summary for data/<demo> (e.g. demo_0)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=3,
        help="With --demo (no --actions-out), show first N rows of each dataset (0 to disable)",
    )
    parser.add_argument(
        "--actions-out",
        type=Path,
        default=None,
        help="With --demo: only export dataset(s) named 'actions' to this .txt file (no other demo output)",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Skip printing full file tree (visititems)",
    )
    args = parser.parse_args()

    if not args.h5_path.is_file():
        print(f"Not a file: {args.h5_path}", file=sys.stderr)
        sys.exit(1)

    if args.actions_out is not None and not args.demo.strip():
        print("使用 --actions-out 时必须同时指定 --demo", file=sys.stderr)
        sys.exit(1)

    try:
        import h5py  # noqa: F401
    except ImportError:
        print("需要安装 h5py: pip install h5py", file=sys.stderr)
        sys.exit(1)

    if not args.no_structure:
        print_structure(args.h5_path)
    if args.demo.strip():
        print_demo_summary(args.h5_path, args.demo.strip(), args.head, args.actions_out)


if __name__ == "__main__":
    main()
