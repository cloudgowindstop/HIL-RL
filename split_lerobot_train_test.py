#!/usr/bin/env python3
"""按 episode 将本地 LeRobot v2.1 数据集划分为 train / eval

整条 episode 只会进入其中一个子集，避免同一轨迹泄漏到两边
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_parquet_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def copy_episode_videos(
    src_root: Path,
    dst_root: Path,
    old_ep: int,
    new_ep: int,
    chunks_size: int,
    video_keys: list[str],
) -> int:
    """Copy per-episode mp4 files if present. Returns number of files copied."""
    if not video_keys:
        return 0

    copied = 0
    old_chunk = old_ep // chunks_size
    new_chunk = new_ep // chunks_size
    for key in video_keys:
        # Common LeRobot layouts
        candidates = [
            src_root / "videos" / f"chunk-{old_chunk:03d}" / key / f"episode_{old_ep:06d}.mp4",
            src_root / "videos" / key / f"episode_{old_ep:06d}.mp4",
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            continue

        if (src_root / "videos" / f"chunk-{old_chunk:03d}" / key).exists():
            dst = dst_root / "videos" / f"chunk-{new_chunk:03d}" / key / f"episode_{new_ep:06d}.mp4"
        else:
            dst = dst_root / "videos" / key / f"episode_{new_ep:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def build_subset(
    src_root: Path,
    out_root: Path,
    selected_eps: list[int],
    info: dict,
    episodes: list[dict],
    ep_stats_by_idx: dict[int, dict],
    split_name: str,
) -> dict:
    if out_root.exists():
        raise SystemExit(f"输出目录已存在，请先备份/删除: {out_root}")

    chunks_size = int(info.get("chunks_size", 1000))
    features = info["features"]
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    out_meta = out_root / "meta"
    out_meta.mkdir(parents=True)

    # tasks.jsonl 原样复制
    tasks_src = src_root / "meta" / "tasks.jsonl"
    if tasks_src.exists():
        shutil.copy2(tasks_src, out_meta / "tasks.jsonl")

    episodes_by_idx = {e["episode_index"]: e for e in episodes}
    episodes_out: list[dict] = []
    ep_stats_out: list[dict] = []
    frame_offset = 0
    videos_copied = 0

    pbar = tqdm(
        enumerate(selected_eps),
        total=len(selected_eps),
        desc=f"Building {split_name}",
        unit="ep",
    )
    for new_idx, old_idx in pbar:
        src_parquet = episode_parquet_path(src_root, old_idx, chunks_size)
        if not src_parquet.exists():
            raise FileNotFoundError(f"缺少 episode parquet: {src_parquet}")

        new_chunk = new_idx // chunks_size
        dst_parquet = (
            out_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_idx:06d}.parquet"
        )
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_parquet)
        cols = table.to_pydict()
        n = len(cols["index"])
        cols["episode_index"] = [new_idx] * n
        cols["index"] = list(range(frame_offset, frame_offset + n))
        pq.write_table(pa.Table.from_pydict(cols), dst_parquet)

        ep_meta = dict(episodes_by_idx[old_idx])
        ep_meta["episode_index"] = new_idx
        if "length" not in ep_meta:
            ep_meta["length"] = n
        episodes_out.append(ep_meta)

        if old_idx in ep_stats_by_idx:
            ep_stats_out.append({"episode_index": new_idx, "stats": ep_stats_by_idx[old_idx]})

        videos_copied += copy_episode_videos(
            src_root, out_root, old_idx, new_idx, chunks_size, video_keys
        )
        frame_offset += n
        pbar.set_postfix(src_ep=old_idx, frames=frame_offset, refresh=False)

    write_jsonl(out_meta / "episodes.jsonl", episodes_out)
    if ep_stats_out:
        write_jsonl(out_meta / "episodes_stats.jsonl", ep_stats_out)

    out_info = dict(info)
    out_info["total_episodes"] = len(selected_eps)
    out_info["total_frames"] = frame_offset
    out_info["total_chunks"] = max(1, (len(selected_eps) + chunks_size - 1) // chunks_size)
    out_info["total_videos"] = videos_copied
    out_info["splits"] = {split_name: f"0:{len(selected_eps)}"}
    (out_meta / "info.json").write_text(json.dumps(out_info, indent=4, ensure_ascii=False) + "\n")

    # 可选：聚合全局 stats.json
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.utils import write_stats

        if ep_stats_out:
            stats = aggregate_stats([e["stats"] for e in ep_stats_out])
            write_stats(stats, out_root)
    except Exception as e:
        print(f"[{split_name}] skip stats.json: {e}")

    summary = {
        "split": split_name,
        "root": str(out_root),
        "num_episodes": len(selected_eps),
        "num_frames": frame_offset,
        "source_episode_indices": selected_eps,
    }
    (out_meta / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"[{split_name}] -> {out_root}  episodes={len(selected_eps)} frames={frame_offset} videos={videos_copied}"
    )
    return summary


def split_indices(num_episodes: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 需在 (0,1) 内，收到: {train_ratio}")

    indices = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(round(num_episodes * train_ratio))
    # 保证两边都非空（至少各 1 条，前提是总 episode >= 2）
    if num_episodes >= 2:
        n_train = min(max(n_train, 1), num_episodes - 1)

    train_eps = sorted(indices[:n_train])
    test_eps = sorted(indices[n_train:])
    return train_eps, test_eps


def main() -> None:
    parser = argparse.ArgumentParser(description="按 episode 8:2 划分 LeRobot 数据集为 train/test")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/install_handle"),
        help="源 LeRobot 数据集 root",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="输出根目录（其下会创建 train/ 与 test/）。默认: {root}_split",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集 episode 比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印划分结果，不写盘",
    )
    args = parser.parse_args()

    src_root = args.root.resolve()
    out_dir = (args.out_dir or Path(str(src_root) + "_split")).resolve()

    info_path = src_root / "meta" / "info.json"
    episodes_path = src_root / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise SystemExit(f"不是有效的 LeRobot 数据集: {src_root}")

    info = json.loads(info_path.read_text())
    episodes = load_jsonl(episodes_path)
    num_episodes = len(episodes)
    if num_episodes == 0:
        raise SystemExit("源数据集没有 episode")

    ep_stats_path = src_root / "meta" / "episodes_stats.jsonl"
    ep_stats_by_idx: dict[int, dict] = {}
    if ep_stats_path.exists():
        for row in load_jsonl(ep_stats_path):
            ep_stats_by_idx[int(row["episode_index"])] = row["stats"]

    train_eps, test_eps = split_indices(num_episodes, args.train_ratio, args.seed)

    print(f"source: {src_root}")
    print(f"total episodes: {num_episodes}")
    print(f"train: {len(train_eps)} ({len(train_eps) / num_episodes:.1%})")
    print(f"eval : {len(test_eps)} ({len(test_eps) / num_episodes:.1%})")
    print(f"seed : {args.seed}")

    if args.dry_run:
        print("dry_run=True，不写盘")
        print(f"train episode indices (前20): {train_eps[:20]}")
        print(f"eval  episode indices (前20): {test_eps[:20]}")
        return

    train_summary = build_subset(
        src_root=src_root,
        out_root=out_dir / "train",
        selected_eps=train_eps,
        info=info,
        episodes=episodes,
        ep_stats_by_idx=ep_stats_by_idx,
        split_name="train",
    )
    test_summary = build_subset(
        src_root=src_root,
        out_root=out_dir / "eval",
        selected_eps=test_eps,
        info=info,
        episodes=episodes,
        ep_stats_by_idx=ep_stats_by_idx,
        split_name="eval",
    )

    manifest = {
        "source_root": str(src_root),
        "out_dir": str(out_dir),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "train": train_summary,
        "test": test_summary,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Done. manifest -> {out_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
