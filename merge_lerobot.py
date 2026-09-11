#!/usr/bin/env python3
"""Merge multiple LeRobot v2.1 datasets (local) into one."""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats

SRC_ROOTS = [
    Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260803_pm_failure"),
    Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260803_pm_success"),
    Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260804_am_failure"),
    Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260804_am_success"),
]
OUT = Path("/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/install_handle")  # 训练用的 root


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if OUT.exists():
        raise SystemExit(f"输出目录已存在，请先备份/删除: {OUT}")

    info0 = json.loads((SRC_ROOTS[0] / "meta/info.json").read_text())
    for root in SRC_ROOTS[1:]:
        info = json.loads((root / "meta/info.json").read_text())
        assert info["features"] == info0["features"], f"features 不一致: {root}"
        assert info["fps"] == info0["fps"]

    out_data = OUT / "data" / "chunk-000"
    out_meta = OUT / "meta"
    out_data.mkdir(parents=True)
    out_meta.mkdir(parents=True)

    # tasks（三个都是 pick_toy）
    shutil.copy(SRC_ROOTS[0] / "meta/tasks.jsonl", out_meta / "tasks.jsonl")

    ep_offset = 0
    frame_offset = 0
    all_ep_stats = []
    episodes_out = []

    for root in SRC_ROOTS:
        episodes = load_jsonl(root / "meta/episodes.jsonl")
        ep_stats = load_jsonl(root / "meta/episodes_stats.jsonl")
        stats_by_ep = {e["episode_index"]: e["stats"] for e in ep_stats}

        for ep in episodes:
            old_idx = ep["episode_index"]
            new_idx = ep_offset + old_idx
            src = root / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet"
            dst = out_data / f"episode_{new_idx:06d}.parquet"

            table = pq.read_table(src)
            cols = table.to_pydict()
            n = len(cols["index"])
            cols["episode_index"] = [new_idx] * n
            cols["index"] = list(range(frame_offset, frame_offset + n))
            pq.write_table(pa.Table.from_pydict(cols), dst)

            episodes_out.append({
                "episode_index": new_idx,
                "tasks": ep["tasks"],
                "length": ep["length"],
            })
            all_ep_stats.append({"episode_index": new_idx, "stats": stats_by_ep[old_idx]})
            frame_offset += n

        ep_offset += len(episodes)
        print(f"merged {root.name}: +{len(episodes)} eps")

    with open(out_meta / "episodes.jsonl", "w") as f:
        for e in episodes_out:
            f.write(json.dumps(e) + "\n")
    with open(out_meta / "episodes_stats.jsonl", "w") as f:
        for e in all_ep_stats:
            f.write(json.dumps(e) + "\n")

    info = dict(info0)
    info["total_episodes"] = ep_offset
    info["total_frames"] = frame_offset
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{ep_offset}"}
    (out_meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    # 可选：写全局 stats（部分代码会读 meta/stats.json）
    try:
        from lerobot.datasets.utils import write_stats
        stats = aggregate_stats([e["stats"] for e in all_ep_stats])
        write_stats(stats, OUT)
    except Exception as e:
        print(f"skip stats.json: {e}")

    print(f"Done -> {OUT}  episodes={ep_offset} frames={frame_offset}")


if __name__ == "__main__":
    main()