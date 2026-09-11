#!/bin/bash
# ============================================================
# 从 BOS 下载 raw HDF5 数据
#
# 使用方式:
#   1. 下载全部:     bash download_raw_data.sh
#   2. 下载指定任务: bash download_raw_data.sh --task "plug_in_ethernet"
#   3. 预览不下载:   bash download_raw_data.sh --dry-run
#   4. 只统计已有数据: bash download_raw_data.sh --stats-only
# ============================================================

set -euo pipefail

# ---- 配置 ----
BCECMD=/media/linux-bcecmd-0.5.1/bcecmd
BOS_BASE="bos:/bd-dp-ten-6spt6-scjd"
RAW_DIR="/media/jushen/project-rl-dataset/cosmos_raw_data"
EXCEL="/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/具身智能数据交付表.xlsx"

# ---- 参数 ----
DRY_RUN=false
FILTER=""
SHOW_STATS=true
STATS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --task) FILTER="$2"; shift 2 ;;
        --output) RAW_DIR="$2"; shift 2 ;;
        --no-stats) SHOW_STATS=false; shift ;;
        --stats-only) STATS_ONLY=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- 统计可用训练数据 ----
# 1 个可正常读取的 trajectory.hdf5 计为 1 条 demo。
# 训练帧数为 metadata/trajectory_length 之和。
print_training_stats() {
    python3 - "$RAW_DIR" <<'PYEOF'
import os
import sys

try:
    import h5py
except ImportError:
    print("统计功能需要 h5py，请先执行: pip install h5py", file=sys.stderr)
    sys.exit(0)

root = sys.argv[1]
tasks = set()
complete_tasks = set()
success_demos = 0
failed_demos = 0
frames = 0
invalid = 0
total_bytes = 0

if not os.path.isdir(root):
    print(f"统计目录不存在: {root}")
    sys.exit(0)

for task_name in os.listdir(root):
    task_dir = os.path.join(root, task_name)
    if not os.path.isdir(task_dir):
        continue

    tasks.add(task_name)
    if os.path.isfile(os.path.join(task_dir, ".download_complete")):
        complete_tasks.add(task_name)

    for dirpath, _, filenames in os.walk(task_dir):
        for filename in filenames:
            if filename != "trajectory.hdf5":
                continue

            path = os.path.join(dirpath, filename)
            try:
                with h5py.File(path, "r") as f:
                    length = int(f["metadata/trajectory_length"][()])
                frames += length
                total_bytes += os.path.getsize(path)
                if task_name.endswith("-fail"):
                    failed_demos += 1
                else:
                    success_demos += 1
            except (OSError, KeyError, TypeError, ValueError):
                # 下载中、损坏或缺少必要元数据的文件不计入有效 demo。
                invalid += 1

valid_demos = success_demos + failed_demos

print("")
print("----------------------------------------")
print(" 当前训练数据统计")
print("----------------------------------------")
print(f"统计目录:       {root}")
print(f"发现任务数:     {len(tasks)}")
print(f"完整下载任务数: {len(complete_tasks)}")
print(f"有效 demo 总数: {valid_demos}")
print(f"成功 demo:        {success_demos}")
print(f"失败 demo:        {failed_demos}")
print(f"训练帧总数:     {frames}")
print(f"损坏/下载中:    {invalid}")
print(f"HDF5 总大小:     {total_bytes / 1024**3:.2f} GiB")
print("----------------------------------------")
PYEOF
}

# ---- 从 Excel 解析任务列表 ----
parse_excel() {
    python3 - "$1" << 'PYEOF'
import sys, json

try:
    import openpyxl
except ImportError:
    print("pip install openpyxl", file=sys.stderr)
    sys.exit(1)

wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]

    # 找 BOS 地址列 和 标注状态列
    try:
        bos_col = next(i for i, h in enumerate(headers) if h and "BOS" in str(h))
        status_col = next(i for i, h in enumerate(headers) if h and "标注状态" in str(h))
    except StopIteration:
        continue

    for row in ws.iter_rows(min_row=2, values_only=True):
        bos_addr = row[bos_col] if row[bos_col] else None
        task_name = row[2] if len(row) > 2 and row[2] else None
        status = str(row[status_col]) if row[status_col] else ""

        if not bos_addr or not task_name:
            continue
        if "标注完成" not in status:
            continue

        # BOS::lerobot_data_v3/{robot_type}/{task_name}/success
        # → raw_data/{robot_type}/{task_name}/success_episodes
        bos_addr = str(bos_addr).replace("BOS::", "").strip()
        bos_addr = bos_addr.replace("lerobot_data_v3", "raw_data")
        bos_addr = bos_addr.replace("/success", "/success_episodes")
        if not bos_addr.endswith("/"):
            bos_addr += "/"

        robot_type = bos_addr.split("/")[1] if len(bos_addr.split("/")) > 1 else "unknown"

        print(json.dumps({
            "task_name": str(task_name),
            "robot_type": robot_type,
            "bos_path": bos_addr,
            "sheet": sheet_name,
        }))
PYEOF
}

mkdir -p "$RAW_DIR"

if $STATS_ONLY; then
    print_training_stats
    exit 0
fi

echo "========================================"
echo " 下载 raw HDF5 数据"
echo " BOS: $BOS_BASE"
echo " 本地: $RAW_DIR"
echo "========================================"
echo ""

TOTAL=0
DOWNLOADED=0
SKIPPED=0
FAILED=0

while IFS= read -r line; do
    TASK_NAME=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['task_name'])")
    BOS_PATH=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['bos_path'])")
    ROBOT=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['robot_type'])")

    # 过滤
    if [ -n "$FILTER" ] && ! echo "$TASK_NAME" | grep -qi "$FILTER"; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    LOCAL_DST="$RAW_DIR/$TASK_NAME"
    COMPLETE_MARKER="$LOCAL_DST/.download_complete"

    echo "[$TOTAL] $TASK_NAME"
    echo "     BOS:  $BOS_PATH"
    echo "     本地: $LOCAL_DST"

    # 只有 bcecmd 完整同步成功后才会创建此标记。
    # 仅存在部分 HDF5 的中断任务会重新同步。
    if [ -f "$COMPLETE_MARKER" ]; then
        echo "    → 已完整下载，跳过"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "    → [DRY RUN] 会下载"
        continue
    fi

    echo "    → 开始下载..."
    mkdir -p "$LOCAL_DST"

    if SYNC_RESULT=$($BCECMD bos sync "$BOS_BASE/$BOS_PATH" "$LOCAL_DST" 2>&1 | tail -1); then
        echo "$SYNC_RESULT"
        touch "$COMPLETE_MARKER"
        HDF5_COUNT=$(find "$LOCAL_DST" -name "*.hdf5" 2>/dev/null | wc -l)
        echo "    → 完成, $HDF5_COUNT 个 hdf5 文件"
        DOWNLOADED=$((DOWNLOADED + 1))
        if $SHOW_STATS; then
            print_training_stats
        fi
    else
        echo "$SYNC_RESULT"
        echo "    → 下载失败或中断，未标记为完成" >&2
        FAILED=$((FAILED + 1))
    fi
done < <(parse_excel "$EXCEL")

echo ""
echo "========================================"
echo " 完成: 共 $TOTAL 个任务"
echo " 已下载: $DOWNLOADED | 已跳过: $SKIPPED | 失败: $FAILED"
echo "========================================"

if $SHOW_STATS; then
    print_training_stats
fi
