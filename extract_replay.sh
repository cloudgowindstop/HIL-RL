#!/bin/bash
# ============================================================
# 批量提取 LeRobot + HDF5 → 真机复现数据
#
# 用法:
#   bash extract_replay.sh \
#       --parquet-dir dataset_cosmos/pick_spoon_cosmos/data/chunk-000/ \
#       --hdf5-dir dataset_raw/pick_spoon/ \
#       --action-scale 0.0061540074,0.0077076681,0.1641568679 \
#       --out pose_delta/pick_spoon/
# ============================================================
set -euo pipefail

PARQUET_DIR=""
HDF5_DIR=""
ACTION_SCALE=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parquet-dir) PARQUET_DIR="$2"; shift 2 ;;
        --hdf5-dir) HDF5_DIR="$2"; shift 2 ;;
        --action-scale) ACTION_SCALE="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$PARQUET_DIR" ] || [ -z "$HDF5_DIR" ] || [ -z "$ACTION_SCALE" ] || [ -z "$OUT_DIR" ]; then
    echo "用法: bash extract_replay.sh --parquet-dir ... --hdf5-dir ... --action-scale ... --out ..."
    exit 1
fi

mkdir -p "$OUT_DIR"

# 复制一份 action_scale 供参考
echo "$ACTION_SCALE" > "$OUT_DIR/action_scale.txt"

for pq in "$PARQUET_DIR"/episode_*.parquet; do
    echo "--- $pq ---"
    python3 "$(dirname "$0")/extract_replay_data.py" \
        --parquet "$pq" \
        --hdf5-dir "$HDF5_DIR" \
        --action-scale "$ACTION_SCALE" \
        --out "$OUT_DIR"
done

echo ""
echo "完成 → $OUT_DIR"
ls -la "$OUT_DIR"/
