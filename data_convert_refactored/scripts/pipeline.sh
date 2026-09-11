#!/bin/bash
# ============================================================
# 一键数据处理链条：下载 raw HDF5 → 转换为 Cosmos LeRobot
#
# 用法:
#   bash data_convert_refactored/scripts/pipeline.sh --episode-outcome success
#   bash data_convert_refactored/scripts/pipeline.sh --task "plug_in_ethernet" --episode-outcome success
#   bash data_convert_refactored/scripts/pipeline.sh --task "plug_in_ethernet" --episode-outcome success --action-encoding cosmos_rotation_6d --max-episodes 1
#   bash data_convert_refactored/scripts/pipeline.sh --episode-outcome success --dry-run
#   追加 --trace-shapes 可在正式转换的关键阶段打印shape和dtype
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
LEGACY_DATA_CONVERT_DIR="$PROJECT_DIR/data_convert"
VENV_ACTIVATE="/media/jushen/mingbo-ge/.venv/bin/activate"
PYTHON_BIN="/media/jushen/mingbo-ge/.venv/bin/python3"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[ERROR] Project virtual environment is missing: $VENV_ACTIVATE" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Project Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
source /media/jushen/mingbo-ge/.venv/bin/activate
echo "  python:  $PYTHON_BIN"
cd "$PROJECT_DIR"

# ---- 路径配置 ----
BCECMD=/media/linux-bcecmd-0.5.1/bcecmd
BOS_BASE="bos:/bd-dp-ten-6spt6-scjd"
EXCEL="$PROJECT_DIR/具身智能数据交付表.xlsx"
T5_EMBEDDINGS="${T5_EMBEDDINGS:-}"    # 必须包含与 --task 完全匹配的文本键
SKIP_T5=false

# ---- Python 环境 (参考 collect_data.sh) ----
setup_env() {
    export PYTHONPATH="${PYTHONPATH:-}"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_DIR/cosmos-policy/"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_DIR/cosmos-policy/cosmos_policy/"
    export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/lerobot/src/"
    export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/rl_envs/"
    export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/"
    export PYTHONPATH="$PYTHONPATH:$LEGACY_DATA_CONVERT_DIR"

    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"

    # 绕过 pynput 的 X display 依赖（服务器无显示器）
    export DISPLAY="${DISPLAY:-:99}"

    # 减少 GPU 显存碎片，防止 OOM
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    unset http_proxy https_proxy
    export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,192.168.0.0/16
    export NO_PROXY=$no_proxy
}

validate_existing_output() {
    local OUTPUT_DIR="$1"
    local EXPECTED_TASK="$2"
    local METADATA="$OUTPUT_DIR/cosmos_dataset_metadata.json"
    if [[ ! -f "$METADATA" ]]; then
        echo "  [ERROR] existing output has no cosmos_dataset_metadata.json" >&2
        return 1
    fi
    "$PYTHON_BIN" -c 'import json, math, sys
path, task, outcome, stats_sha, stats_mode, source, encoding, ts, rs, gs, skip_t5, camera_state, crop_mode, crop_fraction, left_offset, right_offset, head_crop_top = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    meta = json.load(f)
action = meta.get("action", {})
language = meta.get("language_conditioning", {})
images = meta.get("images", {})
labeling = meta.get("episode_labeling", {})
statistics = meta.get("statistics", {})
timing = meta.get("timing", {})
pairs = {
    "task_description": (meta.get("task_description"), task),
    "episode_labeling.outcome": (labeling.get("outcome"), outcome),
    "episode_labeling.source": (labeling.get("source"), "explicit_cli"),
    "episode_labeling.success_tail_frames": (labeling.get("success_tail_frames"), 5),
    "episode_labeling.reward_positive": (labeling.get("reward_positive"), 10.0),
    "episode_labeling.reward_negative": (labeling.get("reward_negative"), -0.05),
    "statistics.sha256": (statistics.get("sha256"), stats_sha),
    "statistics.mode": (statistics.get("mode"), stats_mode),
    "action.source": (action.get("source"), source),
    "action.encoding": (action.get("encoding"), encoding),
    "action.last_action_padding_strategy": (action.get("last_action_padding_strategy"), "repeat_last_valid" if source == "puppet_next_frame" else "none"),
    "timing.cosmos_conditioning_fps": (timing.get("cosmos_conditioning_fps"), 16),
    "action.translation_scale": (action.get("translation_scale"), float(ts)),
    "action.gripper_scale": (action.get("gripper_scale"), float(gs)),
    "t5_enabled": (language.get("t5_embedding_included_during_conversion"), skip_t5 == "false"),
    "images.camera_state": (images.get("camera_state"), camera_state),
    "images.wrist_crop_mode": (images.get("wrist_crop_mode"), crop_mode),
    "images.wrist_crop_fraction": (images.get("wrist_crop_fraction"), float(crop_fraction)),
    "images.wrist_left_center_offset_x": (images.get("wrist_left_center_offset_x"), int(left_offset)),
    "images.wrist_right_center_offset_x": (images.get("wrist_right_center_offset_x"), int(right_offset)),
    "images.head_crop_top_pixels": (images.get("head_crop_top_pixels"), int(head_crop_top)),
}
if encoding == "legacy_euler":
    pairs["action.rotation_scale"] = (action.get("rotation_scale"), float(rs))
bad = []
for key, (actual, expected) in pairs.items():
    equal = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12) if isinstance(actual, (int, float)) and isinstance(expected, float) else actual == expected
    if not equal:
        bad.append(f"{key}: existing={actual!r}, requested={expected!r}")
if bad:
    print("  [ERROR] existing output metadata mismatch:", file=sys.stderr)
    print("\n".join("    " + item for item in bad), file=sys.stderr)
    sys.exit(1)' "$METADATA" "$EXPECTED_TASK" "$EPISODE_OUTCOME" "$DATASET_STATS_SHA256" "$STATS_MODE" "$ACTION_SOURCE" "$ACTION_ENCODING" \
        "$TRANSLATION_SCALE" "$ROTATION_SCALE" "$GRIPPER_SCALE" "$SKIP_T5" \
        "$CAMERA_STATE" \
        "$WRIST_CROP_MODE" "$WRIST_CROP_FRACTION" \
        "$WRIST_LEFT_CENTER_OFFSET_X" "$WRIST_RIGHT_CENTER_OFFSET_X" \
        "$HEAD_CROP_TOP_PIXELS"
}

# ---- 参数 ----
DRY_RUN=false
FILTER=""
MAX_EPISODES=0
ENCODE_BATCH_SIZE=16
ENCODE_WORLD_SIZE="${ENCODE_WORLD_SIZE:-1}"
ENCODE_DEVICE_IDS="${ENCODE_CUDA_DEVICES:-}"
BATCH_SIZE=1
ACTION_ENCODING="cosmos_rotation_6d"
ACTION_SOURCE="puppet_next_frame"
EPISODE_OUTCOME=""
STATS_MODE="generated"
DATASET_STATS=""
OFFICIAL_DATASET_STATS=""
KINEMATICS_CONFIG=""
TRANSLATION_SCALE="0.02"
ROTATION_SCALE="0.06"
GRIPPER_SCALE="1.0"
RAW_DIR="$PROJECT_DIR/dataset_raw"
COSMOS_DIR="$PROJECT_DIR/dataset_cosmos"
PREFLIGHT_ONLY=false
COSMOS_CONFIG="$PROJECT_DIR/train_config_cosmos.json"
CAMERA_STATE="normal"
WRIST_CROP_MODE="center_width"
WRIST_CROP_FRACTION="0.75"
WRIST_LEFT_CENTER_OFFSET_X="64"
WRIST_RIGHT_CENTER_OFFSET_X="64"
HEAD_CROP_TOP_PIXELS="100"
FK_MAX_POSITION_ERROR_M="0.005"
FK_MAX_ROTATION_ERROR_DEG="1.0"
TRACE_SHAPES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        --trace-shapes) TRACE_SHAPES=true; shift ;;
        --task) FILTER="$2"; shift 2 ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --encode-batch-size) ENCODE_BATCH_SIZE="$2"; shift 2 ;;
        --encode-world-size) ENCODE_WORLD_SIZE="$2"; shift 2 ;;
        --encode-device-ids) ENCODE_DEVICE_IDS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --action-encoding) ACTION_ENCODING="$2"; shift 2 ;;
        --action-source) ACTION_SOURCE="$2"; shift 2 ;;
        --episode-outcome) EPISODE_OUTCOME="$2"; shift 2 ;;
        --stats-mode) STATS_MODE="$2"; shift 2 ;;
        --dataset-stats) DATASET_STATS="$2"; shift 2 ;;
        --official-dataset-stats) OFFICIAL_DATASET_STATS="$2"; shift 2 ;;
        --kinematics-config) KINEMATICS_CONFIG="$2"; shift 2 ;;
        --fk-max-position-error-m) FK_MAX_POSITION_ERROR_M="$2"; shift 2 ;;
        --fk-max-rotation-error-deg) FK_MAX_ROTATION_ERROR_DEG="$2"; shift 2 ;;
        --cosmos-config) COSMOS_CONFIG="$2"; shift 2 ;;
        --camera-state) CAMERA_STATE="$2"; shift 2 ;;
        --wrist-crop-mode) WRIST_CROP_MODE="$2"; shift 2 ;;
        --wrist-crop-fraction|--wrist-crop-bottom) WRIST_CROP_FRACTION="$2"; shift 2 ;;
        --wrist-left-center-offset-x) WRIST_LEFT_CENTER_OFFSET_X="$2"; shift 2 ;;
        --wrist-right-center-offset-x) WRIST_RIGHT_CENTER_OFFSET_X="$2"; shift 2 ;;
        --head-crop-top-pixels) HEAD_CROP_TOP_PIXELS="$2"; shift 2 ;;
        --translation-scale) TRANSLATION_SCALE="$2"; shift 2 ;;
        --rotation-scale) ROTATION_SCALE="$2"; shift 2 ;;
        --gripper-scale) GRIPPER_SCALE="$2"; shift 2 ;;
        --t5-embeddings) T5_EMBEDDINGS="$2"; shift 2 ;;
        --skip-t5) SKIP_T5=true; shift ;;
        --raw-dir) RAW_DIR="$2"; shift 2 ;;
        --cosmos-dir) COSMOS_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if $TRACE_SHAPES; then
    export COSMOS_TRACE_SHAPES=1
else
    unset COSMOS_TRACE_SHAPES
fi

if ! [[ "$ENCODE_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --encode-world-size must be a positive integer" >&2
    exit 2
fi
if [[ -n "$ENCODE_DEVICE_IDS" ]] && ! [[ "$ENCODE_DEVICE_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] --encode-device-ids must be comma-separated non-negative integers" >&2
    exit 2
fi
if [[ "$ACTION_SOURCE" == "master_joint_fk_same_frame" && -z "$KINEMATICS_CONFIG" ]]; then
    echo "[ERROR] --kinematics-config is required for master_joint_fk_same_frame" >&2
    exit 2
fi
if [[ "$EPISODE_OUTCOME" != "success" && "$EPISODE_OUTCOME" != "failure" ]]; then
    echo "[ERROR] --episode-outcome is required and must be success or failure" >&2
    exit 2
fi
if [[ "$STATS_MODE" != "generated" && "$STATS_MODE" != "official" ]]; then
    echo "[ERROR] --stats-mode must be generated or official" >&2
    exit 2
fi
if [[ "$STATS_MODE" == "generated" ]]; then
    ACTIVE_DATASET_STATS="$DATASET_STATS"
    STATS_PATH_ARG=(--dataset-stats "$DATASET_STATS")
else
    ACTIVE_DATASET_STATS="$OFFICIAL_DATASET_STATS"
    STATS_PATH_ARG=(--official-dataset-stats "$OFFICIAL_DATASET_STATS")
fi
if [[ ! -f "$ACTIVE_DATASET_STATS" ]]; then
    echo "[ERROR] stats file required by --stats-mode $STATS_MODE does not exist: $ACTIVE_DATASET_STATS" >&2
    exit 2
fi
if [[ "$STATS_MODE" == "generated" && ! -f "${DATASET_STATS%.json}.metadata.json" ]]; then
    echo "[ERROR] stats semantic metadata does not exist: ${DATASET_STATS%.json}.metadata.json" >&2
    exit 2
fi
DATASET_STATS_SHA256=$("$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ACTIVE_DATASET_STATS")
if [[ "$ACTION_SOURCE" != "master_same_frame" && "$ACTION_SOURCE" != "master_joint_fk_same_frame" && "$ACTION_SOURCE" != "puppet_next_frame" ]]; then
    echo "[ERROR] unsupported --action-source: $ACTION_SOURCE" >&2
    exit 2
fi

if [[ "$ACTION_ENCODING" != "legacy_euler" && "$ACTION_ENCODING" != "cosmos_rotation_6d" ]]; then
    echo "[ERROR] --action-encoding must be legacy_euler or cosmos_rotation_6d" >&2
    exit 2
fi
if [[ "$WRIST_CROP_MODE" != "center_width" && "$WRIST_CROP_MODE" != "bottom" && "$WRIST_CROP_MODE" != "none" ]]; then
    echo "[ERROR] --wrist-crop-mode must be center_width, bottom, or none" >&2
    exit 2
fi
if [[ "$CAMERA_STATE" != "normal" ]]; then
    echo "[ERROR] --camera-state currently only supports normal" >&2
    exit 2
fi
if [[ -n "$KINEMATICS_CONFIG" && ! -f "$KINEMATICS_CONFIG" ]]; then
    echo "[ERROR] kinematics config does not exist: $KINEMATICS_CONFIG" >&2
    exit 2
fi
if [[ ! -f "$COSMOS_CONFIG" ]]; then
    echo "[ERROR] Cosmos config does not exist: $COSMOS_CONFIG" >&2
    exit 2
fi
if ! $SKIP_T5 && [[ -n "$T5_EMBEDDINGS" && ! -f "$T5_EMBEDDINGS" ]]; then
    echo "[ERROR] T5 embeddings do not exist: $T5_EMBEDDINGS" >&2
    exit 2
fi

"$PYTHON_BIN" -c 'import sys
max_episodes, encode_batch, batch = map(int, sys.argv[1:4])
translation, rotation, gripper, crop, fk_pos, fk_rot = map(float, sys.argv[4:])
assert max_episodes >= 0, "max-episodes must be >= 0"
assert encode_batch >= 1 and batch >= 1, "batch sizes must be >= 1"
assert translation > 0 and rotation > 0 and gripper > 0, "action scales must be > 0"
assert 0 < crop <= 1, "wrist-crop-fraction must be in (0,1]"
assert fk_pos > 0 and fk_rot > 0, "FK thresholds must be > 0"' \
    "$MAX_EPISODES" "$ENCODE_BATCH_SIZE" "$BATCH_SIZE" \
    "$TRANSLATION_SCALE" "$ROTATION_SCALE" "$GRIPPER_SCALE" \
    "$WRIST_CROP_FRACTION" "$FK_MAX_POSITION_ERROR_M" "$FK_MAX_ROTATION_ERROR_DEG"
if ! $DRY_RUN && ! $SKIP_T5 && [ -z "$T5_EMBEDDINGS" ]; then
    echo "[ERROR] --t5-embeddings is required (zero-vector fallback is disabled)" >&2
    exit 2
fi
if ! [[ "$HEAD_CROP_TOP_PIXELS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --head-crop-top-pixels must be a non-negative integer" >&2
    exit 2
fi

# ---- 从 Excel 解析任务 ----
parse_tasks() {
    "$PYTHON_BIN" - "$EXCEL" "$FILTER" << 'PYEOF'
import sys, json
try:
    import openpyxl
except ImportError:
    print("pip install openpyxl", file=sys.stderr)
    sys.exit(1)

filter_str = sys.argv[2].strip() if len(sys.argv) > 2 else ""
wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    try:
        bos_col = next(i for i, h in enumerate(headers) if h and "BOS" in str(h))
        status_col = next(i for i, h in enumerate(headers) if h and "标注状态" in str(h))
    except StopIteration:
        continue
    for row in ws.iter_rows(min_row=2, values_only=True):
        task_name = str(row[2]) if len(row) > 2 and row[2] else None
        bos_addr = str(row[bos_col]) if row[bos_col] else None
        status = str(row[status_col]) if row[status_col] else ""
        if not task_name or not bos_addr or "BOS::" not in bos_addr:
            continue
        if "标注完成" not in status:
            continue
        if filter_str and filter_str.lower() not in task_name.lower():
            continue
        # lerobot_data_v3/{robot_type}/{task_name}/success → raw 路径
        bos_raw = bos_addr.replace("BOS::", "").strip()
        bos_raw = bos_raw.replace("lerobot_data_v3", "raw_data")
        bos_raw = bos_raw.replace("/success", "/success_episodes")
        if not bos_raw.endswith("/"):
            bos_raw += "/"
        robot_type = bos_raw.split("/")[1] if len(bos_raw.split("/")) > 1 else "unknown"
        task_instruction = task_name.replace("_", " ").strip()
        print(json.dumps({
            "task_name": task_name,
            "task_instruction": task_instruction,
            "robot_type": robot_type,
            "bos_path": bos_raw,
        }))
PYEOF
}

# ---- 主流程 ----
setup_env
mkdir -p "$RAW_DIR" "$COSMOS_DIR"

echo "========================================"
echo " Pipeline: raw HDF5 → Cosmos LeRobot"
echo " action:   $ACTION_ENCODING"
echo "========================================"
echo ""

TOTAL=0
OK=0
SKIP=0
FAIL=0

while IFS= read -r line; do
    TASK_NAME=$(echo "$line" | "$PYTHON_BIN" -c "import json,sys; print(json.loads(sys.stdin.read())['task_name'])")
    TASK_INSTRUCTION=$(echo "$line" | "$PYTHON_BIN" -c "import json,sys; print(json.loads(sys.stdin.read())['task_instruction'])")
    BOS_PATH=$(echo "$line" | "$PYTHON_BIN" -c "import json,sys; print(json.loads(sys.stdin.read())['bos_path'])")

    TOTAL=$((TOTAL + 1))
    RAW_LOCAL="$RAW_DIR/$TASK_NAME"
    COSMOS_OUTPUT="$COSMOS_DIR/${TASK_NAME}_${EPISODE_OUTCOME}_${ACTION_SOURCE}_${ACTION_ENCODING}"

    echo "────────────────────────────────────────"
    echo "[$TOTAL] $TASK_NAME"
    echo "  instruction: $TASK_INSTRUCTION"
    echo ""

    # ---- Stage 1: 下载 ----
    if [ -d "$RAW_LOCAL" ] && [ "$(find "$RAW_LOCAL" -name "*.hdf5" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [下载] 已存在，跳过"
    else
        echo "  [下载] 开始..."
        if $DRY_RUN; then
            echo "  [下载] DRY RUN: $BOS_PATH → $RAW_LOCAL"
        else
            mkdir -p "$RAW_LOCAL"
            $BCECMD bos sync "$BOS_BASE/$BOS_PATH" "$RAW_LOCAL" 2>&1 | tail -1
            HDF5_COUNT=$(find "$RAW_LOCAL" -name "*.hdf5" | wc -l)
            echo "  [下载] 完成, $HDF5_COUNT 个 hdf5"
        fi
    fi

    # ---- Stage 2: 转换 ----
    if ! $PREFLIGHT_ONLY && [ -d "$COSMOS_OUTPUT" ] && [ -f "$COSMOS_OUTPUT/meta/info.json" ]; then
        if validate_existing_output "$COSMOS_OUTPUT" "$TASK_INSTRUCTION"; then
            echo "  [转换] 已存在且配置一致，跳过"
            SKIP=$((SKIP + 1))
        else
            echo "  [转换] 已存在但配置不一致，拒绝复用"
            FAIL=$((FAIL + 1))
        fi
        continue
    fi

    if $DRY_RUN; then
        echo "  [转换] DRY RUN"
        OK=$((OK + 1))
        continue
    fi

    echo "  [转换] 开始..."
    EXTRA_ARGS=()
    EXTRA_ARGS+=(--encode-world-size "$ENCODE_WORLD_SIZE")
    [ -n "$ENCODE_DEVICE_IDS" ] && EXTRA_ARGS+=(--encode-device-ids "$ENCODE_DEVICE_IDS")
    if [ "$MAX_EPISODES" -gt 0 ]; then
        EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
    fi
    if $SKIP_T5; then
        EXTRA_ARGS+=(--skip-t5)
    else
        EXTRA_ARGS+=(--t5-embeddings "$T5_EMBEDDINGS")
    fi
    [ -n "$KINEMATICS_CONFIG" ] && EXTRA_ARGS+=(--kinematics-config "$KINEMATICS_CONFIG")
    $PREFLIGHT_ONLY && EXTRA_ARGS+=(--preflight-only)

    if "$PYTHON_BIN" -m data_convert_refactored.convert \
        --input "$RAW_LOCAL" \
        --output "$COSMOS_OUTPUT" \
        --task "$TASK_INSTRUCTION" \
        --episode-outcome "$EPISODE_OUTCOME" \
        --stats-mode "$STATS_MODE" \
        "${STATS_PATH_ARG[@]}" \
        --action-encoding "$ACTION_ENCODING" \
        --action-source "$ACTION_SOURCE" \
        --translation-scale "$TRANSLATION_SCALE" \
        --rotation-scale "$ROTATION_SCALE" \
        --gripper-scale "$GRIPPER_SCALE" \
        --encode-batch-size "$ENCODE_BATCH_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --cosmos-config "$COSMOS_CONFIG" \
        --camera-state "$CAMERA_STATE" \
        --wrist-crop-mode "$WRIST_CROP_MODE" \
        --wrist-crop-fraction "$WRIST_CROP_FRACTION" \
        --wrist-left-center-offset-x "$WRIST_LEFT_CENTER_OFFSET_X" \
        --wrist-right-center-offset-x "$WRIST_RIGHT_CENTER_OFFSET_X" \
        --head-crop-top-pixels "$HEAD_CROP_TOP_PIXELS" \
        --fk-max-position-error-m "$FK_MAX_POSITION_ERROR_M" \
        --fk-max-rotation-error-deg "$FK_MAX_ROTATION_ERROR_DEG" \
        "${EXTRA_ARGS[@]}"; then
        if $PREFLIGHT_ONLY; then
            echo "  [转换] 预检通过"
        else
            echo "  [转换] 完成 → $COSMOS_OUTPUT"
        fi
        OK=$((OK + 1))
    else
        echo "  [转换] 失败!"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done < <(parse_tasks)

echo "========================================"
echo " Pipeline 完成"
echo " 总计: $TOTAL | 成功: $OK | 跳过: $SKIP | 失败: $FAIL"
echo "========================================"
if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
