#!/bin/bash
# ============================================================
# 仅转换：raw HDF5 → Cosmos LeRobot（不含下载）
#
# 用法:
#   bash data_convert_refactored/scripts/convert.sh --episode-outcome success
#   bash data_convert_refactored/scripts/convert.sh --task "pick_spoon" --episode-outcome success
#   bash data_convert_refactored/scripts/convert.sh --task "pick_spoon" --episode-outcome success --dry-run
#   bash data_convert_refactored/scripts/convert.sh --task "pick_spoon" --episode-outcome success --max-episodes 1 --action-encoding cosmos_rotation_6d
#   追加 --trace-shapes 可在每个关键阶段打印action、proprio、VAE和Parquet维度
#   bash data_convert_refactored/scripts/convert.sh --roundtrip-hdf5 dataset_raw/pick_spoon/0731_103412/trajectory.hdf5
# ============================================================
set -euo pipefail

CALL_DIR="$PWD"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
LEGACY_DATA_CONVERT_DIR="$PROJECT_DIR/data_convert"
VENV_ACTIVATE="${VENV_ACTIVATE:-$WORKSPACE_DIR/.venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/.venv/bin/python}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[ERROR] Project virtual environment is missing: $VENV_ACTIVATE" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Project Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
source "$VENV_ACTIVATE"
echo "  python:  $PYTHON_BIN"
cd "$PROJECT_DIR"

# ---- 路径配置 ----
EXCEL="$PROJECT_DIR/具身智能数据交付表.xlsx"
T5_EMBEDDINGS="${T5_EMBEDDINGS:-}"
SKIP_T5=false

# ---- Python 环境 ----
export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$PYTHONPATH:$WORKSPACE_DIR/cosmos-policy/"
export PYTHONPATH="$PYTHONPATH:$WORKSPACE_DIR/cosmos-policy/cosmos_policy/"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/lerobot/src/"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/rl_envs/"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/"
export PYTHONPATH="$PYTHONPATH:$LEGACY_DATA_CONVERT_DIR"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export DISPLAY="${DISPLAY:-:99}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,192.168.0.0/16
export NO_PROXY=$no_proxy
# ---- 参数 ----
DRY_RUN=false
FILTER=""
MAX_EPISODES=0
ENCODE_BATCH_SIZE=16
ENCODE_WORLD_SIZE="${ENCODE_WORLD_SIZE:-1}"
ENCODE_DEVICE_IDS="${ENCODE_CUDA_DEVICES:-}"
BATCH_SIZE=1
SAVE_CLEAN_RESTORE_LATENT=false
ACTION_ENCODING="cosmos_rotation_6d"
ACTION_SOURCE="puppet_next_frame"
KINEMATICS_CONFIG=""
TRANSLATION_SCALE="0.02"
ROTATION_SCALE="0.06"
GRIPPER_SCALE="1.0"
ROUNDTRIP_HDF5=""
RAW_DIR="$WORKSPACE_DIR/raw_data"
COSMOS_DIR="$WORKSPACE_DIR/cosmos_data"
INPUT_OVERRIDE=""
OUTPUT_OVERRIDE=""
TASK_DESCRIPTION=""
EPISODE_OUTCOME=""
STATS_MODE="generated"
DATASET_STATS=""
OFFICIAL_DATASET_STATS=""
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
MONITOR_MEMORY=false
MEMORY_SAMPLE_INTERVAL="2.0"
MONITOR_STORAGE=false
STORAGE_SAMPLE_INTERVAL="5.0"
STORAGE_PROBE_MIB="8"
TRACE_SHAPES=false
RESUME=false
AUTO_RESTART=false
MAX_RESTARTS=3
RESTART_DELAY=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --task) FILTER="$2"; shift 2 ;;
        --task-description) TASK_DESCRIPTION="$2"; shift 2 ;;
        --episode-outcome) EPISODE_OUTCOME="$2"; shift 2 ;;
        --stats-mode) STATS_MODE="$2"; shift 2 ;;
        --dataset-stats) DATASET_STATS="$2"; shift 2 ;;
        --official-dataset-stats) OFFICIAL_DATASET_STATS="$2"; shift 2 ;;
        --input) INPUT_OVERRIDE="$2"; shift 2 ;;
        --output) OUTPUT_OVERRIDE="$2"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --encode-batch-size) ENCODE_BATCH_SIZE="$2"; shift 2 ;;
        --encode-world-size) ENCODE_WORLD_SIZE="$2"; shift 2 ;;
        --encode-device-ids) ENCODE_DEVICE_IDS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --save-clean-restore-latent) SAVE_CLEAN_RESTORE_LATENT=true; shift ;;
        --action-encoding) ACTION_ENCODING="$2"; shift 2 ;;
        --action-source) ACTION_SOURCE="$2"; shift 2 ;;
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
        --monitor-memory) MONITOR_MEMORY=true; shift ;;
        --memory-sample-interval) MEMORY_SAMPLE_INTERVAL="$2"; shift 2 ;;
        --monitor-storage) MONITOR_STORAGE=true; shift ;;
        --storage-sample-interval) STORAGE_SAMPLE_INTERVAL="$2"; shift 2 ;;
        --storage-probe-mib) STORAGE_PROBE_MIB="$2"; shift 2 ;;
        --trace-shapes) TRACE_SHAPES=true; shift ;;
        --resume) RESUME=true; shift ;;
        --auto-restart) AUTO_RESTART=true; RESUME=true; shift ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --restart-delay) RESTART_DELAY="$2"; shift 2 ;;
        --translation-scale) TRANSLATION_SCALE="$2"; shift 2 ;;
        --rotation-scale) ROTATION_SCALE="$2"; shift 2 ;;
        --gripper-scale) GRIPPER_SCALE="$2"; shift 2 ;;
        --t5-embeddings) T5_EMBEDDINGS="$2"; shift 2 ;;
        --skip-t5) SKIP_T5=true; shift ;;
        --roundtrip-hdf5) ROUNDTRIP_HDF5="$2"; shift 2 ;;
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
if [[ "$ACTION_ENCODING" != "legacy_euler" && "$ACTION_ENCODING" != "cosmos_rotation_6d" ]]; then
    echo "[ERROR] --action-encoding must be legacy_euler or cosmos_rotation_6d" >&2
    exit 2
fi
if [[ -z "$ROUNDTRIP_HDF5" && "$EPISODE_OUTCOME" != "success" && "$EPISODE_OUTCOME" != "failure" ]]; then
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
if [[ -z "$ROUNDTRIP_HDF5" && ! -f "$ACTIVE_DATASET_STATS" ]]; then
    echo "[ERROR] stats file required by --stats-mode $STATS_MODE does not exist: $ACTIVE_DATASET_STATS" >&2
    exit 2
fi
if [[ -z "$ROUNDTRIP_HDF5" && "$STATS_MODE" == "generated" && ! -f "${DATASET_STATS%.json}.metadata.json" ]]; then
    echo "[ERROR] stats semantic metadata does not exist: ${DATASET_STATS%.json}.metadata.json" >&2
    exit 2
fi
if [[ -n "$ACTIVE_DATASET_STATS" ]]; then
    DATASET_STATS_SHA256=$("$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ACTIVE_DATASET_STATS")
else
    DATASET_STATS_SHA256=""
fi
if [[ "$WRIST_CROP_MODE" != "center_width" && "$WRIST_CROP_MODE" != "bottom" && "$WRIST_CROP_MODE" != "none" ]]; then
    echo "[ERROR] --wrist-crop-mode must be center_width, bottom, or none" >&2
    exit 2
fi
if [[ "$CAMERA_STATE" != "normal" ]]; then
    echo "[ERROR] --camera-state currently only supports normal" >&2
    exit 2
fi
if [[ "$ACTION_SOURCE" != "master_same_frame" && "$ACTION_SOURCE" != "master_joint_fk_same_frame" && "$ACTION_SOURCE" != "puppet_next_frame" ]]; then
    echo "[ERROR] unsupported --action-source: $ACTION_SOURCE" >&2
    exit 2
fi
if [[ "$ACTION_SOURCE" == "master_joint_fk_same_frame" && -z "$KINEMATICS_CONFIG" ]]; then
    echo "[ERROR] --kinematics-config is required for master_joint_fk_same_frame" >&2
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
if [[ -n "$OUTPUT_OVERRIDE" && -z "$INPUT_OVERRIDE" ]]; then
    echo "[ERROR] --output requires an explicit --input" >&2
    exit 2
fi

validate_numeric_args() {
    "$PYTHON_BIN" -c 'import sys
max_episodes, encode_batch, batch = map(int, sys.argv[1:4])
translation, rotation, gripper, crop, fk_pos, fk_rot = map(float, sys.argv[4:])
assert max_episodes >= 0, "max-episodes must be >= 0"
assert encode_batch >= 1 and batch >= 1, "batch sizes must be >= 1"
assert translation > 0 and gripper > 0, "translation/gripper scales must be > 0"
assert rotation > 0, "rotation scale must be > 0"
assert 0 < crop <= 1, "wrist-crop-fraction must be in (0,1]"
assert fk_pos > 0 and fk_rot > 0, "FK thresholds must be > 0"' \
        "$MAX_EPISODES" "$ENCODE_BATCH_SIZE" "$BATCH_SIZE" \
        "$TRANSLATION_SCALE" "$ROTATION_SCALE" "$GRIPPER_SCALE" \
        "$WRIST_CROP_FRACTION" "$FK_MAX_POSITION_ERROR_M" "$FK_MAX_ROTATION_ERROR_DEG"
}
validate_numeric_args
if ! [[ "$HEAD_CROP_TOP_PIXELS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --head-crop-top-pixels must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "$MAX_RESTARTS" =~ ^[0-9]+$ && "$RESTART_DELAY" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --max-restarts and --restart-delay must be non-negative integers" >&2
    exit 2
fi

check_existing_metadata() {
    local OUTPUT_DIR="$1"
    local EXPECTED_TASK="$2"
    local METADATA="$OUTPUT_DIR/cosmos_dataset_metadata.json"
    local INFO="$OUTPUT_DIR/meta/info.json"
    if [[ ! -f "$OUTPUT_DIR/meta/info.json" ]]; then
        return 1
    fi
    if [[ ! -f "$METADATA" ]]; then
        echo "[ERROR] completed output has no cosmos_dataset_metadata.json: $OUTPUT_DIR" >&2
        return 2
    fi
    local TOTAL_EPISODES
    local PARQUET_COUNT
    TOTAL_EPISODES=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("total_episodes", 0)))' "$INFO")
    PARQUET_COUNT=$(find "$OUTPUT_DIR/data" -type f -name 'episode_*.parquet' 2>/dev/null | wc -l)
    if [[ "$TOTAL_EPISODES" -le 0 || "$PARQUET_COUNT" -le 0 || "$PARQUET_COUNT" -ne "$TOTAL_EPISODES" ]]; then
        echo "[ERROR] existing output is incomplete: $OUTPUT_DIR" >&2
        echo "        total_episodes=$TOTAL_EPISODES, parquet_count=$PARQUET_COUNT" >&2
        echo "        choose a new --output directory before retrying" >&2
        return 2
    fi
    "$PYTHON_BIN" -c 'import json, math, sys
path, task, outcome, stats_sha, stats_mode, source, encoding, ts, rs, gs, skip_t5, camera_state, crop_mode, crop_fraction, left_offset, right_offset, head_crop_top, save_clean_restore = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    meta = json.load(f)
action = meta.get("action", {})
language = meta.get("language_conditioning", {})
images = meta.get("images", {})
labeling = meta.get("episode_labeling", {})
statistics = meta.get("statistics", {})
timing = meta.get("timing", {})
vae_encode = meta.get("vae_encode", {})
expected = {
    "task_description": task,
    "episode_labeling.outcome": outcome,
    "episode_labeling.source": "explicit_cli",
    "episode_labeling.success_tail_frames": 5,
    "episode_labeling.reward_positive": 10.0,
    "episode_labeling.reward_negative": -0.05,
    "statistics.sha256": stats_sha,
    "statistics.mode": stats_mode,
    "action.source": source,
    "action.encoding": encoding,
    "action.last_action_padding_strategy": "repeat_last_valid" if source == "puppet_next_frame" else "none",
    "timing.cosmos_conditioning_fps": 16,
    "action.translation_scale": float(ts),
    "action.gripper_scale": float(gs),
    "t5_enabled": skip_t5 == "false",
    "images.camera_state": camera_state,
    "images.wrist_crop_mode": crop_mode,
    "images.wrist_crop_fraction": float(crop_fraction),
    "images.wrist_left_center_offset_x": int(left_offset),
    "images.wrist_right_center_offset_x": int(right_offset),
    "images.head_crop_top_pixels": int(head_crop_top),
    "vae_encode.clean_restore_latent.enabled": save_clean_restore == "true",
}
actual = {
    "task_description": meta.get("task_description"),
    "episode_labeling.outcome": labeling.get("outcome"),
    "episode_labeling.source": labeling.get("source"),
    "episode_labeling.success_tail_frames": labeling.get("success_tail_frames"),
    "episode_labeling.reward_positive": labeling.get("reward_positive"),
    "episode_labeling.reward_negative": labeling.get("reward_negative"),
    "statistics.sha256": statistics.get("sha256"),
    "statistics.mode": statistics.get("mode"),
    "action.source": action.get("source"),
    "action.encoding": action.get("encoding"),
    "action.last_action_padding_strategy": action.get("last_action_padding_strategy"),
    "timing.cosmos_conditioning_fps": timing.get("cosmos_conditioning_fps"),
    "action.translation_scale": action.get("translation_scale"),
    "action.gripper_scale": action.get("gripper_scale"),
    "t5_enabled": language.get("t5_embedding_included_during_conversion"),
    "images.camera_state": images.get("camera_state"),
    "images.wrist_crop_mode": images.get("wrist_crop_mode"),
    "images.wrist_crop_fraction": images.get("wrist_crop_fraction"),
    "images.wrist_left_center_offset_x": images.get("wrist_left_center_offset_x"),
    "images.wrist_right_center_offset_x": images.get("wrist_right_center_offset_x"),
    "images.head_crop_top_pixels": images.get("head_crop_top_pixels"),
    "vae_encode.clean_restore_latent.enabled": vae_encode.get("clean_restore_latent", {}).get("enabled", False),
}
if encoding == "legacy_euler":
    expected["action.rotation_scale"] = float(rs)
    actual["action.rotation_scale"] = action.get("rotation_scale")
mismatch = []
for key, value in expected.items():
    got = actual.get(key)
    equal = math.isclose(got, value, rel_tol=0, abs_tol=1e-12) if isinstance(value, float) and isinstance(got, (int, float)) else got == value
    if not equal:
        mismatch.append(f"{key}: existing={got!r}, requested={value!r}")
if mismatch:
    print("[ERROR] existing output metadata does not match requested conversion:", file=sys.stderr)
    print("\n".join("  " + item for item in mismatch), file=sys.stderr)
    sys.exit(2)' "$METADATA" "$EXPECTED_TASK" "$EPISODE_OUTCOME" "$DATASET_STATS_SHA256" "$STATS_MODE" "$ACTION_SOURCE" "$ACTION_ENCODING" \
        "$TRANSLATION_SCALE" "$ROTATION_SCALE" "$GRIPPER_SCALE" "$SKIP_T5" \
        "$CAMERA_STATE" \
        "$WRIST_CROP_MODE" "$WRIST_CROP_FRACTION" \
        "$WRIST_LEFT_CENTER_OFFSET_X" "$WRIST_RIGHT_CENTER_OFFSET_X" \
        "$HEAD_CROP_TOP_PIXELS" "$SAVE_CLEAN_RESTORE_LATENT"
}

# 无 GPU 的单 episode action round-trip：HDF5 -> 10D Parquet -> BaseEnv 7D command。
if [ -n "$ROUNDTRIP_HDF5" ]; then
    if [ "$ACTION_ENCODING" != "cosmos_rotation_6d" ]; then
        echo "[ERROR] --roundtrip-hdf5 only supports cosmos_rotation_6d" >&2
        exit 2
    fi
    if [[ "$ROUNDTRIP_HDF5" != /* ]]; then
        ROUNDTRIP_HDF5="$CALL_DIR/$ROUNDTRIP_HDF5"
    fi
    if [ ! -f "$ROUNDTRIP_HDF5" ]; then
        echo "[ERROR] HDF5 does not exist: $ROUNDTRIP_HDF5" >&2
        exit 2
    fi
    STEM=$(basename "$(dirname "$ROUNDTRIP_HDF5")")
    DEBUG_DIR="$PROJECT_DIR/debug_outputs"
    DEBUG_PARQUET="$DEBUG_DIR/${STEM}_rotation6d.parquet"
    DEBUG_REPORT="$DEBUG_DIR/${STEM}_rotation6d_report.csv"
    "$PYTHON_BIN" "$LEGACY_DATA_CONVERT_DIR/debug_replay_cosmos_6d.py" \
        --hdf5 "$ROUNDTRIP_HDF5" \
        --parquet "$DEBUG_PARQUET" \
        --create_debug_parquet \
        --report "$DEBUG_REPORT" \
        --translation_scale "$TRANSLATION_SCALE" \
        --rotation_scale "$ROTATION_SCALE" \
        --gripper_scale "$GRIPPER_SCALE"
    exit 0
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
        task_instruction = task_name.replace("_", " ").strip()
        bos_raw = bos_addr.replace("BOS::", "").strip()
        bos_raw = bos_raw.replace("lerobot_data_v3", "raw_data")
        bos_raw = bos_raw.replace("/success", "/success_episodes")
        if not bos_raw.endswith("/"): bos_raw += "/"
        robot_type = bos_raw.split("/")[1] if len(bos_raw.split("/")) > 1 else "unknown"
        print(json.dumps({
            "task_name": task_name,
            "task_instruction": task_instruction,
            "robot_type": robot_type,
            "bos_path": bos_raw,
        }))
PYEOF
}

# ---- 转换单个任务 ----
convert_one() {
    local TASK_NAME="$1"
    local TASK_INSTRUCTION="${2:-$(echo "$TASK_NAME" | tr '_' ' ')}"
    local RAW_INPUT="${3:-$RAW_DIR/$TASK_NAME}"
    local COSMOS_OUT="${4:-$COSMOS_DIR/${TASK_NAME}_${EPISODE_OUTCOME}_${ACTION_SOURCE}_${ACTION_ENCODING}}"

    if [ ! -d "$RAW_INPUT" ] || [ "$(find "$RAW_INPUT" -name "*.hdf5" 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "[跳过] 原始 HDF5 数据不存在: $RAW_INPUT"
        return 1
    fi
    if ! $PREFLIGHT_ONLY && ! $RESUME && [ -f "$COSMOS_OUT/meta/info.json" ]; then
        if check_existing_metadata "$COSMOS_OUT" "$TASK_INSTRUCTION"; then
            echo "[跳过] 已存在且配置一致: $COSMOS_OUT"
            return 10
        else
            local metadata_status=$?
            if [[ "$metadata_status" -eq 2 ]]; then
                return 2
            fi
        fi
    fi

    echo "  raw:     $RAW_INPUT"
    echo "  cosmos:  $COSMOS_OUT"
    echo "  task:    $TASK_INSTRUCTION"
    echo "  action:  $ACTION_ENCODING ($ACTION_SOURCE)"
    echo "  outcome: $EPISODE_OUTCOME"

    if ! $SKIP_T5 && [ -z "$T5_EMBEDDINGS" ]; then
        echo "  ❌ --t5-embeddings is required (zero-vector fallback is disabled)" >&2
        return 1
    fi

    local EXTRA_ARGS=()
    EXTRA_ARGS+=(--encode-world-size "$ENCODE_WORLD_SIZE")
    [ -n "$ENCODE_DEVICE_IDS" ] && EXTRA_ARGS+=(--encode-device-ids "$ENCODE_DEVICE_IDS")
    [ "$MAX_EPISODES" -gt 0 ] && EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
    if $SKIP_T5; then
        EXTRA_ARGS+=(--skip-t5)
    else
        EXTRA_ARGS+=(--t5-embeddings "$T5_EMBEDDINGS")
    fi
    [ -n "$KINEMATICS_CONFIG" ] && EXTRA_ARGS+=(--kinematics-config "$KINEMATICS_CONFIG")
    $PREFLIGHT_ONLY && EXTRA_ARGS+=(--preflight-only)
    $SAVE_CLEAN_RESTORE_LATENT && EXTRA_ARGS+=(--save-clean-restore-latent)
    $MONITOR_MEMORY && EXTRA_ARGS+=(--monitor-memory --memory-sample-interval "$MEMORY_SAMPLE_INTERVAL")
    $MONITOR_STORAGE && EXTRA_ARGS+=(
        --monitor-storage
        --storage-sample-interval "$STORAGE_SAMPLE_INTERVAL"
        --storage-probe-mib "$STORAGE_PROBE_MIB"
    )

    local restart_count=0
    local force_resume="$RESUME"
    while true; do
        local RUN_ARGS=("${EXTRA_ARGS[@]}")
        $force_resume && RUN_ARGS+=(--resume)
        "$PYTHON_BIN" -m data_convert_refactored.convert \
            --input "$RAW_INPUT" \
            --output "$COSMOS_OUT" \
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
            "${RUN_ARGS[@]}"
        local status=$?
        if [[ "$status" -eq 0 ]]; then
            if $PREFLIGHT_ONLY; then
                echo "  ✅ 预检通过"
            else
                echo "  ✅ 完成 → $COSMOS_OUT"
            fi
            return 0
        fi
        if $AUTO_RESTART && [[ "$status" -eq 137 || "$status" -eq 143 ]] \
            && [[ "$restart_count" -lt "$MAX_RESTARTS" ]]; then
            restart_count=$((restart_count + 1))
            force_resume=true
            local wait_seconds=$((RESTART_DELAY * restart_count))
            echo "[WARN] converter exited $status; resume attempt $restart_count/$MAX_RESTARTS in ${wait_seconds}s" >&2
            sleep "$wait_seconds"
            continue
        fi
        echo "  ❌ 失败! exit=$status" >&2
        return "$status"
    done
}

# ---- 主流程 ----
mkdir -p "$COSMOS_DIR"

# 显式输入模式：无需依赖目录命名或Excel。
if [ -n "$INPUT_OVERRIDE" ]; then
    RAW_EXPLICIT="$INPUT_OVERRIDE"
    [[ "$RAW_EXPLICIT" != /* ]] && RAW_EXPLICIT="$CALL_DIR/$RAW_EXPLICIT"
    if [[ -f "$RAW_EXPLICIT" && "$(basename "$RAW_EXPLICIT")" == "trajectory.hdf5" ]]; then
        RAW_EXPLICIT="$(dirname "$RAW_EXPLICIT")"
    fi
    EXPLICIT_NAME="${FILTER:-$(basename "$RAW_EXPLICIT")}"
    EXPLICIT_TASK="${TASK_DESCRIPTION:-${FILTER:-${EXPLICIT_NAME//_/ }}}"
    EXPLICIT_OUTPUT="${OUTPUT_OVERRIDE:-$COSMOS_DIR/${EXPLICIT_NAME}_${EPISODE_OUTCOME}_${ACTION_SOURCE}_${ACTION_ENCODING}}"
    [[ "$EXPLICIT_OUTPUT" != /* ]] && EXPLICIT_OUTPUT="$CALL_DIR/$EXPLICIT_OUTPUT"
    if convert_one "$EXPLICIT_NAME" "$EXPLICIT_TASK" "$RAW_EXPLICIT" "$EXPLICIT_OUTPUT"; then
        exit 0
    else
        status=$?
        [[ "$status" -eq 10 ]] && exit 0
        exit "$status"
    fi
fi

# 指定 --task 时，先尝试直接转换（不依赖 Excel）
if [ -n "$FILTER" ]; then
    RAW_CANDIDATE="$RAW_DIR/$FILTER"
    if [ -d "$RAW_CANDIDATE" ] && [ "$(find "$RAW_CANDIDATE" -name "*.hdf5" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "========================================"
        echo " Convert (直接模式): $FILTER"
        echo "========================================"
        echo ""
        if $DRY_RUN; then
            echo "  [DRY RUN] raw=$RAW_CANDIDATE action=$ACTION_ENCODING source=$ACTION_SOURCE"
        else
            if convert_one "$FILTER" "${TASK_DESCRIPTION:-${FILTER//_/ }}" "$RAW_CANDIDATE"; then
                :
            else
                status=$?
                [[ "$status" -eq 10 ]] || exit "$status"
            fi
        fi
        echo ""
        echo "========================================"
        echo " Convert 完成"
        echo "========================================"
        exit 0
    fi
fi

if [ -z "$FILTER" ]; then
    echo "[INFO] 处理 Excel 中全部标注完成的任务"
fi

echo "========================================"
echo " Convert: raw HDF5 → Cosmos LeRobot"
echo " raw:     $RAW_DIR"
echo " cosmos:  $COSMOS_DIR"
echo " action:  $ACTION_ENCODING"
echo "========================================"
echo ""

TOTAL=0; OK=0; SKIP=0; FAIL=0

while IFS= read -r line; do
    TASK_NAME=$(echo "$line" | "$PYTHON_BIN" -c "import json,sys; print(json.loads(sys.stdin.read())['task_name'])")
    TASK_INSTRUCTION=$(echo "$line" | "$PYTHON_BIN" -c "import json,sys; print(json.loads(sys.stdin.read())['task_instruction'])")

    TOTAL=$((TOTAL + 1))
    echo "────────────────────────────────────────"
    echo "[$TOTAL] $TASK_NAME"
    echo ""

    if $DRY_RUN; then
        echo "  [DRY RUN]"
        OK=$((OK + 1))
        continue
    fi

    if convert_one "$TASK_NAME" "$TASK_INSTRUCTION" "$RAW_DIR/$TASK_NAME"; then
        OK=$((OK + 1))
    else
        status=$?
        if [[ "$status" -eq 10 ]]; then
            SKIP=$((SKIP + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    fi
    echo ""
done < <(parse_tasks)

echo "========================================"
echo " Convert 完成"
echo " 总计: $TOTAL | 成功: $OK | 跳过: $SKIP | 失败: $FAIL"
echo "========================================"
if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
