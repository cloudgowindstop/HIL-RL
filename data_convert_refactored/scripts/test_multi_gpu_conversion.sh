#!/bin/bash
# Test the multi-GPU conversion implementation on the development machine.
#
# Quick test (no dataset required):
#   bash data_convert_refactored/scripts/test_multi_gpu_conversion.sh
#
# End-to-end single-GPU versus multi-GPU comparison:
#   bash data_convert_refactored/scripts/test_multi_gpu_conversion.sh \
#     --input /path/to/trajectory.hdf5 \
#     --stats /path/to/dataset_statistics.json \
#     --task "task instruction"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
VENV_ACTIVATE="/media/jushen/mingbo-ge/.venv/bin/activate"
PYTHON_BIN="/media/jushen/mingbo-ge/.venv/bin/python3"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[ERROR] Project virtual environment is missing: $VENV_ACTIVATE" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
source /media/jushen/mingbo-ge/.venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR:$PROJECT_DIR/lerobot/src"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/rl_envs:$WORKSPACE_DIR/cosmos-policy"
export PYTHONPATH="$PYTHONPATH:$WORKSPACE_DIR/cosmos-policy/cosmos_policy"

INPUT=""
STATS=""
TASK=""
EPISODE_OUTCOME="success"
DEVICE_IDS="0,1,2,3,4,5,6,7"
ENCODE_BATCH_SIZE=16
MAX_EPISODES=1
OUTPUT_ROOT=""

usage() {
    sed -n '2,12p' "$0"
    cat <<'EOF'

Options:
  --input PATH             One trajectory.hdf5 or an episode/task directory.
  --stats PATH             Official dataset_statistics.json.
  --task TEXT              Task instruction stored in the converted dataset.
  --episode-outcome VALUE  success or failure (default: success).
  --device-ids IDS         Process-local CUDA IDs (default: 0,1,2,3,4,5,6,7).
  --encode-batch-size N    VAE micro-batch size (default: 16).
  --max-episodes N         Episodes used by each conversion (default: 1).
  --output-root PATH       Parent of single/ and multi/ outputs. Must not exist.
  -h, --help               Show this help.

With no --input/--stats/--task, only fast regression tests are run.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --stats) STATS="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --episode-outcome) EPISODE_OUTCOME="$2"; shift 2 ;;
        --device-ids) DEVICE_IDS="$2"; shift 2 ;;
        --encode-batch-size) ENCODE_BATCH_SIZE="$2"; shift 2 ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$DEVICE_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] --device-ids must be comma-separated non-negative integers" >&2
    exit 2
fi
if ! [[ "$ENCODE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --encode-batch-size must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --max-episodes must be a positive integer" >&2
    exit 2
fi
if [[ "$EPISODE_OUTCOME" != "success" && "$EPISODE_OUTCOME" != "failure" ]]; then
    echo "[ERROR] --episode-outcome must be success or failure" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICE_IDS"
WORLD_SIZE=${#DEVICE_ARRAY[@]}
VISIBLE_GPUS=$(
    "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())'
)
for DEVICE_ID in "${DEVICE_ARRAY[@]}"; do
    if (( DEVICE_ID >= VISIBLE_GPUS )); then
        echo "[ERROR] cuda:$DEVICE_ID is not visible; visible GPU count: $VISIBLE_GPUS" >&2
        exit 2
    fi
done

cd "$PROJECT_DIR"

echo "[1/4] Python and shell syntax checks"
"$PYTHON_BIN" -m py_compile \
    data_convert/convert_raw_to_cosmos.py \
    data_convert/multi_gpu_vae.py \
    data_convert_refactored/encoding/multi_gpu_vae.py \
    data_convert_refactored/tools/compare_conversion_outputs.py
bash -n \
    data_convert_refactored/scripts/convert.sh \
    data_convert_refactored/scripts/pipeline.sh \
    data_convert_refactored/scripts/jobs/batch_convert_back_handle.sh

echo "[2/4] Configuration, adapter, output comparator, and VAE wrapper-pool tests"
"$PYTHON_BIN" -m unittest \
    data_convert_refactored.tests.test_gpu_config \
    data_convert_refactored.tests.test_compare_conversion_outputs \
    data_convert_refactored.tests.test_multi_gpu_encode_adapter \
    data_convert_refactored.tests.test_multi_gpu_vae \
    -v

if [[ -z "$INPUT" && -z "$STATS" && -z "$TASK" ]]; then
    echo "[PASS] Fast multi-GPU tests completed; visible GPUs: $VISIBLE_GPUS"
    echo "Provide --input, --stats, and --task to run the end-to-end VAE comparison."
    exit 0
fi

if [[ -z "$INPUT" || -z "$STATS" || -z "$TASK" ]]; then
    echo "[ERROR] End-to-end mode requires --input, --stats, and --task together" >&2
    exit 2
fi
if [[ ! -e "$INPUT" ]]; then
    echo "[ERROR] Input does not exist: $INPUT" >&2
    exit 2
fi
if [[ ! -f "$STATS" ]]; then
    echo "[ERROR] Stats file does not exist: $STATS" >&2
    exit 2
fi
INPUT=$(realpath "$INPUT")
STATS=$(realpath "$STATS")

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT=$(mktemp -d /tmp/hil_rl_multigpu_test.XXXXXX)
elif [[ -e "$OUTPUT_ROOT" ]]; then
    echo "[ERROR] --output-root already exists; use a new path: $OUTPUT_ROOT" >&2
    exit 2
else
    mkdir -p "$OUTPUT_ROOT"
fi

SINGLE_OUTPUT="$OUTPUT_ROOT/single"
MULTI_OUTPUT="$OUTPUT_ROOT/multi"
COMMON_ARGS=(
    --input "$INPUT"
    --task-description "$TASK"
    --episode-outcome "$EPISODE_OUTCOME"
    --stats-mode official
    --official-dataset-stats "$STATS"
    --skip-t5
    --max-episodes "$MAX_EPISODES"
    --encode-batch-size "$ENCODE_BATCH_SIZE"
)

echo "[3/4] Single-GPU baseline conversion"
CUDA_VISIBLE_DEVICES="${DEVICE_ARRAY[0]}" \
    bash data_convert_refactored/scripts/convert.sh \
    "${COMMON_ARGS[@]}" \
    --output "$SINGLE_OUTPUT" \
    --encode-world-size 1

echo "[3/4] ${WORLD_SIZE}-GPU candidate conversion"
# CUDA_VISIBLE_DEVICES remaps the selected physical cards to process-local 0..N-1.
LOCAL_DEVICE_IDS=$(
    "$PYTHON_BIN" -c 'import sys; print(",".join(map(str, range(int(sys.argv[1])))))' \
    "$WORLD_SIZE"
)
CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
    bash data_convert_refactored/scripts/convert.sh \
    "${COMMON_ARGS[@]}" \
    --output "$MULTI_OUTPUT" \
    --encode-world-size "$WORLD_SIZE" \
    --encode-device-ids "$LOCAL_DEVICE_IDS"

echo "[4/4] Compare all episode Parquet columns"
"$PYTHON_BIN" -m data_convert_refactored.tools.compare_conversion_outputs \
    "$SINGLE_OUTPUT" "$MULTI_OUTPUT"

echo "[PASS] Single-GPU and multi-GPU conversion outputs match."
echo "Test artifacts were kept at: $OUTPUT_ROOT"
