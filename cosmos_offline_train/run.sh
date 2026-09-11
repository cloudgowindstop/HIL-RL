#!/bin/bash
# Cosmos 离线训练统一入口
#
# bash cosmos_offline_train/run.sh env-check
# bash cosmos_offline_train/run.sh preflight
# bash cosmos_offline_train/run.sh prepare-splits
# bash cosmos_offline_train/run.sh test-cpu
# bash cosmos_offline_train/run.sh smoke-single 0
# bash cosmos_offline_train/run.sh smoke-ddp 0,1
# bash cosmos_offline_train/run.sh train 0,1
# bash cosmos_offline_train/run.sh resume 0,1 /path/to/checkpoint
# bash cosmos_offline_train/run.sh validate 0 /path/to/checkpoint
# bash cosmos_offline_train/run.sh validate-base 0
# bash cosmos_offline_train/run.sh plot /path/to/metrics.jsonl [/path/to/output]
# bash cosmos_offline_train/run.sh evaluate 0 /path/to/checkpoint
# bash cosmos_offline_train/run.sh evaluate-base 0
# CONFIG=... bash cosmos_offline_train/run.sh visualize 0 --base
# CONFIG=... bash cosmos_offline_train/run.sh visualize 0 --checkpoint /path/to/checkpoint
# bash cosmos_offline_train/run.sh summarize [/path/to/run_root]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
readonly SCRIPT_DIR PROJECT_DIR WORKSPACE_DIR

VENV_DIR="${VENV_DIR:-/media/jushen/mingbo-ge/.venv}"
PYTHON=$VENV_DIR/bin/python
TORCHRUN=$VENV_DIR/bin/torchrun
CONFIG_INPUT="${CONFIG:-$SCRIPT_DIR/configs/cosmos_offline_6d.yaml}"

resolve_config() {
    local candidate
    for candidate in \
        "$CONFIG_INPUT" \
        "$PROJECT_DIR/$CONFIG_INPUT" \
        "$SCRIPT_DIR/$CONFIG_INPUT"; do
        if [[ -f "$candidate" ]]; then
            realpath "$candidate"
            return 0
        fi
    done
    echo "[ERROR] 配置不存在: $CONFIG_INPUT" >&2
    echo "[ERROR] 已从当前目录、$PROJECT_DIR、$SCRIPT_DIR 查找" >&2
    return 1
}

CONFIG=$(resolve_config) || exit 2
readonly CONFIG

if [[ ! -f "$VENV_DIR/bin/activate" || ! -x "$PYTHON" || ! -x "$TORCHRUN" ]]; then
    echo "[ERROR] 无效虚拟环境: $VENV_DIR" >&2
    exit 2
fi
source "$VENV_DIR/bin/activate"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/lerobot/src:$WORKSPACE_DIR/cosmos-policy"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
hash -r
LAUNCH_CWD=$(pwd)
cd "$PROJECT_DIR"
echo "[CONFIG] $CONFIG"

MODE=${1:-}
GPUS=${2:-0}
RESUME_PATH=${3:-}

resolve_existing_path() {
    local input=$1
    local candidate
    if [[ -z "$input" ]]; then
        return 1
    fi
    for candidate in \
        "$input" \
        "$LAUNCH_CWD/$input" \
        "$WORKSPACE_DIR/$input" \
        "$PROJECT_DIR/$input"; do
        if [[ -f "$candidate" ]]; then
            realpath "$candidate"
            return 0
        fi
    done
    echo "[ERROR] 文件不存在: $input" >&2
    echo "[ERROR] 已从启动目录、$WORKSPACE_DIR、$PROJECT_DIR 查找" >&2
    return 1
}

if [[ -z "$MODE" ]]; then
    echo "用法: bash cosmos_offline_train/run.sh MODE [GPUS] [RESUME_PATH]" >&2
    exit 2
fi

case "$MODE" in
    env-check)
        "$PYTHON" -c 'import sys, torch; print("python=" + sys.executable); print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); print("gpus=" + str(torch.cuda.device_count()))'
        ;;
    preflight|prepare-splits|test-cpu)
        "$PYTHON" -m cosmos_offline_train.train --config "$CONFIG" --mode "$MODE"
        ;;
    plot)
        METRICS_PATH=$GPUS
        PLOT_OUTPUT=${RESUME_PATH:-$(dirname "$METRICS_PATH")/plots}
        "$PYTHON" -m cosmos_offline_train.plot_metrics \
            --metrics "$METRICS_PATH" --output-dir "$PLOT_OUTPUT"
        ;;
    smoke-single)
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.train \
            --config "$CONFIG" --mode "$MODE"
        ;;
    smoke-ddp|train)
        NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
        CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node "$NPROC" \
            -m cosmos_offline_train.train --config "$CONFIG" --mode "$MODE"
        ;;
    resume)
        if [[ -z "$RESUME_PATH" ]]; then
            echo "[ERROR] resume 需要 checkpoint 路径。" >&2
            exit 2
        fi
        RESUME_PATH=$(resolve_existing_path "$RESUME_PATH") || exit 2
        echo "[RESUME] $RESUME_PATH"
        NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
        CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node "$NPROC" \
            -m cosmos_offline_train.train --config "$CONFIG" --mode resume \
            --resume "$RESUME_PATH"
        ;;
    validate)
        if [[ -z "$RESUME_PATH" ]]; then
            echo "[ERROR] validate 需要 checkpoint 路径。" >&2
            exit 2
        fi
        RESUME_PATH=$(resolve_existing_path "$RESUME_PATH") || exit 2
        echo "[RESUME] $RESUME_PATH"
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.train \
            --config "$CONFIG" --mode validate --resume "$RESUME_PATH"
        ;;
    validate-base)
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.train \
            --config "$CONFIG" --mode validate-base
        ;;
    evaluate)
        if [[ -z "$RESUME_PATH" ]]; then
            echo "[ERROR] evaluate 需要 checkpoint 路径。" >&2
            exit 2
        fi
        RESUME_PATH=$(resolve_existing_path "$RESUME_PATH") || exit 2
        echo "[RESUME] $RESUME_PATH"
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.evaluation.runner \
            evaluate --config "$CONFIG" --checkpoint "$RESUME_PATH" \
            --objectives policy world value inverse_dynamics \
            --inverse-ablations normal current_only future_only future_shuffle
        ;;
    evaluate-base)
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.evaluation.runner \
            evaluate --config "$CONFIG" --base \
            --objectives policy world value inverse_dynamics \
            --inverse-ablations normal current_only future_only future_shuffle
        ;;
    visualize)
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m cosmos_offline_train.evaluation.visualize_predictions \
            --config "$CONFIG" "${@:3}"
        ;;
    summarize)
        SUMMARIZE_ROOT=${GPUS:-/media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/action_representation_20260903}
        if [[ "$SUMMARIZE_ROOT" == "0" ]]; then
            SUMMARIZE_ROOT=/media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/action_representation_20260903
        fi
        "$PYTHON" -m cosmos_offline_train.tools.summarize_compare_matrix --root "$SUMMARIZE_ROOT"
        ;;
    *)
        echo "[ERROR] 未知模式: $MODE" >&2
        exit 2
        ;;
esac
