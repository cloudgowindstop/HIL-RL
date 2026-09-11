#!/usr/bin/env bash
set -euo pipefail

PROJECT=/media/jushen/mingbo-ge/HIL-RL-Project
PYTHON=/media/jushen/mingbo-ge/.venv/bin/python
CONFIG="$PROJECT/HIL-RL/cosmos_offline_train/configs/back_handle_inverse_dynamics_6d.yaml"
OUTPUT_ROOT="$PROJECT/train_outputs/back_handle_full_evaluation"
CHECKPOINT_DIR="$PROJECT/train_outputs/back_handle_joint_6d/checkpoints"
MAX_BATCHES="${MAX_BATCHES:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

export PYTHONPATH="$PROJECT/HIL-RL:$PROJECT/HIL-RL/lerobot/src:$PROJECT/cosmos-policy"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON" -m cosmos_offline_train.evaluation.runner evaluate-suite \
  --config "$CONFIG" \
  --base \
  --checkpoint "step_000005000=$CHECKPOINT_DIR/step_000005000.pt" \
  --checkpoint "step_000010000=$CHECKPOINT_DIR/step_000010000.pt" \
  --checkpoint "step_000015000=$CHECKPOINT_DIR/step_000015000.pt" \
  --checkpoint "step_000020000=$CHECKPOINT_DIR/step_000020000.pt" \
  --objectives policy world value inverse_dynamics \
  --inverse-ablations normal current_only future_only future_shuffle \
  --max-batches "$MAX_BATCHES" \
  --output-dir "$OUTPUT_ROOT"

"$PYTHON" -m cosmos_offline_train.evaluation.runner compare \
  --summary "Base=$OUTPUT_ROOT/base/validation_summary_step_000000000.json" \
  --summary "Step5K=$OUTPUT_ROOT/step_000005000/validation_summary_step_000005000.json" \
  --summary "Step10K=$OUTPUT_ROOT/step_000010000/validation_summary_step_000010000.json" \
  --summary "Step15K=$OUTPUT_ROOT/step_000015000/validation_summary_step_000015000.json" \
  --summary "Step20K=$OUTPUT_ROOT/step_000020000/validation_summary_step_000020000.json" \
  --output-dir "$OUTPUT_ROOT/comparison"

"$PYTHON" -m cosmos_offline_train.evaluation.runner compare-episodes \
  --summary "Base=$OUTPUT_ROOT/base/validation_episode_summary_step_000000000.json" \
  --summary "Step5K=$OUTPUT_ROOT/step_000005000/validation_episode_summary_step_000005000.json" \
  --summary "Step10K=$OUTPUT_ROOT/step_000010000/validation_episode_summary_step_000010000.json" \
  --summary "Step15K=$OUTPUT_ROOT/step_000015000/validation_episode_summary_step_000015000.json" \
  --summary "Step20K=$OUTPUT_ROOT/step_000020000/validation_episode_summary_step_000020000.json" \
  --output-dir "$OUTPUT_ROOT/comparison"
