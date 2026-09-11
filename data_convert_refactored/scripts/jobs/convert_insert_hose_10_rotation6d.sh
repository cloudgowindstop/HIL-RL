#!/bin/bash
# Convert 10 insert-hose episodes using adapted legacy official statistics.
set -euo pipefail

JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=$(cd "$JOB_DIR/.." && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
cd "$PROJECT_DIR"
source /media/jushen/mingbo-ge/.venv/bin/activate

MODE=${1:-preflight}
if [[ "$MODE" != "preflight" && "$MODE" != "convert" ]]; then
  echo "用法: bash data_convert_refactored/scripts/jobs/convert_insert_hose_10_rotation6d.sh [preflight|convert]" >&2
  exit 2
fi

SOURCE_STATS=/media/jushen/project-rl-dataset/raw_data_0804download/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/dataset_statistics.json
INPUT=/media/jushen/project-rl-dataset/cosmos_raw_data/tienyi_prod2_dualArm-gripper-3cameras_193_insert_hose_into_motor_20260715_am
OUTPUT=/media/jushen/project-rl-dataset/cosmos_dataset

EXTRA_ARGS=()
if [[ "$MODE" == "preflight" ]]; then
  EXTRA_ARGS+=(--preflight-only)
fi

bash "$SCRIPT_DIR/convert.sh" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --task-description "insert hose into motor" \
  --episode-outcome success \
  --stats-mode official \
  --official-dataset-stats "$SOURCE_STATS" \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --translation-scale 0.02 \
  --gripper-scale 1.0 \
  --max-episodes 10 \
  --skip-t5 \
  --encode-batch-size 8 \
  --save-clean-restore-latent \
  "${EXTRA_ARGS[@]}"
