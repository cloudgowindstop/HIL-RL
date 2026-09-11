#!/bin/bash
# Preflight for converting 50 insert-hose episodes to dual-arm 20D rotation-6D Cosmos data.
set -euo pipefail

JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=$(cd "$JOB_DIR/.." && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
cd "$PROJECT_DIR"
source /media/jushen/mingbo-ge/.venv/bin/activate

bash "$SCRIPT_DIR/convert.sh" \
  --input /media/jushen/project-rl-dataset/cosmos_raw_data/tienyi_prod2_dualArm-gripper-3cameras_193_insert_hose_into_motor_20260715_am \
  --output /media/jushen/project-rl-dataset/cosmos_dataset \
  --task-description "insert hose into motor" \
  --episode-outcome success \
  --dataset-stats /media/jushen/project-rl-dataset/cosmos_dataset_stats/insert_hose_20260715_am_rotation6d.json \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --translation-scale 0.02 \
  --gripper-scale 1.0 \
  --max-episodes 50 \
  --skip-t5 \
  --encode-batch-size 8 \
  --preflight-only
