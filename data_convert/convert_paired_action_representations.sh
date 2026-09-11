#!/usr/bin/env bash
set -euo pipefail

PROJECT=/media/jushen/mingbo-ge/HIL-RL-Project
PYTHON=/media/jushen/mingbo-ge/.venv/bin/python
SPLITS=$PROJECT/HIL-RL/cosmos_offline_train/experiments/action_representation_20260903/source_splits
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/paired_action_data_20260903}
STATS_EULER=${STATS_EULER:-$PROJECT/cosmos_data_3drpy/back_handle_20260803_pm_success/dataset_statistics.json}
STATS_6D=${STATS_6D:-$PROJECT/cosmos_data_6drotation_monitor_20260819/back_handle_20260803_pm_success/dataset_statistics.json}
T5=$PROJECT/t5_embeddings_by_task/back_handle_installation_20260803_pm/t5_embeddings.pkl
TASK='Pick up the left black handle and attach it to the white back panel. Then pick up the two black screws one by one and place them onto the handle.'
MODE=${1:-preflight}
SCOPE=${2:-full}

SOURCE_0803=$PROJECT/HIL-RL/dataset_raw/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm
SOURCE_0804=$PROJECT/HIL-RL/dataset_raw/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260804_am

case "$MODE" in
  preflight) EXTRA=(--preflight-only) ;;
  full) EXTRA=(); SCOPE=full ;;
  visualization) EXTRA=(--save-clean-restore-latent); SCOPE=visualization ;;
  *) echo "usage: $0 {preflight|full|visualization} [full|visualization]" >&2; exit 2 ;;
esac

if [[ "$SCOPE" != "full" && "$SCOPE" != "visualization" ]]; then
  echo "usage: $0 {preflight|full|visualization} [full|visualization]" >&2
  exit 2
fi

stats_for_encoding() {
  case "$1" in
    legacy_euler) echo "$STATS_EULER" ;;
    cosmos_rotation_6d) echo "$STATS_6D" ;;
    *) echo "[ERROR] unsupported encoding: $1" >&2; return 2 ;;
  esac
}

convert_one() {
  local encoding=$1 collection=$2 input=$3 manifest=$4 subset=$5
  "$PYTHON" "$PROJECT/HIL-RL/data_convert/convert_raw_to_cosmos.py" \
    --input "$input" \
    --output "$OUTPUT_ROOT/$subset/$encoding/$collection" \
    --episode-manifest "$manifest" \
    --task-description "$TASK" \
    --episode-outcome success \
    --dataset-stats "$(stats_for_encoding "$encoding")" \
    --t5-embeddings "$T5" \
    --action-source puppet_next_frame \
    --action-encoding "$encoding" \
    --translation-scale 0.02 \
    --rotation-scale 0.06 \
    --gripper-scale 1.0 \
    --encode-batch-size 8 \
    "${EXTRA[@]}"
}

if [[ "$SCOPE" == visualization ]]; then
  SUBSET=visualization
  PREFIX=visualization_val
else
  SUBSET=full
  PREFIX=all
fi

for encoding in legacy_euler cosmos_rotation_6d; do
  convert_one "$encoding" 0803_pm_success "$SOURCE_0803" "$SPLITS/${PREFIX}_0803_pm_success.json" "$SUBSET"
  convert_one "$encoding" 0804_am_success "$SOURCE_0804" "$SPLITS/${PREFIX}_0804_am_success.json" "$SUBSET"
done

if [[ "$MODE" == full ]]; then
  for encoding in legacy_euler cosmos_rotation_6d; do
    "$PYTHON" -m cosmos_offline_train.tools.build_paired_dataset_manifests \
      --dataset-root "$OUTPUT_ROOT/full/$encoding" \
      --source-train "$SPLITS/train.json" \
      --source-val "$SPLITS/val.json" \
      --output-dir "$OUTPUT_ROOT/full/$encoding/manifests"
  done
fi
