#!/bin/bash
# Batch-convert the four back-handle success/failure directories sequentially.
set -uo pipefail

JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=$(cd "$JOB_DIR/.." && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$PACKAGE_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
RAW_ROOT="$WORKSPACE_DIR/raw_data"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_DIR/cosmos_data}"
CONVERT_SH="$SCRIPT_DIR/convert.sh"

RESOURCE_DIR="$RAW_ROOT/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm"
OFFICIAL_STATS="$RESOURCE_DIR/dataset_statistics.json"
T5_EMBEDDINGS="$RESOURCE_DIR/t5_embeddings.pkl"
TASK_DESCRIPTION="Pick up the left black handle and attach it to the white back panel. Then pick up the two black screws one by one and place them onto the handle."

# Known malformed episodes are excluded explicitly. Paths are relative to RAW_ROOT.
# The source data is never moved or modified; conversion uses a temporary symlink view.
EXCLUDED_HDF5=(
    "tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/0804_014423/data/trajectory.hdf5"
)

PREFLIGHT_ONLY=false
SKIP_PREFLIGHT=false
DRY_RUN=false
FAIL_FAST=false
ENCODE_BATCH_SIZE=8
ENCODE_WORLD_SIZE="${ENCODE_WORLD_SIZE:-1}"
ENCODE_DEVICE_IDS="${ENCODE_CUDA_DEVICES:-}"
BATCH_SIZE=1
ACTION_ENCODING="legacy_euler"
MONITOR_MEMORY=false
MEMORY_SAMPLE_INTERVAL="2.0"
MONITOR_STORAGE=false
STORAGE_SAMPLE_INTERVAL="5.0"
STORAGE_PROBE_MIB="8"
RESUME=false
AUTO_RESTART=false
MAX_RESTARTS=3
RESTART_DELAY=10

usage() {
    sed -n '2,32p' "$0"
    cat <<'EOF'

Usage:
  bash data_convert_refactored/scripts/jobs/batch_convert_back_handle.sh [options]

Options:
  --preflight-only       Check all four directories without loading VAE.
  --skip-preflight       Start formal conversion without the preflight pass.
  --dry-run              Print the resolved conversion plan only.
  --fail-fast            Stop after the first failed directory.
  --encode-batch-size N  VAE micro-batch size (default: 8).
  --encode-world-size N  Number of visible GPUs used for VAE encode (default: 1).
  --encode-device-ids S  Process-local CUDA IDs, for example 0,2,4,6.
  --batch-size N         Dataset batch setting (default: 1).
  --output-root PATH     Output dataset root (default: workspace/cosmos_data).
  --action-encoding S   legacy_euler or cosmos_rotation_6d (default: legacy_euler).
  --monitor-memory       Record key-stage and periodic CPU/cgroup memory logs.
  --memory-sample-interval SEC  Sampling interval (default: 2.0).
  --monitor-storage      Record compact GPFS/process logs and run fsync probes.
  --storage-sample-interval SEC  Storage sampling interval (default: 5.0).
  --storage-probe-mib N  Real allocation probe size (default: 8 MiB).
  --resume               Continue each incomplete output dataset.
  --auto-restart         Resume after converter exits with SIGKILL/SIGTERM.
  --max-restarts N       Maximum automatic restarts per dataset (default: 3).
  --restart-delay SEC    Linear backoff base seconds (default: 10).
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --fail-fast) FAIL_FAST=true; shift ;;
        --encode-batch-size) ENCODE_BATCH_SIZE="$2"; shift 2 ;;
        --encode-world-size) ENCODE_WORLD_SIZE="$2"; shift 2 ;;
        --encode-device-ids) ENCODE_DEVICE_IDS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --action-encoding) ACTION_ENCODING="$2"; shift 2 ;;
        --monitor-memory) MONITOR_MEMORY=true; shift ;;
        --memory-sample-interval) MEMORY_SAMPLE_INTERVAL="$2"; shift 2 ;;
        --monitor-storage) MONITOR_STORAGE=true; shift ;;
        --storage-sample-interval) STORAGE_SAMPLE_INTERVAL="$2"; shift 2 ;;
        --storage-probe-mib) STORAGE_PROBE_MIB="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --auto-restart) AUTO_RESTART=true; RESUME=true; shift ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --restart-delay) RESTART_DELAY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if $PREFLIGHT_ONLY && $SKIP_PREFLIGHT; then
    echo "[ERROR] --preflight-only and --skip-preflight are mutually exclusive" >&2
    exit 2
fi
if [[ ! "$ENCODE_BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] batch sizes must be positive integers" >&2
    exit 2
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
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$PROJECT_DIR/$OUTPUT_ROOT"
fi
for required in "$RAW_ROOT" "$CONVERT_SH" "$OFFICIAL_STATS" "$T5_EMBEDDINGS"; do
    if [[ ! -e "$required" ]]; then
        echo "[ERROR] required path does not exist: $required" >&2
        exit 2
    fi
done

mapfile -t CANDIDATE_INPUT_DIRS < <(
    find "$RAW_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*back-handle-installation*' -print | sort
)
if [[ ${#CANDIDATE_INPUT_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] no back-handle input directories found under $RAW_ROOT" >&2
    exit 2
fi

declare -a INPUT_DIRS
for candidate in "${CANDIDATE_INPUT_DIRS[@]}"; do
    if find "$candidate" -type f -name trajectory.hdf5 -print -quit | grep -q .; then
        INPUT_DIRS+=("$candidate")
    else
        echo "[SKIP] no trajectory.hdf5: $candidate"
    fi
done
if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] no back-handle trajectory.hdf5 files found" >&2
    exit 2
fi

declare -a NAMES OUTCOMES OUTPUT_DIRS
for input_dir in "${INPUT_DIRS[@]}"; do
    name=$(basename "$input_dir")
    if [[ "${name,,}" == *fail* ]]; then
        outcome="failure"
    else
        outcome="success"
    fi
    timestamp=$(sed -n 's/.*_\(2026[0-9]\{4\}_[ap]m\).*/\1/p' <<<"$name")
    [[ -n "$timestamp" ]] || timestamp="$name"
    NAMES+=("$name")
    OUTCOMES+=("$outcome")
    OUTPUT_DIRS+=("$OUTPUT_ROOT/back_handle_${timestamp}_${outcome}")
done

echo "============================================================"
echo "Back-handle Cosmos batch conversion"
echo "raw root:    $RAW_ROOT"
echo "output root: $OUTPUT_ROOT"
echo "stats:       $OFFICIAL_STATS"
echo "T5:          $T5_EMBEDDINGS"
echo "action:      $ACTION_ENCODING"
echo "============================================================"
for i in "${!INPUT_DIRS[@]}"; do
    count=$(find "${INPUT_DIRS[$i]}" -type f -name trajectory.hdf5 | wc -l)
    excluded=0
    for relative_path in "${EXCLUDED_HDF5[@]}"; do
        [[ -f "$RAW_ROOT/$relative_path" && "$RAW_ROOT/$relative_path" == "${INPUT_DIRS[$i]}"/* ]] \
            && excluded=$((excluded + 1))
    done
    selected=$((count - excluded))
    printf '[%d] %-7s episodes=%-4s selected=%-4s excluded=%-2s %s\n' \
        "$((i + 1))" "${OUTCOMES[$i]}" "$count" "$selected" "$excluded" "${NAMES[$i]}"
    echo "    -> ${OUTPUT_DIRS[$i]}"
done

if [[ ${#EXCLUDED_HDF5[@]} -gt 0 ]]; then
    echo "Excluded malformed episodes:"
    printf '  %s\n' "${EXCLUDED_HDF5[@]}"
fi

if $DRY_RUN; then
    echo "[DRY RUN] no conversion was started"
    exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"

STAGING_ROOT=$(mktemp -d /tmp/cosmos_back_handle_filtered.XXXXXX)
cleanup_staging() {
    rm -rf -- "$STAGING_ROOT"
}
trap cleanup_staging EXIT INT TERM

declare -a FILTERED_INPUT_DIRS
for i in "${!INPUT_DIRS[@]}"; do
    source_dir="${INPUT_DIRS[$i]}"
    filtered_dir="$STAGING_ROOT/${NAMES[$i]}"
    while IFS= read -r hdf5_path; do
        relative_to_root="${hdf5_path#"$RAW_ROOT/"}"
        skip=false
        for excluded_path in "${EXCLUDED_HDF5[@]}"; do
            if [[ "$relative_to_root" == "$excluded_path" ]]; then
                skip=true
                break
            fi
        done
        $skip && continue
        relative_to_task="${hdf5_path#"$source_dir/"}"
        link_path="$filtered_dir/$relative_to_task"
        mkdir -p "$(dirname "$link_path")"
        ln -s "$hdf5_path" "$link_path"
    done < <(find "$source_dir" -type f -name trajectory.hdf5 -print | sort)
    FILTERED_INPUT_DIRS+=("$filtered_dir")
done

run_conversion() {
    local index="$1"
    local phase="$2"
    local input_dir="${FILTERED_INPUT_DIRS[$index]}"
    local output_dir="${OUTPUT_DIRS[$index]}"
    local outcome="${OUTCOMES[$index]}"
    local log_file="$OUTPUT_ROOT/logs/${NAMES[$index]}_${phase}.log"
    local extra=()
    extra+=(--encode-world-size "$ENCODE_WORLD_SIZE")
    [ -n "$ENCODE_DEVICE_IDS" ] && extra+=(--encode-device-ids "$ENCODE_DEVICE_IDS")
    [[ "$phase" == "preflight" ]] && extra+=(--preflight-only)
    $MONITOR_MEMORY && extra+=(--monitor-memory --memory-sample-interval "$MEMORY_SAMPLE_INTERVAL")
    $MONITOR_STORAGE && extra+=(
        --monitor-storage
        --storage-sample-interval "$STORAGE_SAMPLE_INTERVAL"
        --storage-probe-mib "$STORAGE_PROBE_MIB"
    )
    $RESUME && extra+=(--resume)
    $AUTO_RESTART && extra+=(
        --auto-restart
        --max-restarts "$MAX_RESTARTS"
        --restart-delay "$RESTART_DELAY"
    )

    echo "------------------------------------------------------------"
    echo "[$phase][$((index + 1))/${#INPUT_DIRS[@]}] ${NAMES[$index]}"
    echo "source: ${INPUT_DIRS[$index]}"
    echo "filtered input: $input_dir"
    echo "log: $log_file"
    bash "$CONVERT_SH" \
        --input "$input_dir" \
        --output "$output_dir" \
        --task-description "$TASK_DESCRIPTION" \
        --episode-outcome "$outcome" \
        --stats-mode official \
        --official-dataset-stats "$OFFICIAL_STATS" \
        --action-source puppet_next_frame \
        --action-encoding "$ACTION_ENCODING" \
        --translation-scale 0.02 \
        --rotation-scale 0.06 \
        --gripper-scale 1.0 \
        --t5-embeddings "$T5_EMBEDDINGS" \
        --camera-state normal \
        --wrist-crop-mode center_width \
        --wrist-crop-fraction 0.75 \
        --wrist-left-center-offset-x 64 \
        --wrist-right-center-offset-x 64 \
        --head-crop-top-pixels 100 \
        --encode-batch-size "$ENCODE_BATCH_SIZE" \
        --batch-size "$BATCH_SIZE" \
        "${extra[@]}" 2>&1 | tee "$log_file"
    return "${PIPESTATUS[0]}"
}

declare -a READY
PREFLIGHT_FAILED=0
if $SKIP_PREFLIGHT; then
    for i in "${!INPUT_DIRS[@]}"; do READY[$i]=1; done
else
    for i in "${!INPUT_DIRS[@]}"; do
        if run_conversion "$i" preflight; then
            READY[$i]=1
        else
            READY[$i]=0
            PREFLIGHT_FAILED=$((PREFLIGHT_FAILED + 1))
            echo "[ERROR] preflight failed: ${NAMES[$i]}" >&2
            $FAIL_FAST && exit 1
        fi
    done
fi

if $PREFLIGHT_ONLY; then
    echo "============================================================"
    echo "Preflight complete: passed=$((${#INPUT_DIRS[@]} - PREFLIGHT_FAILED)), failed=$PREFLIGHT_FAILED"
    [[ "$PREFLIGHT_FAILED" -eq 0 ]]
    exit $?
fi

CONVERTED=0
FAILED=$PREFLIGHT_FAILED
SKIPPED_PREFLIGHT=$PREFLIGHT_FAILED
for i in "${!INPUT_DIRS[@]}"; do
    if [[ "${READY[$i]:-0}" -ne 1 ]]; then
        echo "[SKIP] preflight did not pass: ${NAMES[$i]}"
        continue
    fi
    if run_conversion "$i" convert; then
        CONVERTED=$((CONVERTED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "[ERROR] conversion failed: ${NAMES[$i]}" >&2
        $FAIL_FAST && exit 1
    fi
done

echo "============================================================"
echo "Batch complete"
echo "successful/already-complete: $CONVERTED"
echo "skipped after preflight:     $SKIPPED_PREFLIGHT"
echo "failed:                      $FAILED"
echo "logs:                        $OUTPUT_ROOT/logs"
echo "============================================================"
[[ "$FAILED" -eq 0 ]]
