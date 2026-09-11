#!/bin/bash
# Run deterministic synthetic conversion with the project environment and real VAE.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_DIR=$(cd "$PROJECT_DIR/.." && pwd)
PYTHON_BIN="/media/jushen/mingbo-ge/.venv/bin/python3"

source /media/jushen/mingbo-ge/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$WORKSPACE_DIR/cosmos-policy:$WORKSPACE_DIR/cosmos-policy/cosmos_policy:$PROJECT_DIR/lerobot/src:$PROJECT_DIR/rl_envs:$PROJECT_DIR:$PROJECT_DIR/data_convert"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m data_convert_refactored.tools.check_synthetic_real_vae "$@"
