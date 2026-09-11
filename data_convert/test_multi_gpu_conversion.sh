#!/bin/bash
# Compatibility entry point. Canonical implementation lives under data_convert_refactored/scripts.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PROJECT_DIR/data_convert_refactored/scripts/test_multi_gpu_conversion.sh" "$@"
