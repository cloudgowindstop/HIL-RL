#!/bin/bash
# Compatibility entry point. Canonical implementation lives under data_convert_refactored/scripts/jobs.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PROJECT_DIR/data_convert_refactored/scripts/jobs/convert_insert_hose_10_rotation6d.sh" "$@"
