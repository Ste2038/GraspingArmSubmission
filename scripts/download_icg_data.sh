#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

icg_require_repo_dirs
icg_activate_env

icg_info "Downloading official benchmark datasets and checkpoints."
icg_info "Target: ${ICG_BENCHMARK_DIR}/data"
(
  cd "${ICG_BENCHMARK_DIR}"
  python scripts/download_data.py
)

icg_info "Download step complete. Check ${ICG_BENCHMARK_DIR}/data for checkpoints and config files."

