#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

SCENES="${SCENES:-packed pile}"
SMOKE_RUNS="${SMOKE_RUNS:-1}"
SMOKE_ROUNDS="${SMOKE_ROUNDS:-5}"
LOGDIR="${LOGDIR:-${ICG_LOG_ROOT}/icg_smoke}"
CONFIG="${CONFIG:-$(icg_default_config)}"

icg_require_repo_dirs
icg_activate_env
icg_require_cuda

for scene in ${SCENES}; do
  icg_info "Running smoke benchmark for scene=${scene}."
  python "${SCRIPT_DIR}/run_icg_eval.py" \
    --scene "${scene}" \
    --object-set "${scene}/test" \
    --config "${CONFIG}" \
    --logdir "${LOGDIR}" \
    --name "icgnet_smoke" \
    --num-runs "${SMOKE_RUNS}" \
    --num-rounds "${SMOKE_ROUNDS}"
done

python "${SCRIPT_DIR}/summarize_icg_logs.py" "${LOGDIR}"
