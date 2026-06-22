#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

SCENES="${SCENES:-packed pile}"
FULL_RUNS="${FULL_RUNS:-4}"
FULL_ROUNDS="${FULL_ROUNDS:-100}"
LOGDIR="${LOGDIR:-${ICG_LOG_ROOT}/icg_full}"
CONFIG="${CONFIG:-$(icg_default_config)}"

icg_require_repo_dirs
icg_activate_env
icg_require_cuda

for scene in ${SCENES}; do
  icg_info "Running full benchmark for scene=${scene}."
  python "${SCRIPT_DIR}/run_icg_eval.py" \
    --scene "${scene}" \
    --object-set "${scene}/test" \
    --config "${CONFIG}" \
    --logdir "${LOGDIR}" \
    --name "icgnet_full" \
    --num-runs "${FULL_RUNS}" \
    --num-rounds "${FULL_ROUNDS}"
done

python "${SCRIPT_DIR}/summarize_icg_logs.py" "${LOGDIR}"
