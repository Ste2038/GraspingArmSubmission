#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

MIN_FREE_GB="${MIN_FREE_GB:-50}"

icg_info "Repository root: ${ICG_REPO_ROOT}"
icg_check_supported_ubuntu

if icg_is_wsl; then
  icg_info "Execution environment: WSL2/Linux compatibility layer."
else
  icg_info "Execution environment: native Linux."
fi

if [[ "${ICG_REPO_ROOT}" == /mnt/* ]]; then
  icg_warn "Repo is on a Windows-mounted path. This works, but CUDA extension builds are usually faster on the Linux filesystem."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  icg_info "GPU visibility:"
  if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    icg_warn "nvidia-smi is installed, but GPU access failed. CUDA evaluation will not work until GPU access is restored."
  fi
else
  icg_warn "nvidia-smi not found. GPU acceleration may not be available."
fi

if command -v python3 >/dev/null 2>&1; then
  icg_info "Python: $(python3 --version)"
else
  icg_warn "python3 not found."
fi

if command -v gcc >/dev/null 2>&1; then
  icg_info "GCC: $(gcc --version | head -n 1)"
else
  icg_warn "gcc not found."
fi

if command -v g++ >/dev/null 2>&1; then
  icg_info "G++: $(g++ --version | head -n 1)"
else
  icg_warn "g++ not found."
fi

if micromamba_path="$(icg_micromamba_bin 2>/dev/null)"; then
  icg_info "micromamba: ${micromamba_path}"
elif command -v conda >/dev/null 2>&1; then
  icg_info "conda: $(command -v conda)"
else
  icg_warn "No micromamba/conda detected. setup_icg_env.sh can install micromamba."
fi

available_gb="$(df -BG "${ICG_REPO_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
icg_info "Free space at repo path: ${available_gb} GB"
if [[ "${available_gb}" -lt "${MIN_FREE_GB}" ]]; then
  icg_warn "Less than ${MIN_FREE_GB} GB free. Dataset/checkpoint downloads and builds may fail."
fi

icg_info "System check complete."
