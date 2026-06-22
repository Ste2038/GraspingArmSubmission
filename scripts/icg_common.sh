#!/usr/bin/env bash
set -Eeuo pipefail

ICG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ICG_REPO_ROOT="${ICG_REPO_ROOT:-$(cd -- "${ICG_SCRIPT_DIR}/.." && pwd)}"
ICG_ENV_NAME="${ICG_ENV_NAME:-icg_cuda121}"
ICG_BENCHMARK_DIR="${ICG_BENCHMARK_DIR:-${ICG_REPO_ROOT}/third_party/icg_benchmark}"
ICG_NET_DIR="${ICG_NET_DIR:-${ICG_REPO_ROOT}/third_party/icg_net}"
ICG_LOG_ROOT="${ICG_LOG_ROOT:-${ICG_REPO_ROOT}/logs}"
ICG_MAMBA_ROOT="${ICG_MAMBA_ROOT:-${HOME}/micromamba}"

icg_info() {
  printf '[icg] %s\n' "$*"
}

icg_warn() {
  printf '[icg:warn] %s\n' "$*" >&2
}

icg_fail() {
  printf '[icg:error] %s\n' "$*" >&2
  exit 1
}

icg_run() {
  if [[ "${ICG_DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %q' "$1"
    shift || true
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

icg_ubuntu_version() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    printf '%s\n' "${VERSION_ID:-unknown}"
  else
    printf 'unknown\n'
  fi
}

icg_check_supported_ubuntu() {
  local version
  version="$(icg_ubuntu_version)"
  case "$version" in
    22.04|24.04)
      icg_info "Ubuntu ${version} detected."
      ;;
    *)
      icg_fail "Unsupported Linux version '${version}'. Use Ubuntu 22.04 or 24.04 for this workflow."
      ;;
  esac
}

icg_is_wsl() {
  grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

icg_micromamba_bin() {
  if command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
  elif [[ -x "${HOME}/.local/bin/micromamba" ]]; then
    printf '%s\n' "${HOME}/.local/bin/micromamba"
  else
    return 1
  fi
}

icg_activate_env() {
  if [[ "${ICG_DRY_RUN:-0}" == "1" ]]; then
    export CONDA_PREFIX="${CONDA_PREFIX:-${ICG_MAMBA_ROOT}/envs/${ICG_ENV_NAME}}"
    icg_info "Dry run: would activate environment ${ICG_ENV_NAME} at ${CONDA_PREFIX}."
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${ICG_ENV_NAME}" ]]; then
    return 0
  fi

  if micromamba_bin="$(icg_micromamba_bin 2>/dev/null)"; then
    export MAMBA_ROOT_PREFIX="${ICG_MAMBA_ROOT}"
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
    # shellcheck disable=SC1090
    eval "$("${micromamba_bin}" shell hook -s bash)"
    set +u
    micromamba activate "${ICG_ENV_NAME}"
    set -u
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    set +u
    conda activate "${ICG_ENV_NAME}"
    set -u
    return 0
  fi

  icg_fail "Could not find micromamba or conda. Run scripts/setup_icg_env.sh first."
}

icg_default_config() {
  local expected="${ICG_BENCHMARK_DIR}/data/icgnet/51--0.656/config.yaml"
  if [[ -f "${expected}" ]]; then
    printf '%s\n' "${expected}"
    return 0
  fi

  local legacy="${ICG_BENCHMARK_DIR}/data/51--0.656/config.yaml"
  if [[ -f "${legacy}" ]]; then
    printf '%s\n' "${legacy}"
    return 0
  fi

  local found
  found="$(find "${ICG_BENCHMARK_DIR}/data" -path '*/config.yaml' -print 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi

  icg_fail "No ICG-Net config.yaml found. Run scripts/download_icg_data.sh after fetching the repos."
}

icg_require_repo_dirs() {
  [[ -d "${ICG_NET_DIR}" ]] || icg_fail "Missing ${ICG_NET_DIR}. Run scripts/fetch_icg_repos.sh."
  [[ -d "${ICG_BENCHMARK_DIR}" ]] || icg_fail "Missing ${ICG_BENCHMARK_DIR}. Run scripts/fetch_icg_repos.sh."
}

icg_require_cuda() {
  python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.stderr.write(
        "[icg:error] torch.cuda.is_available() is False. Restore CUDA/WSL GPU access before running smoke/full evaluation.\n"
    )
    raise SystemExit(1)

print(f"[icg] CUDA ready: {torch.cuda.get_device_name(0)}")
PY
}
