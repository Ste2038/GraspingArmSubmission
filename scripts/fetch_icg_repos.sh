#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

ICG_DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      ICG_DRY_RUN=1
      shift
      ;;
    --repo-root)
      ICG_REPO_ROOT="$2"
      ICG_BENCHMARK_DIR="${ICG_REPO_ROOT}/third_party/icg_benchmark"
      ICG_NET_DIR="${ICG_REPO_ROOT}/third_party/icg_net"
      shift 2
      ;;
    *)
      icg_fail "Unknown argument: $1"
      ;;
  esac
done

clone_or_update() {
  local url="$1"
  local branch="$2"
  local dir="$3"

  if [[ -d "${dir}/.git" ]]; then
    icg_info "Updating ${dir}"
    icg_run git -C "${dir}" fetch origin "${branch}"
    icg_run git -C "${dir}" checkout "${branch}"
    icg_run git -C "${dir}" pull --ff-only origin "${branch}"
  elif [[ -e "${dir}" ]]; then
    icg_fail "${dir} exists but is not a git repository. Move it aside or inspect it manually."
  else
    icg_info "Cloning ${url} into ${dir}"
    icg_run mkdir -p "$(dirname "${dir}")"
    icg_run git clone --branch "${branch}" "${url}" "${dir}"
  fi
}

clone_or_update "https://github.com/renezurbruegg/icg_net.git" "main" "${ICG_NET_DIR}"
clone_or_update "https://github.com/renezurbruegg/icg_benchmark.git" "master" "${ICG_BENCHMARK_DIR}"

icg_info "Official ICG repositories are ready under third_party/."

