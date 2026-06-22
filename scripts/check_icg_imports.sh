#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

icg_require_repo_dirs
icg_activate_env

python - <<'PY'
import torch
import MinkowskiEngine
import open3d
import pybullet
import icg_net
import icg_benchmark

print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print("imports_ok=MinkowskiEngine,open3d,pybullet,icg_net,icg_benchmark")
PY
