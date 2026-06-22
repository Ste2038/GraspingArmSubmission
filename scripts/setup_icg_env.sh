#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/icg_common.sh
source "${SCRIPT_DIR}/icg_common.sh"

ICG_DRY_RUN=0
SKIP_APT=0
SKIP_MINKOWSKI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      ICG_DRY_RUN=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-minkowski)
      SKIP_MINKOWSKI=1
      shift
      ;;
    --env-name)
      ICG_ENV_NAME="$2"
      shift 2
      ;;
    *)
      icg_fail "Unknown argument: $1"
      ;;
  esac
done

icg_check_supported_ubuntu

if [[ "${ICG_REPO_ROOT}" == /mnt/* ]]; then
  icg_warn "Building CUDA extensions from /mnt/* can be slow. Native Linux filesystem is faster if available."
fi

install_apt_packages() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    icg_info "Skipping apt package installation."
    return
  fi

  local packages=(
    build-essential
    bzip2
    ca-certificates
    cmake
    curl
    git
    libegl1
    libgl1
    libglib2.0-0
    libopenblas-dev
    libx11-6
    libxcursor1
    libxext6
    libxi6
    libxinerama1
    libxrandr2
    libxrender1
    mesa-utils
    ninja-build
    unzip
    wget
    xvfb
  )

  icg_info "Installing Ubuntu packages needed by CUDA builds, Open3D, and PyBullet."
  icg_run sudo apt-get update
  icg_run sudo apt-get install -y "${packages[@]}"
}

install_micromamba_if_needed() {
  if icg_micromamba_bin >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
    icg_info "micromamba/conda already available."
    return
  fi

  icg_info "Installing micromamba into ${HOME}/.local/bin."
  icg_run mkdir -p "${HOME}/.local/bin"
  if [[ "${ICG_DRY_RUN}" == "1" ]]; then
    icg_run bash -lc "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C '${HOME}/.local/bin' --strip-components=1 bin/micromamba"
  elif command -v bzip2 >/dev/null 2>&1; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C "${HOME}/.local/bin" --strip-components=1 bin/micromamba
  else
    icg_warn "bzip2 executable not found; extracting micromamba with Python tarfile."
    python3 - "${HOME}/.local/bin" <<'PY'
import os
import stat
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
dest_dir = Path(sys.argv[1])
dest_dir.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(url) as response:
    archive_data = response.read()

tmp_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(archive_data)
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "r:bz2") as archive:
        member = archive.getmember("bin/micromamba")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("bin/micromamba was not extractable from the downloaded archive")
        output = dest_dir / "micromamba"
        output.write_bytes(extracted.read())
        output.chmod(output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"[icg] wrote {output}")
finally:
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
PY
  fi
}

ensure_env() {
  local create_args=(
    -y
    -n "${ICG_ENV_NAME}"
    -c pytorch
    -c nvidia
    -c conda-forge
    python=3.10
    pip
    cuda-nvcc=12.4.99
    cuda-cudart=12.1.105
    cuda-cudart-dev=12.1.105
    cuda-cccl=12.4.99
    gcc_linux-64=12.4.0
    gxx_linux-64=12.4.0
    openblas
    cmake
    make
    ninja
  )

  if micromamba_bin="$(icg_micromamba_bin 2>/dev/null)"; then
    export MAMBA_ROOT_PREFIX="${ICG_MAMBA_ROOT}"
    if "${micromamba_bin}" env list | awk '{print $1}' | grep -qx "${ICG_ENV_NAME}"; then
      icg_info "Environment ${ICG_ENV_NAME} already exists."
      icg_info "Ensuring core environment packages are pinned."
      icg_run "${micromamba_bin}" install "${create_args[@]}"
    else
      icg_info "Creating micromamba environment ${ICG_ENV_NAME}."
      icg_run "${micromamba_bin}" create "${create_args[@]}"
    fi
  elif command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "${ICG_ENV_NAME}"; then
      icg_info "Environment ${ICG_ENV_NAME} already exists."
      icg_info "Ensuring core environment packages are pinned."
      icg_run conda install "${create_args[@]}"
    else
      icg_info "Creating conda environment ${ICG_ENV_NAME}."
      icg_run conda create "${create_args[@]}"
    fi
  elif [[ "${ICG_DRY_RUN}" == "1" ]]; then
    icg_info "Dry run: would create ${ICG_ENV_NAME} with micromamba after installing it."
  else
    icg_fail "No micromamba or conda found after installer step."
  fi
}

install_python_packages() {
  icg_activate_env

  icg_info "Installing PyTorch 2.2.2 with CUDA 12.1 wheels."
  icg_run python -m pip install --upgrade "pip<24.1" setuptools wheel
  icg_run python -m pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

  icg_info "Installing PyG CUDA 12.1 wheels and project Python dependencies."
  icg_run python -m pip install torch_geometric==2.5.2
  icg_run python -m pip install \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
  icg_run python -m pip install -r "${ICG_BENCHMARK_DIR}/requirements.txt"
  icg_run python -m pip install -r "${ICG_NET_DIR}/requirements.txt"
  icg_info "Installing ICG-Net runtime packages listed in the upstream CUDA 12.1 environment."
  icg_run python -m pip install \
    antlr4-python3-runtime==4.8 \
    freetype-py==2.4.0 \
    gdown==5.1.0 \
    hydra-core==1.0.5 \
    loguru==0.7.2 \
    mako==1.3.10 \
    omegaconf==2.0.6 \
    plyfile \
    pyaml==23.12.0 \
    pycollada==0.6 \
    pyglet==2.0.15 \
    pyopengl==3.1.0 \
    pyrender==0.1.45 \
    pytorch-lightning==2.2.4 \
    rtree==1.2.0 \
    scikit-image==0.23.2 \
    terminaltables==3.1.10 \
    torchmetrics==1.4.0 \
    trimesh==4.2.4
  icg_run python -m pip install --no-deps urdfpy==0.0.22
  icg_run python -m pip install "numpy<2"
}

install_minkowski() {
  if [[ "${SKIP_MINKOWSKI}" == "1" ]]; then
    icg_info "Skipping MinkowskiEngine installation."
    return
  fi

  icg_activate_env
  local minkowski_dir="${ICG_REPO_ROOT}/third_party/MinkowskiEngine"
  if [[ -d "${minkowski_dir}/.git" ]]; then
    icg_info "Updating patched MinkowskiEngine."
    icg_run git -C "${minkowski_dir}" pull --ff-only
  elif [[ -e "${minkowski_dir}" ]]; then
    icg_fail "${minkowski_dir} exists but is not a git repository."
  else
    icg_info "Cloning patched MinkowskiEngine."
    icg_run git clone "https://github.com/renezurbruegg/MinkowskiEngine.git" "${minkowski_dir}"
  fi

  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
  local nvidia_pkg_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
  local nvidia_include_path=""
  local nvidia_library_path=""
  local include_dir
  local lib_dir
  for include_dir in "${nvidia_pkg_dir}"/*/include; do
    [[ -d "${include_dir}" ]] && nvidia_include_path="${nvidia_include_path}:${include_dir}"
  done
  for lib_dir in "${nvidia_pkg_dir}"/*/lib; do
    [[ -d "${lib_dir}" ]] && nvidia_library_path="${nvidia_library_path}:${lib_dir}"
  done
  export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/targets/x86_64-linux/include/cccl${nvidia_include_path}:${CPATH:-}"
  export LIBRARY_PATH="${nvidia_library_path#:}:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${nvidia_library_path#:}:${LD_LIBRARY_PATH:-}"

  icg_info "Building MinkowskiEngine for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}."
  icg_run bash -lc "cd '${minkowski_dir}' && python setup.py install --force_cuda --blas=openblas --blas_include_dirs='${CONDA_PREFIX}/include' --cuda_home='${CUDA_HOME}'"
}

install_local_packages() {
  icg_activate_env
  icg_info "Installing pointnet2 and icg_benchmark; adding icg_net to the environment path."
  icg_run bash -lc "cd '${ICG_NET_DIR}/icg_net/third_party/pointnet2' && python setup.py install"
  icg_run env ICG_NET_DIR_FOR_PTH="${ICG_NET_DIR}" python -c "import os, site; from pathlib import Path; p = Path(site.getsitepackages()[0]) / 'icg_net_repo.pth'; p.write_text(os.environ['ICG_NET_DIR_FOR_PTH'] + '\n', encoding='utf-8'); print(f'[icg] wrote {p}')"
  icg_run python -m pip install -e "${ICG_BENCHMARK_DIR}"
}

install_apt_packages
install_micromamba_if_needed
if [[ "${ICG_DRY_RUN}" == "1" ]]; then
  bash "${SCRIPT_DIR}/fetch_icg_repos.sh" --dry-run
else
  bash "${SCRIPT_DIR}/fetch_icg_repos.sh"
fi
ensure_env
install_python_packages
install_minkowski
install_local_packages

icg_info "ICG-Net environment setup complete."
icg_info "Next: bash scripts/download_icg_data.sh"
