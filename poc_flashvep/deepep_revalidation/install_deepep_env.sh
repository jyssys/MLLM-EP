#!/usr/bin/env bash
set -euo pipefail

# vLLM 0.20.0-compatible DeepEP environment. This script deliberately uses a
# separate venv and workspace; it never installs into flashvep-poc itself.
export CUDA_VISIBLE_DEVICES=4,5,6,7

BASE_PYTHON=${BASE_PYTHON:-/home/esjung/anaconda3/envs/flashvep-poc/bin/python}
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
NVSHMEM_VERSION=3.3.24
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
RESULT_DIR=${1:-}

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Missing base Python: ${BASE_PYTHON}" >&2
  exit 1
fi

mkdir -p "$(dirname "${DEEPEP_VENV}")" "${DEEPEP_WORKSPACE}"
if [[ ! -x "${DEEPEP_VENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${DEEPEP_VENV}"
fi
export PATH="${DEEPEP_VENV}/bin:${PATH}"

"${DEEPEP_VENV}/bin/python" -m pip install --upgrade \
  pip 'setuptools>=77,<81' wheel cmake
# A system-site-packages venv can see the base ninja module without receiving
# its console script. Force a venv-local install because CUDA dlink requires
# the executable, not merely the importable package.
"${DEEPEP_VENV}/bin/python" -m pip install --ignore-installed ninja
test -x "${DEEPEP_VENV}/bin/ninja"

NVSHMEM_ARCHIVE="libnvshmem-linux-x86_64-${NVSHMEM_VERSION}_cuda12-archive.tar.xz"
NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/${NVSHMEM_ARCHIVE}"
NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
if [[ ! -f "${NVSHMEM_DIR}/include/nvshmem.h" ]]; then
  archive_path="${DEEPEP_WORKSPACE}/${NVSHMEM_ARCHIVE}"
  extracted_path="${DEEPEP_WORKSPACE}/${NVSHMEM_ARCHIVE%.tar.xz}"
  if [[ ! -f "${archive_path}" ]]; then
    curl -fSL --retry 3 --retry-delay 2 "${NVSHMEM_URL}" -o "${archive_path}"
  fi
  if [[ -e "${NVSHMEM_DIR}" || -e "${extracted_path}" ]]; then
    echo "Incomplete NVSHMEM path already exists; refusing to overwrite it." >&2
    exit 1
  fi
  tar -xf "${archive_path}" -C "${DEEPEP_WORKSPACE}"
  mv "${extracted_path}" "${NVSHMEM_DIR}"
fi

DEEPEP_SOURCE="${DEEPEP_WORKSPACE}/DeepEP"
if [[ ! -d "${DEEPEP_SOURCE}/.git" ]]; then
  if [[ -e "${DEEPEP_SOURCE}" ]]; then
    echo "Non-Git DeepEP source path already exists; refusing to overwrite it." >&2
    exit 1
  fi
  git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_SOURCE}"
fi

current_commit=$(git -C "${DEEPEP_SOURCE}" rev-parse HEAD)
if [[ "${current_commit}" != "${DEEPEP_COMMIT}" ]]; then
  if [[ -n "$(git -C "${DEEPEP_SOURCE}" status --porcelain)" ]]; then
    echo "DeepEP source is dirty and at ${current_commit}; refusing checkout." >&2
    exit 1
  fi
  git -C "${DEEPEP_SOURCE}" fetch origin "${DEEPEP_COMMIT}"
  git -C "${DEEPEP_SOURCE}" checkout "${DEEPEP_COMMIT}"
fi

export NVSHMEM_DIR CUDA_HOME
export TORCH_CUDA_ARCH_LIST=9.0
export DISABLE_AGGRESSIVE_PTX_INSTRS=1
export MAX_JOBS=${MAX_JOBS:-16}
"${DEEPEP_VENV}/bin/python" -m pip install \
  --no-build-isolation --no-deps --verbose "${DEEPEP_SOURCE}"

if [[ -n "${RESULT_DIR}" ]]; then
  mkdir -p "${RESULT_DIR}"
  DEEPEP_SOURCE="${DEEPEP_SOURCE}" \
  DEEPEP_COMMIT="${DEEPEP_COMMIT}" \
  NVSHMEM_DIR="${NVSHMEM_DIR}" \
  NVSHMEM_VERSION="${NVSHMEM_VERSION}" \
  DEEPEP_VENV="${DEEPEP_VENV}" \
  "${DEEPEP_VENV}/bin/python" - "${RESULT_DIR}/install_manifest.json" <<'PY'
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

source = Path(os.environ["DEEPEP_SOURCE"])
payload = {
    "status": "ok",
    "deep_ep_commit": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "requested_commit": os.environ["DEEPEP_COMMIT"],
    "deep_ep_source": str(source),
    "deep_ep_import": str(importlib.util.find_spec("deep_ep").origin),
    "nvshmem_version": os.environ["NVSHMEM_VERSION"],
    "nvshmem_dir": os.environ["NVSHMEM_DIR"],
    "venv": os.environ["DEEPEP_VENV"],
    "python": sys.version.replace("\n", " "),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
fi
