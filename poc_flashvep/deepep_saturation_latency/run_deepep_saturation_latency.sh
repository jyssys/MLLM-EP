#!/usr/bin/env bash
set -euo pipefail

# This PoC is intentionally isolated from vLLM model execution.  Only physical
# GPUs 1,2,3,4 are exposed to the four DeepEP ranks.
export CUDA_VISIBLE_DEVICES="1,2,3,4"
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DEEPEP_PYTHON:-/home/esjung/.venvs/flashvep-deepep-v020/bin/python}"
RESULT_DIR="${1:?usage: $0 RESULT_DIR [warmups] [iterations]}"
WARMUPS="${2:-5}"
ITERATIONS="${3:-20}"
mkdir -p "${RESULT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/poc_flashvep/deepep_saturation_latency/replay.py" prepare \
  --output "${RESULT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "${REPO_ROOT}/poc_flashvep/deepep_saturation_latency/replay.py" run \
  --output "${RESULT_DIR}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}"
"${PYTHON_BIN}" "${REPO_ROOT}/poc_flashvep/deepep_saturation_latency/replay.py" aggregate \
  --output "${RESULT_DIR}"
