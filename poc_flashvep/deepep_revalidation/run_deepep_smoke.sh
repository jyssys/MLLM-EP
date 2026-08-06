#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
RESULT_DIR=${1:?usage: run_deepep_smoke.sh RESULT_DIR}

export NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_DIR}"

"${DEEPEP_VENV}/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  "${REPO_ROOT}/poc_flashvep/deepep_revalidation/deepep_smoke.py" \
  --output-dir "${RESULT_DIR}" 2>&1 | tee "${RESULT_DIR}/smoke.log"

"${DEEPEP_VENV}/bin/python" - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
rows = [json.loads((directory / f"smoke_rank{rank}.json").read_text()) for rank in range(4)]
summary = {
    "status": "ok" if all(row["all_ranks_pass"] for row in rows) else "failed",
    "all_ranks_pass": all(row["all_ranks_pass"] for row in rows),
    "ranks": rows,
}
(directory / "smoke_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY
