#!/usr/bin/env bash
set -euo pipefail
repo="/home/esjung/MLLM-EP-long-reasoning-signal"
root="${repo}/poc_mllm_mixed_domain_online_calibration"
python_bin="/home/esjung/.venvs/flashvep-deepep-v020/bin/python"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="${root}/hooks:${repo}"
export MLLM_MDOC_TRACE=1
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0
stop_record="${root}/artifacts/gpu_ep4_stop.json"
start_record="${root}/artifacts/gpu_ep4_restart.json"
cleanup() {
  PYTHONPATH="${repo}" python3 -m poc_mllm_mixed_domain_online_calibration.gpu_guard start \
    --gpus 4,5,6,7 --record "${start_record}" || true
}
trap cleanup EXIT INT TERM
PYTHONPATH="${repo}" python3 -m poc_mllm_mixed_domain_online_calibration.gpu_guard stop \
  --gpus 4,5,6,7 --record "${stop_record}"
cd "${repo}"
"${python_bin}" -m poc_mllm_mixed_domain_online_calibration.scripts.run_physical \
  --topology ep4 --physical-gpus 4,5,6,7 \
  --pool "${root}/artifacts/manifests/frozen_domain_pool.json" \
  --stream "${root}/artifacts/manifests/frozen_iid_stream.json" \
  --output "${root}/artifacts/runs/ep4_mixed160" \
  --trace-dir "${root}/artifacts/raw_traces/ep4_mixed160" \
  --control "${root}/artifacts/ep4_control.json" \
  --ledger "${root}/artifacts/execution_ledger.jsonl" --max-tokens 192
