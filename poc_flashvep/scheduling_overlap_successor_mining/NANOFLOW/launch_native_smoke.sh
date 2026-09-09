#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/esjung/.venvs/scheduling-nanoflow-py310/bin:/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAX_JOBS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
exec /home/esjung/.venvs/scheduling-nanoflow-py310/bin/python \
  /home/esjung/MLLM-EP-github/poc_flashvep/scheduling_overlap_successor_mining/NANOFLOW/run_official_smoke.py "$@"
