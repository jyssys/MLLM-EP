#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/esjung/.venvs/scheduling-layered-py310/bin:/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export NVCC_THREADS=2
export PYTORCH_VERSION=2.8.0
export PYTORCH_CUDA_VERSION=12.8
export CMAKE_POLICY_VERSION_MINIMUM=3.5
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs
# Official pinned flash-attention, with the official Layered Prefill patch.
python -m pip install -e "$research_ref/layered-flash-attention" --no-build-isolation --no-deps -v
python -m pip install -e "$research_ref/layered-prefill" --no-build-isolation --no-deps -v
python -c 'import torch, nanovllm.ops, vllm_flash_attn; print("LAYERED_NATIVE_IMPORT_PASS", torch.__version__)'
