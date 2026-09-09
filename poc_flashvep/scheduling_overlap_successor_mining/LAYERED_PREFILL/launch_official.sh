#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_HOME=/usr/local/cuda-12.8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=1
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs/layered-prefill
research_model=/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
research_graph_args=()
if [[ "${SCHEDULING_LP_EAGER:-1}" == "1" ]]; then
  research_graph_args+=(--enforce-eager)
fi
exec /home/esjung/.venvs/scheduling-layered-py310/bin/python \
  "$research_ref/nanovllm/entrypoints/api_server.py" \
  --model "$research_model" --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --schedule-mode layered-prefill --num-stages 16 --moe-dtype bfloat16 \
  "${research_graph_args[@]}" --host 127.0.0.1 --port 31800 --nccl-port 31991 \
  --log-level info "$@"
