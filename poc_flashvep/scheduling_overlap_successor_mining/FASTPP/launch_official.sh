#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE=${SCHEDULING_NCCL_P2P_DISABLE:-1}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
research_python=/home/esjung/.venvs/scheduling-fastpp-py310/bin/python
research_model=/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
exec "$research_python" -m sglang.launch_server \
  --model-path "$research_model" --dtype bfloat16 \
  --port 31800 --host 127.0.0.1 --context-length 16384 \
  --mem-fraction-static 0.9 --schedule-policy fcfs \
  --disable-radix-cache --disable-cuda-graph --enable-mixed-chunk \
  --chunked-prefill-size 2048 --disable-overlap-schedule --pp 4 \
  --random-seed 20260908 "$@"
