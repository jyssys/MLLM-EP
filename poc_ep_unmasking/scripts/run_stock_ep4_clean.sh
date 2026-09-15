#!/usr/bin/env bash
set -euo pipefail

dataset="$1"
label="$2"
submitted="$3"
mini="$4"
generation="$5"
study_root=/home/esjung/MLLM-EP-unmasking/poc_ep_unmasking
venv=/home/esjung/.venvs/llada2-flash-sglang-053
runtime=/home/esjung/external/dinfer-llada2-flash-poc
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
result_dir="${study_root}/results/${label}"
mkdir -p "${result_dir}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
unset EP_UNMASK_ATLAS_DIR LLADA_DENOISE_TRACE LLADA_EP_TRACE_DIR \
  LLADA_EP_TRACE_LAYERS LLADA_EP_TRACE_ROUTES LLADA_EP_SHAPE_TRACE_DIR
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

nvidia-smi -i 4,5,6,7 --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | tee "${result_dir}/physical_gpu_inventory.csv"
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits | tee "${result_dir}/processes_before.csv"
nvidia-smi topo -m | tee "${result_dir}/topology.txt"
nvidia-smi nvlink -s -i 4,5,6,7 | tee "${result_dir}/nvlink.txt"
"${venv}/bin/python" -c 'import torch; print("logical->physical: 0->4 1->5 2->6 3->7; device_count=", torch.cuda.device_count())' \
  | tee "${result_dir}/logical_mapping.txt"

"${venv}/bin/python" "${runtime}/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" \
  --dataset "$dataset" \
  --gpu 0,1,2,3 \
  --batch_size "$submitted" \
  --gen_len "$generation" \
  --block_length 32 \
  --threshold 0.9 \
  --config 42 \
  --model_type flash \
  --mini_batch_size "$mini" \
  --ep_size 4 \
  --moe_a2a_backend deepep \
  --deepep_mode normal \
  --output_dir "${result_dir}/benchmark" \
  --exp_name "$label" 2>&1 | tee "${result_dir}/benchmark.log"
