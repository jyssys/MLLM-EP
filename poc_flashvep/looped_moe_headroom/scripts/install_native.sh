#!/usr/bin/env bash
set -euo pipefail
repo=/home/esjung/MLLM-EP-github
python=/home/esjung/.venvs/looped-moe-headroom-py312/bin/python
export PATH=/home/esjung/.venvs/looped-moe-headroom-py312/bin:$PATH
source_dir=/home/esjung/external/xllm-loop-ep-poc
report="$repo/poc_flashvep/reports/looped_moe_headroom"
mkdir -p "$report/logs"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export ENABLE_FLASH_ATTENTION_4=true
export ENABLE_FLASH_ATTENTION_3=false
"$python" -m pip install flash-attn-4==4.0.0b30 flash-linear-attention==0.6.0
"$python" -m pip install tilelang==0.1.15 apache-tvm-ffi==0.1.12
"$python" -m pip install -r "$source_dir/requirements.txt"
"$python" -m pip install -e "$source_dir" --no-build-isolation --config-settings editable_mode=compat
"$python" -m pip freeze > "$report/environment_freeze.txt"
"$python" -c 'import torch, xllm_extension; from xllm.paper_part1.native_inference import load_native_model; print(torch.__version__, "NATIVE_IMPORT_PASS")'
