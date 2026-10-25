#!/usr/bin/env bash
set -euo pipefail
export ENABLE_FLASH_ATTENTION_4=true
export ENABLE_FLASH_ATTENTION_3=false
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -c 'import torch, tilelang; from fla.modules.conv.triton import causal_conv1d_fwd; from xllm.paper_part1.native_inference import load_native_model; print("NATIVE_IMPORT_PASS")'
