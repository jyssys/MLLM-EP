#!/usr/bin/env bash
set -euo pipefail
cd /home/esjung/external/flash-attention-affinity-poc/hopper
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES=''
export MAX_JOBS=8
export NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST=9.0a
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m pip wheel . \
  --no-deps --no-build-isolation --wheel-dir /home/esjung/looped-moe-affinity-deps/wheels
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m pip install \
  --no-deps --target /home/esjung/looped-moe-affinity-deps/site \
  /home/esjung/looped-moe-affinity-deps/wheels/flash_attn_3-*.whl
