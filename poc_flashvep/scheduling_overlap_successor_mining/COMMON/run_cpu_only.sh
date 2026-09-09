#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export TORCH_CUDA_ARCH_LIST=9.0
exec "$@"
