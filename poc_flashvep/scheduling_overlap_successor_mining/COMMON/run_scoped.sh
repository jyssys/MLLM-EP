#!/usr/bin/env bash
set -euo pipefail
research_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$research_script_dir/GPU_PAUSED_BY_USER.md" ]]; then
  echo "GPU work is paused by the user; use COMMON/run_cpu_only.sh for CPU analysis." >&2
  exit 73
fi
# Even an apparently CPU-only third-party import can initialize CUDA.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TORCH_CUDA_ARCH_LIST=9.0
exec "$@"
