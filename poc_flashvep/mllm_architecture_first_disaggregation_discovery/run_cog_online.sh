#!/usr/bin/env bash
set -euo pipefail

arch_runtime=$1
arch_snapshot=$2
arch_manifest=$3
arch_out=$4
shift 4

export CC="$(dirname "$arch_runtime")/x86_64-conda-linux-gnu-gcc"
export CXX="$(dirname "$arch_runtime")/x86_64-conda-linux-gnu-g++"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

exec "$arch_runtime" -u -m poc_flashvep.mllm_architecture_first_disaggregation_discovery.cog_online \
  --snapshot "$arch_snapshot" --manifest "$arch_manifest" --out "$arch_out" "$@"
