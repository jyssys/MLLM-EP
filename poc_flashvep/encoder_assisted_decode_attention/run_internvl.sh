#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 || $3 == "$4" ]]; then
  printf 'Usage: %s RUNTIME RUN_ROOT DECODE_GPU HELPER_GPU RUN_LABEL\n' "$0" >&2
  exit 2
fi

eada_runtime=$(realpath "$1")
eada_runs=$(realpath "$2")
eada_decode=$3
eada_helper=$4
eada_trial="$eada_runs/$5"
eada_model="$eada_runs/models/internvl3_8b"
eada_manifest=../mllm_epd_autosetup/manifests_1795600931728841878/main100_manifest_template_v2.json
eada_module=poc_flashvep.encoder_assisted_decode_attention
eada_lease=poc_flashvep.mllm_specific_disaggregation_discovery.gpu_epoch
eada_python="$eada_runtime/bin/python"
eada_source="${BASH_SOURCE[0]%/*}"

mkdir "$eada_trial"
sha256sum "$eada_model/config.json" "$eada_model/model.safetensors.index.json" "$eada_manifest" \
  "$eada_source/internvl_native.py" "$eada_source/operator_bench.py" "$eada_source/peer_attention.py" \
  "$eada_source/run_internvl.sh" > "$eada_trial/source.sha256"
export PATH="$eada_runtime/bin:$PATH"
export CUDA_HOME="$eada_runtime/targets/x86_64-linux"
export CC="$eada_runtime/bin/x86_64-conda-linux-gnu-cc" CXX="$eada_runtime/bin/x86_64-conda-linux-gnu-c++"
export FLASHINFER_NVCC="$eada_runtime/bin/nvcc"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_MODULES_CACHE="$eada_trial/hf_modules"
"$eada_python" - <<'PY' > "$eada_trial/software.json"
from importlib.metadata import version
import json
import sys

software = {package: version(package) for package in ("vllm", "torch", "transformers", "triton", "safetensors")}
print(json.dumps({"python": sys.version.split()[0], **software}, indent=2))
PY

"$eada_python" -m "$eada_lease" --run-dir "$eada_trial/native_lease" \
  --devices "$eada_decode" --wall-seconds 420 -- \
  "$eada_python" -u -m "$eada_module.internvl_native" --model "$eada_model" \
  --manifest "$eada_manifest" --output "$eada_trial/native.json" \
  --tokens 16 --repeats 3 --capture-qkv "$eada_trial/real_layer0_qkv.pt"

"$eada_python" -m "$eada_lease" --run-dir "$eada_trial/operator_lease" \
  --devices "$eada_decode,$eada_helper" --wall-seconds 420 -- \
  "$eada_python" -u -m "$eada_module.operator_bench" --out "$eada_trial/operator.jsonl" \
  --decode-device 0 --helper-device 1 --fixture "$eada_trial/real_layer0_qkv.pt" \
  --ratios .25 .5 .75 1 --local-splits 0 1 2 4 8 --repeats 3 --calls 64 --require-split-graph
