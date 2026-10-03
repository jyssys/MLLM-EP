#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=2,3,6,7
cpu_python=/home/esjung/anaconda3/bin/python
"$cpu_python" -m pytest -q poc_dllm_ep_comm/tests
# prepare is append-only and refuses to overwrite a frozen manifest.
if [[ ! -f poc_dllm_ep_comm/results/frozen_manifest.json ]]; then
  "$cpu_python" -m poc_dllm_ep_comm.scripts.prepare
fi
run_name="${1:-reproduction_v1}"
"$cpu_python" -m poc_dllm_ep_comm.scripts.launch --name "$run_name" --limit 32 --secondary
"$cpu_python" -m poc_dllm_ep_comm.scripts.analyze --run "$run_name" --output "$run_name"
"$cpu_python" -m poc_dllm_ep_comm.scripts.figures --result "$run_name"
"$cpu_python" -m poc_dllm_ep_comm.scripts.supplement --result "$run_name"
# The scoped GPU transaction restores the user's burner on success/failure.
