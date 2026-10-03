#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo_root"
task_cpu_python=${DLLM_PERSIST_CPU_PYTHON:-/home/esjung/anaconda3/bin/python}
task_run_root=$(mktemp -d /tmp/dllm-request-persistent-reproduce.XXXXXX)
mkdir -p "$task_run_root/results"
cp poc_dllm_request_persistent_replica/results/frozen_manifest.json "$task_run_root/results/frozen_manifest.json"
export DLLM_PERSIST_ARTIFACT_ROOT="$task_run_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-offline}" in
  offline)
    "$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.pack_evidence --unpack-to "$task_run_root/raw/long_v1"
    ;;
  capture)
    /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_dllm_request_persistent_replica.scripts.collect_long_horizon
    ;;
  *) echo 'Usage: reproduce.sh offline|capture' >&2; exit 2 ;;
esac
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.reuse_short_traces
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.extract_long_traces
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.physical_validation
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.simulate_persistent_replica --lane short --workers 4
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.simulate_persistent_replica --lane long --workers 4
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.analyze_persistence
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.copy_amortization
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.analyze_results
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.hotspot_audit
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.figures
"$task_cpu_python" -m poc_dllm_request_persistent_replica.scripts.report
echo "Reproduction artifacts: $task_run_root"
