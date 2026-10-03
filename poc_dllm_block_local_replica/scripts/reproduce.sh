#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DLLM_REPLICA_ARTIFACT_ROOT="$(mktemp -d /home/esjung/dllm-replica-repro.XXXXXX)"
export CUDA_VISIBLE_DEVICES=2,3,6,7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
replica_python=/home/esjung/anaconda3/bin/python
"$replica_python" -m poc_dllm_block_local_replica.scripts.prepare
"$replica_python" -m poc_dllm_block_local_replica.scripts.physical_audit
"$replica_python" -m poc_dllm_block_local_replica.scripts.validate_correctness
"$replica_python" -m poc_dllm_block_local_replica.scripts.reuse_or_extract_traces
"$replica_python" -m poc_dllm_block_local_replica.scripts.copy_upper_bound
"$replica_python" -m poc_dllm_block_local_replica.scripts.simulate_replication --workers 4
"$replica_python" -m poc_dllm_block_local_replica.scripts.estimate_copy_cost
"$replica_python" -m poc_dllm_block_local_replica.scripts.analyze_results
"$replica_python" -m poc_dllm_block_local_replica.scripts.figures
"$replica_python" -m poc_dllm_block_local_replica.scripts.report
"$replica_python" -m pytest -q poc_dllm_block_local_replica/tests
echo "Append-only output: $DLLM_REPLICA_ARTIFACT_ROOT"
