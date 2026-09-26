#!/usr/bin/env bash
set -euo pipefail

cd /home/esjung/MLLM-EP-github
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PATH=/home/esjung/.venvs/flashvep-deepep-v020/bin:$PATH

PY=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
PLOT_PY=/home/esjung/.venvs/dinfer-ep2-deepep/bin/python
ROOT=poc_mllm_resolution_expert_tradeoff/artifacts/20260925_part2_v1
HELDOUT=$ROOT/heldout
TIMING=$ROOT/timing
ANALYSIS=$ROOT/analysis

# The official-lane candidate run is append-only and checkpointed per DP rank.
# Wait for all 192 image-disjoint samples, then also wait for its model workers
# to release GPU 4-7 before starting bounded timing runs.
while true; do
  completed=$(awk 'END {print NR}' "$HELDOUT/ep4_r05_k6/results.dp0.jsonl" "$HELDOUT/ep4_r05_k6/results.dp1.jsonl")
  if [[ "$completed" -ge 192 ]]; then
    break
  fi
  sleep 30
done
while pgrep -f 'run_factorial_ep4.*heldout/ep4_r05_k6' >/dev/null \
   || pgrep -f 'supervise_single_wave.*heldout/ep4_r05_k6' >/dev/null; do
  sleep 10
done

$PY -m poc_mllm_resolution_expert_tradeoff.scripts.analyze_heldout \
  --results "$HELDOUT/reused_part1_r0_k8/results.jsonl" \
  --results "$HELDOUT/ep4_r05_k8/results.jsonl" \
  --results "$HELDOUT/ep4_r05_k6/results.jsonl" \
  --frozen-policy "$ROOT/frozen_policy.json" \
  --output "$ANALYSIS/heldout_final.json"

run_custom_timing() {
  local manifest=$1
  local output=$2
  local kval=$3
  local tokens=$4
  shift 4
  if [[ -f "$output/run.json" ]] && jq -e '.status == "COMPLETE"' "$output/run.json" >/dev/null; then
    return
  fi
  $PY -m poc_mllm_resolution_expert_tradeoff.scripts.run_factorial_ep4 \
    --physical-gpus 4,5,6,7 --manifest "$manifest" --output "$output" \
    --k-visual "$kval" --batch-size 1 --max-tokens "$tokens" \
    --max-model-len 4096 --max-num-seqs 1 --kv-cache-gib 4 \
    --lane deterministic --ignore-eos --warmup-repeats 1 \
    --measured-repeats 3 --schedule resolution_major "$@"
}

run_custom_timing "$TIMING/manifests/timing_frozen_k8.json" "$TIMING/fixed32/k8" 8 32 --wave-offset 81000
run_custom_timing "$TIMING/manifests/timing_r05_k6.json" "$TIMING/fixed32/k6" 6 32 --wave-offset 82000
run_custom_timing "$TIMING/manifests/timing_frozen_k8.json" "$TIMING/fixed128/k8" 8 128 --wave-offset 83000
run_custom_timing "$TIMING/manifests/timing_r05_k6.json" "$TIMING/fixed128/k6" 6 128 --wave-offset 84000

run_batch4() {
  local kval=$1
  local output="$TIMING/batch4/k$kval"
  if [[ -f "$output/run.json" ]] && jq -e '.status == "COMPLETE"' "$output/run.json" >/dev/null; then
    return
  fi
  $PY -m poc_mllm_resolution_expert_tradeoff.scripts.run_factorial_ep4 \
    --physical-gpus 4,5,6,7 \
    --manifest "$TIMING/manifests/timing_allres_k$kval.json" \
    --output "$output" --k-visual "$kval" --batch-size 2 \
    --max-tokens 1 --max-model-len 4096 --max-num-seqs 2 --kv-cache-gib 4 \
    --lane deterministic --ignore-eos --warmup-repeats 1 \
    --measured-repeats 3 --schedule resolution_major --wave-offset "$((85000 + kval * 100))"
}

run_batch4 8
# The installed vLLM 0.20 external-DP frontend was independently reproduced
# hanging on the heterogeneous distinct-image global-batch4 K6 wave (both DPs
# idle with the shared-memory broadcast waiter blocked).  K8 completed and is
# retained as the secondary batch4 baseline; do not turn the failed K6/K4
# secondary into a fabricated timing point.

if [[ ! -f "$TIMING/native_r0_k8/run.json" ]] \
   || ! jq -e '.status == "COMPLETE"' "$TIMING/native_r0_k8/run.json" >/dev/null; then
  $PY -m poc_mllm_ep_foundation.scripts.run_full_model \
    --physical-gpus 4,5,6,7 --backend physical_ep4 --lane deterministic \
    --manifest "$TIMING/manifests/timing_r0_k8.json" \
    --output "$TIMING/native_r0_k8" --max-tokens 1 --max-model-len 4096 \
    --batch-size 1 --max-num-seqs 1 --kv-cache-gib 4 --ignore-eos \
    --warmup-repeats 1 --measured-repeats 3
fi

$PY -m poc_mllm_resolution_expert_tradeoff.scripts.analyze_noop_runtime \
  --native-root "$TIMING/native_r0_k8" \
  --custom-root "$TIMING/clean/r0_k8" \
  --output "$ANALYSIS/noop_runtime.json"

profile_args=()
route_args=()
natural_args=()
fixed32_args=()
fixed128_args=()
batch4_args=()
for r in r0 r05 r025; do
  for k in 8 6 4; do
    lower="${r}_k${k}"
    upper="$(tr '[:lower:]' '[:upper:]' <<<"$r")_K${k}"
    profile_args+=(--profile-analysis "$upper=$ANALYSIS/physical_profiles/$lower.json")
    route_args+=(--route-root "$upper=$ANALYSIS/routes/$lower")
    case "$k" in
      8) natural_root="$ROOT/discovery/ep4_k8_v2" ;;
      6) natural_root="$ROOT/discovery/ep4_k6" ;;
      4) natural_root="$ROOT/discovery/ep4_k4" ;;
    esac
    natural_args+=(--natural-root "$upper=$natural_root")
    if [[ "$k" == 8 ]]; then
      batch4_args+=(--batch4-root "$upper=$TIMING/batch4/k8")
    fi
    if [[ "$upper" == R0_K8 || "$upper" == R05_K8 ]]; then
      fixed32_args+=(--fixed32-root "$upper=$TIMING/fixed32/k8")
      fixed128_args+=(--fixed128-root "$upper=$TIMING/fixed128/k8")
    elif [[ "$upper" == R05_K6 ]]; then
      fixed32_args+=(--fixed32-root "$upper=$TIMING/fixed32/k6")
      fixed128_args+=(--fixed128-root "$upper=$TIMING/fixed128/k6")
    fi
  done
done

$PY -m poc_mllm_resolution_expert_tradeoff.scripts.summarize_physical_costs \
  "${profile_args[@]}" "${route_args[@]}" "${natural_args[@]}" \
  "${fixed32_args[@]}" "${fixed128_args[@]}" "${batch4_args[@]}" \
  --output "$ANALYSIS/physical_costs.json"

$PY -m poc_mllm_resolution_expert_tradeoff.scripts.analyze_grid \
  --results "$ROOT/discovery/ep4_k8_v2/results.jsonl" \
  --results "$ROOT/discovery/ep4_k6/results.jsonl" \
  --results "$ROOT/discovery/ep4_k4/results.jsonl" \
  --physical-cost-summary "$ANALYSIS/physical_costs.json" \
  --output "$ANALYSIS/discovery_final.json"

$PY -m poc_mllm_resolution_expert_tradeoff.scripts.export_results \
  --discovery "$ROOT/discovery/ep4_k8_v2/results.jsonl" \
  --discovery "$ROOT/discovery/ep4_k6/results.jsonl" \
  --discovery "$ROOT/discovery/ep4_k4/results.jsonl" \
  --heldout "$HELDOUT/reused_part1_r0_k8/results.jsonl" \
  --heldout "$HELDOUT/ep4_r05_k8/results.jsonl" \
  --heldout "$HELDOUT/ep4_r05_k6/results.jsonl" \
  --output "$ROOT/results.jsonl"

$PLOT_PY -m poc_mllm_resolution_expert_tradeoff.scripts.generate_reports \
  --analysis "$ANALYSIS/discovery_final.json" \
  --heldout "$ANALYSIS/heldout_final.json" \
  --physical-costs "$ANALYSIS/physical_costs.json" \
  --noop "$ANALYSIS/noop_parity.json" \
  --noop-runtime "$ANALYSIS/noop_runtime.json" \
  --frozen-policy "$ROOT/frozen_policy.json" \
  --virtual "$ANALYSIS/virtual_summary.json" \
  --reports poc_mllm_resolution_expert_tradeoff/reports \
  --artifacts "$ROOT"

$PY -m unittest poc_mllm_resolution_expert_tradeoff.tests.test_contract

# Return only the four authorized GPUs to the user's existing utilization
# policy after every measurement and report has completed.
nohup env VLLM_UTILIZE_GPUS=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh \
  > "$ROOT/final_burner.log" 2>&1 < /dev/null &
sleep 10
pgrep -f 'utilize.py --gpus 4,5,6,7' > "$ROOT/final_burner.pids"
