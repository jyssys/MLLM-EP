# Vision-conditioned decode reduction bounded PoC

This directory contains two independently gated discovery tracks for the
frozen Qwen3-VL-30B-A3B-Instruct baseline:

- Track A: exact measurement of the local error introduced by consolidating
  secondary Top-8 expert branches into already-executed primary branches.
- Track B: exact measurement of temporal same-expert recurrence and branch
  output stability before any cached contribution is propagated.

The shared routing analysis removes the known idle-DP dummy row and redundant
TP copy. `decode_vector_capture.py` is diagnostic only: it returns the original
fused-MoE result and stores unit-weight selected-branch outputs. It is not an
optimization runtime and its wall time is never used as latency evidence.

Run the bounded discovery capture with burner cleanup/restoration:

```bash
bash poc_mllm_vision_conditioned_decode_reduction/scripts/run_bounded_capture.sh
```

Then run the offline analyses and tests:

```bash
PYTHONPATH=. python -m poc_mllm_vision_conditioned_decode_reduction.scripts.track_a_analyze \
  --capture-dir poc_mllm_vision_conditioned_decode_reduction/artifacts/vector_capture_discovery_v1/raw \
  --artifact poc_mllm_vision_conditioned_decode_reduction/artifacts/track_a/oracle_headroom.json \
  --report poc_mllm_vision_conditioned_decode_reduction/reports/track_a_oracle_headroom.md

PYTHONPATH=. python -m poc_mllm_vision_conditioned_decode_reduction.scripts.track_b_analyze \
  --compact-routes poc_mllm_vision_conditioned_decode_reduction/artifacts/condition_a_tp0_routes.jsonl.gz \
  --pool poc_mllm_vision_aware_decode_coscheduling/artifacts/frozen_request_pool.json \
  --output-root poc_mllm_vision_conditioned_decode_reduction \
  --vector-capture poc_mllm_vision_conditioned_decode_reduction/artifacts/vector_capture_discovery_v1

PYTHONPATH=. pytest -q poc_mllm_vision_conditioned_decode_reduction/tests
```

EP8 outputs are structural route replays only. No EP8 latency, TPOT, E2E, or
speedup is estimated.
