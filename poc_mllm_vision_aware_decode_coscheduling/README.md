# Vision-aware decode co-scheduling bounded PoC

This isolated PoC evaluates Stage-A working-set headroom before any runtime scheduler is built. It reuses the frozen Qwen3-VL TRUE-EP4 long-reasoning traces and changes no token, route, expert, weight, precision, or model computation.

Authoritative specification: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/MLLM_Vision_Aware_Decode_CoScheduling_TRUE_EP4_PoC.md`.

The primary pool contains every previously frozen, normally terminated response with at least 128 natural tokens. The shortage relative to the recommended 24 requests is reported rather than relaxed. S2/S3/S4 use the same exact partitioner and top-32 per-layer signature; only the prefill feature differs. S5 sees future decode traces and is an offline oracle.

See `artifacts/reproducible_commands.sh` and `reports/final_summary.md`.

Run from the repository root:

```bash
PYTHONPATH=. python -m poc_mllm_vision_aware_decode_coscheduling.scripts.run_stage_a \
  --cohort poc_mllm_vision_aware_decode_coscheduling/artifacts/frozen_request_pool.json \
  --compact-trace poc_mllm_vision_aware_decode_coscheduling/artifacts/source_pool_trace.jsonl.gz \
  --prior-analysis poc_mllm_vision_aware_decode_coscheduling/artifacts/predictor_contract.json \
  --expert-contract poc_mllm_vision_aware_decode_coscheduling/artifacts/expert_tensor_contract.json \
  --output poc_mllm_vision_aware_decode_coscheduling/artifacts

PYTHONPATH=. python -m poc_mllm_vision_aware_decode_coscheduling.scripts.render_reports \
  --root poc_mllm_vision_aware_decode_coscheduling \
  --baseline-audit poc_mllm_vision_aware_decode_coscheduling/artifacts/baseline_audit_reused.json \
  --source-spec /home/esjung/MLLM-EP-github/poc_flashvep/reports/MLLM_Vision_Aware_Decode_CoScheduling_TRUE_EP4_PoC.md

PYTHONPATH=. python -m poc_mllm_vision_aware_decode_coscheduling.scripts.verify_artifacts \
  --root poc_mllm_vision_aware_decode_coscheduling
```

The frozen Stage-A gate failed, so this PoC intentionally contains no Stage-B runtime
scheduler. `artifacts/clean_timing.json` explicitly records the unexecuted state.
