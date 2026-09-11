# dLLM MoE EP Temporal PoC

Oracle-first study of two diffusion-specific EP opportunities on the official
`inclusionAI/dInfer` and `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` substrate:

- predictive, overlapped transient GPU expert replication (1A);
- iteration-aware rank-complementary request batching (1B).

All task-owned GPU launches are restricted to physical GPUs 4, 5, 6, and 7.
The attached working contract is tracked by path and digest in `SPEC.md`.

This tree intentionally contains analysis/instrumentation code only. It does
not contain a production replica manager or serving scheduler.

## Result

- 1A transient GPU expert replication: **NO-GO**.
- 1B iteration-aware rank-complementary batching: **NO-GO**.

The dLLM temporal signal is strong, but the request-level economic mass is not:
the impossible zero-cost absolute E2E ceilings are 2.63% and 2.36%. See
`reports/dllm_moe_ep_1A_1B_poc.md` for the synthesis and
`reports/final_decision.md` for evidence boundaries.

## Reproduce CPU analysis

From the repository root, with the isolated environment already installed:

```bash
PY=/home/esjung/anaconda3/envs/dinfer-ep-poc/bin/python
ROOT=poc_dllm_ep_temporal/results/dllm_moe_ep_20260911_151500

$PY poc_dllm_ep_temporal/scripts/run_oracle_analysis.py \
  --runs "$ROOT/runs" \
  --copy-csv "$ROOT/peer_copy.csv" \
  --output "$ROOT/oracle_summary.json"

$PY poc_dllm_ep_temporal/scripts/make_characterization.py \
  --trace "$ROOT/runs/trace_1/rank0_trace.jsonl" \
  --copy "$ROOT/peer_copy.csv" \
  --oracle "$ROOT/oracle_summary.json" \
  --output-dir "$ROOT/analysis"

PYTHONPATH=$PWD $PY -m pytest -q poc_dllm_ep_temporal/tests
```

GPU scripts hard-fail unless `CUDA_VISIBLE_DEVICES` is exactly `4,5,6,7`.
They are retained for reproducibility, but no GPU rerun is needed for the
final NO-GO decision.
