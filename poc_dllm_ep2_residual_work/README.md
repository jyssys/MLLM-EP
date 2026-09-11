# dLLM EP2 residual-work removal PoC

Mechanism-screening study on physical GPUs 6 and 7 only.  The experiment asks
whether repeated dLLM refinement leaves material MoE/EP work after accounting
for Epoch's block plan, Liveness-Shard Parallelism, decoded-output cache, and
FreshLane dispatch.

Positive results are capped at `HOLD-FOR-EP4`; latency, payload, fanout, and
backend effects require a later true-EP4 validation.

## Outcome

`EP2 NO-GO`: after excluding Epoch's stable-decoded FreshLane opportunity,
the largest independent perfect request oracle is complete router elimination
at 2.69%, below the 5% kill gate.

Reports are in `reports/`.  Reproducible capture and analysis programs are in
`scripts/`, and calculation sanity tests are in `tests/`.

## Reproduce CPU analysis

```bash
/home/esjung/anaconda3/envs/dinfer-ep-poc/bin/python \
  poc_dllm_ep2_residual_work/scripts/analyze_temporal_work.py \
  --result-root poc_dllm_ep2_residual_work/results/ep2_residual_20260911_191336 \
  --prior-scaling poc_dllm_ep2_policy/results/ep2_20260911_170000/analysis/scaling_by_physical_m.csv

/home/esjung/anaconda3/envs/dinfer-ep-poc/bin/python \
  poc_dllm_ep2_residual_work/scripts/finalize_analysis.py \
  --result-root poc_dllm_ep2_residual_work/results/ep2_residual_20260911_191336 \
  --prior-scaling poc_dllm_ep2_policy/results/ep2_20260911_170000/analysis/scaling_by_physical_m.csv
```

GPU capture programs fail closed unless `CUDA_VISIBLE_DEVICES=6,7` and exactly
two CUDA devices are visible.
