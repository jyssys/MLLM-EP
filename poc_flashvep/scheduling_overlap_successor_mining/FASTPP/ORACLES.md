# What the available FastPP oracles do and do not establish

## Native existing-policy lower envelope

Dense Qwen2.5-32B PP4: PP-only, greedy, ALP, ALP+rebalancing, three independent
paired restarts. Best-static greedy to per-workload median-cost selection:
0.247% request-mix-weighted / 0.642% equal-workload mean request E2E improvement.

Native Qwen3-30B-A3B PP4: PP-only, greedy, ALP, three independent paired restarts,
7,200 measured requests. Corresponding envelope: 0.587% / 2.351%. Leave-one-block-
out configuration selection yields approximately +0.646%, -0.750%, 0%, not a
stable material positive. The in-sample oracle must not hide selection noise.

On nine predeclared all-token SLOs (TTFT 1/2/5 s × TBT 50/100/200 ms), the native
MoE finite-policy envelope's largest attained goodput advantage is 12.31% at
TTFT2/TBT100. This uses successful requests / actual fixed-arrival trace elapsed
time. It is not maximum SLO-satisfying arrival-rate capacity, nor a new method.
Mean TPOT is never substituted for every-token TBT compliance.

These bound only selection among the tested existing policies. They are **not**
upper bounds on untested chunks, partitions, native PP×EP or native Qwen3-VL.
Sources: `qwen3_screen_20260909_v1/` and the earlier dense restart screens.

## Measured stage / partition diagnostics

The subsequent 15-engine common-KV/full-trace-warmup control adds chunk128,
chunk512 and chunk2048 to the greedy/ALP comparison on steady and bursty inputs.
PP-only/chunk2048 wins both median mean-E2E workloads; finite-policy additional
envelope and all three held-out selections are **0%**. The largest attained
goodput addition across the nine SLOs is **6.967%** (not capacity goodput).
All fifteen short-answer checks pass; long-continuation quality remains scoped
as a diagnostic rather than certified benchmark equivalence.

The v3 native mechanism run has 3,950 complete four-PP-rank invocation identities
and 127 complete 48-layer profiles. Perfect contiguous minimax partition gives
approximately 23–26% lower stage-max proxy. This is **not a request oracle**:
cost may move with rank/runtime state, stage waits overlap, admission decisions
change, and inter-stage dependencies remain. No multiplication of this number
by MoE share is reported as direct E2E gain.

The completed uninstrumented existing-static-partition control (8/12/14/14 versus
equal12) tests actual request consequences with full identical-trace warmup,
common KV capacity, randomized order and three restarts. The measured-cost
partition is worse in **all three pairs**: median E2E reduction -14.91% steady
and -11.98% bursty. Restart-bootstrap 95% median-effect intervals are
[-17.54,-6.93]% and [-16.42,-11.74]%, respectively. These small-n intervals remain
descriptive. This is a negative causal intervention against the naive proxy,
not a proof that no other partition or coupled execution can help.

## Exact chunk and joint PP–EP limits

The current ALP itself walks all candidate chunks until its modeled SLO/capacity
constraint is met; replaying its own predicted table is not an independent exact
execution oracle. A true best-chunk counterfactual requires new scheduler state
and request timelines, not independent minima from unrelated batches.

Native Qwen3's MoE path does not shard experts across PP ranks. Therefore a
joint PP–EP oracle cannot be claimed from these PP4 runs. Separate DeepEP4 VL
traces are labelled transfer diagnostics. An unverified native port is not a
method failure, and layer/wave improvements do not certify request benefit.
