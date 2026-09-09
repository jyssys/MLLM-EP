# Native finite-plan envelope, not paper-optimal regret

Completed clean pure-prefill portfolio: plain/split2/split4, M4096 and M8192,
three independent engines per cell, 72 measured requests. All twelve paired
comparisons have exact first-token agreement. Best static and per-workload
selection both choose plain: **0% additional request E2E**. Within these options,
a small portfolio or per-regime selection cannot improve the median costs.

The decode-volume portfolio completed **36 engines / 3,888 requests**, B4/B64/
B256 with 16 output tokens. Only two of 27 whole-cohort paired comparisons are
token-exact; no alternative plan qualifies across all three restarts/workloads.
Its finite additional envelope is therefore **null / INSUFFICIENT**, not zero.
This does not prove the generated answers are semantically wrong; the prior
same-prefix/HF tests find near-tie sensitivity, but do not certify this full pool.

First-decode ITL accounts for median 58–94% of these short-cohort E2Es across
conditions; that interval includes setup/capture and required work. It is not
all removable runtime waste. The same-input 96-output-token control completed
18 engines / 612 requests. Only 1/12 cohort pairs is exact; its envelope also
remains null. First-decode share falls to about 30–71%, without subtracting any
work. Steady ITL cannot substitute for actual request completion or warmed
continuous-serving performance. See DECODE_PORTFOLIO.md.

`analyze_plan_envelope.py` requires three output-exact paired repetitions for
each workload before a plan is eligible. If fewer than two plans qualify, the
answer is **INSUFFICIENT**, not zero. No missing/failed cell is imputed.

Original searched-plan, dynamic-shape MLLM and arbitrary operation/SM allocation
request oracles are **NOT ESTABLISHED**. Missing native MoE profiling hooks
prevent faithful optimized-search reproduction without a further port. An empty
profile hook must not supply zero-cost experts. No feasible successor headroom,
Kimi promotion or new-method gain is claimed from this limited portfolio.
