# Native large-prefill portfolio — 18 fresh engines

`prefill_portfolio_20260909_v1`: plain / FFN split2 / FFN split4, M4096 and
M8192, three randomized independent restarts each. Native Qwen1.5-MoE FP16 EP4
on physical 4–7. Four real-content requests per cohort, one generated token;
request TTFT equals completion latency. Two exact-shape prefill cohorts warm
each engine. Eager path, so no decode graph-capture cost enters these results;
prefill plan construction happens during warmup, not the measured request.

All twelve paired comparisons agree on all four generated tokens. This is a
first-token sanity check on these inputs, not benchmark accuracy certification.

| Aggregate prompt tokens | Plain median E2E | Split2 median E2E | Split4 median E2E |
|---|---:|---:|---:|
| 4096 | 50.738 ms | 60.636 ms | 99.093 ms |
| 8192 | 96.784 ms | 97.357 ms | 108.269 ms |

Median paired E2E reduction versus plain: split2 **-17.46% / -0.58%** at
M4096/M8192; split4 **-90.66% / -11.96%**. One M8192 split4 run is much slower;
all runs are retained. No positive effect is claimed from a favorable sample.

Best-static and per-workload finite-plan oracle both choose **plain**: additional
E2E envelope **0%**. The small-portfolio oracle cannot do better within these
three options. This is **not** the original paper's searched optimum, a full
resource-allocation search, or continuous-batching performance. In particular,
the native H100 Qwen MoE profile hooks are incomplete and prompt-wide final-head
materialization is present. These constraints prevent concluding that the paper
method fails on MLLMs. A bounded larger decode-cohort check is prepared next.
