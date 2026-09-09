# Native existing partition attack — completed 2026-09-09 12:35 KST

Three independent paired restarts, randomized equal versus cost-static order,
same arrivals and 48-request steady/bursty traces, same fixed 262,144-token KV
capacity, identical short warmup plus one full workload warmup immediately before
measurement. Greedy dynamic chunks in both arms. No mechanism hooks. Native
Qwen3-30B-A3B BF16 PP4 on physical 4–7, **not EP4**. All short-answer sanity passes.

Equal 12/12/12/12 versus modal separate-profile partition 8/12/14/14:

| Request metric, restart median | Equal: steady | Cost-static: steady | Equal: bursty | Cost-static: bursty |
|---|---:|---:|---:|---:|
| Mean E2E, s | 7.1812 | 8.3967 | 7.0874 | 8.0690 |
| Mean TTFT, s | 0.4315 | 0.5154 | 0.5099 | 0.5522 |
| Mean TPOT, s | 0.04568 | 0.05351 | 0.04482 | 0.05129 |
| Tokens/s, fixed trace | 209.35 | 200.94 | 228.00 | 215.85 |

The median *paired* E2E reduction is **-14.91% steady / -11.98% bursty**.
All three pairs worsen: steady -17.54%..-6.93%, bursty -16.42%..-11.74%.
Ratios of separate medians above need not equal the median paired ratio.

This falsifies using the 23–26% instrumented layer-partition makespan proxy as
a direct request-level saving in this control. It does not establish a universal
optimal partition or localize the exact reason for cost non-portability. Rank0
was critical in 3,885/3,950 instrumented invocations, and instrumentation/state,
rank placement, and changed scheduler dynamics are not isolated by that proxy.
Do not promote the negative partition result itself as a new MoE phenomenon.

Existing same-capacity equal partition is the obvious winning static control.
No new scheduler or successor implementation is justified by this result.
Long free continuations are stored with same-config restart diagnostics; exact
text hashes are not used as a substitute for a full quality benchmark.

Artifacts: `fastpp_runs/static_partition_control_20260909_v2/` includes all six
run logs, warmup and request timelines, paired summaries and correctness checks.
The earlier v1 had an incorrect local snapshot argument, failed during config
loading, and is excluded. A prelaunch local-config existence check now prevents
this harness mistake. No weights or model semantics were changed.
