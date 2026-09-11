# Final decision

## Independent outcomes

- **1A Predictive + overlapped transient GPU replication: NO-GO**
- **1B Iteration-aware rank-complementary batching: NO-GO**

## One-sentence result

dLLM denoising does create highly predictable expert/rank reuse, but on the
measured LLaDA-MoE EP4 path the removable physical imbalance has too little
request-critical mass: even impossible, cost-free absolute oracles are only
2.63% E2E for replication and 2.36% for batching.

## Gate table

| Gate | 1A | 1B |
|---|---:|---:|
| Temporal/diffusion-specific signal | Pass: t+1 hot retention 82.0% | Pass: next rank cosine 0.9967 |
| Required diversity/economic substrate | Weak imbalance-latency coupling | Fail: pairwise request cosine 0.9995 median |
| Costed best future-aware oracle | 0.746% projected E2E | 0.166% projected E2E |
| Causal policy | 0.459% E2E, 61.6% oracle recovery | 0.049% E2E, 29.3% recovery |
| Impossible absolute upper bound | 2.6266% E2E | 2.3557% E2E |
| Required continuation gate | ≥8–10% request gain | ≥8–10% request gain |
| Decision | **NO-GO** | **NO-GO** |

## Evidence classification

Measured facts:

- true TP1/DP4/EP4 remote dispatch/combine after the minimal dInfer patch;
- exact expert size and all-pair copy/overlap cost;
- 3,792 logical route records across 30 requests and three trace restarts;
- clean request timings across three independent restarts;
- strong temporal persistence and weak between-request diversity;
- no positive load-to-latency slope on the tested path.

Projected upper bounds:

- costed H=1/2/4/8 replica leases;
- offline complementary batching;
- proportional expert-time E2E mapping;
- zero-copy/unlimited-replica and fractional-rank absolute counterfactuals.

No live replica manager or scheduler was implemented, because both early kill
tests fail even under assumptions more favorable than any feasible method.

## Limitations that do not reverse the decision

1. The trace has 30 rather than the requested 32–64 measured requests because
   the GPUs were returned. This weakens generalization but not the absolute
   per-trace E2E bounds.
2. The vLLM `NaiveAll2AllManager` is a portable correctness/debug path, not
   DeepEP. A faster production backend could reduce communication and make the
   expert share no larger; it cannot turn a 2.63% zero-cost imbalance bound into
   8–10% on the same routes.
3. Cross-topology BF16 logits drift substantially. Consequently these artifacts
   are not a production-quality dInfer EP patch. Clean/trace outputs are stable,
   and the decision is an economic rejection rather than a quality claim.
4. Larger blocks, expert counts, EP degree, or a much more diverse request pool
   could define a different regime. That would be a new substrate study, not a
   reason to build the two proposed methods from this result.

## Recommendation

Do not continue either 1A or 1B on this four-H100 LLaDA-MoE configuration. If
dLLM MoE is revisited, prioritize mechanisms that eliminate repeated required
work across the diffusion block (the larger mass addressed by TEAM/Epoch/DES)
or first find a regime where expert execution is a dominant, positively
load-sensitive part of request latency. Do not spend engineering time on a
replica manager or complementary scheduler before that substrate gate passes.

GPU 4–7 were released before final CPU analysis, and no burn process was
restarted.
