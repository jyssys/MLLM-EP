# F1 generalization on GSM8K

## Verdict

**FAIL at the mandatory n=32 gate.** The fixed F1 policy (`P2 previous-confidence-high`, active ratio 0.75, max freeze age 1) completed all 32 requests without a mask leak, but changed two baseline-correct answers to wrong answers. The contract allows at most one additional wrong answer. Consequently GSM8K-128 was not run.

All generation and quality values below are `MEASURED_QUALITY` from the same checkpoint, prompt, request IDs, threshold 0.95, block length 32, generation cap 2048, and deterministic seed protocol. F0/F1 execute dense semantic emulation; their wall time is not a sparse-runtime result.

| Method | Correct | Accuracy | baseline correct→wrong | baseline wrong→correct | paired exact p | Mean NFE | NFE reduction | W_active | W_active reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 31/32 | 96.875% | — | — | — | 98.250 | — | 49,906 | — |
| F0 P2-75%-age1 | 31/32 | 96.875% | 0 | 0 | 1.000 | 92.938 | 5.41% | 36,709 | 26.44% |
| F1 P2-75%-age1 | 29/32 | 90.625% | 2 | 0 | 0.500 | 95.719 | 2.58% | 38,261 | 23.33% |

The paired p-value does not establish a population-level quality difference at n=32. It also does not rescue the pre-registered safety gate: IDs 15 and 21 were correct under Vanilla and incorrect under F1, so `additional wrong <= 1` is false. There was no benchmark retuning after observing this result.

## Failure inspection

- ID 15: baseline parsed 125; F1 parsed 5000 and produced internally inconsistent arithmetic/conclusion.
- ID 21: baseline parsed 14; F1 parsed 2 after changing the age reasoning trajectory.
- Termination: 32/32 EOS, zero remaining masks.
- Parsed-answer identity with Vanilla: 30/32 (93.75%).

## Work and EP projection

The common S1 accounting removes only rows F1 actually reuses; stable decoded/prefix caching is not granted for free.

| Method | fresh expert-token pairs | reduction vs Vanilla | unique experts/invocation | EP4 routed-MoE gain | EP8 routed-MoE gain |
|---|---:|---:|---:|---:|---:|
| Vanilla | 788,011,776 | — | 221.539 | — | — |
| F0 | 723,984,512 | 8.13% | 220.975 | 5.98% | 5.90% |
| F1 | 747,224,248 | 5.18% | 221.097 | 3.13% | 3.03% |

EP4/EP8 entries are `SIMULATED-EP4-EP2-CALIBRATED` and `SIMULATED-EP8-EP2-CALIBRATED` routed-MoE-stage projections, not E2E latency. Even before applying the quality gate, F1 is dominated by F0 on this cohort: it has lower quality, less NFE/work reduction, and less projected EP gain.

## Age-1 diagnostics

| Metric | Value |
|---|---:|
| post-layer hidden cosine p50 / p10 | 0.99776 / 0.98404 |
| post-layer relative-L2 p50 / p90 | 0.06754 / 0.18063 |
| expert-route Jaccard mean | 0.79995 |
| EP4 / EP8 destination-rank Jaccard | 0.93336 / 0.88971 |
| confidence-cache absolute error mean | 0.14997 |
| commit disagreement | 1.635% |

High cosine similarity therefore did not certify final trajectory safety. This is consistent with the observed answer regressions.

## Gate outcome

`F1_GSM8K32: FAIL`  
`F1_GSM8K128: NOT_RUN`

