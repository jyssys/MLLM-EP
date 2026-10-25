# Executive decision

## Verdict

**NO_GO_HEADROOM**, limited to the tested native B1 runtimes and candidate grid.
Do not implement a new expert-reuse system from these results.
The experiment does not prove that all future expert policies lack headroom.

| Model | Best tested NLL-eligible arm | Omitted work | ΔNLL [95% CI], nats/token | Optimistic timing model | Achieved E2E speedup | P2 / P3 |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | O1_drop_0.2_renorm0 | 16.84% | 0.0136 [0.0112, 0.0159] | 1.0462× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| IFM MoE Huginn | O1_drop_0.3_renorm0 | 19.06% | 0.0133 [0.0109, 0.0156] | 1.0525× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| Foil-1 | O1_drop_0.2_renorm0 | 9.16% | 0.0122 [0.0072, 0.0175] | 1.0151× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |


## Why P3 did not run

The best tested points satisfy an upper-CI NLL sensitivity target of +0.02 nats/token.
Neither IFM timing estimate reaches the 1.10× discovery threshold.
The two IFM models therefore do not meet the 1.15× GO_PROTOTYPE target.
These are favorable estimates with no selector, cache, or gather overhead.
The native-weight microbenchmark further rejects exact proportional timing for sparse row removal.

Unconditional same-expert reuse increased held-out NLL by 3.3565, 4.3506, and 1.5990 nats/token.
The order is Ouro, Huginn, then Foil-1.
Shared expert identity does not imply interchangeable expert outputs.

## Evidence and limits

All three native no-op paths and q=0 paths passed their logit checks.
All three official generation APIs matched their adapters on an eight-token smoke prompt.
The IFM FA4-versus-math whole-model check failed its original threshold.
Later strict FP32 diagnostics also changed one of eight generated tokens on another prompt.
These failures remain visible.
Comparisons within the same official FA4 path remain internally controlled.

There are 13,440 primary full-model quality records and 696 free-running sequences.
The language held-out set has 128 passages per model.
GSM8K and MATH-500 natural-generation pilots contain only eight questions each.
Weak MATH and Foil baselines cannot certify the 1 percentage point task-quality margin.
The study did not expand these pilots after the latency gate failed.

All native generation paths recompute the complete prefix without a KV cache.
The study does not describe production cached-decode serving or physical EP.
No positive loop-specific acceleration contribution is established.
