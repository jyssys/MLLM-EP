# Track A3/A4 — compression and economic oracle

## Reconstruction and local codec cost

| Codec | Wire ratio vs BF16 | Hidden reconstruction rel-L2 | Unfused sender+receiver p50, 32--1024 rows |
|---|---:|---:|---:|
| FP8 delta | 0.50 | 0.00271 | 0.037--0.047 ms |
| Row-scaled INT8 delta | about 0.50 | 0.00062 | 0.124--0.143 ms |

INT4 was not attempted because FP8 already fails the economic gate.  Sparse
delta was rejected by the measured density/entropy rather than assumed useful.

## Request oracle

The measured DeepEP payload surface, not observer-contaminated dispatch events,
is used.  Five traced layers are extrapolated to all routed layers and divided
by the median clean request BCT.  The full calibrated dispatch share is only
about 4.72% (GSM8K) and 4.55% (HumanEval); only its payload-sensitive portion
can be removed.

| Task | K | Effective hit | Raw dispatch bytes saved | Gross E2E oracle | Feasible E2E |
|---|---:|---:|---:|---:|---:|
| GSM8K | 2 | 42.33% | 21.16% | 0.50% | 0% |
| GSM8K | 16 | 79.36% | 39.68% | **0.72%** | 0% |
| HumanEval | 2 | 40.51% | 20.26% | 0.23% | 0% |
| HumanEval | 16 | 75.96% | 37.98% | **0.37%** | 0% |

Even an ideal zero-cost codec is below the 5% kill gate.  The measured unfused
codec costs are larger than the gross saving, so the feasible oracle clips to
zero.  A future fused codec cannot turn a sub-1% perfect payload oracle into an
8% candidate.

Decision: **KILL / Track A NO-GO**.
