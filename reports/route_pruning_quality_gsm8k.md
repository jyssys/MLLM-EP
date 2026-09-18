# Route-pruning quality: GSM8K Stage A

## Verdict

The 1% P4 run was completed for all 32 paired requests, but it failed the
pre-registered gate: accuracy changed from 31/32 to 29/32 and two
baseline-correct requests (IDs 15 and 21) became wrong. Request 21 also rose
from 44 to 108 refinement forwards (2.45x). All requests terminated by EOS and
had zero remaining masks, so this is a semantic/trajectory failure rather than
a malformed-generation failure.

P2 met the permissive 32-sample gate exactly (one additional wrong), but this
is only a provisional utility-only point: one sample equals 3.125 percentage
points and no GSM8K-128 confirmation was allowed after P4 failed.

| Method | Correct | Accuracy | Additional wrong | Parsed identity | Mean NFE | NFE delta | Actual mass | AvgK | Route reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 31/32 | 96.875% | 0 | 100% | 98.250 | 0% | 0% | 8.000 | 0% |
| P2 utility-only 1% | 30/32 | 93.750% | 1 | 96.875% | 83.281 | -15.235% | 0.998% | 7.7529 | 3.088% |
| P4 EP-aware 1% | 29/32 | 90.625% | 2 | 93.750% | 84.125 | -14.377% | 0.999% | 7.8012 | 2.486% |

The paired exact p-values are 1.000 (P2) and
0.500 (P4). They are not evidence of equivalence;
the explicit additional-wrong gate, not an underpowered p-value, controls
promotion. Exact generation identity was only 3.125%
for P2 and 3.125% for P4, showing that even
1% route mass omission perturbs most trajectories.

## Protocol and scope

- Actual BF16 LLaDA2.0-mini rollouts, pinned revision, threshold 0.95,
  block length 32, max 32 refinements/block, no weight renormalization, min-k 4.
- Vanilla reused the already-validated identical GSM8K IDs 0--31 rollout;
  a smoke parity run matched its token sequence exactly before pruning runs.
- P2/P4 were independently rolled out; baseline routes were not edited offline.
- Dense semantic emulation computes all branches and zeroes omitted branch
  coefficients. Its wall time is not a sparse-runtime speedup.

## Stop decision

The 2.5% and 5% budgets, GSM8K-128, and HumanEval were not run. This follows
the specification's stop rule for clear P4 regression already at 1%.
