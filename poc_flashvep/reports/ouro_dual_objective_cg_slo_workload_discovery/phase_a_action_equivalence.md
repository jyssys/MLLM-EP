# Phase A: why the previous plans matched

Status: PASS for the captured selection decisions. Rejected admission opportunities remain only partially observed.
This diagnosis preserves the previous NO_GO_INCREMENTAL verdict.
It does not measure any new policy speedup.

## Reproduction

The parent commit is `821b11653695d0a470bd88bfcc997ae1bbcf5bc9`.
The analysis checks committed input hashes and every quiet attempt's completed-artifact hashes.
It reproduces the frozen selector's complete feasible candidate lists and selected actions exactly.
Ordered native scheduling descriptors match on 144/144 G-only/Joint request pairs.
All 144 output token sequences also match.
The descriptor comparison preserves per-request order. It does not sort away boundary differences.
The quiet lane has no heavy per-dispatch GPU observer, so these descriptors are not independent device-span measurements.

| Prompt stratum | B, ms | Selector calls | Fresh chunk | Multiple legal actual C | Multiple budget-feasible C | C differs from G-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2033/N4 | 40 | 120 | 24 | 8 | 8 | 0 |
| P2033/N4 | 120 | 54 | 24 | 8 | 8 | 0 |
| P8177/N8 | 40 | 330 | 66 | 30 | 30 | 0 |
| P8177/N8 | 120 | 156 | 66 | 30 | 30 | 0 |
| Total | — | 660 | 180 | 76 | 76 | 0 |

The call denominator is 660. It is not the number of unique requests or independent statistical samples.
Continuation calls account for 480/660, or 72.73%.
They cannot legally resize an active chunk.
Another 104 fresh calls collapse to one actual span after prompt clipping.
Multiple actual C choices remain in 76/180 fresh calls, or 42.22%.
Those calls are 11.52% of all captured selector calls.
All 76 retain multiple budget-feasible C choices.

## Mechanism

The matched G-only policy fixes configured C2048.
At every captured fresh state with multiple feasible C choices, U2 selects the same actual C as that control.
The objective maximizes eventual full-R4 chunk tokens divided by estimated remaining chunk service.
It does not directly charge current decode-gap exposure or a measured complete-token SLO risk.
Its extra yield penalty is zero in these runs.
The largest efficiency-favoring chunk therefore reproduces the fixed-C control in these states.
This is a measured selector-replay result, not a universal law for other states or utilities.

Budget filtering removes some G alternatives in 12 fresh calls with multiple C choices.
It does not eliminate their multiple-C opportunity.
The exclusive cause table therefore does not mislabel those calls as C-infeasible.
No overbudget fallback or cross-C utility tie explains the 76 multiple-C calls.
The shared one-prefill-span rule limits packing for every adaptive arm.
This dataset does not independently establish packing as the cause of C nonselection.

## Limits and next test

The selected shape is outside the lookup's exact support in 420/660 captured calls.
Most clipped short or remainder spans therefore use explicit enclosing-shape estimates.
That limits claims about physical optimality. It does not invalidate exact reproduction of the implemented selector.
Rejected admission checks do not have exhaustive historical state snapshots.
The funnel covers actual recorded selector invocations, not every scheduler poll.
The next study must log those two denominators separately.

The new experiment will test efficiency-only, risk-only, and coupled utilities on the same legal candidate surface.
Alternative scores are OFFLINE_ESTIMATE until the corresponding action executes on a GPU.
No objective will receive credit for unfinished R1/R2 prompt tokens.

Data: `prior_plan_reproduction.csv`, `decision_opportunity_funnel.csv`, `eligibility_collapse.csv`, and `candidate_score_audit.csv.gz`.
`phase_a_provenance.json` records parent hashes and missing fields.
