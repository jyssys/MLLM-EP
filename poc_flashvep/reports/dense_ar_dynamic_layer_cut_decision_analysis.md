# Dynamic layer selection: audit and evaluation protocol

This report belongs to the decode-safe elastic-layer campaign.
The evidence directory is `dense_ar_decode_safe_elastic_layer_pd_20261113/`.
Fresh confirmation results appear below and in the main summary.

## Previous selection mechanism

`TRACE_ANALYSIS`: five previously viewed COST starts contained 174 group decisions.
The policy selected Last0 in 100 decisions and Last12 in 74 decisions.
Last8 and Last12 each passed admission in 74 decisions.
The final ranking preferred the largest predicted P saving.
That ranking selected Last12 instead of Last8 in every shared feasible case.

Last2 and Last4 each failed `NO_COMPLETION_BENEFIT` 145 times.
The audit found no previous case where Last12 failed admission and a smaller cut passed.
Thus, the old logs do not prove that the old policy discarded an admitted smaller cut after rejecting Last12.

See [the candidate audit](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/previous_candidate_audit.json).

## Revised selection

The revised policy evaluates every allowed cut before P submits its prefix.
The candidate set remains Last0/2/4/8/12.
The policy uses the observed P backlog, known D work, KV credits, and measured cost tables.
It does not read future arrivals or held-out completion times.

The score combines a predicted request benefit and a bounded backlog benefit.
It divides that benefit by predicted D occupancy.
The policy does not maximize the number of migrated layers.
It retains Native execution when no candidate passes the gates.

Admission budget and execution budget remain separate fields.
The runtime also retains a hard ingress guard.
It does not admit a new suffix while an existing Decode cohort remains active.
This guard protects existing Decode work without assuming kernel preemption.
It can also reduce concurrency and delay continuation completion.
The earliest-token-deadline field is observational in Revised selection.
No Revised rejection predicate directly compares that field.
Do not describe the current heuristic as a complete per-token deadline optimizer.

## Serialized handoff correction

`STRUCTURAL_ACCOUNTING`: revision1 allowed handoff to overlap an existing Decode cohort in its prediction.
The runtime did not support that overlap.
It submitted the destination request only after the cohort retired.

Revision2 uses:

`D_start = max(P_cut_ready, D_available) + handoff_cost`

The regression fixture omitted 145.150 ms with the old formula.
That number describes the fixture, not measured CUDA execution.
The red test and the earlier source remain preserved.

See [the regression evidence](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/serial_handoff_red_reproduction.json).

## Destination geometry correction

The previous source token budget was 32000 tokens.
The destination inherited that budget when no separate value existed.
At 8000 tokens per request, suffix batches could contain at most four requests.
Increasing `D_admission_batch` alone did not remove this limit.

The new discovery separates the two budgets.
It evaluates destination batches8/16 with row budgets64000/128000.
P remains batch4 with its original 32000-token budget.
Both Static and Dynamic receive the same legal destination options.
Every constructed scheduler records its actual token budget and sequence limit.

## Interpretation rules

Actual completed request records determine migrated work.
An accepted reservation is not a completed migration.
Forced-cut qualification does not count as an online dynamic decision.
Matched-work caps do not establish a pure time-budget fallback.

The decision audit distinguishes these cases:

- Last12 fails the observed-state time gate and a smaller cut passes.
- An experimental layer-work cap excludes Last12 or Last8.
- All candidates fail, so the request retains Native execution.

Last2/4 can remain uneconomic even when their execution costs are small.
Their hidden-state and control costs do not decrease in proportion to the layer count.
The final report will retain zero observed selections when that outcome occurs.
It will not force small cuts to create a favorable distribution.

## Limits

The budget is a Host-state prediction.
It is not a measured physical idle interval.
The empirical margins are not WCET bounds.
Unknown future arrivals can invalidate a predicted opportunity.
The final decision requires independent-start latency intervals and matched-budget controls.

## Actual fresh decisions

`TRACE_ANALYSIS`: the frozen Revised policy selected Last0/8/12 on fresh online requests.
It did not select Last2 or Last4 in the completed Output16/64 confirmation cells.
The forced Last2/4 state qualification remains separate evidence.
Zero online selections do not mean those forced cuts failed correctness.

At Output16/QPS4.8, the three N128 starts completed the following work.

| Arrival seed | Native requests | Last8 requests | Last12 requests | Completed migrated layers |
|---|---:|---:|---:|---:|
| 2026111381 | 20 | 100 | 8 | 896 |
| 2026111382 | 24 | 92 | 12 | 880 |
| 2026111383 | 32 | 88 | 8 | 800 |
| Total | 76 | 280 | 28 | 2576 |

Migration completed for 308/384 requests, or80.21%.
Migrated work was 2576/13824 layer visits, or18.63%.
The work contained no skipped or repeated prefix/suffix layer visits in the qualified execution path.

Four committed groups selected Last8 after all larger cuts failed predicted time gates.
Those groups did not use an experimental migration cap to reject Last12.
They demonstrate actual state-aware fallback from Last12 to Last8.
They do not demonstrate Last8-to-Last4 or Last4-to-Last2 fallback.

## A concrete smaller-cut decision

Seed2026111381 contains a four-request decision at Host time9266446.896325167.
The observed P backlog was 8.
Known pending D requests numbered 4.
No Decode request was active at the observation.
The state age was 16.800 ms.

Last2 failed the aggregate completion-benefit gate.
Last4 passed admission but received a lower efficiency score than Last8.
Last12 failed the bounded-admission-wait gate.
The policy selected and committed Last8.
These are recorded predictions and actions, not reconstructed physical idle intervals.

The policy can admit a request whose own predicted completion is slower than Native.
It requires positive aggregate benefit after a bounded P-backlog benefit term.
Thus, it does not satisfy a stronger requirement that every migrated request individually finishes earlier.
This distinction is an implementation limitation, not a measured structural impossibility.

## Low-load and longer-output behavior

At Output16/QPS2.5, Revised migrated0/384 requests.
The lack of useful P backlog favored Native execution.
At Output16/QPS6.4, Revised completed 344/384 migrated requests and 2816 migrated layers.
The completed distribution was 40 Native,328 Last8, and 16 Last12.

Revised migrated0/384 requests at each tested Output64 QPS2.5/4.8/6.4.
Any latency difference there is not a Layer Offloading gain.
Those differences expose controller and independent-start variation.
The final Output200 distribution appears in the sweep evidence.

See [actual decisions](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/actual_cut_decision_audit.json).
See [matched controls](dense_ar_static_dynamic_matched_migration_comparison.md).
