# Predictor safety decision

Phase 3 status: **BLOCKED for online deadline control**.
Aggregate blind coverage passes the provisional 95% target.
Decision-relevant prefill strata do not pass it.

| E1 held-out stratum | Quanta | Coverage of expected plus frozen margin | MAE |
| --- | --- | --- | --- |
| Atomic prefill | 1,117 | 79.141% | 25.943 ms |
| Staged prefill | 12,254 | 90.901% | 9.702 ms |
| Mixed | 2,278 | 98.990% | 9.776 ms |
| Decode recurrent | 69,413 | 99.824% | 0.545 ms |

Numerous inexpensive decode calls dominate aggregate coverage.
The aggregate hides the operations that a prefill controller must admit safely.
The miss fractions are not measured false-feasible scheduling rates.
False feasibility also depends on deadlines and selected actions.

The D1 margin is 230.204 ms.
The registered Moderate TBT limit is 271.873 ms.
That uncertainty reserve leaves little room for a prefill decision.
It cannot certify a useful adaptive policy.
It also does not prove that the physical system has no useful slack.

Native `loops_done` records issued work and can precede GPU completion.
Pending-event count alone does not prove that each event completed.
Early observer schemas lack full readiness snapshots before policy selection.
D labels include stock-policy future interruptions.
They are valid retrospective targets, not direct counterfactual completion costs.

The main cost includes queued waits.
Summing those costs through a new DAG can count one wait twice.
Reconstructing a baseline from its own endpoints would not validate counterfactual prediction.
No such reconstruction is claimed as a validated simulator.

Do not relax frozen thresholds after the blind result.
Do not tune a margin on HELDOUT errors.
Keep Phase 4 limited to labeled offline characterization.
Do not start an SLO-preserving A3 or qualified capacity evaluation.

A future registration can test path-stratified calibration on fresh source families.
It must separate queued work from service cost and validate complete-token readiness.
That is a next experiment, not a completed correction.
