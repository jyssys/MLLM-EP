# Phase B: objective response and matched controls

This document separates score geometry from executed serving results.
No score improvement receives credit as measured serving speedup.
The original parent NO_GO_INCREMENTAL verdict remains unchanged.

## Offline survey

The survey evaluates O0–O13 at eight lambda settings each.
Some families ignore lambda. The 112 settings are not 112 independent algorithms.
The retained historical TRAIN data contain1,724 fresh selector states.
Only 360 states have a budget-feasible action. Of those,132 allow multiple actual C values.
The 1,364 infeasible states remain visible in the output tables.
These states come from earlier development traffic, not a fresh blind test.

The score uses normalized efficiency E and exposure-risk R.
E pays estimated remaining Full-R4 service for the same chunk.
R combines planned quantum exposure, observed delivered-token age, and current decode occupancy.
R does not estimate a calibrated future violation probability.
No proposed policy receives a complete-token deadline guarantee.

| Feature difference | Rows | Positive E difference | Positive R difference | Mean E difference | Mean R difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Increase C at fixed G |324 |324 |272 |0.82205 |0.15699 |
| Increase G at fixed C |768 |684 |696 |0.03967 |0.36355 |

Evidence: OFFLINE_ESTIMATE on frozen TRAIN feature surfaces.
Rows are repeated state/action differences, not independent requests or GPU interventions.
E and R use different frozen normalization scales.
Their raw magnitudes cannot establish a physical separability law.
Increasing C tends to improve the implemented efficiency score, but also increases modeled exposure.
Increasing G often improves efficiency and often increases exposure.
The intended axis bias is therefore coupled, not exclusive.

There are 86 legal four-corner C×G interaction comparisons.
All four cost shapes have empirical support in 76/86 comparisons.
Mean E/R interaction differences are 0.03061/0.17620 at P2033 and 0.05976/0.19843 at P8177.
These are model-feature interactions. They are not new latency measurements.
The constructed-state audit adds8,892 explicitly simulated rows across budgets20/40/80/120/240ms and decode-pressure bins.
It does not add live evidence or change finalist selection.

## Endpoint ablations on the same360 states

| Family / frozen weight | C differs from E-only | G differs from E-only | G agrees with R-only |
| --- | ---: | ---: | ---: |
| O0 /1 |0/360 |0/360 |288/360 |
| O2 /8 |52/360 |72/360 |360/360 |
| O4 /1 |44/360 |72/360 |360/360 |
| O6 /2 |0/360 |72/360 |360/360 |
| O10 /8 |57/360 |72/360 |360/360 |

Endpoint agreement describes these states only.
It does not show that the risk-only action satisfies an actual output-token SLO.
O6 changes G without changing E-only C on this historical sample.
O2 and O10 produce more C variation under larger risk weights.
The mechanism therefore depends on objective structure and pressure, not merely the presence of two variables.

## Fresh validation and frozen selection

The live screen contains 22 arms,352 cohorts, and 2,112 offered requests.
There are four independent model-process starts per configured width4/8.
Each process runs two source cohorts and every policy on the same GPU.
All matched adaptive arms share budget 120ms, completion-aware fairness, cost information, and one-prefill-span packing.
Native Mixed can co-pack rows and remains a separately labeled strong serving baseline.

| Family | Fresh C difference versus fixed-C2048 | Worst-stratum Joint / stronger G-only goodput |
| --- | ---: | ---: |
| O2 /8 |23.457% |0.957× |
| O4 /1 |3.125% |0.999× |
| O6 /2 |3.125% |1.052× |
| O10 /8 |23.457% |0.932× |

These are measured validation point estimates, not qualified capacity or accuracy-matched speedups.
Some natural outputs and cap counts differ across policies.
The same-objective G1024 control supplements G2048 for every family.
O6 maximizes the prespecified worst-stratum validation criterion.
Its3.125% C-change rate does not pass the registered 5% diversity screen.
The provisional promotion flag is therefore false despite the point goodput ratio.
FINAL characterization still tests transfer without retuning the objective.

Fixed controls were selected from 24 fresh TRAIN points.
The best tested fixed point is staged G4/C512. The best tested Mixed C is 512.
These are finite tested optima, not globally optimal schedulers.
O3 risk cap0.5 and O5 efficiency floor0.5 are prospective constants.
The supplementary offline audit evaluates O3/O5 thresholds at TRAIN feature percentiles10/25/50/75/90.
It retains the original first-captured-state denominator and does not change the online shortlist.
This is not an exhaustive threshold search or a prospective online constraint-family test.

## Mechanism controls and limits

The no-C-change selector replay agrees with G-only's G choice on 2,492 same-C validation states.
That is a deterministic selection-identity check, not a counterfactual live outcome.
Eight forced C alternatives start from exact matched valid KV/hidden state.
Smaller C improves background P99 in 6/8 pairs but improves target TTFT in only 1/8.
The controlled observations support a trade-off, not an adaptive Joint win.

The current online risk model does not estimate unresolved decode/coda service.
A full SLOWeave-style complete-next-token deadline baseline is NOT_IMPLEMENTED.
P-PAS pressure-budget control is NOT_IMPLEMENTED in this fixed-budget isolation study.
Neither missing baseline justifies a novel-method claim against those papers.

## Data links

- `objective_grid.csv`, `objective_action_transitions.csv.gz`, and `objective_gradients.csv`.
- `objective_interactions.csv` and `objective_endpoint_diagnostics.csv`.
- `constructed_state_sensitivity.csv.gz` and `sensitivity_scope.json`.
- `constraint_threshold_sensitivity.csv` and `constraint_threshold_scope.json`.
- `objective_shortlist.json`, `main_table.csv`, and `paired_comparisons.csv`.
- `no_C_change_identity.csv` and `confirmed_alternative_states.md`.

The final workload and quality results belong in `analysis.md` after the frozen queues complete.
