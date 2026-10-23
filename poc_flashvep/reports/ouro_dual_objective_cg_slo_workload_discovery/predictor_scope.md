# Cost and risk scope

The native path lookup remains the frozen historical v1 model.
The study does not retrain it on the previous HELDOUT errors.
Old TRAIN service records determine normalization and conditional service-tail factors.
New input families remain separate for validation and final traffic.

A query estimates one single-span native service unit.
The full-chunk score pays prelude and all remaining recurrent groups.
The model does not know later co-batching, future arrivals, or future output lengths.
Unknown actual shapes retain explicit support flags and enclosing/inflated estimates.

The risk feature is a causal exposure proxy.
It combines predicted planned-unit duration with already observed output-gap age and active decode state.
It does not claim a conditional future-ITL probability or a complete-token latency bound.
O7 uses empirical TRAIN service tails only.
No old D1 deadline model enters the new selector.

Past output timestamps come from completed native `engine.step` outputs.
Issued loop frontiers are not treated as physically completed output tokens.
CUDA events and native runner dependencies remain unchanged.
Poor risk calibration limits safety claims. It does not automatically block labeled exploratory comparisons.

## New measured audit

The separate CALIBRATION event audit contains 770 matched selected units.
The largest observed host/device alignment bound is 0.1412ms.
The timer starts after native dependency waits and includes preparation and host-submission gaps.
It is not an isolated GEMM timer or a client-visible complete-token predictor.
The recurrent MAE at B120 is 6.733ms for Joint,7.098ms for G-only, and 7.105ms for C-only.
There are no B120 budget overruns in those 55/51/38 selected recurrent audit units.
The small sample does not establish a future upper bound or universal safety.

At B40, measured recurrent overruns are 3/64 for Joint and 2/60 for G-only.
C-only uses the explicit infeasible-budget progress fallback and overruns201/201 recurrent units.
An advantage over that infeasible control does not establish incremental C value.
The primary live comparison remains B120 with matched information and fairness.

Cost-shape support is empirical in 498/770 audit units.
The rest use enclosing/inflated estimates.
Graph readiness, actual decode interference, and mixed packing are not fully conditioned in this inherited lookup.
The constructed sensitivity grid cannot repair those missing physical measurements.
No previous HELDOUT residual or new FINAL outcome refits the frozen model.

Data: `event_budget_residuals.csv`, `optimization_overhead.md`, and `state_support.md`.
