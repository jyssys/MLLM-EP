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
