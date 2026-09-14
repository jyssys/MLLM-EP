# Mixed refinement stream

`REFINEMENT_STREAMS.jsonl` contains 146 chronological layer-16 workload states
from measured true-EP4 LLaDA2 GSM8K and HumanEval trajectories. Expert IDs and
source ownership are measured; the fresh-row filter is a future-known
Epoch/FreshLane sensitivity, not a measured Epoch implementation.

GSM8K spans M=1--1024 and HumanEval M=1--802. Multiple requests are offset in
refinement age so that large, medium, and tiny shapes are ready together. This
preserves each selected route's M, expert identities, rank load, fanout, and
remote assignment geometry.

Mixed-M did not expose a reversal. At Q=16, LL two-slot delivered 11,705 waves/s
versus 4,268 for Normal. Its advantage remains between the homogeneous-small
and homogeneous-large controls; mixedness did not introduce a separate backend
pathology. Figures 08--09 visualize the source stream and age offsets.
