# Existing-control attack

Completed: PP-only/greedy/ALP/rebalancing dense screen, P2P transport control,
Qwen3 PP-only/greedy/ALP screen, and a native existing static-partition pair.
Also completed: equal-capacity, full-identical-workload warmup with PP chunks
128/512/2048, greedy and ALP, three randomized restarts each. Existing PP-only
is best for both median mean-request E2E workloads. Greedy recovers 94.97%,
18.12%, 82.77% of ALP bursty excess in the three blocks; PP-only removes it.
No non-trivial residual is established by this finite policy control.

ALP already models context/prefix interactions and updates online. “Add context
length” would attack a straw baseline. Existing greedy and SLO-appropriate
options must be compared before a successor predictor can be credited.

Uneven partition required an opt-in KV-layer-count compatibility correction;
actual function CPU regression was red before and green after. Correcting that
bug is not research headroom. The safe cost-derived partition was slower in
all three request-level comparisons; see STATIC_PARTITION_RESULTS.md.

No trivial-recovery percentage is invented for an unmeasured MLLM failure.
