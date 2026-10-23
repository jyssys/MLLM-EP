# Staging, graph paths, and memory boundaries

The adaptive staged engine reserves C2048 hidden rows per request slot.
For hidden size 2048 and BF16, this is 8,388,608 bytes per request slot.
The stage pool is 33,554,432 bytes at N4 and 67,108,864 bytes at N8.
These are structural allocation sizes, not measured HBM traffic.
Actual peak allocated and reserved bytes remain in each cohort record.

The KV pool uses 18,432 blocks of 16 token positions.
The shared physical-layer cache retains depth-specific last_exited planes.
The four loops have 96 independent layer/depth namespaces.
All matched policies use the same KV pool reservation.
Atomic prefill does not allocate the staged hidden pool.

Native CUDA Graphs, static buffers, and multiple streams remain enabled.
The graph batch bound equals the configured request concurrency.
The graph-entry bound is 32.
The harness records actual capture, replay, and fallback counters around each submission.
A configuration flag alone does not prove that a particular prefill quantum used graph replay.

The observer starts after native request-dependency waits.
GPU preparation, model forward, and hidden-state publication inside native execution remain in its service interval.
Earlier CPU preparation and queued waits remain separate fields.
The end event does not represent every later host bookkeeping operation.
Clean request latency includes the actual bookkeeping and scheduler overhead.

Graph/path state and pending work can affect service variance.
The v1 lookup does not model every one of those recorded features.
This is one reason its estimates remain experimental.
No stream-service sum is presented as an additive whole-request latency breakdown.
No tensor reads occur before the native completion dependencies permit them.
