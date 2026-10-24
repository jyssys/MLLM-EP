# Exact KV movement and break-even

The test transfers original BF16 model-derived K and V.
Both host buffers use pinned memory.
CUDA completion waits enter the wall interval.
Each size has twelve technical repetitions in one process.
Every completed roundtrip is byte-exact.

| Valid positions | K+V MiB | H2D median ms | D2H median ms | Roundtrip median ms |
| --- | --- | --- | --- | --- |
| 1024 | 8.0 | 0.1737 | 0.1726 | 0.3345 |
| 2048 | 16.0 | 0.3262 | 0.3265 | 0.6373 |
| 8192 | 64.0 | 1.2367 | 1.2456 | 2.4672 |
| 16384 | 128.0 | 2.4521 | 2.4684 | 4.9016 |


The 16K payload is one physical layer at one recurrence depth.
Moving a full depth requires 24 such payloads.
The conditional roundtrip sum is 117.64 ms per depth.
Moving all four depths gives 470.55 ms per output token.
Those sums are extrapolations, not measured full-model transfers.
The native B1/16K decode interval is about 40 ms.
Repeated full-cache CPU tiering therefore fails this bounded break-even screen.

NCCL P2P records use the same original KV values.
Their latency includes completion barriers and software overhead.
It is not raw NVLink bandwidth.
The distributed attention operator instead transfers a small query and partial-output summaries.
It does not transfer the historical KV on every attention call.

Native RLT already keeps its whole pool in HBM.
It has no CPU KV-fetch stall for a depth prefetch policy to remove.
The study therefore does not construct an offload engine.
Future memory-pressure exploration must charge legal cache placement and append costs.
