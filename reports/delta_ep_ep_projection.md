# DeltaEP routed-MoE projection

The projection applies the held-out exact codec only to measured layers 1/5/10/14/19 on top of B1. Expert compute is unchanged. Stage gain is an optimistic **zero codec-overhead upper bound**, not a measured implementation.

| Target | dispatch-byte reduction | combine-byte reduction | 5-layer stage upper bound | all-layer linear extrapolation | total overhead budget (ms/16 req) |
|---|---|---|---|---|---|
| EP4 | 0.407% | 0.298% | 0.0157% | 0.0596% | 3.6136 |
| EP8 | 0.373% | 0.275% | 0.0203% | 0.0768% | 4.8532 |

Because even zero-overhead gain is below the gate, any encode/decode/cache lookup cost can only reduce it; a GPU microkernel is not justified. A branch-keyed selected-five-layer cache has a worst-case 5.0 MiB/request/direction bound (10.0 MiB for dispatch+combine); extrapolating to all 19 layers gives 19.0 MiB/direction. Token-destination deduplication can lower this, but does not change the latency conclusion.
