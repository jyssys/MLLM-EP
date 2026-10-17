# Track B: depth–sequence wavefront

## Decision

The structural verdict is `B_GENERIC_ONLY`.
Legal causal prefill concurrency exists.
However, the tested dependency graph and schedules map directly to generic unrolled causal PP.
No additional recurrent-specific scheduling gain is demonstrated.

Physical pipeline performance remains unverified.
Serial chunking has only `REFERENCE_PARTIAL` numerical qualification.
The study therefore reports conditional modeled schedules, not a validated method or measured speedup.

## Physical inputs

The reference measures serial chunk sizes 128, 256, 512, and 1,024.
The prompt contexts are 1K, 4K, and 8K.
Each configuration has five clean timing repetitions.
A separate observer records actual per-layer, per-depth, and per-chunk intervals.
Nested intervals are not added twice.

The six-layer task cost sums its layer intervals.
The last task in each pass also includes the terminal normalization.
The model retains any positive residual between observed task totals and clean serial wall time.
Task costs do not assume an unsplit GEMM retains its efficiency after chunking.
The observed chunked work is substantially slower in many configurations.

An activation transfer has `chunk_rows * 2048 * 2` BF16 bytes at B1.
The communication input is the slowest measured directed P2P median at that payload size.
No full KV transfer occurs when the owning stage retains its own depth/layer cache.

## Placement and generic controls

| Placement | Device ownership | Cross-device hidden edges per chunk | Storage implication |
|---|---|---:|---|
| Layer-stationary | Six physical layers per GPU, with recurrent wraps | 15 | Physical layer weights remain local |
| Loop-stationary | One full 24-layer pass per GPU | 3 | Full shared block is replicated across four GPUs |

Both schedules retain all four depths and all prompt chunks.
Each GPU serializes its compute tasks.
Each directed link serializes its transfers.
The conservative scenario also serializes communication against both endpoint compute streams.

The scheduler compares chunk-FIFO and depth-prioritized ASAP list policies.
It reports the better tested policy for each placement and cost scenario.
This is not a proof of the globally optimal fine-grained schedule.
The coarse dependency requires an earlier chunk's complete stage output before its successor advances.
That condition is stricter than the fine graph's KV-publication boundary.

With only one whole-prompt chunk, the same request cannot fill a multi-stage pipeline.
Chunking creates legal overlap but adds launch and shape costs.
The unrolled generic control uses identical resources, operation costs, and terminal norms.
Its modeled latency equals the recurrent candidate's latency by graph identity.
The resulting 0% recurrent-specific delta is analytical, not a measured pipeline comparison.

## Conditional cost-model examples

These rows use component lanes that pass the 10% observer gate.
They still fail to qualify a complete physical pipeline because smaller-chunk numerical comparisons are partial.
The comparison denominator is the exact HF single-GPU prefill, not optimized native TP4.

| Context / chunk | Placement | Measured-cost modeled prefill, ms | Model reduction versus HF single GPU |
|---|---|---:|---:|
| 1K / 1,024 | Layer-stationary | 81.431 | 4.24% |
| 1K / 1,024 | Loop-stationary | 80.550 | 5.28% |
| 4K / 1,024 | Layer-stationary | 104.244 | 41.23% |
| 4K / 1,024 | Loop-stationary | 149.910 | 15.49% |
| 8K / 1,024 | Layer-stationary | 333.395 | 11.40% |
| 8K / 1,024 | Loop-stationary | 370.571 | 1.52% |

These are not measured speedups.
The corresponding native TP4 TTFT measurements are lower.

| Context | Native TP4 diagnostic TTFT, ms |
|---|---:|
| 4K | 73.037 |
| 8K | 107.446 |

Those native comparisons remain diagnostic because cross-backend fidelity is partial.
No free transition to a different decode layout is assumed.
No future-token decode overlap or full-request E2E improvement is claimed.

`oracle_b.csv` retains every modeled context, chunk, scenario, and validity label.
`modeled_schedule.png` is explicitly a simulated timeline.
It is not a capture of four GPUs executing a new pipeline.

## Stop condition

The tested wavefront does not exceed the matched generic schedule.
Numerical chunk qualification is also incomplete.
These results fail the loop-specific architecture gate.
The study does not implement a PP engine, hidden-state predictor, or speculative rescue policy.
