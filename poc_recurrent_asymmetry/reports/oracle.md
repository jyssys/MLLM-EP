# Physical communication and architecture oracles

## Actual communication measurements

The physical bundle uses only GPUs 4–7 and checks their UUIDs.
It measures 108 P2P cases across all 12 directed pairs and nine payload sizes.
Each case has five warmups and 30 repetitions.
Destination payload checks establish that copies actually complete.

| Payload | Directed-path median host-completion range, ms |
|---|---:|
| 4 KiB | 0.05503–0.06465 |
| 128 KiB | 0.05593–0.06386 |
| 512 KiB | 0.05465–0.06822 |
| 1 MiB | 0.05474–0.06367 |
| 2 MiB | 0.05399–0.06475 |
| 4 MiB | 0.05865–0.07344 |
| 32 MiB | 0.12986–0.14932 |
| 128 MiB | 0.38455–0.40473 |
| 256 MiB | 0.72986–0.74859 |

The host intervals include software and synchronization costs.
The separate destination-stream intervals produce `ApparentThroughput`, not pure-link bandwidth.
The study does not divide HBM capacity by latency to infer a physical traffic counter.

Real four-process NCCL AllReduce and broadcast tests cover five payload sizes.
Their median cumulative-critical-rank intervals span approximately 0.043–0.082 ms.
Raw records retain every rank and sample.
Tensor payloads are not NCCL wire-byte measurements.

## Measured compute/copy interference

The operator uses the pretrained first-layer MLP on 1,024 rows.
A disjoint peer copy transfers 64 MiB.
The concurrent run checks exact MLP output parity.

| Mode | Host completion, ms |
|---|---:|
| Isolated MLP | 0.16677 |
| Isolated copy | 0.20606 |
| Sum of isolated medians | 0.37283 |
| Actual concurrent completion | 0.33668 |

Concurrency saves only about 9.7% relative to the isolated sum in this operator experiment.
It does not deliver the ideal maximum-of-two-times bound.
This is a contention diagnostic, not a physical pipeline speedup.
The scheduling model's conservative case therefore does not assume free overlap.

## Track A oracle provenance

| Scenario | Attention replacement | Important limit |
|---|---|---|
| Zero-cost ideal | Matched measured attention interval divided by four | Assumes ideal shard efficiency and free communication |
| Measured-operator cost | 96 actual four-GPU operator medians | Generic CP microbenchmark, not integrated E2E |
| Conservative cost | 96 operator p95 intervals | Still excludes nonnegative historical KV redistribution |

| Context | Matched reference, ms | Ideal, ms | Measured-cost model, ms | Conservative model, ms |
|---|---:|---:|---:|---:|
| 1K | 66.997 | 63.346 | 115.519 | 118.037 |
| 16K | 89.089 | 84.182 | 135.598 | 137.257 |

No Track A model establishes a recurrent-specific advantage.

## Track B oracle provenance

| Scenario | Compute input | Transfer input | Overlap assumption |
|---|---|---|---|
| Zero-cost ideal | Actual observed chunk task costs | Zero | Independent ready tasks overlap |
| Measured-cost | Same observed costs | Worst directed median for the exact payload | Compute and links are separate resources |
| Conservative | Observed costs multiplied by 1.10 | Twice that measured transfer cost | Transfers block endpoint compute |

The 1.10 and 2.0 multipliers are declared sensitivity assumptions.
They are not measured hardware slowdowns.
The model retains fill/drain bubbles, recurrent wraps, and chunk prerequisites.
It retains the measured loss of efficiency from smaller chunk shapes.
It does not account for an unimplemented scheduler as zero-cost measured execution.
The serial positive wall residual remains in the modeled total.

The equal-resource generic schedule has identical graph edges and task costs.
The recurrence-specific delta is therefore 0% within this model.
No oracle row is a measured full-request E2E result.

## Decision table

| Track | Four-GPU baseline status | Ideal / cost-aware finding | Novelty status | Quality scope | Decision |
|---|---|---|---|---|---|
| A | Native TP4 measured, REFERENCE_PARTIAL | Core-only ideal reduction approximately 5.5%. Physical CP operator is slower | Conventional CP only | Operator tolerance passes, no candidate model | NO_GO |
| B | Native TP4 measured, REFERENCE_PARTIAL | Conditional generic prefill overlap, no qualified E2E gain | Generic PP equivalence | Smaller-chunk numerical gate fails | GENERIC_ONLY, physical benefit INCONCLUSIVE |

The strongest exact same-four-GPU performance gate remains unestablished.
This limitation prevents a positive architecture decision.
It does not invalidate the exact single-GPU reference or the physical communication microbenchmarks.
