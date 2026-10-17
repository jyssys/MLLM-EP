# Track A: recurrent KV-aware distributed attention

## Decision

Track A is `NO_GO` for a new recurrent-specific architecture.
The attention-core sub-reason is `A_NO_BOTTLENECK` in the valid light decode lane.
The measured conventional CP operator is also slower after launch and merge costs.
Reference cache management has a larger cost, but native paged KV already addresses that generic runtime problem.

This conclusion applies to the measured contexts and implementation.
It does not exclude a more efficient conventional CP operator on another workload.
The study does not certify all attention/KV costs as small.

## Matched Amdahl evidence

| History | Clean cached forward, ms | Attention interval, ms | Infinite-core-speed ceiling | Ideal four-way-core reduction |
|---|---:|---:|---:|---:|
| 1K | 66.997 | 4.869 | 7.268% | 5.451% |
| 16K | 89.089 | 6.544 | 7.345% | 5.509% |

The denominator comes from the matched light lane.
The component intervals pass its observer gate.
The study measures one calibration request with five technical repetitions per family.
These bounds do not establish a population-wide confidence interval.
They apply only to replacing the attention core while other costs remain fixed.
They do not include a hypothetical simultaneous cache-manager replacement.

At 16K, prefill attention is 325.764 ms of a matched 879.832 ms reference forward.
That larger prefill fraction does not establish a new recurrent-specific distributed attention method.
Track B separately tests legal causal prefill scheduling.

## Physical exact conventional CP operator

The operator uses Q/K/V captured from the actual pretrained model.
It splits historical KV across GPUs 4–7 along sequence length.
It broadcasts the fresh query and calculates one local attention partial per shard.
It merges outputs using global logsumexp normalization.
The current token's KV appears exactly once.
It does not add independently normalized partial outputs.

Both the first and fourth recurrent depth are tested at each context.
All eight cases pass fixed maximum-absolute and relative-L2 limits of 0.02.
The operator includes temporary buffers, device transfers, launches, and completion synchronization.
Historical KV placement is measured separately.
This is a physical microbenchmark, not a new inference runtime.

| History | Native operator, ms | Four-GPU CP operator, ms | Native-time / CP-time |
|---|---:|---:|---:|
| 1K | approximately 0.055 | approximately 0.556 | approximately 0.10× |
| 4K | approximately 0.067 | approximately 0.550 | approximately 0.12× |
| 8K | approximately 0.085 | approximately 0.553 | approximately 0.15× |
| 16K | approximately 0.101 | approximately 0.553 | approximately 0.18× |

These host-completion intervals include Python launch overhead.
They are not pure attention-kernel times.
A C++ or fused-collective implementation could change the result.
The result does not justify constructing that implementation without additional headroom.

## Cost-aware oracle

The oracle replaces the measured attention interval with 96 serial CP operator invocations.
It retains all other reference wall cost.
It does not predict a physically executed model request.

| History | Zero-cost ideal, ms | Measured-operator model, ms | Conservative model, ms |
|---|---:|---:|---:|
| 1K | 63.346 | 115.519 | 118.037 |
| 16K | 84.182 | 135.598 | 137.257 |

The ideal uses a four-way attention speedup with zero communication.
The measured-cost case uses physical median operator completion times.
The conservative case uses the measured operator p95.
The latter two do not include every historical KV redistribution or fresh KV placement cost.
Those omitted costs are nonnegative and cannot rescue these negative modeled results.

## Novelty and fair-baseline limit

The tested sequence sharding and exact softmax merge are conventional CP operations.
The test exposes no extra recurrent-specific delta over that generic operator.
The optimized TP4 control is physically measured but numerically `REFERENCE_PARTIAL` against HF.
A strongest-exact-four-GPU performance comparison therefore remains `UNESTABLISHED`.
No architecture GO or measured E2E speedup follows from this track.
