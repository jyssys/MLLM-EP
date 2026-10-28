# B: loop-frontier handoff discovery

Status: immutable-fragment transfer passed. Repeated serving evaluation is in progress.

## Validity and ownership

B keeps every prefill loop on P and every decode loop on D.
It does not use A's intermediate compute handoff.
The comparison uses unchanged staged G1, chunk size, depth, attention backend, and cache semantics.

The primary cache layout has independent storage for each recurrent depth.
A completed prompt range at one depth is immutable for that request.
The connector waits for its producing CUDA event before NIXL reads that range.
It preserves upstream destination reservations, transfer sequences, and source leases.
The destination admits decode only after the final commit and transfer completion.

Huginn and Dory use prelude and coda KV at depth zero only.
Their recurrent KV remains depth-indexed.
The coda prefix is valid only after the final recurrent loop.
The last prompt token's coda executes on D, as in the matched PD control.
Unused boundary-layer planes are not valid transfer payload.

## Measured identity

The following 512-token checks include three imports per model, including warmup.
All checks passed with reversed fragment order.

| Model | Valid KV bytes | Final hidden bytes | Exact imports |
|---|---:|---:|---:|
| Ouro R4 | 402,653,184 | 4,096 | 3/3 |
| Huginn R32 | 1,427,331,840 | 21,120 | 3/3 |
| Dory R2 | 117,407,744 | 10,240 | 3/3 |

Evidence: `MEASURED_NIXL_BYTE_IDENTITY_HEAVY_CORRECTNESS_ONLY`.
The heavy SHA checks are excluded from clean serving benchmarks.

The 100-prompt bounded generation pilots also matched each model's staged PD control on 100/100 outputs.
This is internal RLT fidelity. Official-reference fidelity remains conditional.

## Existing overlap is the main control

Upstream RLT exports a completed full-depth token chunk.
It can transfer that chunk while P computes another chunk.
B exports immutable depth planes earlier within that chunk.
The main comparison therefore retains upstream overlap in the control.
It does not compare B only against a deliberately synchronous end-of-prompt copy.

Earlier availability can reduce the final uncovered transfer interval.
More fragments can also increase submission, polling, event, and descriptor costs.
Copy traffic can interfere with model computation.
The experiment must measure both effects on request latency.

`transfer_completion_observation_seconds` is cumulative host-observed transfer duration.
Overlapping transfers can contribute multiple durations.
This counter is not GPU DMA time. It is not an E2E latency component.
The host phase partition separately measures the final uncovered handoff interval.
GPU timeline analysis remains necessary for an actual overlap fraction.

## Invalid alternatives and novelty limit

The shared or last-loop current-token slot can be overwritten by later loops.
B does not export an overwritten intermediate value as final KV.
Changing the cache mode would create a different quality configuration.

Loop-frontier transfer is a structured form of immutable layerwise transfer.
Generic layerwise streaming is therefore a required novelty counterexample.
The current implementation does not establish superiority over every generic layerwise connector.
No accepted loop-specific serving gain is established yet.
