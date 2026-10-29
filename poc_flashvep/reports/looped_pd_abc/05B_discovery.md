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
For boundary models, the final recurrent submission also writes the valid prompt-prefix coda KV.
Its completion event follows that write.
The frontier transfer waits for that complete event, not merely the recurrent operation's submission.

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

## Recorder-confounded Dory SLO region

One completed frozen-runtime start showed a possible operating region for further controls.
Subsequent analysis found synchronous evidence I/O in its request path.
Its timing status is now `TIMING_INVALID_FOR_CLEAN_METHOD_COMPARISON`.
The following values remain as historical observations, not clean performance evidence.
The workload uses P2048, concurrency 8, 24 requests, and output cap 32.
The SLO requires TTFT at most 10 seconds, E2E at most 30 seconds, and every request ITL at most 100ms.
This SLO cell belongs to the grid frozen before the main serving campaign.

| Method | Qualified requests / N | SLO goodput, requests/s | Request p95 E2E, ms |
|---|---:|---:|---:|
| Colocated DP2 | 2/24 | 0.210 | 3189.0 |
| Colocated Mixed DP2 | 2/24 | 0.270 | 2489.1 |
| Matched overlapped G1 PD | 5/24 | 0.452 | 4574.9 |
| B frontier handoff | 18/24 | 1.875 | 3414.9 |

Evidence: `MEASURED_INSTRUMENTED_EVIDENCE_IO_NOT_CLEAN_SERVING`.
The frontier and the three listed controls match on all 24 output sequences and output lengths.
The point goodput ratio against matched G1 PD is 4.151×.
That number is not an accepted multi-start speedup.
It also does not establish sustainable open-loop capacity.
Mixed DP2 remains faster in p95 E2E despite its lower attainment in this strict-ITL cell.
The trade-off must remain visible.

The source runs are `dory_*_P2048_C8_s0` under `raw/`.
The paired SLO analysis uses the same request IDs and conditions on observed cohort spans.
It reports undefined ratio draws when a resampled baseline has zero goodput.
One start cannot quantify between-start uncertainty.

If the signal survives three nonblocking, frozen, output-matched starts, a generic NIXL pacing sweep runs before C.
It changes only transfer-packet bytes and maximum inflight bytes.
It never changes the 256-token prefill chunk or G1 computation.
The confirmation then compares both mechanisms at the same selected transport budget.
`configs/B_transport_followup_plan.json` records this pilot-guided secondary experiment before tuning.
Generic pacing is a control, not a new loop-specific contribution.

Any surviving gain also needs a boundary-only versus recurrent-frontier streaming ablation.
Generic immutable layerwise streaming remains another competing explanation.
The current adapter's 80/100 official-reference agreement remains unresolved.
No performance signal removes that fidelity or novelty limitation.
`timing_correction.md` records the recorder defect, CPU tests, and required canonical retiming.
