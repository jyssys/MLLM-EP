# Original-gate CDB outer observation: timing validity failure

## Result

The bounded Base1.4B adaptive cell completed on physical GPU5.
All four request triplets preserve output IDs, exit depths, NFE counts, and finish reasons.
All measured graph-capture counts are zero.
The recorded recurrent, coda, and terminal-publication event counts match the actual work counters.
The observer restores every instance override after each trace.

However, every request fails the frozen observer-overhead or clean-drift guard.
Zero of four requests qualify for quantitative bottleneck attribution.
The remaining three size/mode diagnostic cells are `NOT_RUN`.
This lane stops without changing its thresholds or inference semantics.

This result invalidates timing attribution from this observer configuration.
It does not show an architecture failure, quality loss, or absence of an adaptive bottleneck.
No candidate speedup follows from these observations.

## Fixed protocol

The [source contract](../artifacts/cdb_exposed_cost_trace_contract_v1.json) precedes implementation and GPU execution.
It uses the original Base1.4B checkpoint and its original early-exit gate.
The policy remains q=0.5, minimum R2, maximum R4, and `last_exited` KV.
BF16, greedy sampling, native EOS, and frozen stop strings remain unchanged.
The config enables decode graphs and disables prefill graphs.
It disables async batching, delayed gate consumption, and synthetic replay.

Each request runs an untimed same-request warmup followed by clean-before, trace, and clean-after.
Each generation permits at most 128 new tokens.
This is not the 8192-token natural-quality protocol.
It supplies no benchmark-accuracy result.
The four IDs come from metadata-only selection within the existing calibration cohort.
No held-out request enters this lane.

The request clock includes native generation and the existing request-boundary synchronization.
Trace export occurs after that clock ends.
Every CUDA interval uses a distinct, preallocated, materialized event pair on the native compute stream.
No observer enters a graph body or adds a per-layer synchronization.

## Observed request walls

| Request | Output tokens | Clean before, ms | Trace, ms | Clean after, ms | Clean drift | Trace / clean mean − 1 | Attribution |
|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K106 | 56 | 967.694 | 1335.092 | 1037.440 | 6.957% | +33.167% | UNRESOLVED |
| GSM8K1074 | 85 | 1574.318 | 1628.405 | 1896.051 | 18.542% | −6.154% | UNRESOLVED |
| MATH142 | 128 | 2149.382 | 2658.067 | 2210.268 | 2.793% | +21.939% | UNRESOLVED |
| MATH156 | 128 | 2464.231 | 2864.466 | 2619.034 | 6.091% | +12.702% | UNRESOLVED |

The frozen clean-drift limit is 5% of the two-clean-wall mean.
The frozen overhead guard rejects trace wall above 105% of the larger clean wall.
The negative signed ratio for GSM8K1074 does not establish negative observer overhead.
Its clean walls drift by 18.542%.
This analysis estimates no confidence interval from one triplet per request.

The [request CSV](../analysis/cdb_cost_observer_validity_v1/request_validity.csv) retains full precision.
The [wall plot](../analysis/cdb_cost_observer_validity_v1/observer_wall_triplets.svg) shows the three clocks separately.
The [summary](../analysis/cdb_cost_observer_validity_v1/summary.json) records all validity decisions and source hashes.

## What remains useful

The trace records actual per-pass graph launches rather than one replay multiplied by a count.
It distinguishes containing recurrent-batch host intervals from nested metadata and recurrent-device intervals.
It also distinguishes inclusive exit-decision wall, whole terminal-publication windows, and combined coda windows.
Terminal-publication coverage equals the number of observed exits below R4 in this B1, no-refill lane.
These checks establish coverage and preserved discrete execution within the bounded requests.

The [nested interval CSV](../analysis/cdb_cost_observer_validity_v1/nested_intervals_diagnostic_only.csv) retains observed span sums.
Those sums remain invalidated diagnostics, not clean service costs.
CPU intervals and device intervals overlap or contain each other.
They cannot form a 100% E2E breakdown.
Host wall minus device span is not a measured pure-CPU remainder.

Pure terminal-copy kernel time remains unresolved.
The publication window includes row preparation, gathers, copies, and host enqueue gaps.
The outer coda hook cannot separate LM-head projection from argmax.
The exit-decision hook includes dispatch and the unchanged policy's device-to-host scalar read.
The analysis computes neither apparent bandwidth nor pure HBM traffic.

## Observer qualification and retained failures

The [independent review](cdb_cost_observer_independent_review_v1.md) records the implementation findings and their corrections.
The current CPU suite has 17 passing tests.
Tests cover exact restoration, unique events, nested errors, overflow, graph-hit coverage, publication coverage, and parity rejection.
The implementation also preserves partial trace rows after an exception without querying CUDA again.

CPU qualification did not guarantee acceptable physical observer overhead.
The completed GPU lane therefore remains a negative measurement-validity result.
The result does not justify more intrusive hooks, a different gate, or a new scheduler.

## Reproduction

Run from `/home/esjung/MLLM-EP-ouro-deep-discovery`.
Choose a fresh output directory.
The guard must check process ownership before it releases GPU5.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session --gpus 5 --output outputs/NEW_CDB_COST_TRACE --timeout 1200 -- /home/esjung/.venvs/ouro-deep-discovery/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.cdb_exposed_cost_trace --size 1.4B --mode adaptive
/home/esjung/.venvs/ouro-deep-discovery/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.analyze_cdb_exposed_cost_trace --job outputs/NEW_CDB_COST_TRACE --output analysis/NEW_CDB_COST_ANALYSIS
/home/esjung/.venvs/ouro-deep-discovery/bin/python -m unittest poc_ouro_parallelism_deep_discovery.tests.test_cdb_cost_observer_cpu
```

The retained run is `outputs/cdb_cost_base14_adaptive_gpu5_v1`.
Its guarded child exits with code zero.
Its guard records a scoped GPU5 burner restart.
Successful process completion does not override the failed timing-validity gates.

## Campaign implication

This is baseline diagnosis, not C07 fixture recovery or an architecture implementation.
The campaign still has no measured candidate E2E gain.
Keep the unresolved costs explicit in every candidate model.
Do not treat their absence from a clean breakdown as zero cost.
Continue the independent architecture search without using this lane for a positive claim.
