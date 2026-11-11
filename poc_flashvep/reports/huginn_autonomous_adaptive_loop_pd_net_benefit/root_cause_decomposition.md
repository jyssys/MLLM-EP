# Root-cause decomposition

This is a developing evidence audit. It is not the final campaign verdict.
The final report separates selection results from held-out confirmation.

## 1. The earlier comparison changed two transport settings

The optimized Zero configuration specifies 8 GiB of in-flight wire credit.
It also specifies zero extra draining requests.
The recovery generator replaces the nested dictionary with a shutdown timeout.
Native defaults then supply 256 MiB and eight extra draining requests.

The draining setting changes the source transfer-slot provision.
It is not only a timing threshold.
The wire setting changes how much transfer work can remain outstanding.
This campaign does not yet isolate their separate effects.

The earlier recovery measurements remain valid observations of their actual configurations.
They do not isolate Loop checkpoint overhead against the tuned Zero.
The initial new checkpoint control has the same limitation.

Evidence:

- `state/wire_configuration_confound.json`
- `raw/selection_matchedwire_nomigration_seed2026110940/resolved_native_PD_*.json`
- `raw/selection_fasttoken_seed2026110940/effective_PD_config_*.json`
- The preserved `completion_aware_loop_pd/recovery_confirmation.py` generator

The matched no-migration selection measured TTFT p95 of 9824.4005 ms.
The adjacent fixed-depth Zero measured 9676.2237 ms.
Both runs use Arrival Seed 2026110940 and N128.
This one-start comparison does not establish a confidence interval.

The later, wire-matched checkpoint-only control measures TTFT p95 of 24067.6195 ms.
It exposes all six source checkpoints and performs no migration.
Thus, correcting the transfer confound does not remove checkpoint overhead.
The selective no-migration control preserves the atomic path instead.
These are different source-execution controls.

## 2. Submission count is not CPU time

The instrumented checkpoint control issued 896 P prompt submissions.
Its summed P CUDA Event enclosures measured 75.3971 seconds.
The instrumented Zero issued 128 P prompt submissions.
Its summed enclosures measured 76.3246 seconds.

These values are execution enclosures, not exclusive kernel times.
They can include gaps inside the instrumented call.
The checkpoint and Zero transport settings also differ.
Neither the submission count nor their ratio measures checkpoint cost.

The checkpoint request enclosure can also include interleaved work for another request.
A longer per-request enclosure does not establish a physical GPU idle gap.
Nested Host counters overlap with each other and with device work.
Do not add them into TTFT.

Evidence: `costs/controls.json` and the corresponding `raw/cost_*_events_seed2026110931/` directories.

## 3. Fixed R32 permits fewer Host-dependent Decode decisions

The previous synchronous path collects an unused recurrent Gate result.
The fixed-depth contract does not consume that Gate result.
The new submission extension retains all graph and model operations.
It advances only the exact fixed-depth lifecycle under Native completion events.

The grouped extension uses the existing explicit multi-loop execution path.
It does not fuse or remove recurrent model computation.
Native CODA still collects real output tokens.
Global asynchronous Huginn execution remains disabled.

The combined N128 selection measured ITL p99 of 91.5166 milliseconds.
The three original-runtime Zero starts measured 145.1623, 148.4301, and 123.0373 milliseconds.
These are independent starts of one selection Arrival trace.
They are not held-out confirmation of a Loop benefit.

The separate N8 final-state observer also matched final valid Decode KV and hidden state.
Its synchronization and hashing exclude it from clean latency evidence.

Evidence: `fixed_decode.py`, `fixed_submit.py`, and the final-state qualification namespaces.

## 4. Logical source progress can lead device completion

The inherited source-pressure check reads the asynchronous in-flight list.
That list omits pending work in the synchronous fixed-depth Runtime.
The new check also reads Native per-request producer events.
It counts only live source-owned requests.

A pending event can represent queued work.
It does not prove physical kernel saturation.
The distinct-request pressure threshold remains unchanged.

Evidence: `source_events.py` and its per-run receipts.

## 5. Separate streams alone do not protect Decode

The first separate-stream extension still shares bounded submission buffers.
The next extension adds distinct prompt inputs and submission storage.
Both retain Native per-request dependencies.
Their state checks do not establish a serving speedup.

Synchronous CODA collection also blocks Host submission.
The delayed collector removes that particular wait.
Its first version permits another Decode group before the previous cohort delivers tokens.
That version regresses Zero ITL p99 to 149.8726 ms.
The campaign rejects it as a stronger baseline.

Version 2 preserves Native Decode cohort retirement.
Its matched N128 Zero selection measures ITL p99 of 88.4754 milliseconds.
Static Loop31 on the same execution policy measures 107.5914 milliseconds.
Loop TTFT p95 improves from 9584.1625 to 7398.9598 milliseconds.
All 128 requests perform their last full-prompt Loop on D.
This still fails the Native-relative five-percent Decode target.

The result motivates the resource-budget experiments.
It does not establish adaptive scheduling or Loop-specific novelty.

## 6. A strict serial budget can reject every Heavy offload

The matched adaptive Loop selection completes N128 with zero measured migrations.
Its TTFT p95 is 9658.9196 ms.
The controlled qualification does perform real online boundary negotiation.
The N32 controlled run moves four Loops for each of 15 requests.
Those controlled arrivals are not the Primary Poisson trace.

The reserve allocates 93.1062 ms to a complete Decode cycle.
Its Native-relative ITL target is 97.7615 milliseconds.
The smallest calibrated terminal quantum costs 6.1148 milliseconds.
The repeated serial margin can be smaller than that terminal quantum.
These values explain a policy rejection, not a measured physical idle limit.

The qualified Native numerical fix also adds shape-dependent padded dense work.
Recurrent token rows alone do not measure this cost.
Loop31 moves valid CODA work as well as one recurrent Loop.
The structural work account includes both components.

## 7. Completion prediction still needs direct validation

The matched adaptive Token selection moves one 256-token suffix.
It predicts 253.7948 ms from commit to first output.
The observed Host interval is 318.8645 ms.
The observed import-to-first-output interval is 129.4879 ms.
The observed largest inter-quantum wait is 8.0655 ms.

These observations do not show the previous multi-second starvation in this request.
They also do not establish a transfer or device-kernel duration.
The cohort TTFT remains worse than Zero in this selection start.
Separate controls and cost diagnostics must identify the source of that difference.

## 8. Physical idle capacity remains unmeasured

The first short Nsight control exceeds the interference target.
Moving collector start before Warm-up does not prevent a later NIXL progress stall.
The preserved source stack stops inside `nixl_cu12._api.transfer`.
The cause below that API remains inconclusive.

The graph-level trace is diagnostic only.
Its graph intervals contain internal gaps.
They are not a clean physical kernel union.
Sampled NVML utilization cannot replace that union.
The campaign therefore does not publish a physical idle-capacity Oracle from these traces.

Actual GPU interventions and marker-free serving remain valid separate evidence paths.
Their results cannot establish unmeasured physical idle percentages.

## 9. Host backlog and Decode batch geometry

The Zero selection has Prefill slot pressure in 193/193 central samples.
The central window covers 20% through 90% of the arrival span.
It contains no sample with a Decode-only KV block.
The coordinator admission wait measures 8390.0408 ms at p95.
These observations establish a Prefill-side slot bottleneck, not physical GPU saturation.

Static Loop31 reduces the same admission-wait percentile to 6125.7615 milliseconds.
The observed D receive-to-first-output percentile increases from 73.9120 to 113.0609 milliseconds.
These are Host receipts, not isolated P or D GPU compute durations.
The two percentile differences must not be added into a request-level decomposition.

Three-request Decode groups increase from 14/599 to 121/571 in the central submission histogram.
The larger Decode cohorts accompany the earlier Prefill release.
They may contribute to the observed Decode tail change.
The measured contrast does not isolate batch geometry from continuation interference.

The source pipeline also changes the timing of the Host compute-release receipt.
The first forecasted Loop selection performs no measured migration.
Its P start-to-compute-release p95 decreases, but its compute-release-to-wire-commit p95 increases.
Do not interpret that shorter first Host interval as GPU compute saved.

Evidence: the `runtime_observations/` JSON files for the matched selection runs.

## 10. Paired request differences separate queueing from later work

Compare full-device deferred CODA v2 Zero and forced Loop31 at Seed 2026110940.
All 128 requests have matching input tokens and complete Host milestones.
The mean Loop-minus-Zero TTFT difference is −1558.6351 milliseconds.
The mean arrival-to-admit difference is −1541.6206 milliseconds.
The mean P start-to-compute-release receipt difference is −106.3098 milliseconds.
The mean D receive-to-output difference is +39.5627 milliseconds.
The other intervals appear in the paired JSON and CSV.
The signed intervals sum to each request's TTFT.
They do not give isolated kernel time or DMA time.

The ITL p99 for two Host-visible live Decode requests is 88.7837 ms for Zero and 107.7652 ms for Loop31.
The two strata contain 1718 and 1456 intervals.
Live Decode means after first output delivery and before last output delivery.
It is not the physical batch or the number of Decode-ready requests.
This descriptive control does not remove all workload feedback.
It shows that the larger three-request cohort count alone does not explain the observed ITL increase.

Evidence: `paired_host_attribution/deferred2_loop31_seed2026110940_vs_deferred2_zero_seed2026110940.json`.
