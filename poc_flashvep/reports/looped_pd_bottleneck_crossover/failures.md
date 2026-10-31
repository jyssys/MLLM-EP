# Failures and Measurement Limits

## Scope

This file records failures in the current study.
It does not replace earlier frozen reports.
Raw namespaces, failed commands, source archives, and logs remain intact.

## Huginn cache allocation during the chunk sweep

`phase_a_huginn_chunk_start0` failed while allocating its second KV pool.
A local `cache` reference retained the previous pool after engine closure.
The next allocation therefore overlapped two large pools.
An exception handler then reported an additional `UnboundLocalError`.

The harness now releases `engine`, `cache`, and `barrier` references before the next allocation.
It records allocated and reserved memory before every new pool.
`phase_a_huginn_chunk_retry_start0` completed the three-group regression.
The private engine, checkpoint, and model mathematics did not change.
See `state/chunk_sweep_OOM_diagnosis.json`.

## Initial barrier pilots

The V1 barrier pilots ended before a normal terminal status update.
Their raw status files still say RUNNING, but their recorded PIDs are absent.
The experiment matrix labels these namespaces as incomplete rather than active.
The V2 pilots then failed with `KeyError('issued_loop_rows')` in the qualification harness.
The V3 gates completed eight exact output and state groups per model.
The failed and incomplete pilot namespaces remain intact.
Their latencies are not clean Serving evidence.

## Qwen unsupported-mode pilot

The initial Qwen pilot disabled chunked prefill.
Native vLLM emitted a warning for that unsupported configuration.
The owned coordinator of the initial NIXL pilot received SIGINT.
These pilot results are not primary performance evidence.

The supported retry enables chunked prefill.
Its native NIXL gate passed two output comparisons against native single-engine execution.
Timed independent Prefill cohorts fit within their declared token budget.
Their chunking-enabled flag does not imply a mixed Prefill/Decode timed phase.
Native AR PD Serving can use native mixed scheduling.
This difference is an explicit engine confound.

## Source guard rejected an analysis-only edit

`phase_b_huginn_g2_tuned_chunk_start0` failed before model inference.
The startup guard included every study Python file.
An edit to `additional_analysis.py` therefore triggered the source guard.
The failed log identifies that file.
The dispatcher recorded return code 1.
The raw worker did not reach creation of its status file.

The first campaign controller treated the missing status as an unknown active namespace.
It therefore stopped instead of advancing to the next job.
The controller now recognizes a recorded nonzero dispatcher result as a failed start.
It does not rewrite the failed raw namespace.

The worker guard still checks actual GPU execution sources.
The dispatcher still archives all study sources.
Analysis-only files no longer block an unrelated worker startup.
The CPU regression failed before this change and passed afterward.
The new namespace `phase_b_huginn_g2_tuned_chunk_retry_start0` completed successfully.
This failure was not an output divergence or a model correctness failure.
See `state/source_guard_diagnosis.json`.

## Prepared Decode alignment diagnostic: excluded V1 timing

The V1 alignment diagnostic read configuration files inside its per-request scheduling callback.
This added avoidable CPU and I/O work.
It could confound a comparison between simultaneous and staggered releases.
V1 outputs remain correctness evidence.
V1 timings are excluded from alignment performance attribution.

V2 caches the configuration outside the callback.
It restores the original fast path for already released requests.
A zero-offset control uses the release gate without intentional delay.
All V2 smoke cases passed native output comparisons.
The zero-offset TPOT ratios were 1.0177 for Ouro and 0.9407 for Huginn.
Each ratio uses one start per variant.
They do not establish a statistical overhead bound.
The larger Ouro B24 diagnostic includes another zero-offset control.
See `state/alignment_v1_measurement_exclusion.json` and `state/alignment_no_op_gate.json`.

## Heavy profiling changes latency

CUPTI profiling and CPU operator tracing can substantially increase request latency.
Disabling shape recording alone did not remove this effect.
Profiled runs have a distinct measurement mode.
They are excluded from clean Serving comparisons and SLO-goodput claims.

Recorded kernel unions are genuine profile observations.
Their denominator is a profiled interval, not an unprofiled Serving interval.
Do not interpret their idle fraction as a clean-server GPU idle fraction.
Do not add overlapping kernel, copy, and host spans.

## UCX messages during worker teardown

Several Huginn runs emitted `cuDevicePrimaryCtxGetState` error-code-4 messages during teardown.
The messages followed completed output delivery in the observed logs.
The affected completed runs recorded return code 0.
Their associated output and state checks passed where those checks were enabled.

The study does not prove the teardown error is harmless in every deployment.
Its root cause remains unverified.
Raw logs retain the complete messages.
Request timing ends before offline engine shutdown.

## Analysis and preflight corrections

The first percentage validator grouped serial calibration requests by sample ID alone.
Each calibration subrun reused sample ID 0.
The validator therefore combined several different requests.
The derived CSV now includes the subrun identifier.
Each reconstructed host breakdown sums to 100% within floating-point error.
No raw timestamp changed.

The initial 64-request generator inherited only 32 Prompt IDs.
The preflight test found this before those pending configurations ran.
The generator now selects the frozen 64-record test file.
The original pending configurations remain in `state/pending_prompt_id_repair.json`.
No executed request stream changed.

## Inter-queue idle interval

The Phase A queue completed normally at 09:44:53 UTC on 2026-10-31.
Its last worker returned code 0.
The configured queue did not generate the Phase B plan automatically.
Phase B started after the resume request at approximately 11:02 UTC.
No GPU experiment ran during that inter-queue interval.
This interval counts toward total elapsed study time.

The current queue is durable and skips completed namespaces.
An independent resource watcher waits for all owned dispatchers to stop.
It requests burner restoration only when no GPU process remains active.
This mechanism does not authorize automatic advancement through a scientific gate.

## Remaining limits

- Full-Prefill PD uses C256 or C2048 in this campaign. Whole-prompt atomic PD Serving was not measured.
- Exact 4K/100 and Ouro 8K conditions were not measured. The common grid cannot establish an all-context limit.
- The Qwen3-1.7B common-grid diagnostic is not a faithful DuetServe Qwen3-8B 8,000/200 reproduction.
- Internal parity does not establish original-HF equivalence or downstream answer accuracy.
- Controlled EOS-ignored generation does not establish natural output-length behavior.
- Finite cohorts do not establish infinite-horizon sustainable capacity.
- A prepared all-ready batch can overestimate achievable online Decode capacity.
- Co-occurring admission reasons are not exclusive causal waiting-time shares.
- Host request percentages are not pure GPU computation or DMA percentages.
- Native AR and RLT use different engines and kernels.
- Parallel-lane contention was not quantified. Primary timing uses one isolated lane.
