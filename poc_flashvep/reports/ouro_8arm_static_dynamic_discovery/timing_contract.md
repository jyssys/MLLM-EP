# Timing contract

## Separate lanes

Task generation uses actual greedy feedback. It measures time to each independently generated answer.
Outputs can differ between backends. Report output length with every task timing comparison.
Task instrumentation can include a finite-logit observer. This lane is not an observer-free hardware speedup.

Custom runtimes currently rebuild request-bound graph storage for each cohort.
The serving clock starts after that preparation. It is not a cold-cohort admission-to-answer clock.
Later records include a separate resident-model clock from cohort preparation through service completion.
This clock includes per-cohort graph capture, allocation, staging, and waits. It excludes initial model loading.
Earlier custom TP pilot records lack this field. Do not reconstruct it by dividing command wall time among requests.
Earlier PP records have a preparation duration, but not the exact shared admission timestamp.
Report incomplete coverage as PARTIAL. Do not call service-only ratios cold E2E gains.
The command completion wall time includes initial loading, preparation, service, logging, and teardown.
It is an experiment-job measurement, not a serving-only latency.

Hardware replay uses frozen prompt IDs and frozen output IDs.
Each arm computes all model outputs but feeds the same frozen token into the next step.
The replay never counts as benchmark accuracy.
The replay excludes graph construction and weight loading from its serving clock.
The current implementation does not retain request-bound graphs across cohorts or repeats.
Replay therefore isolates an already-prepared service path. It does not demonstrate deployable steady-state graph reuse.
Repeat zero warms the serving path. Five subsequent repeats form the primary replay sample.

Trace runs use NVTX, CUDA events, and optional Nsight Systems.
Trace wall time never substitutes for clean serving time.
Per-rank CUDA event durations are service intervals. They are not additive E2E percentages.

## Admission and metrics

Each cohort admits all requests at a common rank-zero host time.
TTFT ends when the first token becomes available at the controller.
E2E ends when the final token becomes available at the controller.
TPOT is `(last publication - first publication) / (output tokens - 1)`.
Cohort makespan ends when the last request completes.
Request throughput divides the completed request count by cohort makespan.
Output throughput divides the generated token count by cohort makespan.

Single uses physical GPU4. Distributed arms use physical GPUs4,5,6,7.
Authorized unused GPUs run the requested burners. The environment record must identify those concurrent burners.
There is no performance claim about GPU efficiency from a raw four-GPU versus one-GPU ratio alone.

## Engine differences

Native vLLM uses its own batching, attention kernels, paged KV, graph compiler, and scheduler.
Custom PP uses one GPU-affine process and a FIFO queue per rank.
Custom stages execute one request at a time. Independent requests can occupy different stages.
Custom graph decode uses a full-capacity causal mask. Its allocated key length differs from exact-length eager attention.
The custom TP diagnostic shares this mask policy, but its four-head attention geometry differs from unsharded attention.
Production-oriented and custom engine-matched comparisons require separate tables.
Custom serving preloads prompt IDs onto each GPU before admission. Native frontend staging remains part of native task E2E.
Custom TP retains the generated token on the GPU for the next forward. It does not require a second CPU-to-GPU token upload.

## Validity

The first native TP1 quality run emitted only final results.
Its recorded TTFT and TPOT are invalid. Its generated outputs and total completion times remain available.
The corrected native frontend records each token publication. Earlier raw records remain unchanged.
See `state/measurement_corrections.json`.

Use paired question bootstrap for accuracy. Use paired cohort or repeat bootstrap for timing ratios.
Do not treat tokens, layers, or recurrent loops as independent serving samples.
Numerically unresolved TP timing remains exploratory. It cannot establish a qualified architecture speedup.
