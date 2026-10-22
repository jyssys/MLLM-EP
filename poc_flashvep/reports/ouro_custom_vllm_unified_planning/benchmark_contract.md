# Natural-generation benchmark contract

## Status and evidence labels

This contract defines future experiments. All new performance and accuracy cells are **NOT_RUN**.

| Label | Meaning |
| --- | --- |
| SOURCE_VERIFIED | Direct inspection of a pinned source or configuration |
| HISTORICAL_MEASURED | Measurement from the preserved eight-arm study |
| LIVE_MEASURED | Future execution in the unified engine, with a traceable raw record |
| REPLAY_MEASURED | GPU execution with frozen output IDs or depth schedules |
| MODELED | A cost model or dependency bound, not live speedup |
| TO_VERIFY / NOT_RUN | Unqualified mechanism or unexecuted experiment |

The main table uses LIVE_MEASURED natural generation only.
Replay, component timing, and heavy profiles occupy separate tables.

## Frozen model, prompts, and evaluation

Use the model and tokenizer revision in [dependency_lock.md](dependency_lock.md).
Use the historical `evaluation_subset400.json` selection and `frozen_prompt_ids.json` inputs for backward comparison.
Do not reselect questions from candidate scores.
Keep the historical prompt text, demonstrations, stop strings, evaluator, and answer normalization.
The recipe uses base-model completions: three-shot CoT for GSM8K and zero-shot boxed-answer MATH.
Do not substitute the Thinking model's chat template.
Hash each artifact before execution. Compare the input IDs through the HTTP token-input path where supported.
If the server only accepts text, test re-tokenization equality and record any mismatch before benchmarking.

Quality generation uses temperature 0, natural EOS, the frozen stop strings, and max_new_tokens=4096.
Record all remaining sampling fields explicitly. Do not rely on changing server defaults.
Preserve parse failures and truncated outputs in the accuracy denominator.
Stop strings and EOS are part of the fixed policy. Do not extend a candidate's cap only after observing its errors.

Quality stages:

1. Numerical calibration: eight disjoint requests. No candidate-result tuning on the quality set.
2. Natural correctness smoke: 24 frozen requests, spanning page and prompt-length boundaries.
3. Pilot: historical 64 GSM8K and 50 MATH questions.
4. Main quality anchor: historical 300 GSM8K and 100 MATH questions.
5. Conditional extension: all 1,319 GSM8K and 500 MATH-500 questions.

The 400 questions are not a fresh held-out discovery set. Earlier experiments already used them.
Keep new calibration disjoint. Freeze any additional held-out IDs before candidate tuning.
Never describe a 64-token bounded trace as a reasoning accuracy evaluation.

Dynamic policy is identical across architectures: original gate, q=.5, R2–R4, full-R4 prefill, and `last_exited`.
Begin with no-refill. Keep refill, delayed publication, and other admission modes in separate rows.
Do not use RLT delayed exit or future depth traces in live Dynamic evaluation.

## Comparison families

### Main serving comparison

Use one pinned engine and the same optimized attention/GEMM/sampling path.
Compare the four placements in each depth mode.
Use common request arrivals, workload manifests, memory-safety rules, and admission limits.
Do not cap the native baseline to a slow B1 custom executor merely to favor PP.

Different placements can create different stage batches. Record that difference instead of claiming identical packing.
Report active rows, padded rows, stage batch sizes, graph buckets, queue rules, and actual layer work.
Include a controlled matched-admission lane and the strongest qualified production-oriented lane.
Do not merge results across these lanes.

### Required controls

- Unmodified native vLLM TP1 and TP4.
- Extension disabled, to measure fork overhead on native execution.
- Static graph-cut execution, to isolate depth-level graph fragmentation.
- Generic tied-unrolled contiguous PP with the same boundaries as Loop PP.
- Synchronous and asynchronous repair within the same admission/barrier mode.
- Optional DP4 capacity and throughput, identified as four independent replicas.

An eight-arm result must not replace these controls with historical custom timings.

## Workload shapes and arrival patterns

Main PP uses full prompts as stage units. Disable prompt chunking in the matched lane.
If native optimized chunked prefill helps a production baseline, report it as an additional explicitly different baseline.
Freeze semantic prompt bins near 128, 512, and 2048 tokens before execution.
Use existing natural text and record exact lengths. Do not call random-token padding a natural request.
Measure the actual distribution of output lengths rather than forcing uniform output lengths.

Closed-loop concurrency is C1/C4/C8/C16. C1 and C4 are mandatory.
Probe C32/C64 only after capacity and lifetime tests pass.
Closed-loop clients issue a replacement after a request completes. Record any client think time.

Open-loop arrivals use a frozen Poisson seed and a frozen burst trace.
Determine a reference rate `lambda0` from independent native TP4 calibration under the pre-registered SLO grid.
Test offered rates `0.25/0.50/0.75/0.90/1.00/1.10 * lambda0` where safe.
Use identical scheduled arrival timestamps across arms. Do not delay arrivals until the candidate has room.
Record both scheduled and actual send time to detect client overload.
Do not relabel an overloaded rate as sustainable throughput.

Start key comparisons with three independent server starts and ten paired cohort repetitions.
Increase samples if uncertainty remains large. Report the actual count and early termination reason.
For tails, target at least 1,000 completed requests per arm and load point across repeats.
Prefer 5,000 when practical. Smaller runs must label P99 as low-sample exploratory.
The 24-request smoke does not qualify P99 or stable goodput.

## Timing boundaries

Use a client monotonic clock for all client latency metrics.
Record server clocks separately. Clock alignment is required only for cross-host causal attribution.
Use one local clock domain where possible.

| Metric | Exact definition |
| --- | --- |
| E2E | Client request arrival to final completion event, including queue and normal serving overhead |
| TTFT | Client request arrival to first output token |
| TPOT | `(completion - first_token_time) / (generated_tokens - 1)`, for at least two generated tokens |
| ITL | Difference between successive client token timestamps |
| Cohort makespan | Final cohort completion minus first scheduled cohort arrival |
| Requests/s | Successful completed requests divided by the stated measured window |
| Output tokens/s | Actual generated tokens divided by that same stated window |
| SLO goodput | Requests in the arrival cohort that satisfy completion and frozen latency limits, divided by arrival-window duration |

Exclude one-token requests from TPOT only. Report their count and E2E/TTFT.
Report streaming chunks, token counts per chunk, and tied timestamps.
Do not invent token-level ITL within a multi-token network chunk.
Report ITL tails only for observed token intervals or label them as chunk-level intervals.

Measure process cold start, model load, initial graph preparation, and readiness separately.
Normal request-bound capture or staging stays inside client E2E.
For warmed service, freeze the warm-up manifest and graph-reuse policy before measurement.
A recurring per-request graph capture cannot disappear from the headline clock.
Record end-of-request processing and client transport costs rather than assuming they are zero.

Throughput has two explicit windows:

- Arrival-window goodput uses requests submitted during `[a,b)` and drains their outcomes afterward.
- Drained cohort throughput divides those requests by `last_completion - a`.

Do not substitute one denominator for the other.
Timeouts, server errors, OOMs, and missing completions remain failures in the arrival cohort.
Report the timeout value and unfinished requests.

## SLO registration

Choose application limits before candidate measurements.
If no application limit exists, use independent native calibration to freeze a grid.
A proposed grid uses native-calibration P95 TTFT and TPOT multiplied separately by 1.0, 1.5, and 2.0.
Store the resulting numeric seconds and the calibration manifest.
Never recompute the grid from each candidate.
Require successful EOS/stop completion within the cap, plus TTFT and TPOT limits, for the main latency goodput row.
Report truncation-sensitive goodput separately if capped completions count for a different application.
Do not require task labels online. Report benchmark answer quality beside service goodput.

## Uncertainty, speedup, and quality

Use paired request IDs and common arrival seeds.
Resample requests within independent runs. Treat repeated copies of one prompt as a cluster.
Resample server runs as a second level when enough runs exist.
Use at least 5,000 bootstrap draws with a recorded analysis seed.
Do not treat tokens, layers, or recurrent visits as independent accuracy or request samples.
For C1 timing, repeat requests across runs. One request does not provide a request-population CI.

Latency speedup is `T(reference) / T(candidate)` for the same metric, mode, and cohort.
Report raw seconds and `1.xx×` speedup. A slowdown remains `0.xx×`.
Report both same-mode Single speedup and Dynamic speedup against Single Static.
Also compare the strongest qualified same-four-GPU baseline.
Single uses one GPU. Four-rank arms use four GPUs. Report output tokens/s/GPU and memory capacity separately.
Different natural outputs can change work. Show lengths, depths, truncation, and token totals beside timing.
Use frozen-work replay only to attribute that confound, not to replace main generation.

Quality reports correct/N, accuracy%, and paired accuracy differences in percentage points.
Use Single Static and same-mode Single as separate references.
Report baseline-correct/candidate-wrong counts and the opposite counts.
A zero observed delta on 100 MATH questions does not establish small-margin equivalence.
Apply [acceptance_gates.md](acceptance_gates.md) without post-hoc tolerance changes.

## GPU measurements and memory

Clean timing has no per-layer synchronization or heavy profiler.
Use a separate bounded trace for CUDA events, NVTX, Nsight, kernels, and copy activity.
Check traced/untraced output, depth, and work parity. Quantify observer overhead.
Kernel-service sums, busy interval unions, and non-overlapped critical path are different quantities.
Do not sum overlapped compute and collectives into a false E2E percentage.

Record native TP collective type, logical tensor bytes, call count, and selected implementation.
Native custom all-reduce kernels must not become an invented NCCL-only cost or an EP All-to-All cost.
Record PP hidden, residual, token feedback, and exit-KV payload separately.
Software payload is not link wire bytes.

Record per-rank weight storage, logical KV, page pool reserve, active pages, graph workspace, allocator allocated/reserved/peak, and NVML HBM.
Do not sum overlapping memory accounting categories.
Compare capacity using the same safety headroom and sustained request policy, not just configured pool sizes.

## Future raw schema and figures

Request records need arrival/send/first-token/final timestamps, raw outputs, tokens, stop reason, correctness, depth sequence, and policy ID.
Stage records need request/allocation IDs, depth/layer range, queue entry, dispatch, device completion, and publication.
Repair records need source/destination pages, token position, enqueue, receiver ready, first read, bytes, and source lease release.
Environment records need every revision, runtime flag, backend, graph mode, UUID, and software library identity.

Required figures include SLO curves, latency distributions, memory/capacity, communication payload, actual repair overlap, and numerical divergence.
Keep source analysis, modeled bounds, replay, and LIVE E2E in separate plot panels.

## Main-table schema

Every row below remains NOT_RUN in this planning package.

| Mode | Arm | GPUs | GSM correct/N | MATH correct/N | E2E P50/P95/P99 | TPOT P50/P95 | TTFT P50/P95 | ITL P95/P99 | Goodput | Req/s, out tok/s | Speedup | Peak HBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static | Single | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Static | TP4 | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Static | Physical PP4 | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Static | Loop PP4 | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Dynamic | Single | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Dynamic | TP4 | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Dynamic | Physical PP4 | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Dynamic | Loop PP4 sync | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Dynamic | Loop PP4 async ablation | 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
