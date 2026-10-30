# Root causes of B performance regression

## Scope

This report separates three effects:

1. A changed workload and SLO predicate.
2. Additional work from numerical corrections.
3. Transfer granularity and producer scheduling in B itself.

Saved W1 records provide the main long-running evidence.
New GPU measurements use small, finite diagnostic cohorts.
They are not a new serving-capacity campaign.
No new4/8GPU capacity result is inferred from a two-GPU diagnostic.

## A. Workload and SLO changes

The earlier selected condition uses input512, concurrency4 and output cap32.
The later W1 workload uses real ShareGPT prompts, mixed lengths, Poisson arrivals and output cap128.
The old highlighted rule constrains the maximum observed ITL of each request.
The later primary rule constrains request-average TPOT and calibrated TTFT.
Those endpoints are different.

The frozen Huginn B/PD_G1 goodput ratio is1.310× under the old selected SLO.
Its conditional95% CI is[1.125,1.580].
The ratio decomposes into qualified-count ratio1.250000 and cohort-span ratio1.048355.
This accounting is not a1.310× GPU processing speedup.

Rescoring the same old token timestamps under the later SLO reduces the ratio to1.048×.
Against the old colocated DP result, B is0.858× under that later predicate.
The source report preserves both predicates and denominators.
See [the frozen SLO analysis](looped_pd_b_slo_serving/10_why_previous_gain_did_not_transfer.md).

The later W1 result also loses under the old SLO predicate.
Thus the SLO change alone does not explain the reversal.
Prompt lengths, arrival timing, generation duration and batch composition also changed.
The saved studies do not independently vary all those factors.
No causal percentage is assigned to each workload change.

## B. Numerical-correction cost

The previous canonical correction changes all Linear geometries and graph execution.
The new selective correction changes only Down geometry and small-row normalization.
Both can add GPU work, allocations, copies and graph variants.
Their cost must remain in every comparator of the same execution epoch.

Pre-correction and post-correction output sequences can differ.
A cross-epoch latency ratio is therefore not an equal-work speedup.
The new clean comparison applies `down_norm` to PD_G1, B and the diagnostic readiness variants.
The same checkpoint, FA4 backend, fixed depth, prompts, seeds and output cap apply within those comparisons.

`component_timing.py` measures hot native and corrected operators with CUDA events and graph replay.
It alternates native/fixed order and uses10 groups of100 replays.
Norm inputs come from captured failures. Down uses seeded BF16 inputs and real checkpoint weights.
These are component timings, not whole-model E2E estimates.
They cannot be summed into a serving critical path.

| Model | Operator | Real rows | Native mean, microseconds | Corrected mean, microseconds | Corrected/native time ratio |
|---|---|---:|---:|---:|---:|
| Ouro | RMSNorm | 1 | 13.6733 | 19.2063 | 1.405× |
| Ouro | RMSNorm | 8 | 18.5179 | 23.2899 | 1.258× |
| Ouro | Down | 1 | 9.5089 | 20.4899 | 2.155× |
| Ouro | Down | 8 | 8.6469 | 20.5570 | 2.377× |
| Huginn | RMSNorm | 1 | 14.6856 | 26.1429 | 1.780× |
| Huginn | RMSNorm | 8 | 21.8479 | 29.8167 | 1.365× |
| Huginn | Down | 1 | 73.4730 | 92.6111 | 1.260× |
| Huginn | Down | 8 | 75.2484 | 93.0802 | 1.237× |

These are `MEASURED` CUDA-event elapsed times around100 graph replays per group.
They can include GPU gaps between host submissions. They are not kernel-only duration sums.
The small Ouro Down weight fits the hot-cache regime better than the larger Huginn weight.
The table does not predict a full-model cost ratio.
The M16 norm control uses the same reduction path and has nearly equal native/corrected timings.

## C. Transfer fragmentation and scheduling

### Same payload, different partition

| Saved W1 pair | Requests | PD_G1/B payload bytes | PD_G1/B completed fragments | B/G1 fragment ratio | Output agreement |
|---|---:|---:|---:|---:|---:|
| Ouro original | 200 | 131400237056 /131400237056 | 2058 /3242 | 1.575× | 185/200 |
| Huginn original | 200 | 460435430400 /460435430400 | 7183 /11114 | 1.547× | 193/200 |
| Ouro legacy canonical, seed6001 | 400 | 262800474112 /262800474112 | 4116 /6486 | 1.576× | 400/400 |

These are `TRACE_ANALYSIS` results from real GPU records.
The first two rows remain output-confounded.
The third row has exact outputs but only one fully passing independent start.

B sends valid depth/layer ranges when their frontiers advance.
PD_G1 submits completed chunk ranges.
Both eventually transfer the same complete BF16 prompt KV and final hidden state.
Different frontier boundaries change fragment partitioning without reducing the payload.

Fragment count is not descriptor count or CUDA copy count.
B coalesces adjacent descriptors in `FrontierConnector._queue`.
Native Ouro PD_G1 does not apply the same coalescer.
Huginn's boundary connector has a different descriptor path.
More B fragments therefore do not prove greater descriptor submission time.

### Actual source-level producer constraint

RLT `PDStageConnector.can_schedule`, line173, requires an empty unsent fragment queue.
It also requires a completed producer event and available in-flight byte capacity.
`progress`, line342, submits at most one bounded fragment per request per tick.
The byte limit is256MiB in the tested contract.

This rule can delay the next prefill computation while fragments remain unsent.
It does not wait for every submitted transfer to complete before allowing compute.
Calling the original engine fully serialized would be incorrect.
Queued fragments and in-flight transfers are different states.

Producer event ordering, registered-buffer ownership, completion notifications and ACKs remain necessary.
The source retains registered KV and final-hidden storage until both local and remote users finish.
Decode activation rejects incomplete transfer sequences and late writes after readiness.

## Matched-output W1 phase attribution

The strongest saved attribution pair is Ouro canonical seed2026106001.
It has400/400 identical sequences.
Its p95 TTFT changes from876.059ms to1082.067ms.
The difference is206.008ms.
A paired within-run request bootstrap gives95% CI[84.862,282.475]ms.
This interval does not model independent server starts or queue dependence.

| Serial host phase | Mean change, B minus PD_G1, ms |
|---|---:|
| Harness submission | -0.0106845 |
| Coordinator queue | -0.00382335 |
| Reservation/control | +1.43329385 |
| Prefill, including backpressure | +37.71094507 |
| Exposed handoff | +0.83621274 |
| Decode resume through first token | +8.56930057 |
| First-token observation | -0.01264015 |
| Reconstructed mean TTFT change | +48.52260421 |

The arithmetic reconstructs mean TTFT, not p95 TTFT.
The p95 differences must not be added as phase contributions.
The mean exposed handoff is21.028978ms for PD_G1 and21.865191ms for B.
That difference is too small to explain the complete mean TTFT regression alone.
Most of the observed mean change lies inside prefill/backpressure and decode-resume intervals.
Those intervals contain scheduling and synchronization costs. They are not pure kernel execution.

This is an observed phase association, not complete causal attribution.
The saved records do not identify pure DMA time, all scheduler stalls or exact bandwidth contention.
The corrected analysis checks that every request's phase sum reconstructs its TTFT timestamp.
See `saved_phase_pairs.csv` and `phase_delta_attribution.csv`.

## Instrumented host controls

The first instrumented pilots use8 real prompts and output cap8.
Their totals include warmup and the complete process lifetime.
They are not pooled with clean request latency.

| Pilot | PD_G1/B accepted submissions | PD_G1/B accepted descriptors | PD_G1/B host submission seconds |
|---|---:|---:|---:|
| Ouro | 128 /221 | 8270 /3942 | 0.067902 /0.030327 |
| Huginn | 444 /874 | 43642 /49467 | 0.399399 /0.407216 |

The Ouro pilot is a direct counterexample to the claim that more fragments always mean more descriptors or slower submission.
Huginn behaves differently.
Polling counts identify repeated observations. They are not exclusive durations of request stalls.
Host API time can overlap GPU computation and other requests.
It must not be added to GPU copy time or E2E as a separate component.

## Queued-compute diagnostic and generic control

The opt-in queued-compute ablation allows unsent immutable fixed-depth KV fragments to remain queued.
It retains the producer event, fixed-depth checks, in-flight byte limit, storage pins and receiver completion gate.
It does not let Decode consume an incomplete prompt prefix.

The ablation initially changes only B.
That result alone cannot establish a loop-specific advantage.
The same readiness relaxation can also apply to ordinary PD_G1.
The final diagnostic includes this generic control as `PD_G1_queued`.
These variants have small output and byte checks, not full200/400 validation.

The clean finite comparison uses three independent starts per method and model.
Each start handles8 identical prompts with a natural-generation safety cap of8 tokens.
Starts alternate physical pairs4,5 and6,7.
The original three-method order is reversed on the middle start.
The generic queued control runs later and is not fully interleaved with those starts.
This temporal limitation remains explicit in its comparison.
One-second NVML samples check for GPU processes outside the assigned pair.
The experiment does not reserve exclusive CPU cores or eliminate every host background task.
Thus the lane is GPU-isolated, not a proof of zero shared-host interference.

The measured tables are `quiet_latency_raw.csv` and `quiet_paired_comparisons.csv`.
Ratios use PD_G1 or PD_G1_queued from the same declared comparison.
Student-t intervals resample neither tokens nor requests as independent starts.
Only three starts are available, so uncertainty can remain large.
No sustainable request rate follows from these finite cohorts.

### Completed clean finite controls

All cells below use8 source prompts, three independent starts and output safety cap8.
The same eight prompts repeat across starts. There are not24 unique prompts.
Each method agrees with the paired PD_G1 output on24/24 comparisons.
Huginn still fails the separate long qualification.
The p95 columns are percentiles of eight requests per start, then means across starts.
They are not robust population-tail estimates.

| Model | Method | Mean cohort makespan, ms | Mean per-start p95 TTFT, ms | Mean paired makespan ratio versus PD_G1 [Student-t95% CI] |
|---|---|---:|---:|---|
| Ouro | PD_G1 | 1383.386 | 810.776 | 1.000×, reference |
| Ouro | B | 1604.667 | 1044.245 | 0.864× [0.583,1.144] |
| Ouro | B_queued | 1119.146 | 837.678 | 1.246× [1.043,1.449] |
| Ouro | PD_G1_queued | 1292.666 | 849.473 | 1.103× [0.537,1.668] |
| Huginn | PD_G1 | 3236.265 | 2340.539 | 1.000×, reference |
| Huginn | B | 3425.258 | 2328.154 | 0.946× [0.752,1.140] |
| Huginn | B_queued | 3145.961 | 2069.125 | 1.032× [0.797,1.267] |
| Huginn | PD_G1_queued | 3590.278 | 2557.110 | 0.905× [0.666,1.144] |

The ratio is the mean of three paired ratios, not the ratio of the displayed means.
The ordinary B point does not improve makespan in either model.
Its intervals include1.000×, so these small cohorts do not prove a universal slowdown.

Ouro B_queued improves finite makespan against ordinary PD_G1 in this small experiment.
Its p95 TTFT does not show a matching demonstrated improvement.
Against PD_G1_queued, its makespan ratio is1.160× with95% CI[0.557,1.763].
Huginn's corresponding ratio is1.149× with95% CI[0.748,1.550].
Those later generic-control intervals do not establish an additional loop-specific gain.
They also retain the temporal-order limitation stated above.
The queued-readiness result is a conditional mechanism lead, not an accepted B serving improvement.

## Actual GPU activity and overlap

Nsight Systems records CUDA kernel intervals and actual peer-to-peer copy intervals.
Heavy profiling runs are separate from clean latency runs.
Each run captures only the measured serving region with an NVTX range.
The analysis uses interval unions and intersections, not sums of overlapping kernels.

The initial Ouro pilot records856 actual P2P copy events and1854423040 bytes.
The bytes exactly match the measured NIXL payload counter.
The P2P interval union is7.052230ms in a1222.158493ms measured region.
It overlaps the producer kernel union by0.277600ms and the decode kernel union by0.151428ms.
The producer overlap is approximately3.94% of the copy union.

That initial pilot ran beside numerical workers on other pairs.
It is not an isolated W1 throughput or latency comparison.
Final quiet-lane profiles provide the matching PD_G1/B controls.
`configs/GPU_timeline_summaries.json` and `figures/*_timeline.svg` retain their individual results.

| Instrumented quiet run, N8/cap8 | Measured NVTX region, ms | Actual P2P copy events | P2P bytes | P2P union, ms | Overlap with P kernels, ms | Overlap with D kernels, ms |
|---|---:|---:|---:|---:|---:|---:|
| Ouro PD_G1 | 1513.384 | 4135 | 4076109824 | 19.5862 | 0.122720 | 0.671904 |
| Ouro B | 1773.995 | 1890 | 4076109824 | 15.4117 | 0.141824 | 0.881094 |
| Huginn PD_G1 | 4530.548 | 21839 | 14123028480 | 82.9430 | 6.219271 | 20.927012 |
| Huginn B | 3983.660 | 24686 | 14123028480 | 86.4213 | 1.162786 | 9.128898 |

All four actual P2P byte totals match their measured transport-counter deltas.
These use physical GPUs4,5. Visible device0 is P and visible device1 is D.
The producer overlap remains small. B does not improve it in the Huginn profile.
Ouro B uses fewer actual copy events than PD_G1 despite its finer frontier submissions.
Descriptor coalescing and queue boundaries therefore need separate analysis.

The profiler changes elapsed time and scheduling.
For example, its Huginn elapsed-region ordering differs from the clean makespan ordering.
That is why the profile durations do not become serving speedup results.

The GPU activity partition uses four non-overlapping states for each GPU:

- Kernel activity without P2P.
- P2P without kernel activity.
- Simultaneous kernel and P2P activity.
- Neither recorded kernel nor P2P activity.

This partition describes recorded activity, not exclusive request-level E2E components.
The final state is not automatically CPU scheduler overhead.
Other copies, dependencies, graph preparation or idle periods may contribute.
Actual HBM bandwidth and a causally isolated bandwidth-contention effect remain `NOT_MEASURED`.

## E2E percentages

`e2e_component_shares.csv` partitions mean request wall time using serial host timestamps.
Its components sum to100% for each run.
The denominator is mean request E2E, not cohort makespan or GPU busy time.
Prefill includes backpressure. Post-first-token time includes delivery and waiting.
The CSV does not label those intervals as pure computation or pure communication.

GPU overlap percentages use the separate measured NVTX region.
Host cumulative transfer observations are neither pure DMA time nor an additive E2E component.
The report preserves these three distinct denominators.

The following percentages aggregate three clean starts of each finite N8/cap8 condition.
The denominator is summed request E2E wall time across those starts.
They describe the Down-plus-norm runtime, not the later Huginn adapter correction.

| Model and method | Prefill with backpressure | Exposed handoff | D resume to first token | Post-first-token wall time | Other host phases |
|---|---:|---:|---:|---:|---:|
| Ouro PD_G1 | 34.597% | 8.276% | 0.360% | 56.191% | 0.576% |
| Ouro B | 38.162% | 8.217% | 1.719% | 51.308% | 0.594% |
| Huginn PD_G1 | 62.033% | 2.202% | 0.864% | 34.560% | 0.341% |
| Huginn B | 55.329% | 2.931% | 2.118% | 39.346% | 0.276% |

Small negative harness residuals can occur because the timestamp boundaries differ slightly.
The CSV retains those values without clipping. The complete algebraic partition still reconstructs measured E2E.

## Decision boundary

B has not demonstrated a loop-specific, lossless serving-capacity improvement.
Ouro's selective numerical correction passes the finite internal qualification.
Huginn's larger independent tests still fail.
Any queued-readiness benefit must beat the same generic correction in PD_G1.
The final decision uses the completed finite controls without extrapolating a capacity result.
No new A/C method, native-vLLM Loop PP engine or large serving campaign was implemented.
