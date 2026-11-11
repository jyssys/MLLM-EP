# Huginn R32 Loop28 Decode-tail diagnosis

## Result

The reported Loop28 result did not use an online Decode-safe execution budget.
It used a forced static cut and independent prompt execution on D.
Ready-Decode priority did not prevent prompt work from interfering with an already submitted Decode operation.

The strongest measured association is the longer Decode operation enclosure when prompt work overlaps that enclosure.
The effect remains at the same actual Decode batch size.
Earlier Decode arrivals also increase batch size, but that change does not explain the entire observed increase.

Specific SM, HBM, cache, launch, and device-wait contributions remain `NOT_IDENTIFIED`.
This investigation does not prove an intrinsic shortage of physical D idle capacity.
It identifies unsafe overlap in this static execution policy and a smaller contribution consistent with changed batch composition.

This investigation changes neither serving policy nor SLO.
The original results remain frozen at commit `21a067b8bf2242bb8a16fc8c59ed59193e481f54`.

## Scope and evidence

- Model: original BF16 Huginn-0125, fixed R32, input 2048, output 16.
- Primary evidence: original N128 held-out seeds 2026110951, 2026110952, and 2026110953.
- New diagnosis: N64, original arrival seed 2026110940, QPS 1.971, P/D slots 2/8, GPU pair 4/5.
- New controls use the first 64 original controlled prompts and arrival offsets.
- All new N64 outputs match the frozen Native reference. Both workers exited normally.
- The serialized counterfactual passed N8 output, KV, hidden-state, work, and lifetime qualification.
- These short controls do not replace the original N128 comparison or establish sustainable capacity.

Evidence directory: [huginn_loop28_decode_tail_diagnosis_20261111](huginn_loop28_decode_tail_diagnosis_20261111/).
The directory contains frozen configurations, source snapshots, worker logs, raw timing, analysis, and figures.

## 1. Original symptom

The original request logs reproduce the reported ratios.
Ratios below use the geometric mean of three paired starts.

| Method / Native-path Zero | TTFT p95 ratio | ITL p99 ratio | TPOT p95 ratio |
|---|---:|---:|---:|
| Loop28 full prompt | 0.488528 | 1.708787 | 1.209596 |

The per-start ITL ratios were 1.735918, 1.705531, and 1.685292.
This is not a single-run tail anomaly.

Original Loop31 confirmation used 512-token tiling.
Its paired ITL p99 ratio was 1.102679, with bootstrap 95% CI [1.102514, 1.102833].
Its TTFT p95 ratio was 0.827212, with 95% CI [0.789333, 0.861885].
Older full-prompt Loop31 results are secondary diagnostics, not this confirmed arm.

Source: [existing_evidence.json](huginn_loop28_decode_tail_diagnosis_20261111/analysis/existing_evidence.json).

## 2. Execution budget: absent, not an identified prediction overrun

The static configurations set `intervention_method='native'` and a fixed `native_partial_cut`.
The bootstrap installs completion-aware admission only when `intervention_method != 'native'`.
Therefore, the reported static arms never executed that admission policy.

Copied fields such as `require_whole_suffix_budget=True` did not create an active budget check.
A one-loop quantum cap also did not create a Decode deadline.
There is no logged online prediction to compare with a hypothetical 30 ms budget in these arms.

Classification: `NO_ONLINE_EXECUTION_BUDGET`, not a quantified prediction-overrun error.
The duration of a diagnostic quantum is not an admission prediction.

Sources: [bootstrap](../huginn_net_benefit/bootstrap/sitecustomize.py), [frozen configurations](huginn_loop28_decode_tail_diagnosis_20261111/configs/).

## 3. Clean short controls reproduce the trade-off

These values use marker-free N64 request logs from one matched arrival trace.
Units are ms.

| Method | TTFT p95 | TPOT p95 | ITL p99 | Maximum ITL | E2E p95 |
|---|---:|---:|---:|---:|---:|
| Native-path Zero | 6618.743 | 80.339 | 88.401 | 98.220 | 7809.366 |
| Loop28 full prompt | 3000.462 | 99.767 | 159.540 | 318.594 | 4444.446 |
| Loop28 tile256 | 3016.918 | 99.860 | 141.869 | 149.954 | 4485.158 |
| Loop31 tile512 | 5482.453 | 85.194 | 102.784 | 105.697 | 6755.163 |

Tiling reduces Loop28 ITL p99, but does not restore Native-relative TPOT.
All four p99 values satisfy the frozen 192.821 ms ITL gate.
The full-prompt control has an individual ITL above that threshold.
An absolute p99 pass is not Native-relative Decode protection.

Source: [new_diagnostics.json](huginn_loop28_decode_tail_diagnosis_20261111/analysis/new_diagnostics.json).

## 4. Same-batch operation timing

CUDA Events enclose actual model operations on their execution streams.
An enclosure includes possible device waits, launch gaps, and host submission gaps.
It is not exclusive kernel execution time.
“Overlap” below means overlapping operation enclosures, not proven simultaneous kernels.

| Actual batch and condition | Operations | Mean recurrent enclosure, ms |
|---|---:|---:|
| Zero B2, no prompt overlap | 432 | 71.600 |
| Loop28 full B2, no prompt overlap | 46 | 71.449 |
| Loop28 full B2, prompt overlap | 38 | 136.631 |
| Loop28 tile256 B2, prompt overlap | 54 | 95.599 |
| Loop31 tile512 B2, prompt overlap | 75 | 88.828 |
| Serialized Loop28 B2 | 58 | 71.515 |

Full-prompt Loop28 adds approximately 65.2 ms to the mean B2 enclosure in the overlapping subset.
The corresponding non-overlapping B2 enclosure remains near Zero.
This finding rejects batch-size increase as the sole explanation.

Serialization orders model operations by CUDA events.
It removes permission for operation overlap without changing model arithmetic.
It restores the intrinsic B2 enclosure, but does not restore total ITL.
The serialized control still has ITL p99 155.247 ms and TPOT p95 103.004 ms.
Waiting can move before the measured operation instead of disappearing.
This is a diagnostic counterfactual, not an adopted optimization.

The following values describe prompt-operation enclosures.

| Method | Enclosure p95, ms | Maximum enclosure, ms |
|---|---:|---:|
| Loop28 full | 32.554 | 33.157 |
| Loop28 tile256 | 8.022 | 11.226 |
| Loop31 tile512 | 12.524 | 15.124 |

These values are operation enclosures, not non-preemptive kernel maxima.

## 5. Scheduling and workload composition

The explicit priority policy selects ready Decode before prompt work.
However, deferred CODA collection can leave a submitted Decode cohort pending.
Independent prompt work can execute during that interval.
The policy neither preempts running kernels nor reserves a verified future Decode-safe interval.

The original held-out seed 2026110951 also shows a real batch-composition change.
Central half-second submission histograms contain these recurrent Decode batches:

| Method | B1 submissions | B2 submissions | B3 submissions | B4 submissions |
|---|---:|---:|---:|---:|
| Zero | 11 | 442 | 41 | 0 |
| Loop31 tile512 | 1 | 384 | 102 | 0 |
| Loop28 full | 0 | 21 | 376 | 5 |

Loop28 completes Prefill sooner and exposes more Decode work earlier.
Actual batches increase from predominantly B2 to predominantly B3.
Without prompt overlap, B3 recurrent enclosures are approximately 77 ms instead of B2's 71.5 ms.

A descriptive reweighting applies Zero's batch-specific mean costs to the candidate batch mix.
For Loop28 full, Zero's mean changes from 71.792 milliseconds to 75.637 milliseconds.
The observed candidate mean on common support is 85.295 milliseconds.
Common support covers 948 of 960 request-token intervals.
This is not randomized attribution and does not assign causal percentages to ITL p99.

Prompt graph selection hits were present in the original compared starts.
Missing prompt graph coverage does not explain these specific results.
Full Loop28 also moves final prompt CODA work to D.
Recurrent-row counts alone do not describe that additional work.

Sources: [deferred_coda.py](../huginn_net_benefit/deferred_coda.py), [decode_priority.py](../huginn_net_benefit/decode_priority.py), [stream_route.py](../huginn_net_benefit/stream_route.py).

## 6. Contribution accounting and unidentified causes

The next table is an additive **mean observation account** for the instrumented N64 controls.
Each row uses request-level output intervals before calculating the mean.
The columns are not additive p99 components or causal effect estimates.

| Difference from Zero | Mean ITL change | Model enclosure change | Between-enclosure change | Output-observation change |
|---|---:|---:|---:|---:|
| Loop28 full | +15.297 ms | +13.641 ms | +1.260 ms | +0.397 ms |
| Loop28 tile256 | +17.328 ms | +15.713 ms | +1.444 ms | +0.171 ms |
| Loop31 tile512 | +7.828 ms | +5.585 ms | +2.256 ms | −0.012 ms |

The dominant observed location is inside model-operation enclosures.
That location does not distinguish slower kernels from stream waits or intra-operation submission gaps.

| Candidate cause | Result | Quantitative limit |
|---|---|---|
| Predicted execution-budget overrun | Not applicable to these static arms | No active budget or prediction existed |
| Prompt/Decode execution interference | Strong same-batch and serialization evidence | B2 enclosure 71.449 → 136.631 ms with prompt overlap |
| Specific SM/HBM/cache contention | `NOT_IDENTIFIED` | No hardware resource counters |
| Scheduling outside model enclosures | Observed, smaller mean location | +1.260 ms for full Loop28 |
| Larger Decode batches | Observed | Descriptive common-support cost change +3.845 ms |
| Causal percentage of original ITL p99 for each class | `NOT_IDENTIFIED` | No additive tail decomposition or randomized batch control |

## 7. Profiler sensitivity and actual kernel evidence

Native A/A ITL p99 differed by +0.069%, but maximum ITL differed by +69.26%.
Native A/A TTFT p95 differed by +6.12%.
Maximum ITL is unstable in this short cohort.

The final event observer changed Zero ITL p99 by +1.69% and maximum ITL by +5.15%.
The strict two-sided 5% screen therefore fails for Zero.
Full Loop28's maximum changed by approximately −50%, so that pair also fails the strict screen.
Only Loop31 passes all four point checks in that event-observer comparison.
These event traces support diagnosis, not qualified clean tail attribution.

A warmed activity-only CUPTI collector reduced in-measurement activation from approximately 75 ms to 0.041 ms.
The final kernel captures contain no dropped or invalid records.
They cover only the final imported-request tail, not the full cohort or central P-backlog interval.
Their strict maximum-ITL sensitivity screen still fails.

The Zero captured tail contains 1.029 s of kernel union in 1.250 s.
These are actual instrumented kernel intervals, not a qualified clean idle estimate.
Mapped Zero B2 Decode operations contain approximately 66.6 ms of kernel union within 72.5 ms enclosures.
The remaining approximately 5.9 ms is not automatically host delay or reusable slack.
Full Loop28 stream-role lookup occurred inside temporary prompt-stream routing.
Its aliased role mapping cannot identify exclusive Decode kernel cost reliably.
That quantity remains `NOT_IDENTIFIED`.

This report does not claim clean physical idle, SM allocation, HBM saturation, or cache contributions.
Source: [attribution.json](huginn_loop28_decode_tail_diagnosis_20261111/analysis/attribution.json).

## 8. Verdict and next diagnostic

The most supported explanation is unbudgeted prompt execution interfering with in-flight Decode operations.
Larger Decode batches contribute, but cannot explain the same-batch enclosure increase.
Smaller tiles reduce the worst overlapping unit, while repeated prompt work still increases average Decode cost.
Serialization changes where waiting occurs and does not remove the total occupancy cost.

The absent safety gate is an implementation-policy issue.
Whether meaningful migration can preserve Native-relative Decode tails is still a physical-capacity and completion-economics question.
This diagnosis alone cannot declare that question structurally impossible.

The requested Output16/4 capacity diagnosis starts only after this report.
It will not start another scheduler optimization campaign.
All new evidence uses a separate namespace. Original code, results, and SLOs remain unchanged.

Figures: [clean comparison](huginn_loop28_decode_tail_diagnosis_20261111/figures/clean_diagnostic.png), [operation timeline](huginn_loop28_decode_tail_diagnosis_20261111/figures/operation_timeline.png).
The operation timeline shows actual CUDA Event enclosures, output delivery, and a post-hoc frozen threshold.
Blank regions are not measured GPU idle.
