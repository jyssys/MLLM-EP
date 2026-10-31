# Looped PD Bottleneck Crossover: Final Report

## 1. Final result and scope

The GPU campaign and its automatic analysis ended normally.
The study completed finite capacity measurements and native PD screens for Ouro, Huginn, and a dense AR diagnostic.
It did not establish a tuned P-compute bottleneck with simultaneous D compute, KV, and Decode-SLO reserve.
Phase C therefore did not start.

This is not evidence that Looped models can never become Prefill-heavy.
It is not a measured failure of Dynamic A.
No new Dynamic A policy executed in this campaign.
The architectural explanation remains **INCONCLUSIVE**.

| Question | Final answer | Evidence class |
|---|---|---|
| Can independent P service be slower than D service? | Yes, in measured Ouro configurations. | MEASURED finite service cohorts |
| Did an online run show finite P pressure? | Yes, the untuned Ouro G1 screen was a candidate. | TRACE_ANALYSIS |
| Did that establish a tuned, GPU-compute-bound P stage with safe D spare? | No. Generic tuning and admission effects prevent that conclusion. | Final evidence review |
| Does recurrence itself explain the prior Decode bottleneck? | Not established. | INCONCLUSIVE |
| Does Dynamic Loop Split beat Token Split or DP2 here? | Not measured. The Phase C gate was not met. | NOT_RUN_GATE_NOT_MET |
| Is DuetServe Figure 2 reproduced? | No. The model, lengths, and engine settings differ. | NOT_RUN |

The scope-limited decision is `NO_P_HEADROOM_ON_TESTED_GRID` for both Looped models.
Here, the label means no **verified offloading opportunity** in the tested configurations.
It does not mean zero possible Prefill pressure or zero possible Dynamic A headroom elsewhere.
The earlier provisional gate remains preserved.
The final review is [final_phase_decisions.json](looped_pd_bottleneck_crossover/state/final_phase_decisions.json).

## 2. Critical terminology: Full-Prefill PD still used chunks

Full-Prefill PD specifies computation ownership.
P completes every Prompt token at the full recurrent depth.
D then generates every Decode token at the full recurrent depth.
It does **not** specify that P processes the complete Prompt as one chunk.

The measured native `PD_G1` baseline uses chunked Prefill and native NIXL streaming.
The primary screens use C256.
Generic tuning controls use C2048 with the same arrival stream.
Neither baseline moves unfinished Prefill computation to D.
Neither uses shared-GEMM Prefill/Decode Mixed batching.

| 3,500-token condition | Prompt chunks | P recurrent ownership | Raw evidence |
|---|---|---|---|
| C256 | 13 × 256 + 172 | All R4 or all R32 | First/last boundaries `[0,256]`, `[3328,3500]`, count 14 |
| C2048 | 2,048 + 1,452 | All R4 or all R32 | First/last boundaries `[0,2048]`, `[2048,3500]`, count 2 |
| One complete 3,500-token chunk in PD | NOT_MEASURED | Would remain full P ownership | No such Serving control ran |

The boundary evidence comes from `closed_stats.json` in the G3 screen and tuned-control namespaces.
The DP control uses a different atomic/full-prompt configuration.
An atomic DP result is not an atomic PD result.
The missing whole-prompt PD control limits any general conclusion about naive PD.

## 3. Execution, models, and provenance

Branch: `flashvep/looped-pd-bottleneck-crossover-abc`.
Base commit: `d2050debde78c33c4a77fce5fe3ef342fc5f3694`.
The last committed execution checkpoint before final documentation was `6ac5f4102ba1471a492c26cc72f4a1660d388e36`.
Individual source archives identify subsequent uncommitted worker revisions.
Use each run's archive and hash, not only its parent Git SHA.
The publication commit is the commit that contains this report.

| Model | Checkpoint | Fixed depth | Revision |
|---|---|---:|---|
| Ouro Base 1.4B | `ByteDance/Ouro-1.4B` | R4 | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Huginn-0125 | `tomg-group-umd/huginn-0125` | R32 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| Dense AR diagnostic | `Qwen/Qwen3-1.7B` | Not recurrent | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

All weights and KV remain BF16.
The study retains the qualified numerical fixes, original checkpoints, and full recurrent work.
It uses no early exit, training, approximation, TP, EP, or new Loop PP runtime.
RLT remains at private HEAD `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its preserved diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

The machine has eight H100 80GB GPUs with NV18 peer links.
Clean measurements run one isolated lane at a time.
The study uses the pairs 0/1, 2/3, 4/5, and 6/7 in selected runs.
The GPU UUIDs and topology remain in [initial_gpu.json](looped_pd_bottleneck_crossover/state/initial_gpu.json).
Parallel contention was not quantified, so parallel runs were not accepted as clean timing.

The environment uses Python 3.14.6, PyTorch 2.11/CUDA 12.9, vLLM 0.26, and NIXL 1.4.1.
Native Qwen and Looped RLT use different execution paths and attention backends.
Their absolute speed difference does not isolate recurrence.

## 4. Completion and elapsed time

The study started on 2026-10-31 at 06:40:52 UTC, or 15:40:52 KST.
The resumable GPU queue drained on 2026-10-31 at 15:25:01.896 UTC.
That is 2026-11-01 at 00:25:01.896 KST.
Elapsed time to queue completion was **8 h 44 min 9.9 s**.
Automatic analysis finished at 15:28:19.429 UTC, or 00:28:19.429 KST.
These elapsed times include an inter-queue idle interval of 77.63 minutes.
They are not GPU compute durations.

| Scope | Completed | Failed or missing | Interpretation |
|---|---:|---:|---|
| Final resumable plan | 82 configurations | 1 failed startup | A separate retry completed the failed condition. |
| Eligible clean Serving | 56 runs | 0 incomplete clean runs | Excludes calibration, heavy correctness, profiling, and prepared-Decode diagnostics. |
| Clean Ouro requests | 1,016 | 0 | 24 runs |
| Clean Huginn requests | 1,016 | 0 | 24 runs |
| Clean Qwen requests | 512 | 0 | Eight diagnostic runs |
| Total clean requests | 2,544 | 0 | All offered requests in eligible clean runs completed. |
| Capacity summary | 140 groups | Separate blocked/failed records preserved | Groups contain repeated finite cohorts, not 140 server starts. |

Earlier pilot failures and stale raw status files also remain preserved.
They are not active GPU processes.
The [experiment matrix](looped_pd_bottleneck_crossover/experiment_matrix.csv) records reconciled statuses.
The final report was not generated by the GPU queue.
That manual review and publication step remained unfinished after automatic analysis ended.

## 5. Dataset, lengths, and measurement contract

The dataset is ShareGPT, revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`.
The source SHA256 is `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
The workload uses explicit model-tokenizer prefixes from real conversation text.
It is `LENGTH_MATCHED_CONTROLLED_SHAREGPT_PREFIX`, not an Azure trace or natural full-dialogue distribution.
Generation is greedy with EOS ignored and an exact output cap.
Natural-EOS and downstream answer accuracy were not evaluated.

| Cell | Input tokens | Output tokens | Initial requests/model |
|---|---:|---:|---:|
| G1 | 2,048 | 128 | 24 |
| G2 | 3,072 | 128 | 24 |
| G3 | 3,500 | 128 | 24 |
| G4 | 3,072 | 192 | 24 |
| G5 | 3,500 | 192 | 24 |
| G6 | 3,072 | 256 | 24 |
| C1 | 1,536 | 448 | 24 |
| C2 | 1,900 | 448 | 24 |
| D32, secondary only | 3,500 | 32 | 64 |

Each Looped model also has a five-rate G2 sweep with 64 requests per point.
The central rate has three independent starts.
The enlarged generic engine also has three independent N64 starts per model.
The native AR screens use N64 at each of the eight main cells.
All corresponding Looped controls retain the same Prompt IDs, seeds, and arrival offsets.

The target and realized finite Poisson rates differ.
The initial N24 seed gives a realized rate 1.3423 times its target.
The N64 start-0 seed gives a factor of 1.1451.
The analysis reports both rates and does not normalize traces after the result.
Calibration and test prefixes have zero overlap in the recorded audit.
The test pool contains 32 original held-out records and 32 additional held-out records.
The additional source records remain in `data/heldout_extension32.json`.
Some different source records share a truncated test prefix.
The audit preserves the exact duplicate counts.

TTFT and E2E start at scheduled arrival and end at coordinator delivery.
The path includes IPC and polling, but not HTTP.
Engine startup and initial warm-up occur before the observation window.
Any preparation that occurs inside request service remains in its timing.
Goodput counts passing offered requests over arrival-to-drain wall time.
Finite goodput is not proven sustainable capacity.

## 6. Correctness result

| Check | Ouro | Huginn | Scope |
|---|---:|---:|---|
| Internal barrier output/state groups | 8/8 | 8/8 | Same-engine finite qualification |
| Independent native-PD sequence and Prompt-state qualification | 32/32 | 32/32 | Input 1,536, output cap 16 |
| Long partial-tail qualification | 2/2 | 2/2 | Input 3,500, output cap 8 |
| Enlarged geometry output/state qualification | 24/24 | 4/4 | Full 128-token output and valid Prompt KV/hidden state |
| Original HF / downstream accuracy | NOT_MEASURED | NOT_MEASURED | No lossless original-model claim |

Across the clean matching-input comparisons, **264/264 input/seed/cap groups** have identical output sequences.
This denominator counts comparison groups, not individual requests or 264 independent trials.
All **83 completed RLT work-counter records** have the exact expected full-depth issued rows.
That counter measures active recurrent work, not padded physical GEMM rows.
The Qwen native NIXL qualification passed 2/2 native output comparisons.

No observed internal correctness failure blocked Phase C.
The block was the unverified performance opportunity.
The evidence does not establish original-HF equivalence or GSM8K/MATH accuracy.

## 7. Main measured findings

### 7.1 Generic Prefill tuning changes the apparent bottleneck

At 3,072 input tokens, measured per-request Prefill service changed substantially across the chunk sweep.
Ouro changed from approximately 1.1641 s at C256 to 0.2907 s at C2048.
Huginn changed from approximately 2.2624 s to 1.0236 s.
These are finite measurements, not Loop Split gains.

The following G2 controls use the same 24 requests and identical arrivals per model.
Both PD variants retain native NIXL streaming.
The change is generic chunk/budget tuning from 256 to 2,048.
DP2 uses two independent colocated replicas.
The DP chunk settings differ explicitly and do not constitute a fully tuned production Mixed baseline.

| Model / method | Chunk | Joint SLO passed/N | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | E2E p95 ms | Finite goodput requests/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ouro native PD | 256 | 0/24 | 18,608.22 | 44.81 | 75.54 | 23,194.48 | 0.0000 |
| Ouro tuned native PD | 2,048 | 24/24 | 1,398.53 | 34.80 | 292.36 | 5,652.53 | 0.8249 |
| Ouro DP2 | 4,096 | 24/24 | 321.43 | 35.33 | 241.68 | 4,719.53 | 0.8727 |
| Ouro DP2 + separate-phase chunking | 2,048 | 24/24 | 515.73 | 43.34 | 192.26 | 5,836.35 | 0.8503 |
| Huginn native PD | 256 | 8/24 | 44,262.67 | 103.25 | 113.58 | 57,374.07 | 0.0415 |
| Huginn tuned native PD | 2,048 | 11/24 | 36,486.53 | 104.32 | 114.45 | 49,726.97 | 0.0589 |
| Huginn DP2 | 4,096 | 20/24 | 10,343.59 | 115.99 | 113.33 | 24,058.71 | 0.1268 |
| Huginn DP2 + separate-phase chunking | 2,048 | 20/24 | 11,030.25 | 114.22 | 298.90 | 24,665.92 | 0.1263 |

These are single-start observations, not between-start improvement estimates.
Each empirical p95 has limited N24 tail evidence.
Descriptive Wilson intervals appear in `serving_runs.csv`.
Queued requests are dependent, so those intervals do not establish a method-improvement confidence interval.

Tuning can improve TTFT and average TPOT while worsening token tails.
At G2, Ouro's population ITL p99 increased from 75.54 to 292.36 milliseconds.
Only 7/24 tuned requests kept every observed ITL within 100 ms.
The primary criterion is TTFT plus request-average TPOT, not every-token latency.
The report does not switch between these criteria to declare success.

### 7.2 Independent capacity does not guarantee online D spare

Ouro G1 at C256 had finite P pressure with apparent D reserve.
At the same arrivals, C2048 changed primary SLO attainment from 4/24 to 23/24.
The original candidate therefore does not establish a partition-specific opportunity beyond generic tuning.

An enlarged Ouro engine uses 32 slots and 22,528 KV blocks.
At 3,500/128, prepared native-PD Decode completed at 3.706 requests/s.
Nevertheless, its online N64 runs passed only 0/64, 1/64, and 5/64 SLOs.
Their TPOT p95 values were 99.06, 101.91, and 104.61 ms.
The frozen Ouro TPOT limit is 53.23 ms.
D KV-credit refusals also occurred.

Prepared simultaneous Decode and online staggered readiness have different effective batching.
In the qualified Ouro B24 diagnostic, simultaneous TPOT p50 was 48.776 ms.
Staggered readiness increased it to 86.469 ms without timed Prefill or incoming transfer.
Mean valid recurrent rows per submission changed from 23.828 to 12.577.
Mean valid Coda rows changed from 23.303 to 5.532.
These are logical active rows, not fixed graph or padded GEMM dimensions.
The diagnostic also changes runtime control work.
It does not isolate all of that work from GPU batch efficiency.
Its observation-window ratio is not a Serving speedup.

### 7.3 Huginn is strongly constrained by reservation in this runtime

Every one of the 22 completed Huginn screening/control classifications contains the declared D-KV-admission condition.
These include the short-output diagnostic and enlarged-pool runs.
This does not imply that every moment is KV-blocked or that all waiting has one cause.
It shows that D reservation prevents a verified spare-capacity classification.

The default G2 screen assigns 47.14% of summed request E2E to host-observed KV/slot admission waiting.
Its generation interval accounts for another 43.84%.
Pure GPU compute and host scheduling are not separately identified within those intervals.

### 7.4 The short-output diagnostic did not rescue the admission gate

| Model, 3,500/32 | Target requests/s | Joint passed/N | TTFT p95 ms | TPOT p95 ms | D-credit-refusal observation fraction |
|---|---:|---:|---:|---:|---:|
| Ouro | 2.5252 | 3/64 | 16,548.07 | 130.01 | 49.43% |
| Huginn | 0.6195 | 15/64 | 43,136.27 | 120.08 | 94.56% |

The refusal fraction is sampled admission-reason occupancy, not an exclusive time share.
The diagnostic retains the frozen model SLOs from G2 calibration.
It does not replace the primary 128–256 output grid.

## 8. KV geometry and the runtime contribution

| Model | Valid planes | Allocated planes | Valid KV bytes/token | Allocated KV bytes/token |
|---|---:|---:|---:|---:|
| Ouro R4 | 96 | 96 | 786,432 | 786,432 |
| Huginn R32 | 132 | 256 | 2,787,840 | 5,406,720 |
| Qwen3-1.7B | 28 | 28 | 114,688 | 114,688 |

This table is **STRUCTURAL_ACCOUNTING**, not measured reclaimed memory or speedup.
Huginn's unused boundary-depth planes represent 48.4375% of the allocated plane count.
The study did not implement compaction.
Native NIXL already excludes unused boundary planes from the relevant transfer path.
The allocation difference is not equivalent wasted network traffic.
The models also differ in KV-head count, head dimension, and logical depth.
Parameter count alone does not determine legal Decode concurrency.

Shared weights do not permit different recurrent depths to share the same KV values.
However, this storage accounting does not prove that recurrence inevitably causes a Decode bottleneck.
The current layout, reservation policy, workload, and batch readiness also matter.

## 9. E2E percentages and profiling limits

The requested percentage report is available as [host_e2e_component_table.md](looped_pd_bottleneck_crossover/host_e2e_component_table.md).
The [aggregate CSV](looped_pd_bottleneck_crossover/e2e_component_shares_aggregate.csv) preserves exact values.
Every reconstructed request breakdown totals 100% within floating-point error.

| Host interval, G2 N24 | Ouro C256 | Ouro C2048 | Huginn C256 | Huginn C2048 |
|---|---:|---:|---:|---:|
| KV/slot admission | 13.35% | 0.02% | 47.14% | 43.82% |
| Observed P service, not pure compute | 57.03% | 13.22% | 8.33% | 4.87% |
| Exposed observed handoff | 0.94% | 1.21% | 0.06% | 0.28% |
| Observed generation | 24.74% | 83.64% | 43.84% | 50.37% |
| Other measured host intervals | 3.94% | 1.92% | 0.63% | 0.66% |

Each percentage divides summed component time by summed request E2E.
It is not a percentage of a cohort's exclusive GPU critical path.
Observed P service includes host control and possible queueing.
Pure GPU compute, pure DMA, and exclusive synchronization percentages remain **NOT_MEASURED** for clean Serving.
The report does not invent dispatch/computation shares from cumulative counters.

Separately profiled kernel unions are real CUPTI observations.
Heavy tracing changes latency substantially.
Its kernel-busy fraction cannot serve as clean-server utilization.
CUDA forward spans can contain submission gaps.
Cumulative transfer durations can overlap computation and each other.
None of these quantities can be added without an overlap-aware critical-path model.

The transport counters below exclude warm-up through before/after counter differences.
Each row covers 24 G2 requests.
The last column is cumulative host-observed completion time, not exclusive DMA time.

| Model / chunk | Transferred bytes | Completed transfer count | Host-observed cumulative seconds |
|---|---:|---:|---:|
| Ouro C256 | 57,982,156,800 | 888 | 32.447 |
| Ouro C2048 | 57,982,156,800 | 888 | 6.822 |
| Huginn C256 | 205,541,360,640 | 3,168 | 28.284 |
| Huginn C2048 | 205,541,360,640 | 3,096 | 22.768 |

Changing a Prompt chunk boundary does not necessarily change the wire transfer count proportionally.
The equal payloads also do not imply equal exposed handoff latency.
Do not interpret the cumulative-time differences as standalone DMA speedups.

## 10. SLOs, rate sweeps, and uncertainty

| Model | TTFT limit ms | Request-average TPOT limit ms | Secondary ITL limit ms |
|---|---:|---:|---:|
| Ouro | 2,576.074 | 53.233 | 66.031 |
| Huginn | 7,567.570 | 204.961 | 243.072 |
| Qwen diagnostic | 184.982 | 8.631 | 11.490 |

Eight disjoint serial calibration requests determine each model's limits.
The limits remain frozen during held-out evaluation.
The sensitivity grid uses calibration multipliers 1.5, 2, and 4.
Absolute ITL limits are 50, 100, 150, 200, and 250 ms.
Population ITL p99 and request-maximum ITL are separate metrics.

The completed C256 G2 sweeps have no point with at least 90% joint attainment.
The lowest tested rates are 0.3585 requests/s for Ouro and 0.0630 for Huginn.
Their attainment is 26/64 and 39/64, respectively.
This does not establish zero capacity.
Lower-rate searches and full tuned-engine capacity sweeps remain unmeasured.

Three-start mean finite goodput at the central C256 rate was 0.0268 requests/s for Ouro.
Its 95% start-level interval was [0.0000, 0.0857].
Huginn's corresponding mean was 0.0110 requests/s, with interval [0.0000, 0.0262].
These wide intervals show limited repeatability evidence.
They do not estimate Dynamic A speedup.
See [independent_start_intervals.md](looped_pd_bottleneck_crossover/independent_start_intervals.md) for all four three-start settings.

## 11. Dense AR and the DuetServe comparison

Qwen3-1.7B completed all eight native-vLLM NIXL screens, with 512 requests.
Its independent capacity also often favors P over D in the tested length range.
For example, at B32 and 3,072/128, its finite post-first-sample capacity ratio was approximately 2.27.
Its larger-batch token budget differs from its B8 budget.
This is diagnostic evidence, not a matched-kernel architecture experiment.

Native NIXL exposes actual remote block descriptors and completed receive events.
All 64 G2 requests report 3,072 cached Prompt tokens on D.
Its returned process-lifetime telemetry records 23,253,221,376 transferred bytes over 66 observations, including two warm-ups.
These are not 64-request exclusive measured-window counters.
The AR P worker returns empty counters, which mean unavailable telemetry, not zero traffic.

[DuetServe](https://proceedings.mlr.press/v306/gao26m.html) uses Qwen3-8B, two H100 GPUs, 8,000 input tokens, and 200 output tokens in the verified motivation setting.
The source audit and PDF hash remain in `configs/duetserve_primary_source_audit.json`.
This study uses a smaller AR model and a common input grid capped at 3,500 tokens.
It does not replicate that configuration.
DuetServe's spatial P/D isolation is also not Dynamic Recurrent Loop Split.

**Exact 4K/100 was not tested.**
Neither Ouro 8K/100 nor 8K/200 was tested.
Huginn cannot support input 4,096 plus 100 output tokens within its 4,096-token context.
At a reserved output length of 100, its maximum Prompt length is 3,996 tokens.
The study must not use its common-support grid to claim an all-context architectural limit.

## 12. Phase decisions and omitted work

| Phase / model | Status | Reason |
|---|---|---|
| A, Ouro | PASS_FINITE_CHARACTERIZATION | Valid independent P/D cohorts, tuning, and state gates completed. |
| A, Huginn | PASS_FINITE_CHARACTERIZATION | Same scope, with reservation and layout limits recorded. |
| A/B, Qwen | PASS_DIAGNOSTIC_ONLY | Real native NIXL runs completed. Engine and architecture confounds remain. |
| B, Ouro | CONDITIONAL / NO_P_HEADROOM_ON_TESTED_GRID | Finite untuned P candidate, but no verified tuned compute-bound/D-spare regime. |
| B, Huginn | NO_P_HEADROOM_ON_TESTED_GRID | D admission pressure prevents validation of spare capacity. |
| C, Ouro | NOT_RUN_GATE_NOT_MET | No new static or online migration comparison. |
| C, Huginn | NOT_RUN_GATE_NOT_MET | Same gate, assessed independently. |
| Architectural recurrence claim | INCONCLUSIVE | No matched-engine causal control or faithful long-context replication. |

The unexecuted C matrix includes Loop-Static, Loop-Dynamic, Token-Static, Token-Dynamic, and the full-cut Snapshot control.
DP2, DP2+Chunked, and native PD were measured as Phase B controls.
Their measurement does not complete the conditional C matrix.
No new online cut decisions or nonfull continuations exist in this study.
No oracle E2E speedup is reported.
Independent service minima do not form a valid contention-aware Serving oracle.

## 13. What the evidence does and does not justify next

The results justify further **baseline diagnosis**, not a large Dynamic A implementation now.
A minimal next campaign should first test whole-prompt PD and a tuned chunk sweep at the same native transport.
Ouro can then test supported longer inputs, such as 8K, with short controlled output.
A faithful Qwen3-8B 8K/200 diagnostic would separately test the DuetServe motivation setting.
Huginn requires a distinct near-context-limit study with valid KV reservation and explicit memory accounting.
These are proposed experiments, not completed results or guaranteed positive regimes.

Before continuing Dynamic A, require actual P backlog near measured service capacity.
Also require D compute reserve, D KV credit, and protected Decode SLOs.
Use clean timing and independent starts.
Then compare static Loop and Token offloading before a new online controller.
Do not weaken the admission gate merely to increase nonfull-cut counts.

Generic logical-layer cuts include recurrent boundaries.
At the same boundary, the execution mechanisms can be equivalent.
This study neither implements Layer Split nor proves Loop-specific novelty.
It also does not prove that Token Split requires small chunks.

## 14. Files, reproduction, and resource state

- [Phase A report](looped_pd_phase_a_independent_capacity.md).
- [Phase B report](looped_pd_phase_b_bottleneck_crossover.md).
- [Phase C gate report](looped_pd_phase_c_dynamic_a_headroom.md).
- [Reproduction commands and source pins](looped_pd_prefill_decode_bottleneck_reproduction.md).
- [Complete screens](looped_pd_bottleneck_crossover/completed_screen_tables.md).
- [Complete rate sweep](looped_pd_bottleneck_crossover/completed_rate_sweep_tables.md).
- [Raw evidence, CSVs, profiles, and figures](looped_pd_bottleneck_crossover/).
- [Failures and measurement limits](looped_pd_bottleneck_crossover/failures.md).

The request analysis is published as `per_request_analysis.json.gz` without information loss.
The uncompressed 185.7 MB local file remains intact but exceeds GitHub's single-file limit.
Original per-run CSV/JSONL and output token IDs remain separately available.
No previous raw evidence, checkpoint, staged work, or private runtime was removed.

At final audit, the research queue and analysis processes were absent.
Approved `utilize.py` workers were running on physical GPUs 0–7.
All eight GPUs reported 100% utilization in that sample.
These are restored burners, not continuing research experiments.
The source/index preservation guard passed for 122,717 unrelated entries.
