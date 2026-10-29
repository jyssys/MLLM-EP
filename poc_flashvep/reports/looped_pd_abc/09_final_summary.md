# Final summary: Dense Looped PD A/B/C

Finalized with explicit limitations on 2026-10-29T11:01:18.347826+00:00.
Scope: Ouro Base 1.4B R4 and Huginn-0125 R32, BF16, fixed depth, vLLM-RLT.
Dory is deferred by the user. Its historical results remain unchanged.
The GPU campaign finished at 19:26 KST on 2026-10-29.
The final review uses saved measurements. It starts no new inference experiment.

## Decision

No track receives GO for a lossless, original-model-equivalent serving improvement.
A and B retain narrow bounded-workload signals. They receive CONDITIONAL, not deployment acceptance.
C fails the batched natural-output gate. Its service-time ablation also lacks an established incremental gain.

| Track | Ouro | Huginn | Overall |
| --- | --- | --- | --- |
| A | NO_GO_PERFORMANCE | CONDITIONAL | CONDITIONAL |
| B | CONDITIONAL | CONDITIONAL | CONDITIONAL |
| C | NO_GO_CORRECTNESS | NO_GO_CORRECTNESS | NO_GO_CORRECTNESS |

## Main measured E2E comparison

Each row uses three independent starts and 24 requests per start.
Thus N=72 means 72 evaluations of 24 paired prompts, not 72 distinct prompts.
Inputs contain repeated synthetic text. EOS remains enabled. Output cap is 32.
The comparator is the fastest measured equal-two-GPU baseline by pooled p95 E2E.
Selection is retrospective. It is not a deployable oracle.

The 95% intervals resample starts and paired request IDs with 3,000 draws, seed 20261029.
They are conditional intervals, not simultaneous or selection-adjusted intervals.

| Model | Candidate | Input / concurrency / cap | Best measured E2E baseline | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup [95% CI] | Exact outputs / evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | A | 512 / 4 / 32 | DP | 3295.142 | 3849.766 | 0.856× [0.817–0.886×] | 72/72 |
| huginn | B | 512 / 4 / 32 | DP | 3295.142 | 4065.456 | 0.811× [0.761–0.896×] | 72/72 |
| ouro | A | 512 / 4 / 32 | DP_Mixed | 1070.138 | 1215.311 | 0.881× [0.812–1.277×] | 72/72 |
| ouro | B | 512 / 4 / 32 | DP_Mixed | 1070.138 | 981.602 | 1.090× [1.012–1.153×] | 72/72 |

### Supporting latency and throughput measurements

All times below use milliseconds. Rates use summed cohort spans, not mean reciprocal request latency.
The complete mean/p50/p95/p99 table, physical GPU lists, EOS counts, and actual lengths are in `final_serving_metrics.csv`.

| Model | Method | TTFT p50 / p95 | TPOT mean / p95 | E2E p50 / p95 | Requests/s | Output tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | DP | 178.020 / 272.751 | 23.572 / 29.588 | 865.107 / 1102.345 | 3.974 | 127.158 |
| ouro | DP_Mixed | 215.597 / 389.469 | 21.991 / 28.289 | 935.167 / 1070.138 | 4.101 | 131.229 |
| ouro | PD_G1 | 319.687 / 335.425 | 20.739 / 24.593 | 937.068 / 1172.164 | 4.208 | 134.649 |
| ouro | A | 212.006 / 432.979 | 20.984 / 25.249 | 833.769 / 1215.311 | 4.530 | 144.949 |
| ouro | B | 345.189 / 350.261 | 20.018 / 20.322 | 962.476 / 981.602 | 4.181 | 133.805 |
| huginn | DP | 354.440 / 461.316 | 89.184 / 94.008 | 3120.045 / 3295.142 | 1.257 | 40.219 |
| huginn | DP_Mixed | 380.937 / 642.409 | 89.962 / 97.588 | 3170.016 / 3537.789 | 1.234 | 39.495 |
| huginn | PD_G1 | 836.094 / 1013.668 | 97.739 / 108.620 | 3844.931 / 4309.330 | 1.028 | 32.900 |
| huginn | A | 625.924 / 723.444 | 99.949 / 104.175 | 3821.307 / 3849.766 | 1.072 | 34.306 |
| huginn | B | 735.775 / 1009.013 | 94.521 / 97.796 | 3742.032 / 4065.456 | 1.078 | 34.491 |

## Strong-service ablation for C

This comparison uses P2:D2 on four GPUs, 48 natural questions per start, concurrency 8, and cap 64.
N=144 means three evaluations of the same 48 questions.
The source is a local GSM8K training-validation subset, not the GSM8K test benchmark.
The following latency ratios are output-confounded. They are not accepted lossless speedups.

| Model | Candidate | Input / concurrency / cap | Best measured E2E baseline | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup [95% CI] | Exact outputs / evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | C_loop | natural / 8 / 64 | C_service | 4009.910 | 4086.386 | 0.981× [0.716–1.843×] | 125/144 |
| huginn | C_loop | natural / 8 / 64 | C_service | 6741.919 | 6635.428 | 1.016× [0.977–1.093×] | 139/144 |

The strongest full-budget E2E controls are also retained:

| Model | Candidate | Input / concurrency / cap | Best measured E2E baseline | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup [95% CI] | Exact outputs / evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | C_loop | natural / 8 / 64 | C_PD1 | 6297.944 | 6635.428 | 0.949× [0.923–0.966×] | 137/144 |
| ouro | C_loop | natural / 8 / 64 | C_PD3 | 2316.379 | 4086.386 | 0.567× [0.478–1.111×] | 128/144 |

## Frozen-grid SLO signals and negative results

The SLO requires normal bounded termination, TTFT within its limit, every observed ITL within its limit, and E2E at most 30s.
A length stop is valid only for this capped workload. It does not prove a complete answer.
Goodput equals qualified requests divided by summed cohort spans.
The table includes the absolute-150ms cell for both models, not only the favorable Huginn result.
The complete frozen grid is `final_strongest_SLO_grid.csv`.

These cells are exploratory multiple comparisons. No new SLO was selected for acceptance.

| Model | Candidate | Input / cap | ITL / TTFT limits ms | Best goodput baseline | Baseline / candidate qualified | Baseline / candidate requests/s | Ratio [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | A | 512 / 32 | 150.000 / 5000 | PD_G1 | 56 / 67 of 72 | 0.800 / 0.998 | 1.248× [1.009–1.495×] |
| huginn | B | 512 / 32 | 150.000 / 5000 | PD_G1 | 56 / 70 of 72 | 0.800 / 1.048 | 1.310× [1.125–1.580×] |
| huginn | C_loop | natural / 64 | 760.908 / 5000 | C_PD1 | 144 / 144 of 144 | 3.030 / 2.906 | 0.959× [0.934–0.978×] |
| huginn | L128_A | 512 / 128 | 150.000 / 5000 | L128_PD_A_control | 36 / 44 of 48 | 0.237 / 0.284 | 1.197× [0.793–1.742×] |
| huginn | L128_B | 512 / 128 | 150.000 / 5000 | L128_PD_A_control | 36 / 47 of 48 | 0.237 / 0.315 | 1.325× [1.070–1.786×] |
| ouro | A | 512 / 32 | 150.000 / 5000 | DP_Mixed | 71 / 64 of 72 | 4.044 / 4.026 | 0.996× [0.843–1.142×] |
| ouro | B | 512 / 32 | 150.000 / 5000 | DP_Mixed | 71 / 72 of 72 | 4.044 / 4.181 | 1.034× [0.975–1.143×] |
| ouro | C_loop | natural / 64 | 173.728 / 5000 | C_DP4_Mixed | 136 / 85 of 144 | 4.121 / 2.315 | 0.562× [0.475–0.637×] |
| ouro | L128_A | 512 / 128 | 150.000 / 5000 | L128_DP_Mixed | 47 / 44 of 48 | 1.400 / 1.323 | 0.945× [0.744–1.107×] |
| ouro | L128_B | 512 / 128 | 150.000 / 5000 | L128_DP_Mixed | 47 / 48 of 48 | 1.400 / 1.390 | 0.992× [0.969–1.057×] |

Huginn B at P512/C4, cap32, gives 1.310× bounded goodput at ITL150ms and TTFT5s.
Its conditional 95% interval is 1.125–1.580×. Counts are 70/72 versus 56/72.
Huginn A gives 1.248× in the same cell, with a 1.009–1.495× interval.
At cap128, Huginn B gives 1.325×, with a 1.070–1.786× interval and counts 47/48 versus 36/48.
These signals do not establish sustainable arrival-rate capacity.

They coexist with worse E2E than the best DP/Mixed control.
Ouro's P512/C4 B latency signal is 1.090×, or an 8.27% p95 reduction.
Its moderate-SLO goodput ratio is only 1.034×, with a 0.975–1.143× interval.
Other repeated Ouro B regimes lose to the strongest E2E baseline.

## Original-model fidelity and full depth

| Model | RLT / official exact sequences | Wilson 95% CI % | Answer accuracy |
| --- | --- | --- | --- |
| ouro | 88/100 | [80.18790057328857, 93.00059356298051] | NOT_MEASURED |
| huginn | 89/100 | [81.3687034969197, 93.74580364293543] | NOT_MEASURED |

Internal RLT PD/A/B qualification passes 100/100 bounded outputs per active model and control.
Both models pass exact transferred KV/hidden checks, including warmup and reversed fragment order for B.
All 128 canonical replay runs pass the logical full-depth row accounting.
This accounting counts submitted token-loop rows. It does not count physical GEMM padding or prove bit-exact model math.
The official-reference discrepancies remain unresolved. They are not demonstrated accuracy losses or verified harmless rounding.

GSM8K test accuracy and HumanEval accuracy are NOT_MEASURED.
The separate Ouro gate audit did not explain its 12 observed reference mismatches.

## 2/4/8GPU scope

Repeated baselines cover all three GPU budgets.
The main A/B evidence has three starts at two GPUs. The main C evidence has three starts at four GPUs.
Four/eight-GPU candidate and natural-pressure sweeps are one-start discovery.
Their point ratios and exact-output counts are in `scale_track_comparisons.csv` and `final_scale_discovery.csv`.
No confidence interval or accepted speedup follows from one start.

Ouro C at four GPUs has a 1.236× synthetic point ratio versus the best baseline and 1.992× versus service placement.
That isolated signal does not repeat as an accepted natural-workload gain.
Ouro B at eight-GPU natural pressure has a 1.125× point ratio but only 122/128 output agreement.
The report rejects it as a lossless win.
P6:D2 and P7:D1, dynamic exit, C2 migration, and dynamic role switching are NOT_MEASURED.

## Physical mechanism and limits

A transfers compute ownership after a real recurrent boundary. It preserves the total recurrence.
It trades less P work and a smaller snapshot against D-side prefill and interference.
Generic unrolled-depth migration remains a competing explanation for any benefit.
B transfers consumer-valid immutable depth planes before whole-chunk completion.
The matched G1 control already overlaps completed-chunk NIXL transfer with later prefill.

B reduces neither KV values nor the prescribed model work. Extra fragments increase submission and polling costs.
Generic immutable layerwise streaming remains an untested novelty control.
C changes initial request placement only. A and B are disabled.
Its loop score is a pending-work heuristic, not a checked completion-time predictor.
Batch geometry can affect BF16 outputs. The current evidence does not isolate that effect from a state bug.

## E2E component percentages

The denominator is summed per-request arrival-to-final-output wall time.
The phases below partition that denominator. Concurrent requests can overlap.
These values are not GPU utilization or an additive cohort-makespan partition.
Prefill and decode phases include waiting, host control, and GPU execution.
Pure dispatch/combine, attention/FFN, and GPU synchronization E2E percentages are NOT_MEASURED.

Profiler inclusive durations must not fill those missing fields.

| Model / method | coordinator queue % | reservation control % | P assigned prefill including backpressure % | uncovered handoff % | D resume coda first token % | remaining decode phase % | harness delivery poll residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ouro / PD | 0.061 | 2.990 | 29.111 | 2.583 | 0.269 | 64.867 | 0.118 |
| ouro / PD_G1 | 0.086 | 0.404 | 30.425 | 0.956 | 0.256 | 67.708 | 0.165 |
| ouro / A | 0.093 | 0.413 | 8.654 | 2.421 | 14.416 | 73.823 | 0.180 |
| ouro / B | 0.080 | 0.375 | 32.885 | 1.198 | 0.305 | 64.979 | 0.179 |
| huginn / PD | 0.010 | 2.432 | 19.148 | 6.945 | 0.330 | 71.108 | 0.027 |
| huginn / PD_G1 | 0.017 | 0.311 | 19.037 | 2.212 | 0.430 | 77.953 | 0.040 |
| huginn / A | 0.014 | 0.332 | 1.923 | 0.273 | 14.334 | 83.014 | 0.110 |
| huginn / B | 0.021 | 0.162 | 19.327 | 0.802 | 0.663 | 78.980 | 0.044 |

The exact CSV is `final_E2E_component_percentages.csv`.
The frozen-timeline oracle removes only the final observed handoff interval.
It retains all other request intervals. It does not model queue feedback or new streaming schedules.
Its ratios are ORACLE, not measured serving speedups or a global PD upper bound.

## Communication and memory

| Model | Method | Mean cohort transfer GB | Mean cohort transfer count | Maximum sampled per-device HBM GB |
| --- | --- | --- | --- | --- |
| ouro | DP | 0.000 | 0.000 | 17.786 |
| ouro | DP_Mixed | 0.000 | 0.000 | 17.807 |
| ouro | PD | 9.664 | 168.000 | 17.734 |
| ouro | PD_G1 | 9.664 | 168.000 | 17.854 |
| ouro | A | 2.466 | 48.000 | 18.051 |
| ouro | B | 9.664 | 216.000 | 17.860 |
| huginn | DP | 0.000 | 0.000 | 42.397 |
| huginn | DP_Mixed | 0.000 | 0.000 | 42.491 |
| huginn | PD | 34.256 | 528.000 | 43.166 |
| huginn | PD_G1 | 34.256 | 528.000 | 44.364 |
| huginn | A | 1.817 | 48.000 | 45.760 |
| huginn | B | 34.256 | 840.000 | 44.557 |

This table uses P512/C4, cap32, 24 requests per start, and three starts.
GB means decimal 10^9 bytes. Transfer counters exclude warmup.
HBM is sampled every 0.2s. It is not a complete graph-workspace or peak active-KV decomposition.
`worker_memory_transport.csv` separates parameter residency and reserved cache where recorded.
`profile_device_summary.csv` contains GPU event unions and recorded-window idle time.

Those heavy-profile measurements are separate from clean request latency.

## Completion, failures, and scope closure

| Manifest | Total | COMPLETE | FAILED |
| --- | --- | --- | --- |
| frozen_confirmation_jobs.json | 128 | 128 | 0 |
| C_campaign.json | 70 | 70 | 0 |
| scale_campaign.json | 174 | 172 | 2 |
| profile_checks.json | 8 | 8 | 0 |

Manifest counts are operational counts. They must not be added as independent statistical samples.
Two DP Mixed timeline jobs fail because ReplicaPool has no start_profile interface.
Their failure logs remain intact. They supply no measured timeline.
They do not invalidate separately completed unprofiled DP Mixed request measurements.
The final review does not change the frozen runtime or start another long campaign to correct this optional gap.

Legacy foreground-fsync runs remain invalid for clean comparisons. Their evidence stays preserved.
All reported repeated comparisons select the registered nonblocking protocol and frozen runtime.
The recorder still has nonzero producer work and background resource cost.

## Final CPU checks

The synchronous-fixture suite passes 35 tests. The separate asynchronous request-thread seam test passes one test.
Earlier mixed-mode invocations fail because their recorder modes do not match their file-read fixtures.
The logs retain those failures and the corrected commands. No measured runtime source changes during this closeout.
`test_finalize.py` checks comparator selection, partition weighting, missing uncertainty, and empty denominators.

## What this PoC does not conclude

It does not establish official benchmark quality, lossless original-reference equivalence, or sustainable open-loop capacity.
It does not establish that PD always loses, that every loop-specific design is impossible, or that long-context PD has no headroom.
The repeated matrix reaches input2048 and a separate output-cap128 regime.
It lacks representative ShareGPT/document distributions, a long-context serving sweep, and a rate-stability study.
No method is accepted merely because a transfer-only oracle, one-start point, or weak baseline looks favorable.

## Recommended next work, not executed

First isolate the official-reference and batched-output discrepancies without changing model semantics.
Then preregister a held-out Huginn ITL150ms serving test against tuned Mixed and generic layerwise streaming.
Use natural complete answers and a stable open-loop rate sweep if the fidelity gate passes.
Only then consider another implementation. Do not resume the paused native-vLLM loop-PP project.

## Reproduction and evidence

`reproduce.md` contains environment and start commands.
`commands.jsonl` records actual command lines and GPU masks.
Every attempt keeps configuration, request IDs, token IDs, seeds, and source snapshots under `raw/`.
`final_review_input_hashes.json` pins the derived inputs used by this review.
`final_manifest_status.json` records the reviewed queue status.

`configs/reviewed_decisions.json` contains the independent human-readable review conclusions.
`07A_final_decision.md`, `07B_final_decision.md`, and `07C_final_decision.md` contain per-track tables.
`08_comparison_matrix.md` contains model and GPU-budget decisions.
All remaining unexecuted follow-up work is optional and explicit. The current PoC is closed with limitations.