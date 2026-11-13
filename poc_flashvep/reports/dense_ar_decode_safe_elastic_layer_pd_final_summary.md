# Dense AR Decode-safe Elastic Layer PD

The main grid completed all 108 independent arm starts and its separate Native reference.
All 108 arm runs matched the Native reference's output tokens.
The separate reference completed normally.
No main cell remains unexecuted.
Final resource verification and publication checks accompany this report.

## Scientific answer

The requested P-to-D computation pattern works.
P executes a real layer prefix.
D receives partial KV, Hidden, and Residual, executes the remaining layers, and performs Full36 Decode.
The revised policy makes actual state-aware Last0/8/12 decisions before P submits its prefix.

The complete Dynamic superiority claim is not established on the tested envelope.
Output16/QPS6.4 provides a reproducible Native-relative TTFT reduction with Decode 5% non-inferiority.
However, Strong Static Last8 achieves essentially the same result.
At QPS4.8, Revised improves Native TTFT but fails the required Decode confidence criterion.
It also loses TTFT and Balanced goodput to Strong Static.

The tested-envelope verdict for the full requested conjunction is `NO_GO_ON_TESTED_CONDITIONS`.
This is not a claim that exact Layer Offloading is impossible or always unhelpful.
It is a negative result for this frozen Dynamic implementation against the tested strong baselines.
The implementation uses cohort-isolated suffix execution.
It does not test simultaneous spatial multiplexing of suffix and Decode kernels.
The verdict does not rule out that different execution architecture.
The policy logs the earliest-token-deadline field but does not directly use it to select Revised candidates.
Decode protection relies on drain prediction, an execution reserve, and the hard cohort guard.
Thus, the requested direct per-token deadline optimizer remains a partial implementation goal.
The campaign does not declare a structural NO-GO for an unimplemented stronger policy.

## Environment and reproducibility

| Field | Frozen value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Computation | Original BF16, exact Attention, all 36 layers, greedy, thinking disabled |
| Runtime | vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| Primary GPUs | H100 SXM80GB, GPU0/1 |
| GPU0 UUID | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` |
| GPU1 UUID | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` |
| PD mapping | P0, D1, full model on both GPUs |
| Colocated mapping | Full replica0 and replica1 on GPU0/1 |
| KV | 42 GiB per GPU, prefix caching disabled |
| Primary input/output | 8000/16,8000/64,8000/200 |
| Arrival | Open-loop Poisson, paired immutable traces |
| Fresh prompts | 128 distinct LongBench document IDs, separate from viewed tuning IDs and prefixes |
| Arrival seeds | 2026111381,2026111382,2026111383 |
| Source namespace | `poc_flashvep/dense_ar_elastic_layer_pd/` |
| Evidence namespace | `dense_ar_decode_safe_elastic_layer_pd_20261113/` |
| Publication parent | `96414d7baac14dea2d697198b2c595850ae83661` |
| Publication branch | `flashvep/dense-ar-decode-safe-elastic-layer-pd-20261113` |

Every executed job retains its command, source ZIP, source hashes, immutable configuration, GPU mapping, and result receipt.
The fresh prompt SHA256 is `245da8aa5e7c5978771a398a7fb5f6eb5fd091972555281a3e06c6283e170a71`.
The data manifest preserves document IDs, original hashes, and truncation rules.
No filler token creates artificial model work.
The initial administrative deadline left six predeclared starts unexecuted.
The authorized extension completed those same immutable configurations without retuning.
The original deadline snapshot remains preserved.

The experiment uses the predecessor's unchanged research SLO regimes.
The SLO SHA256 is `c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`.
Both Native-relative Decode 95% confidence upper bounds must be at most 1.05.
First ITL remains included.
The evidence also separates first ITL, steady ITL, maximum ITL, and threshold exceedances.
Service TTFT remains `NOT_MEASURED`.
TTFT uses scheduled arrival to coordinator delivery without HTTP.
Native retains the prior contract that releases its P-produced first token after D import and new-token submission.
The PD arms use the same contract.
This is not a measurement of an alternative early-P-token HTTP service.

## Strong Continuous qualification

`CORRECTNESS_ONLY`: standard Continuous Colocated passed N32 output, all36 prompt KV planes, and all8000 final Hidden rows.
The archived Chunk512 N64 failure changed from 8/64 mismatches to 0/64 after a generic numerical correction.

Saved-operand replay identified a batch-dependent fused residual-add RMS Normalization reduction in the installed binary.
Rows below 256 used1024 threads per block.
Larger batches used256 threads per block.
The first captured difference affected1/4096 values after Layer 23 Attention/output projection.
Its maximum absolute difference was 0.001953125.
That difference later changed greedy argmax.

The correction selects the original Native reduction for each token phase.
It does not alter model weights, precision, Attention, GEMMs, tokens, or layer work.
All final methods enable the same optional correction.
The primary Colocated scheduler remains standard Continuous Batching.
No cohort barrier weakens Colocated.

Viewed tuning selected token budgets512/512/384 for Output16/64/200.
Maximum sequences were 32/16/16 per replica.
The router remained Round-robin.
Compilation did not qualify because an optional attempt mismatched4/8 sequences.
Thus, this baseline is strongest within the tested exact, non-compiled family.
It is not globally optimal stock vLLM.

See [Continuous diagnosis and qualification](dense_ar_continuous_colocated_correctness_and_baseline.md).

## Implementation changes and cost attribution

| Finding | Change or evidence | Interpretation |
|---|---|---|
| Old COST selected Last0/12 | All feasible cuts receive an efficiency score | Dynamic policy change |
| Handoff could not overlap admitted Decode | Serialize predicted handoff after known D availability | Model contract correction |
| D inherited32000-token P budget | Separate D budgets64000/128000 and qualify batches8/16 | Generic destination geometry improvement |
|5 ms margin lacked residual support | Freeze measured bias, drain uncertainty, and per-cut margins before fresh evaluation | Empirical risk control |
| Detailed request logging delayed token delivery | Defer final JSON records until worker shutdown | Generic runtime improvement shared by all arms |
| Native cached ingress increased Decode tails | Reject it for protected Native and longer-output primary settings | Safety rejection, not a relaxed baseline |

The logging control reduced Native ITL p99 from 33.45 ms to 28.57 ms in a viewed paired diagnostic.
Detailed handler p95 decreased from 1.935 ms to 0.227 ms.
This improvement is not a Layer Offloading speedup.

The rejected cached-ingress path passed state correctness.
Its Output64 ITL p99 reached 335.78 ms.

The campaign did not use that unsafe Native as a more permissive protection denominator.
Protected Native retains Decode-isolated admission.
Genuine suffixes also wait for existing Decode to retire.
The implementation never assumes kernel preemption.

The final policy can value aggregate queue benefit above a migrated request's own predicted completion benefit.
It therefore does not guarantee an earlier completion for every migrated request.
That stronger per-request requirement remains unmet.
The score is a queue/cost heuristic rather than an explicit request-level SLO-goodput optimizer.

## Fresh Output16 results

`MEASURED`: entries are means of three independent N128 start-level quantiles.
Rates include finite-cohort drain.
They are not sustainable capacity measurements.

| QPS | Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output tokens/s |
|---|---|---:|---:|---:|---:|
|4.8 | Native |10937.125 |26.981 |25.418 |54.11 |
|4.8 | Continuous DP2 |3026.389 |35.836 |29.323 |70.13 |
|4.8 | Static Last8 |4550.111 |28.979 |27.389 |66.29 |
|4.8 | Revised Dynamic |5316.998 |28.263 |27.027 |64.59 |
|6.4 | Native |16696.71 |28.09 |26.16 |54.62 |
|6.4 | Continuous DP2 |7066.23 |35.04 |27.66 |74.04 |
|6.4 | Static Last8 |10022.04 |27.35 |26.01 |66.74 |
|6.4 | Revised Dynamic |10036.89 |27.18 |26.02 |66.95 |

At QPS4.8, Revised/Native TTFT p95 was 0.48614[0.44759,0.51894].
TTFT decreased 51.39%.
ITL p99 was 1.04753[0.95691,1.13825].
TPOT p95 was 1.06330[0.99161,1.12960].
Both Decode confidence upper bounds exceed1.05.
This cell does not pass the requested non-inferiority gate.

At QPS6.4, Revised/Native TTFT p95 was 0.60113[0.58058,0.62200].
TTFT decreased 39.89%.
ITL p99 was 0.96778[0.92903,1.01804].
TPOT p95 was 0.99482[0.96209,1.03815].
Both Decode confidence upper bounds pass1.05.
Static Last8 also passes both bounds and has nearly identical TTFT.
Thus, this is a Native-relative benefit, not an established Dynamic-over-Static benefit.

The frozen SLO regime changes the Colocated comparison.
At Output16/QPS6.4, Revised/Colocated Decode-priority goodput was 1.73537, with 95% CI [1.52379,1.93205].
Balanced goodput was 0.68999 [0.65055,0.72071].
Revised therefore helps a stricter Decode-priority service objective in this cell.
That comparison uses the frozen Balanced-selected Colocated setting.
A separately Decode-priority-tuned Colocated frontier remains `NOT_RUN`.
It does not dominate Colocated across all service objectives.
Its Balanced goodput versus Static was 1.00458 [0.98372,1.01864].
That interval does not establish additional Dynamic goodput.

Maximum ITL remains a separate risk indicator.

| Output16/QPS6.4 arm | Maximum ITL by seed81/82/83 (ms) |
|---|---|
| Native |28.43 /28.47 /28.49 |
| Revised |32.63 /26.35 /30.14 |
| Static |27.96 /29.01 /49.92 |

Revised had no 34 ms Balanced exceedance in these starts.
A passing p99 ratio does not mean every request satisfies the stricter30 ms regime.

Each interval uses27 empirical paired bootstrap draws of three independent starts.
The intervals do not establish production latency guarantees.
The analysis does not treat requests or tokens as independent starts.

## Migration and matched controls

At Output16/QPS4.8, Revised completed 308/384 migrated requests and 2576/13824 migrated layer visits.
The distribution was 76 Native,280 Last8, and 28 Last12.
Last2/4 had zero actual online selections.
Forced N32 Last2/4/8/12 state checks passed separately.
The corresponding migrated work was 20608000 token-layer rows.
This is structural work accounting, not a measured FLOP rate or GPU time.
Across all 27 Revised starts, the distribution was 2804 Native,608 Last8,44 Last12, and zero Last2/4 requests.
Both longer-output sweeps completed zero migrated requests.

Four actual committed groups selected Last8 because larger cuts failed predicted time gates.
No experimental work cap created those four decisions.
This is real smaller-cut adaptation, but only from Last12 to Last8.

The shuffled control matched actual migrated request counts and layer work in all three starts.
Revised/Matched TTFT p95 was 0.87937[0.85359,0.90329].
Balanced goodput was 1.31563[1.21648,1.48526].
The comparison contains batching and timing differences beyond total migrated work.
It does not isolate the predictor's state-aware contribution.
Revised used27/26/24 migrated groups.
The matched control used30/28/30 because runtime collection fragmented some shuffled assignments.
The actual group profiles therefore did not match.
The control is post-hoc aggregate accounting, not a deployable held-out winner.

Against Strong Static, Revised TTFT was 1.16854[1.10985,1.25473].
Balanced goodput was 0.82959[0.72449,0.91070].
Therefore, the matched-control improvement does not satisfy the full research success criteria.

See [matched migration and ablations](dense_ar_static_dynamic_matched_migration_comparison.md).
See [actual decisions](dense_ar_dynamic_layer_cut_decision_analysis.md).

## P bottleneck and actual critical-path evidence

`TRACE_ANALYSIS`: Native Output16/QPS4.8 P-ready queue maxima were33/38/28.
First-half to second-half means increased 8.933→28.091,8.231→26.100, and 2.000→20.917.
Neither P nor D recorded a KV allocation refusal in these starts.
Minimum D free KV blocks remained 17109.

P-only batch4 capacity was 3.58564 requests/s.
Native Host-observed Source-ready production was approximately 3.44 requests/s.
This is near the independent capacity reference.
It supports a Prefill service bottleneck, but does not replace a kernel-union saturation measurement.

| Output16/QPS4.8 arm | GPU0 NVML mean (%) | GPU1 NVML mean (%) |
|---|---:|---:|
| Native |94.080 |48.681 |
| Continuous |85.336 |86.607 |
| Static |88.642 |80.383 |
| Revised |90.388 |76.544 |

These are start-mean sampled NVML values over each measurement window.
They are not physical kernel busy ratios or safe execution budgets.
The report does not interpret their complements as idle capacity.
Continuous recorded 1860/1890/1770 actual mixed measurement batches across its two replicas.
Those counts establish actual continuous Prefill–Decode co-batching, not exclusive kernel overlap time.

Mean final Source-ready time decreased from 37.201 s to 30.976 s with Revised.
Mean clearance after the last arrival decreased from 10.721 s to 4.497 s.
Static completed its Source-ready production in 29.880 s.
These are measured Host critical-path boundaries, not exclusive GPU compute times.

Structural P work decreases exactly by the completed migrated layer count.
Clean exclusive P kernel compute saved remains `NOT_MEASURED`.

The longer-output envelope has different bottlenecks.
At Output200/QPS4.8, Native P recorded 30510/22253/20156 unsuccessful KV allocation attempts.
These are repeated attempts, not rejected requests.
P free KV blocks reached 113 while delayed handoff retained Source state.
That point includes Source KV and Decode pressure.
It is not qualified as a pure P-compute-heavy condition.
Normal completion restored the KV credits.

## Transfer, ownership, and prediction

Each migrated request transfers125 MiB of Hidden plus Residual.
Partial KV plus that state totals1187.5/1125/1000/875 MiB for Last2/4/8/12.
Native already transfers1125 MiB of KV.
The report does not charge the entire KV payload as additional Layer Split overhead.

All308 migrated requests in the Output16/QPS4.8 cell returned normal Hidden ACK and Source credit.
Hidden submit-to-DONE p95 was 0.526/0.921/0.541 ms across starts.
KV NIXL telemetry p95 was approximately 191.9 ms in each start.
These definitions overlap other intervals and do not form an additive wire-time decomposition.

The completion upper estimate covered77/77 committed groups in that cell.
Central completion MAE was 57.522/41.078/59.087 ms.
Absolute error p95 was 78.422/75.228/91.082 ms.
Accepted-outcome coverage does not prove future Decode safety.
Rejected cuts have no measured counterfactual completion.

See [cost model and deadline evidence](dense_ar_layer_cost_model_and_deadline_validation.md).

## Physical measurement and unsupported claims

Clean P kernel union, D physical idle, spatial slack, and request-feasible physical packing capacity remain `NOT_MEASURED`.
The Nsight diagnostic changed TTFT by 2.929× and TPOT by 14.943× against its control.
It also failed strict isolation qualification.
The warmed background collector did not produce qualified capacity evidence.

CUDA event A/A controls also failed the complete5% tail criterion.
Their timelines remain `TRACE_ANALYSIS`, not clean kernel timelines.
Blank plot regions are not measured physical idle.
The campaign therefore does not claim the strict same-window P≥80% physical saturation gate.
The separate marker-free serving results remain usable within their stated protocol.

| Claim | Evidence level | Limit |
|---|---|---|
| N32 tokens, all prompt KV, final Hidden | `CORRECTNESS_ONLY` | Qualified shapes and normal completion |
| N128 arrival-to-output latency | `MEASURED` | Coordinator protocol, no HTTP |
| Actual completed cut counts | `TRACE_ANALYSIS` | Qualified execution path, separate from forced cuts |
| Token-layer rows and byte expectations | `STRUCTURAL_ACCOUNTING` | Not measured speedup |
| NIXL bytes and normal retirement | `TRACE_ANALYSIS` | No new adversarial recovery test |
| Completion prediction error | `TRACE_ANALYSIS` | Accepted realized groups only |
| Clean exclusive kernel busy/idle | `NOT_MEASURED` | Metrology qualification failed |
| Exact isolated state-predictor advantage | `NOT_IDENTIFIED` | Matched controls also change batch geometry |
| Sustainable online capacity | `NOT_MEASURED` | N128 includes overload and drain |

## Figures and complete metrics

- [QPS versus TTFT](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_A_qps_vs_ttft.png).
- [QPS versus TPOT](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_B_qps_vs_tpot.png).
- [QPS versus generation throughput](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_C_qps_vs_generation_throughput.png).
- [Actual cut distribution](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_G_actual_dynamic_cut_distribution.png).
- [Diagnostic P/D timeline](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_I_diagnostic_PD_timeline.png).
- [Interactive observed timeline](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/interactive_observed_PD_timeline.html).
- [Budget and prediction errors](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_J_budget_and_prediction_error.png).
- [Decode-priority goodput](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_K_decode_priority_slo_goodput.png).

The [sweep report](dense_ar_four_method_qps_sweep_results.md) links every start's full metric vector and confidence intervals.
It includes TTFT, E2E, TPOT, ITL, maximum ITL, three SLO regimes, token rates, and queue observations.

## Failures and unexecuted scope

Failed Native/Continuous numerical reproductions remain preserved.
The optional compiled baseline remains unqualified.
The interrupted cost-calibration namespace remains failed, with a separate completed replacement.
A temporary discovery coordinator name error remains a failed implementation experiment.
The corrected state qualification excludes warm-up IDs and preserves the earlier analysis erratum.

Cancellation and GPU failure recovery did not receive new adversarial qualification.
Normal ACK, source lifetime, destination retirement, and credit restoration did receive checks.
Natural EOS, bursty arrivals, Output32/128 confirmation, and additional models remain `NOT_RUN`.
No offline oracle counts as an observed serving gain.

## Research decision

Continue exact layer-continuation infrastructure only if a new hypothesis addresses a failed gate.
Do not claim Dynamic superiority from migrating more requests or from a generic runtime correction.
Useful next hypotheses are lower-cost control, stricter per-request economics, and stronger tail prediction.
Each requires a new frozen validation set.

The fixed8000-token, fixed-output workloads also favor a strong static pipeline balance.
At Output16, Static Last8 transfers more useful work than Revised and reaches similar protected Decode latency at high load.
This observation is consistent with a nearly balanced fixed pipeline.
It does not provide an exclusive causal decomposition of GPU times.
Heterogeneous lengths or bursty arrivals could change the best cut, but this campaign did not test that claim.

This campaign does not prove layer-axis novelty over every generic offloading method.
It does not compare official Kairos, FluidPD, or DuetServe implementations.
It compares the requested Native, qualified Continuous, Strong Static, and Revised Dynamic arms.

## Verification and resource closeout

GPU work is closed.
All eight approved Burner workers returned fresh completed generation after restoration.
The verification records actual process ownership, GPU UUIDs, utilization, and increasing generation counters.
The workers remain active during CPU analysis and publication.

All 24 CPU tests passed.
The report audit checked 188 printed metric values against independent-start records without a rounding mismatch.
Frozen execution sources, predictors, prompts, SLOs, prior sources, and installed runtime sources remain unchanged.
The existing checkout, staged index, and external runtime remain preserved.
Publication uses a private Git index and the separate branch above.

- [Resource restoration proof](dense_ar_decode_safe_elastic_layer_pd_20261113/state/resource_restoration_verified.json).
- [Final source, link, and test verification](dense_ar_decode_safe_elastic_layer_pd_20261113/state/final_verification.json).
- [Numeric and frozen-source audit](dense_ar_decode_safe_elastic_layer_pd_20261113/state/report_numeric_and_freeze_audit.json).
- [Frozen original deadline snapshot](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/initial_deadline_confirmation_results.json).
