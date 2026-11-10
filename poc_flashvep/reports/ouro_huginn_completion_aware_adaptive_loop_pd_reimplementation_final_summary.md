# Completion-aware Adaptive Loop-Split PD: implementation and validation

## 1. TO-BE outcome

The campaign reproduced part of the TO-BE execution mechanism, but not its common serving benefit.
It implemented real boundary negotiation and exact P-to-D continuation.
Recovery moved 6/1536 Ouro Loops and 853/12288 Huginn Loops.
The corresponding request counts were 3/384 and 142/384.
Both models passed all three absolute Decode screens.
Huginn did not preserve Native-relative Decode tails near the 5% target.
The provisioned Native path without any migration also outperformed the Loop path.
It also removed one cause of the old multi-second starvation.
However, successful state movement did not establish useful large-scale P/D load balancing.
Physical capacity remains a separate measurement uncertainty.

The primary and recovery experiments remain separate.
They use different policy revisions and, for Huginn, different KV pool sizes.
The tables below must not become one combined treatment estimate.

## 2. Implementation and previous failure

The source directory is `poc_flashvep/completion_aware_loop_pd/`.
The detailed analyses are `root_cause_report.md`, `reference_code_review.md`, and `implementation_design.md` in the evidence directory.
The optional worker extension preserves the external vLLM-RLT checkout.
It reuses the Native-path connector, exact state frontiers, NIXL, and existing model kernels.
No model parameter, precision, recurrent depth, or Attention approximation changed.

The old source armed only 8/384 Ouro requests and 15/384 Huginn requests.
False initial arming could permanently remove later boundary opportunities.
One accepted continuation could occupy the only slot for most later arrivals.
Persistent admission suspension also prevented execution of already committed work.
The frozen host gaps were 23.207262 s and 85.872055 s.
Those gaps were missing submissions, not measured GPU idle.

The redesign separates source monitoring, completion admission, and quantum execution.
It retries legal Loop boundaries after transient rejection.
It prices complete indivisible quanta, transfer, accepted work, and repeated Decode reserves.
Committed work can progress under live deadlines even after new admission stops.
Eligible owned work uses completion-deadline priority.
A miss stops new borrowing but does not discard state or duplicate computation.

The reservation is an empirical forecast, not a guaranteed future GPU timeslot.
The implementation does not provide exact D-to-P recovery after commit.
It does not guarantee starvation freedom under every future arrival trace.
The direct P-native versus D-completion counterfactual remains incomplete.

## 3. Exactness and ownership

Twenty-six new CPU tests passed.
Thirteen inherited frontier and epoch tests also passed.
The final source compile check passed.
N8 and N32 compare exact output, valid prompt KV hashes, hidden hashes, full recurrent work, and Native lifetimes.
Their reference is an identified frozen Native GPU reference, not a new Native run.
These checks establish internal exactness, not independent original-HF fidelity or task accuracy.

Recovery N32 Loop diagnostics selected 1 and 3 Loops for Ouro.
Huginn's historical-cut diagnostic selected 2, 4, and 8 Loops.
The expanded Huginn diagnostic selected 4, 8, 11, and 16 Loops.
That expanded cohort committed 13/32 requests and completed 157 full-prompt Loop equivalents.
The Scheduler made these decisions online.
They were not a preassigned forced-cut cohort.

N32 means simultaneous submission of 32 requests.
It does not mean 32 simultaneous resident GPU requests.
State diagnostics use observer settings and longer completion deadlines.
They are not clean Decode SLO evidence.

Marker-free N128 runs check exact output, full work, ACK-before-retirement, final credit release, and empty Native allocations.
They do not hash every intermediate tensor during clean timing.
The separate N8/N32 tests qualify those state paths.
GPU error injection, post-commit cancellation recovery, and reverse migration remain `NOT_MEASURED`.
CPU stale-epoch checks are not substitutes for those GPU fault tests.

## 4. Frozen workload and fairness

Ouro uses original BF16 R4, 4096/32, QPS 6.356, and slots 2/8.
Huginn uses original BF16 R32, 2048/16, QPS 1.971, and slots 2/8.
The scientific pairs are 0/1 and 4/5.
Only one pair runs in the clean timing lane.
All approved Burners stop during that lane.

All arms retain 128 distinct controlled ShareGPT/LongBench prompt IDs.
Prefix caching is disabled.
Output caps ignore EOS.
Arrivals are paired open-loop Poisson traces.
These overloaded finite cohorts are not sustainable-capacity measurements.
They are not workloads where every request satisfies TTFT SLO.

The original confirmation uses arrival seeds 2026110911/12/13.
Recovery selection uses 2026110920.
Recovery confirmation uses 2026110931/32/33.
The latter arrivals are held out from recovery calibration and selection.
The texts are not held out.

The frozen absolute Decode SLOs remain unchanged.

| Model | TTFT limit, ms | Request TPOT limit, ms | Additional-work TPOT p95 limit, ms | ITL p99 limit, ms |
| --- | ---: | ---: | ---: | ---: |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

Native-relative ITL p99 non-inferiority remains a separate 5% design target.
Passing an absolute limit does not prove that target.

## 5. Original primary confirmation, before recovery

| Model | Arm | Arrival seed | TTFT p95, ms | ITL p99, ms | Max ITL, ms | TPOT p95, ms | E2E p95, ms | Requests/s | SLO-goodput | Offloaded requests | Full Loops | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | native | 2026110911 | 2843.935 | 43.389 | 60.941 | 38.820 | 4030.261 | 5.200 | 2.194 | 0 | 0 | PASS |
| ouro | old | 2026110911 | 2855.727 | 42.228 | 44.906 | 37.834 | 3989.177 | 5.208 | 2.197 | 1 | 2 | PASS |
| ouro | loop | 2026110911 | 3723.047 | 42.452 | 135.293 | 38.631 | 4884.386 | 5.010 | 1.566 | 1 | 2 | PASS |
| ouro | token | 2026110911 | 25204.882 | 38.696 | 866.017 | 35.262 | 26259.323 | 2.686 | 0.042 | 4 | 0 | PASS |
| ouro | old | 2026110912 | 4271.570 | 42.841 | 152.247 | 38.488 | 5456.135 | 5.107 | 2.035 | 1 | 1 | PASS |
| ouro | loop | 2026110912 | 4721.096 | 41.525 | 46.105 | 37.793 | 5881.059 | 5.023 | 1.688 | 1 | 1 | PASS |
| ouro | token | 2026110912 | 25096.884 | 39.925 | 395.109 | 35.838 | 25977.539 | 2.748 | 0.064 | 6 | 0 | PASS |
| ouro | native | 2026110912 | 4363.009 | 41.419 | 49.126 | 37.844 | 5521.891 | 5.092 | 1.830 | 0 | 0 | PASS |
| ouro | loop | 2026110913 | 8192.772 | 41.915 | 44.829 | 37.590 | 9109.141 | 4.733 | 0.814 | 0 | 0 | PASS |
| ouro | token | 2026110913 | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP | NOT_RUN_TAIL_SAFETY_STOP |
| ouro | native | 2026110913 | 5676.990 | 41.625 | 50.236 | 37.874 | 6821.206 | 5.211 | 0.936 | 0 | 0 | PASS |
| ouro | old | 2026110913 | 6287.036 | 41.569 | 49.824 | 38.030 | 7395.259 | 4.894 | 0.726 | 1 | 2 | PASS |
| huginn | native | 2026110911 | 30209.204 | 134.875 | 135.587 | 126.449 | 32050.887 | 1.289 | 0.091 | 0 | 0 | PASS |
| huginn | old | 2026110911 | 39584.200 | 141.872 | 143.949 | 130.766 | 41192.428 | 1.176 | 0.037 | 1 | 1 | PASS |
| huginn | loop | 2026110911 | 32934.611 | 153.387 | 155.408 | 135.229 | 34762.724 | 1.249 | 0.049 | 2 | 10 | PASS |
| huginn | token | 2026110911 | 63080.353 | 143.641 | 149.464 | 134.958 | 64978.196 | 0.954 | 0.000 | 2 | 0 | PASS |
| huginn | old | 2026110912 | 38728.919 | 133.131 | 136.967 | 120.763 | 40484.223 | 1.215 | 0.047 | 0 | 0 | PASS |
| huginn | loop | 2026110912 | 36798.450 | 148.548 | 155.042 | 132.665 | 38741.348 | 1.240 | 0.048 | 2 | 8 | PASS |
| huginn | token | 2026110912 | 54446.520 | 137.920 | 144.759 | 123.189 | 56039.304 | 1.057 | 0.008 | 1 | 0 | PASS |
| huginn | native | 2026110912 | 43170.743 | 150.596 | 152.853 | 141.468 | 45061.505 | 1.165 | 0.036 | 0 | 0 | PASS |
| huginn | loop | 2026110913 | 42792.326 | 153.531 | 163.529 | 136.067 | 44494.890 | 1.248 | 0.039 | 1 | 4 | PASS |
| huginn | token | 2026110913 | 64080.701 | 123.026 | 165.053 | 110.061 | 65687.537 | 1.024 | 0.008 | 1 | 0 | PASS |
| huginn | native | 2026110913 | 49671.263 | 151.057 | 152.027 | 142.043 | 51673.423 | 1.168 | 0.036 | 0 | 0 | PASS |
| huginn | old | 2026110913 | 43634.253 | 144.021 | 149.292 | 131.176 | 45480.393 | 1.230 | 0.038 | 0 | 0 | PASS |

Ouro Token start 3 did not run after an earlier tail failure.
Its second start was already in progress when the explicit stop record arrived.
The completed Token results also include cold measured graph captures.
They do not establish an intrinsic Loop advantage.

The primary Loop moved 3/1536 Ouro Loops and 22/12288 Huginn Loops.
The corresponding offloaded request counts were 2/384 and 5/384.
Ouro's TTFT p95 ratio was 1.2692, with a paired 95% interval of [1.0821, 1.4432].
Huginn's ratio was 0.9285, with [0.8524, 1.0902].
The Huginn interval does not establish improvement.

## 6. Recovery changes and matched controls

The recovery uses the same full Decode-cycle reserve for Token and Loop.
It protects the next cumulative request-average TPOT with already delivered token counts.
It includes known accepted continuations' future Decode load.
It does not know future request arrivals or output times.

The source filters clearly infeasible offers before a relay.
Loop requests retain later legal boundary opportunities.
Token chooses a feasible initial prefix and revalidates it at the actual boundary.
After a rejection, Token computes its remaining suffix rather than create many new small checkpoints.
This is a bounded single-offer Token adaptation, not the complete official FluidToken algorithm.
It can miss later Token opportunities.
The report does not claim an optimal Token-only Scheduler.

All recovery arms prohibit cold packed-prefill capture during measurement.
They use cached graphs or the qualified eager path.
The common cache budget and warm-up policy apply to Native, Token, and Loop.
This policy does not control every Decode graph in the runtime.
Graph selections and eager fallbacks remain visible in the audit.

All recovery Huginn arms use 24832 D KV blocks, rather than the primary pool's 23552 blocks.
The larger pool admits six requests instead of five, despite eight logical slots.
Startup requires at least 3 GiB of free HBM.
The matched Native control receives the same memory change.
A same-arrival Native pool comparison changed completed QPS by only about 0.3%.
It does not establish KV admission as the unique throughput bottleneck.

The expanded Huginn candidate set passed separate GPU exactness tests.
Its N32 selection passed the absolute Decode screen and reduced TTFT p95 from 4964.819 to 4557.848 ms versus historical-cut Loop.
That selection remained slower than Native and worsened Decode tails.
Only selection data determined the expanded candidate choice.
The confirmation policy then remained fixed.

## 7. Recovery marker-free N128 confirmation

| Model | Arm | Arrival seed | TTFT p95, ms | ITL p99, ms | Max ITL, ms | TPOT p95, ms | E2E p95, ms | Requests/s | SLO-goodput | Offloaded requests | Full Loops | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | native | 2026110931 | 5234.655 | 41.121 | 54.036 | 37.812 | 6399.877 | 5.145 | 1.085 | 0 | 0 | PASS |
| ouro | loop | 2026110931 | 5383.829 | 40.261 | 48.989 | 37.269 | 6522.053 | 5.088 | 1.073 | 1 | 1 | PASS |
| ouro | token | 2026110931 | 4955.073 | 41.585 | 50.741 | 38.079 | 6124.216 | 5.185 | 1.134 | 0 | 0 | PASS |
| ouro | loop | 2026110932 | 7235.867 | 42.520 | 54.200 | 38.061 | 8402.751 | 5.144 | 0.522 | 1 | 3 | PASS |
| ouro | token | 2026110932 | 6609.491 | 41.865 | 44.862 | 37.912 | 7766.677 | 5.257 | 0.575 | 1 | 0 | PASS |
| ouro | native | 2026110932 | 7083.025 | 41.708 | 44.526 | 37.756 | 8183.282 | 5.197 | 0.528 | 0 | 0 | PASS |
| ouro | token | 2026110933 | 6692.716 | 41.283 | 43.893 | 38.006 | 7860.419 | 5.269 | 0.329 | 0 | 0 | PASS |
| ouro | native | 2026110933 | 6782.682 | 41.559 | 44.937 | 37.790 | 7941.599 | 5.241 | 0.287 | 0 | 0 | PASS |
| ouro | loop | 2026110933 | 7209.080 | 42.671 | 50.367 | 38.329 | 8370.937 | 5.126 | 0.320 | 1 | 2 | PASS |
| huginn | native | 2026110931 | 23455.200 | 140.681 | 144.725 | 130.920 | 25122.083 | 1.493 | 0.082 | 0 | 0 | PASS |
| huginn | loop | 2026110931 | 22525.624 | 183.745 | 186.294 | 150.020 | 24533.327 | 1.515 | 0.201 | 81 | 490 | PASS |
| huginn | token | 2026110931 | 29031.784 | 144.279 | 148.127 | 134.558 | 30808.659 | 1.403 | 0.011 | 1 | 0 | PASS |
| huginn | loop | 2026110932 | 30700.825 | 182.298 | 187.361 | 150.570 | 32398.083 | 1.485 | 0.070 | 57 | 333 | PASS |
| huginn | token | 2026110932 | 32647.628 | 150.317 | 163.527 | 135.844 | 34528.603 | 1.465 | 0.023 | 1 | 0 | PASS |
| huginn | native | 2026110932 | 31989.225 | 149.230 | 153.520 | 140.062 | 33822.430 | 1.458 | 0.011 | 0 | 0 | PASS |
| huginn | token | 2026110933 | 32863.040 | 147.992 | 172.088 | 136.139 | 34631.240 | 1.444 | 0.000 | 1 | 0 | PASS |
| huginn | native | 2026110933 | 30575.581 | 143.372 | 149.721 | 134.800 | 32446.205 | 1.497 | 0.000 | 0 | 0 | PASS |
| huginn | loop | 2026110933 | 33140.895 | 146.611 | 165.313 | 118.434 | 34812.168 | 1.443 | 0.023 | 4 | 30 | PASS |

`Service TTFT` and exact GPU-start Queue Latency are `NOT_MEASURED` in marker-free confirmation.
They are not replaced by coordinator admission timestamps.
The report does not add component percentiles to create a total percentile.
Separate CUDA Event boundary diagnostics retain their own scope.

All p50/p95/p99 latency values, input/output token rates, offered/realized QPS, and attainment counts remain in `recovery_serving_results.csv`.
Maximum ITL and threshold exceedances remain separate from the p99 screen.
Completed requests/s and goodput use the full cohort drain window.

## 8. Independent paired-start uncertainty

| Model | Comparison | Metric | Paired starts | Geometric ratio | Start-bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| ouro | loop/native | TTFT_p95_ms | 3 | 1.037 | [1.022, 1.063] |
| ouro | loop/native | E2E_p95_ms | 3 | 1.033 | [1.019, 1.054] |
| ouro | loop/native | ITL_p99_ms | 3 | 1.008 | [0.979, 1.027] |
| ouro | loop/native | SLO_goodput | 3 | 1.030 | [0.989, 1.118] |
| ouro | loop/token | TTFT_p95_ms | 3 | 1.086 | [1.077, 1.095] |
| ouro | loop/token | E2E_p95_ms | 3 | 1.071 | [1.065, 1.082] |
| ouro | loop/token | ITL_p99_ms | 3 | 1.005 | [0.968, 1.034] |
| ouro | loop/token | SLO_goodput | 3 | 0.942 | [0.909, 0.973] |
| huginn | loop/native | TTFT_p95_ms | 3 | 1.000 | [0.960, 1.084] |
| huginn | loop/native | E2E_p95_ms | 3 | 1.001 | [0.958, 1.073] |
| huginn | loop/native | ITL_p99_ms | 3 | 1.177 | [1.023, 1.306] |
| huginn | loop/native | SLO_goodput | 3 | NOT_MEASURED | NOT_ESTABLISHED |
| huginn | loop/token | TTFT_p95_ms | 3 | 0.903 | [0.776, 1.008] |
| huginn | loop/token | E2E_p95_ms | 3 | 0.909 | [0.796, 1.005] |
| huginn | loop/token | ITL_p99_ms | 3 | 1.152 | [0.991, 1.274] |
| huginn | loop/token | SLO_goodput | 3 | NOT_MEASURED | NOT_ESTABLISHED |

Ratios use paired start-level metrics, not independent-token samples.
Three starts provide limited uncertainty coverage.
Missing starts do not become zero values.
A Token arm with no actual offload is a Native-fallback control, not evidence of Token redistribution benefit.

### Native-path zero-offload control

The original Native arm uses its original connector configuration.
Loop and Token enable Native-path continuation buffers and the qualified packed wire format.
This implementation difference can affect even a request with no offload.
Later controls enable that path without boundary negotiation or computation movement.
They pass separate N8/N32 state qualification before clean N128 execution.
Their later execution order remains a temporal limitation.
They do not provide a new randomized policy comparison.

| Model | Arm | Arrival seed | TTFT p95, ms | ITL p99, ms | Max ITL, ms | TPOT p95, ms | E2E p95, ms | Requests/s | SLO-goodput | Offloaded requests | Full Loops | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | zero | 2026110931 | 4975.090 | 43.713 | 46.993 | 38.882 | 6168.489 | 5.171 | 1.131 | 0 | 0 | PASS |
| ouro | zero | 2026110932 | 6723.762 | 42.245 | 44.121 | 38.728 | 7907.265 | 5.220 | 0.530 | 0 | 0 | PASS |
| ouro | zero | 2026110933 | 6662.619 | 41.103 | 44.634 | 38.010 | 7837.707 | 5.246 | 0.369 | 0 | 0 | PASS |
| huginn | zero | 2026110931 | 17052.741 | 133.455 | 140.215 | 124.464 | 18859.978 | 1.618 | 0.240 | 0 | 0 | PASS |
| huginn | zero | 2026110932 | 23203.409 | 125.194 | 130.632 | 117.597 | 24943.627 | 1.626 | 0.064 | 0 | 0 | PASS |
| huginn | zero | 2026110933 | 23096.633 | 125.560 | 132.986 | 117.980 | 24843.147 | 1.631 | 0.038 | 0 | 0 | PASS |

| Model | Metric | Zero / original Native | Paired starts | 95% interval |
| --- | --- | --- | --- | --- |
| ouro | TTFT_p95_ms | 0.961 | 3 | [0.949, 0.982] |
| ouro | ITL_p99_ms | 1.021 | 3 | [0.989, 1.063] |
| ouro | completed_QPS | 1.004 | 3 | [1.001, 1.005] |
| ouro | SLO_goodput | 1.105 | 3 | [1.004, 1.287] |
| huginn | TTFT_p95_ms | 0.736 | 3 | [0.725, 0.755] |
| huginn | ITL_p99_ms | 0.887 | 3 | [0.839, 0.949] |
| huginn | completed_QPS | 1.096 | 3 | [1.084, 1.115] |
| huginn | SLO_goodput | NOT_MEASURED | 3 | NOT_ESTABLISHED |

This control separates a provisioned Native path from the monitored Loop fallback path.
It does not prove that every enabled zero-offload Loop request has Native cost.

### Matched Native-path comparison

| Model | Comparison | Metric | Geometric ratio | 95% interval |
| --- | --- | --- | --- | --- |
| ouro | loop/zero | TTFT_p95_ms | 1.080 | [1.076, 1.082] |
| ouro | loop/zero | E2E_p95_ms | 1.063 | [1.057, 1.068] |
| ouro | loop/zero | ITL_p99_ms | 0.987 | [0.921, 1.038] |
| ouro | loop/zero | completed_QPS | 0.982 | [0.977, 0.985] |
| ouro | loop/zero | SLO_goodput | 0.933 | [0.869, 0.985] |
| huginn | loop/zero | TTFT_p95_ms | 1.359 | [1.321, 1.435] |
| huginn | loop/zero | E2E_p95_ms | 1.333 | [1.299, 1.401] |
| huginn | loop/zero | ITL_p99_ms | 1.328 | [1.168, 1.456] |
| huginn | loop/zero | completed_QPS | 0.911 | [0.885, 0.937] |
| huginn | loop/zero | SLO_goodput | 0.815 | [0.590, 1.096] |

The later A/B/A control uses the same Huginn arrival seed throughout.
Its order is original Native, provisioned zero-offload, original Native.
It tests a simple temporal-drift explanation for the combined path difference.
It does not isolate wire packing, buffer provisioning, or scheduling as the unique cause.

| Model | Arm | Arrival seed | TTFT p95, ms | ITL p99, ms | Max ITL, ms | TPOT p95, ms | E2E p95, ms | Requests/s | SLO-goodput | Offloaded requests | Full Loops | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | native | 2026110931 | 28381.615 | 142.380 | 145.773 | 133.602 | 30097.900 | 1.418 | 0.011 | 0 | 0 | PASS |
| huginn | zero | 2026110931 | 17116.554 | 122.847 | 133.016 | 115.977 | 18832.988 | 1.617 | 0.265 | 0 | 0 | PASS |
| huginn | native | 2026110931 | 25687.244 | 145.137 | 621.386 | 135.422 | 27420.437 | 1.460 | 0.057 | 0 | 0 | PASS |

The complete A/B/A ratios remain in `late_ABA_results.json`.
The B path reduced TTFT p95 by 36.686% versus the mean of the two A controls.
Its ITL p99 was 14.546% lower.
The A controls themselves differed by 9.493% in TTFT p95.
One A control also had a 621.386 ms Maximum ITL.
These observations support a combined path effect without claiming stable Maximum ITL or a unique component cause.

## 9. Actual completed offloading

| Model | Arm | Starts | Requests | Source armed | Actual offloaded requests | Full Loops | D recurrent rows | Loop-count distribution | Logged Prefill-phase misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | native | 3 | 384 | 0 | 0 | 0 | 0 | {} | 0 |
| ouro | loop | 3 | 384 | 384 | 3 | 6 | 24576 | {1: 1, 2: 1, 3: 1} | 0 |
| ouro | token | 3 | 384 | 8 | 1 | 0 | 12288 | {} | 0 |
| huginn | native | 3 | 384 | 0 | 0 | 0 | 0 | {} | 0 |
| huginn | loop | 3 | 384 | 384 | 142 | 853 | 1746944 | {1: 7, 2: 7, 4: 52, 8: 74, 16: 2} | 3 |
| huginn | token | 3 | 384 | 41 | 3 | 0 | 65536 | {} | 3 |

Token recurrent rows are not reported as completed full-prompt Loops.
Each Loop count requires all prompt tokens at that depth.
Recurrent rows are structural work accounting, not FLOPs.
Attention work, Prelude/Coda, and tensor shapes differ between Token and Loop.
Actual migrated FLOPs remain `NOT_MEASURED`.

Huginn exceeded the original 500-Loop exploration target.
Ouro did not reach that target.
Huginn completed 490, 333, and 30 Loops in its three starts.
Its larger follow-up target of 4000 Loops was not reached.
The 853 completed Loops represent 6.942% of Huginn's total prompt recurrent Loops.
Ouro's 6 completed Loops represent 0.391%.
These fractions do not equal FLOP fractions.

Each Ouro N128 start accepted only one request.
Its selected counts were 1, 3, and 2 Loops across independent starts.
Within-run mixed-cut correctness passed in N32, not in these sparse Ouro N128 starts.
Huginn's N128 starts did contain actual online mixed cuts and Native fallbacks.

## 10. Completion latency and transfer

| Model | Loop loans | Commit-to-output p50, ms | p95, ms | p99, ms | Max, ms | Max inter-quantum host wait, ms |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | 3 | 134.704 | 176.449 | 180.160 | 181.087 | 0.287 |
| huginn | 142 | 619.453 | 951.952 | 1532.929 | 2413.390 | 124.874 |

These are host observations.
They include transfer, import, scheduling, computation, and output generation.
They are not CUDA kernel occupancy.
The complete prediction audit is `recovery_completion_audit.csv`.
Its direct first-output audit found four Huginn deadline overruns among 142 Loop loans.
The Runtime logged three misses in its receiving/Prefill phases.
The difference identifies an incomplete end-to-end completion guard.
The report does not equate a missing Runtime miss event with an on-time first output.
The largest inter-quantum host wait was 124.874 ms for Huginn and 0.287 ms for Ouro.
The corresponding maximum commit-to-first-output intervals were 2413.390 ms and 181.087 ms.

| Model | Arm | Arrival seed | Offload KV bytes | Hidden bytes | Fragments | Descriptors | Commit-to-ACK p95 host ms | P host submissions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | loop | 2026110931 | 2415919104 | 16777216 | 37 | 57 | 57.088 | 511 |
| ouro | token | 2026110931 | 0 | 0 | 0 | 0 | NOT_MEASURED | 131 |
| ouro | loop | 2026110932 | 805306368 | 16777216 | 13 | 45 | 13.859 | 509 |
| ouro | token | 2026110932 | 805306368 | 4096 | 13 | 51 | 12.531 | 129 |
| ouro | token | 2026110933 | 0 | 0 | 0 | 0 | NOT_MEASURED | 131 |
| ouro | loop | 2026110933 | 1610612736 | 16777216 | 25 | 35 | 28.349 | 510 |
| huginn | loop | 2026110931 | 370684723200 | 3503554560 | 5616 | 543728 | 340.378 | 616 |
| huginn | token | 2026110931 | 4995809280 | 2724480 | 75 | 7035 | 1592.454 | 155 |
| huginn | loop | 2026110932 | 262896353280 | 2465464320 | 3979 | 385675 | 332.012 | 699 |
| huginn | token | 2026110932 | 4995809280 | 2724480 | 75 | 7021 | 2131.651 | 135 |
| huginn | token | 2026110933 | 1427374080 | 16241280 | 22 | 2010 | 929.830 | 132 |
| huginn | loop | 2026110933 | 17301504000 | 173015040 | 263 | 25351 | 542.529 | 881 |

Payload bytes count successful Native transport regions for committed offload seeds.
They exclude protocol headers and control messages.
Native PD also transfers KV.
The offload payload is not entirely incremental communication over a zero-transfer baseline.
Commit-to-ACK is not DMA latency.
It includes source preparation, transfer progress, destination processing, and ACK observation.

One primary Huginn loan predicted 459.735 ms but completed 1277.638 ms after commit.
Its import-to-output interval was 404.793 ms.
The roughly 873 ms before first continuation submission was therefore not recurrent kernel execution.
The cut moved 4930928640 KV bytes and 43253760 hidden bytes.
It used 75 fragments and 7243 descriptors.
The existing dedicated copy stream was already enabled.
The campaign did not isolate a unique cause for the remaining handoff delay.

## 11. P/D execution and remaining overhead

The source must expose a completed boundary before it can negotiate ownership.
The current atomic graph does not expose earlier hidden state retrospectively.
Checkpoint-capable zero-offload requests therefore still pay extra submission and boundary costs.
The feature-disabled path remains Native Atomic PD.
This distinction prevents a claim of free zero-offload execution.

In the first primary Ouro start, Native recorded 128 source prompt spans.
The new Loop path recorded 510 spans.
Huginn recorded 128 versus 634 spans.
These are host request-span observations, not additional recurrent work.
Full-work checks show no duplicate or missing recurrent rows.

`systems_accounting.csv` reports source submissions, queue samples, graph selections, wire bytes, ACK intervals, and HBM inventories.
Exclusive P compute saved, exclusive D compute added, and GPU critical-path saving remain `NOT_MEASURED` in clean confirmation.
Serial calibration and structural rows cannot supply those missing online quantities.

## 12. Physical headroom and target scale

The separate Oracle report is `ouro_huginn_pd_idle_slack_loop_offloading_oracle_final_summary.md`.
Full D-only captures completed but failed the 5% interference gate.
Both-stage captures also failed progress.
Their exact interaction remains `ROOT_CAUSE_INCONCLUSIVE`.
The campaign does not treat those traces as clean Native physical capacity.

Diagnostic D idle was 32.492% for Ouro and 57.089% for Huginn.
Those values describe perturbed executions only.
Clean Native D idle, Decode-safe capacity, and completion-feasible capacity remain `NOT_MEASURED`.
This limitation prevents attribution of sparse acceptance solely to absent physical slack.

The historical cut ceilings across 384 requests are 1152 Ouro Loops and 3072 Huginn Loops.
The targets are 500 and 4000 Loops.
The historical Huginn set cannot reach its target structurally.
The newly qualified R16 frontier raises its structural ceiling to 6144 Loops.
Neither ceiling is measured serving capacity.
The campaign does not claim either target achieved.

## 13. Figures and interactive comparison

The figure manifest is `figure_manifest.json`.
The primary comparison is `figures/primary_serving_comparison.png`.
The matched recovery comparison is `figures/recovery_serving_comparison.png`.
The actual budget/cut scatter is `figures/recovery_budget_vs_actual_loop_count.png`.
The prediction-error plot is `figures/recovery_completion_prediction_error.png`.
The serial cost plot is `figures/full_prompt_continuation_costs.png`.
The intrusive idle histogram marks its failed gate explicitly.

`figures/native_old_new_observed_timeline.html` compares the same primary arrival seed.
`figures/recovery_native_zero_loop_token_timeline.html` compares the same recovery arrival seed.
Both timelines distinguish source work, destination work, handoff, and output delivery.
Their bars are host enclosures.
Blank space is not measured GPU idle.
Separate GPU-clock diagrams also identify enclosures rather than exclusive kernels.

## 14. Reference reuse and novelty

[FluidPD](https://arxiv.org/html/2610.06917v1) motivates pressure, KV-aware admission, and suffix completion checks.
The inspected sources did not establish an official public implementation for reuse.
Our Token arm remains a FluidToken-style adaptation.
It is not the original FluidPD system.

The reviewed [SGLang decode path](https://github.com/sgl-project/sglang/blob/fe30d1e0fd4ede090b48a4147d23e23571b327f2/python/sglang/srt/disaggregation/decode.py) separates preallocation, transfer pending state, and runnable Decode.
The redesign uses that lifecycle principle without transplanting incompatible request or KV objects.
The local `ref_loopserve` review covered its scheduler, recurrent planner, ownership, graph controller, and grouped execution.
Its Early Exit path and incompatible KV semantics remain disabled.

Generic logical-layer scheduling can reproduce the same recurrent boundaries and state transfers.
This case is `EQUIVALENT_MECHANISM`.
The campaign does not establish a Loop-specific novelty claim from boundary names or recurrent-row counts.
It also does not combine generic deflection and depth scheduling into an unsupported new-method claim.

## 15. Failure scope and next decision

The desired useful large-scale TO-BE remains unconfirmed against the provisioned Native control.
The current implementation still has checkpoint overhead and imperfect completion predictions.
No result establishes Loop superiority over a strong optimal Token-only Scheduler.
The physical-capacity verdict remains `INCONCLUSIVE_MEASUREMENT`.
The implementation verdict is `NO_GO_NET_BENEFIT` for Ouro and Huginn under these tested conditions.
This verdict does not mean `NO_GO_PHYSICAL_HEADROOM`.
The physical Oracle remains `INCONCLUSIVE_MEASUREMENT`.

| Model | Tested serving verdict | Physical-capacity verdict | Main reason |
| --- | --- | --- | --- |
| Ouro | NO_GO_NET_BENEFIT | INCONCLUSIVE_MEASUREMENT | Only 6 Loops moved. TTFT and goodput lose to the provisioned zero-offload path. |
| Huginn | NO_GO_NET_BENEFIT | INCONCLUSIVE_MEASUREMENT | Real 853-Loop movement does not beat the provisioned path. Decode p99 also rises. |

Huginn improved finite-cohort joint goodput versus the original Native path.
Its mean paired gain was 0.066788 requests/s, with a 95% interval of [0.022551, 0.119581].
One original Native start had zero goodput.
The analysis therefore reports a difference rather than an undefined geometric ratio.
The gain did not establish superiority over the provisioned zero-offload path.
Huginn's ITL p99 ratio to original Native was 1.1773, with [1.0226, 1.3061].
It therefore did not establish the requested Native-relative tail protection.

The common Loop-specific research verdict is no GO from this campaign.
The exact continuation mechanism works.
The measured Runtime-path improvement without migration is the stronger practical result.
It should become the no-offload control before further Dynamic Scheduler work.

The run inventory records complete, failed, and not-run cells with source and configuration hashes.
`configuration_inventory.csv` also identifies prepared configurations that never executed.
One Huginn seed-31 Loop start overlapped a 4.79-second host audit near measurement start.
The campaign excluded that start before inspecting its performance.
It retained the raw files and repeated the same policy and arrivals in a new namespace.
The clean replacement contributes to the final table.
The inherited control auditor initially failed on a null partial-cut field.
The Runtime correctly resolves that field to full depth.
A scoped analysis view applies the same resolution without changing the immutable configuration.
The retained GPU outputs then passed the unchanged state and lifetime gates.
Invalid full traces, earlier failed qualifications, and stopped Token starts remain preserved.
Low/medium load recovery comparisons, a clean N512 persistence test, and GPU fault-recovery qualification remain unexecuted.
A separate adaptive hybrid logical-frontier comparison also remains unexecuted.
Token-tiled execution of an accepted Loop suffix is not that separate hybrid comparison.
No sustainable-capacity or general-text claim follows from N128 finite cohorts.

The next step is not a larger Dynamic campaign by default.
It first requires a lower-cost checkpoint path or a demonstrated completion advantage over continued P execution.
It also requires reliable physical measurement if the claim concerns unused hardware capacity.
Increasing continuation slots alone does not resolve either requirement.

## 16. Reproduction, publication, and resource restoration

The evidence directory is `ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation/`.
It records source snapshots, configuration hashes, GPU maps, arrival seeds, request inventories, and qualification results.
The lossless bundle preserves raw evidence without changing the original files.
`evidence_bundle --verify` checks reconstructed file hashes.

The publication branch is `flashvep/completion-aware-adaptive-loop-pd`.
The publication parent is `0adb7491c866d6d66cf71ea30dca63f82c46d135`.
The publication uses a temporary Git index and leaves the active checkout unchanged.
The staged-entry hash and external runtime diff hash receive separate preservation checks.
The publication receipt records the remote verification.
The final Burner receipt checks actual engine ownership, generation progress, and utilization on all eight GPUs.
The live Burner processes continue after the final response.
