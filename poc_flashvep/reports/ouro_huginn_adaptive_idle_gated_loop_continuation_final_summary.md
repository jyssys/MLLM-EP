# Adaptive idle-gated Loop continuation: final evidence and unresolved gates

Campaign status: GPU_EXPERIMENTS_CLOSED.

## 1. Decision

**Ouro: INCONCLUSIVE_SLACK. Huginn: INCONCLUSIVE_SLACK. Common method: INCONCLUSIVE_COMPARISON.**

This campaign did not demonstrate the requested online Adaptive Loop-Split.
The runtime completed exact forced heterogeneous Loop continuations.
It did not select those cuts from a measured, validated online Safe Slack budget.
No reported primitive or forced-cut request establishes an adaptive serving gain.
These results are not proof that useful D headroom cannot exist.

| Gate | Ouro | Huginn |
| --- | --- | --- |
| 1: same-run P pressure and deadline-safe D budget | INCONCLUSIVE_SLACK | INCONCLUSIVE_SLACK |
| 2: variable nonzero online accepted Loop counts | NOT_TESTED | NOT_TESTED |
| 3: adaptive Decode protection | NOT_TESTED | NOT_TESTED |
| 4: net TTFT or goodput gain | NOT_TESTED | NOT_TESTED |
| 5: Dynamic Loop versus Dynamic Token | INCONCLUSIVE_COMPARISON | INCONCLUSIVE_COMPARISON |

The failed Gate 1 evidence contract stopped method expansion.
No new Dynamic Scheduler should use these unqualified budgets as a safety guarantee.
Generic logical-layer execution at identical tied-weight boundaries remains EQUIVALENT_MECHANISM.
No current record establishes a Loop-specific serving advantage.

## 2. Provenance, clock, and isolation

Publication parent: `3b3633dbc64bcd916c2ba1ff513deb4226cc647d`. Branch: `flashvep/adaptive-idle-gated-loop-continuation-poc`.
Preserved checkout HEAD: `f4ecea59f36f620c69c4fef09c590bf790802cd9`.
Preserved private runtime HEAD: `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The private runtime is `/home/esjung/external/looped-pd-abc-rlt`.
The source archives and configuration inventory identify each executed revision.
The new branch uses a separate Git index. The original checkout and staged entries remain unchanged.

The campaign reuses qualified Native graph replay, Token/Loop connectors, NIXL handoff, and exact state validators.
No external runtime source or checkpoint changed.
The `ref_loopserve` code uses a different runtime interface. No compatible boundary-negotiation implementation was established for direct reuse.
Independent D-local probes and cross-depth batching are not substitutes for an actual adaptive P-to-D continuation.

The fixed wall-clock budget started at 2026-11-06 13:05:11 KST.
The final deadline is 21:05:11 KST. The GPU experiment deadline is 20:15:11 KST.
The controller completed its configured calibration list before the 18:58:13 resumption inspection.
The monitor and deadline guard remained active, but they did not schedule further research.
This idle interval remains part of the budget. It is not GPU experiment time.
See [the operational incident](ouro_huginn_adaptive_idle_gated_loop_continuation/resumption_and_burner_incident.md).

The host has eight H100 80GB GPUs. Each scientific job uses an isolated two-GPU pair.
All approved burners stop during those measurements.
Heavy CPU trace analysis uses the same exclusive lane and does not overlap a GPU experiment.
Driver version: 570.211.01. CUPTI collector version: CUDA 12.8.
The environment inventory retains topology, UUIDs, and process ownership.

Python: `3.14.6 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:37:38) [GCC 14.3.0]`.
Torch: `2.11.0+cu129`. Torch CUDA build: `12.9`.
See `software_versions.json` for NIXL, Triton, and other installed package versions.

## 3. Models, requests, and frozen SLO

| Model | Precision / depth | Input / output | Low / medium / heavy QPS | Physical P/D pair | P/D slots |
| --- | --- | --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | Original BF16 / R4 | 4096 / 32 | 1.23460875 / 3.4569045 / 6.356 | 0 / 1 | 2 / 8 |
| tomg-group-umd/huginn-0125 | Original BF16 / R32 | 2048 / 16 | 0.37903355455 / 1.06129395274 / 1.971 | 4 / 5 | 2 / 8 |

The primary pool contains 128 distinct ShareGPT/LongBench controlled-prefix prompt IDs per model.
The campaign reuses that text pool. It does not establish unseen-text generalization.
Prefix caching is disabled. Output caps ignore EOS.
Calibration uses eight disjoint text IDs per model and serial requests.
Poisson offsets regenerate from each recorded QPS and seed. Paired controls retain identical offsets.
No N512 campaign ran. No natural-EOS accuracy or original-HF equivalence test ran.

Pinned Ouro revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Pinned Huginn revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
The native contract inventory records actual precision, depth, revision, and prefix-cache state.
Source defaults alone do not substitute for that runtime record.

| Model | TTFT ms | Request TPOT ms | Additional-work TPOT p95 ms | ITL p99 ms |
| --- | --- | --- | --- | --- |
| ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

The Native-relative ITL p99 design target remains at most 5% degradation.
The observer gate separately requires an absolute difference within 5% for all fixed local-tail windows.
The windows cover TPOT p95, ITL p95/p99, and maximum ITL.
The ITL p99 threshold is not a hard deadline guarantee for every token.

## 4. Native idle evidence and observer interference

| Collector | Matched cohorts | Qualified local tails and coverage |
| --- | --- | --- |
| host | 2 | 1 |
| sparse | 2 | 0 |
| graph | 10 | 0 |
| submit_events | 8 | 0 |

A host-only Huginn control passed the observer screen. It does not measure physical GPU idle.
No graph or complete submission-event trace passed the complete current observer contract.
Some complete traces changed local tails despite small cohort-duration changes.
Some event traces improved measured tails by more than 5%. They also fail the absolute equivalence screen.
A failed screen does not isolate instrumentation as the sole causal source of variation.

| Minimal-host event pilot | Minimum local ratio | Maximum local ratio | Cohort duration ratio |
| --- | --- | --- | --- |
| ouro | 0.9809017180258613 | 1.068204736202201 | 1.006933353856532 |
| huginn | 0.6371727660081676 | 0.8803734567781655 | 0.9922173837982168 |

The final-flush correction recovered complete buffers in some runs.
The Ouro drained pilot returned 2/2 buffers with zero invalid records.
The Huginn drained pilot returned 10/10 buffers but contained one invalid timestamp record.
The latter run completed its response cohort before export failed.
This is not a model-correctness failure. Its invalid-record cause remains ROOT_CAUSE_INCONCLUSIVE.
The retained audit distinguishes collector plumbing, coverage, perturbation, and model correctness.

Physical kernel idle, scheduler-visible idle, and deadline-safe future execution remain separate evidence classes.
Graph envelopes include their internal gaps. CUDA Event enclosures also include waits and host submission gaps.
Their complements are bounds, not exact kernel-idle measurements or future execution guarantees.
The eligible traces remain diagnostic. A valid same-run Safe Slack distribution is NOT_MEASURED.

| Required opportunity distribution | Ouro | Huginn |
| --- | --- | --- |
| Count and duration p50/p90/p95/p99 | NOT_MEASURED | NOT_MEASURED |
| Fractions at least 1/2/4/8/16/32 ms | NOT_MEASURED | NOT_MEASURED |
| Same-run validated P-backlog plus safe future budget | NOT_MEASURED | NOT_MEASURED |

Native A/A repeats diagnose run-to-run variation. They do not relax the frozen observer gate.
They use the earlier minimal-host pilot as their reference. They do not isolate a simultaneous instrumentation effect.
The host-observer ablation removes live HBM queries and optional transport observations without changing model computation.
Its raw GC and long-call records remain diagnostic. Startup GC intervals do not count as measured-cohort pauses.
See `measured_host_observer_intervals.csv` and `state/finish_controls.json`.

| Native A/A repeat | Minimum local ratio | Maximum local ratio | Cohort duration ratio | Inside pointwise 5% band |
| --- | --- | --- | --- | --- |
| diag_ouro_native_minimal_AA_start2_n128 | 1.0214414757855257 | 1.0904275710240205 | 1.0318471125682245 | False |
| diag_huginn_native_minimal_AA_start2_n128 | 0.6132595565720482 | 0.8842766175233213 | 0.9941525742873154 | False |
| diag_ouro_native_minimal_AA_start3_n128 | 0.9952588688809262 | 1.0451909013961742 | 1.0037049126051432 | True |
| diag_huginn_native_minimal_AA_start3_n128 | 0.607223820195756 | 0.8745818715838142 | 0.9949019377697409 | False |

| Model | Window | Metric | Independent starts | Mean observer/Native ratio | 95% CI |
| --- | --- | --- | --- | --- | --- |
| ouro | WHOLE_COHORT | ITL_p99_ms | 3 | 1.014631718102113 | [1.0068930766635906, 1.0296755491104521] |
| ouro | WHOLE_COHORT | max_ITL_ms | 3 | 0.5392603241162214 | [0.2570535585199968, 1.0178425925543382] |
| ouro | BOUNDED_CAPTURE_REGION | ITL_p99_ms | 3 | 1.0181731026535765 | [1.0000578547617913, 1.0431568876056536] |
| ouro | BOUNDED_CAPTURE_REGION | max_ITL_ms | 3 | 1.027253209868432 | [1.0175214340431453, 1.037675275771241] |
| huginn | WHOLE_COHORT | ITL_p99_ms | 3 | 1.06642995093025 | [1.0077902769951557, 1.1599233379030633] |
| huginn | WHOLE_COHORT | max_ITL_ms | 3 | 1.1966094220754453 | [0.9891071904754066, 1.4562860067727268] |
| huginn | BOUNDED_CAPTURE_REGION | ITL_p99_ms | 3 | 1.0623321183540633 | [0.9509453466320379, 1.1605107821242548] |
| huginn | BOUNDED_CAPTURE_REGION | max_ITL_ms | 3 | 1.1804042496688467 | [0.9213615147495055, 1.4784583521178125] |

These intervals resample independent start-level ratios, not tokens or requests.
Three starts provide limited uncertainty evidence. The intervals do not override the frozen pointwise observer gate.
No method-improvement confidence interval exists because no adaptive method cohort ran.

## 5. Full-prompt continuation primitive costs

| Model | Remaining loops on D | P event sum p50 ms | D event sum p50 ms | Host handoff p50 ms | Serial Service TTFT p50 ms |
| --- | --- | --- | --- | --- | --- |
| ouro | 0 | 182.20117950439453 | NOT_MEASURED | 19.057401455938816 | 204.4997438788414 |
| ouro | 1 | 137.42704010009766 | 47.22286415100098 | 12.077894993126392 | 214.1580367460847 |
| ouro | 2 | 92.23527908325195 | 92.1466064453125 | 10.209170170128345 | 218.97671278566122 |
| ouro | 3 | 48.75259208679199 | 136.1998519897461 | 4.518185742199421 | 208.27523060142994 |
| huginn | 0 | 582.2918090820312 | NOT_MEASURED | 81.98598492890596 | 668.0614715442061 |
| huginn | 1 | 554.8459167480469 | 27.318527221679688 | 56.76843784749508 | 645.9848461672664 |
| huginn | 2 | 537.1302185058594 | 44.746511459350586 | 87.8748120740056 | 679.0050072595477 |
| huginn | 4 | 501.95106506347656 | 79.68392181396484 | 90.29810596257448 | 684.6898188814521 |
| huginn | 8 | 431.0397186279297 | 149.8342742919922 | 94.00095976889133 | 694.8589524254203 |

Each row contains eight serial requests from one independent start.
CUDA Events measure model-stream enclosures, not exclusive kernel compute.
D prompt enclosures exclude transfer and separate first-output delivery.
Huginn terminal continuation includes its required prompt Coda-KV work.
Host handoff is an observed elapsed interval, not pure network time or incremental cost over Native.
Do not add the component percentiles into a critical-path latency.
Cross-start confidence intervals are NOT_MEASURED for these primitives.

The original Huginn one-loop serial row has lower Service TTFT than its Native control.
The later one-loop-quantum control has a different handoff interval and higher Service TTFT.
Their D compute enclosures remain similar. Compute timing alone therefore does not establish the request-latency outcome.
These independent single-start observations do not isolate a recurrent redistribution benefit.

A full-prompt loop is much larger than the earlier 16-token independent probe.
Small-tile timing cannot establish whole-prompt admission safety.
A loop longer than one physical idle gap also does not prove that deadline-safe borrowing is impossible.
Host preparation can overlap device execution. A validated budget or actual insertion test must establish that opportunity.

## 6. Matched one-loop execution quantum: Token versus Loop

| Model | Family | D recurrent rows/request | P saved enclosure p50 ms | D total p50 ms | Largest D submission ms | Payload GiB | Serial Service TTFT p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | WHOLE_PROMPT_LOOP | 4096.0 | 44.99879455566406 | 46.198448181152344 | 46.336063385009766 | 2.265625 | 211.3056480884552 |
| ouro | TOKEN_SUFFIX | 4096.0 | 48.24736022949219 | 54.348544239997864 | 14.161408424377441 | 2.2500038146972656 | 207.68184587359428 |
| huginn | WHOLE_PROMPT_LOOP | 2048.0 | 28.26446533203125 | 27.25334358215332 | 28.50271987915039 | 5.115966796875 | 700.3198591992259 |
| huginn | TOKEN_SUFFIX | 2048.0 | 9.7674560546875 | 81.1085913181305 | 5.58131217956543 | 5.151863694190979 | 722.2203640267253 |

Ouro compares Loop3 against Token3072. Huginn compares Loop31 against Token1984.
Both families use the same legal D recurrent execution quantum of one loop.
Both passed their separate N8 state tests.
This is a matched-quantum diagnostic, not a search for the best Token baseline.
Larger Token suffixes and other loop quanta require separate cost and Decode-safety tests.
Equal recurrent token rows do not imply equal attention FLOPs.
A Token suffix attends to the existing prefix at every recurrent depth.
The total D enclosure and the largest scheduling quantum measure different costs.
A lower total Loop cost does not establish lower Decode interference.
These serial records provide no concurrent-Decode or adaptive serving advantage.

| Model | Family | Structural D attention pairs/core layer | D Huginn Prelude tokens | D Huginn Coda-KV tokens |
| --- | --- | --- | --- | --- |
| ouro | WHOLE_PROMPT_LOOP | 8390656 | 0 | 0 |
| ouro | TOKEN_SUFFIX | 14682112 | 0 | 0 |
| huginn | WHOLE_PROMPT_LOOP | 2098176 | 0 | 2047 |
| huginn | TOKEN_SUFFIX | 4129792 | 64 | 63 |

These counts are structural accounting, not measured FLOPs or speedups.
Huginn Loop31 also moves substantially more prompt Coda-KV work than Token1984.
Token1984 moves suffix Prelude work that Loop31 does not move.
Therefore, equal recurrent rows do not provide an equal-total-work causal comparison.
The measured total-cost difference cannot establish a Loop-specific advantage by itself.

## 7. Decode service diagnostic

| Model | Serial Decode steps | Delivery ITL p50 ms | Event-sum p50 ms | Per-step residual p50 ms |
| --- | --- | --- | --- | --- |
| ouro | 248 | 26.896988041698933 | 25.749808210879564 | 1.1602303013205528 |
| huginn | 120 | 94.44984514266253 | 73.29704004526138 | 21.195184908807278 |

The residual is delivery ITL minus the sum of recorded operation enclosures for that token.
It is not physical idle or an available online borrowing budget.
The sum can include stream waits. Medians in this table are not additive.
Huginn has more host-side recurrent submissions than Ouro in this serial path.
This diagnostic does not isolate CPU scheduling as the cause of online tail latency.

## 8. Forced mixed-cut correctness

| Run | Qualified | D prompt recurrent rows | Forced offloaded requests | Remaining-loop counts |
| --- | --- | --- | --- | --- |
| q_ouro_native_state8 | True | 0 | 0 | [0] |
| q_ouro_forced_mixed_state8 | True | 49152 | 6 | [0, 1, 2, 3] |
| q_ouro_native_state32 | True | 0 | 0 | [0] |
| q_ouro_forced_mixed_state32 | True | 196608 | 24 | [0, 1, 2, 3] |
| q_huginn_native_state8 | True | 0 | 0 | [0] |
| q_huginn_forced_mixed_state8 | True | 59392 | 7 | [0, 1, 2, 4, 8] |
| q_huginn_native_state32 | True | 0 | 0 | [0] |
| q_huginn_forced_mixed_state32 | True | 208896 | 26 | [0, 1, 2, 4, 8] |

The cohort uses preassigned cuts. It does not observe a Safe Slack estimator.
The reused connector communicates those cuts during reservation.
It does not renegotiate a cut after observing each completed P loop.
A host graph submission boundary does not establish GPU completion at that boundary.
N32 means simultaneous submission, not 32 GPU-resident requests.
The state tests compare exact output tokens, valid prompt KV planes, and recorded hidden state.
They also validate full recurrent rows, transfer fingerprints, ACK-before-retirement, and empty final native allocations.
Static full-PD branches are forced fallbacks, not live rejection or stale-credit tests.
Real transfer-time Decode arrivals, overruns, queue aging, and adaptive admission remain NOT_TESTED.
No intermediate kernel preemption or D-to-P return is assumed.

## 9. Native finite-cohort serving controls

| Model | Load | Offered QPS | Realized QPS | Completed QPS | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Max ITL ms | E2E p95 ms | Joint goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | low | 1.23460875 | 1.2413652421651662 | 1.2377458495792815 | 545.0737068429588 | 32.50817664027694 | 36.1703459918499 | 39.96513783931732 | 1495.0931481085718 | 1.2377458495792815 |
| ouro | medium | 3.4569045 | 3.4758096239704606 | 3.3980614838776035 | 807.8037423081695 | 35.14630081913164 | 38.54918150231242 | 40.83617031574249 | 1773.5643941909075 | 3.3980614838776035 |
| ouro | heavy | 6.356 | 6.390631543596266 | 5.191182232356035 | 3858.383823465556 | 38.91747108089828 | 42.89916731417179 | 47.80862107872963 | 5047.677669674158 | 0.9327905573764751 |
| huginn | low | 0.37903355455 | 0.38110918998829696 | 0.38146348455024504 | 1679.801720939576 | 112.83407146732013 | 120.76582292094827 | 130.63537515699863 | 3317.785685509443 | 0.36954275065804987 |
| huginn | medium | 1.06129395274 | 1.067100716322969 | 1.0560563646471512 | 2507.9255295917383 | 115.7839812586705 | 122.28915642946959 | 130.24079240858555 | 4223.374410159885 | 0.9075484383686456 |
| huginn | heavy | 1.971 | 1.9817905267180296 | 1.619101223154597 | 13784.632336627692 | 118.94620910907786 | 125.85138382390141 | 130.55100850760937 | 15527.191992290318 | 0.20238765289432462 |

These are Native controls, not Dynamic Loop results. Each complete primary cohort offers 128 requests.
The load grid uses the first declared Phase A arrival seed, 2026110611, at every QPS.
Minimal-host ablations and later observer seeds remain separate controls.
The complete CSV includes p50/p95/p99 latency, token throughput, SLO counts, and threshold exceedances.
All offered requests remain the SLO denominator.
The heavy setting can accumulate a finite overload queue. Completed cohort throughput is not sustainable capacity.
Primary marker-free Service TTFT and actual-first-model Queue Latency are NOT_MEASURED.
Admission timestamps alone do not replace actual model entry.
The separately instrumented serial Service TTFT is not a replacement for clean online Service TTFT.

## 10. Oracle status

The cost-fit diagnostic uses measured serial costs and optimistic recorded gap envelopes.
It removes graph compute and ignores handoff, frontier timing, and admission costs equally for both action families.
It is a clairvoyant relaxed cost-fit bound, not a Feasible Oracle.
No calibrated prediction of online TTFT, E2E, goodput, or Decode tail is reported.
Validation error is NOT_MEASURED. No unvalidated oracle speedup appears as an online result.
No new token-tiled variable-loop primitive ran because whole-prompt infeasibility was not established.
Missing slack evidence cannot trigger a claim that full-prompt loops never fit.

Diagnostic action count: 11. Diagnostic comparison rows: 17.
See `relaxed_opportunity_oracle.csv` and `offline_budget_loop_decisions.csv`.
Their offline cost-fit selections are not accepted handoffs or actual online cut decisions.

## 11. Memory and ownership

| Model | Role | Component | GiB | Percent of NVML HBM |
| --- | --- | --- | --- | --- |
| huginn | prefill | weight_bytes | 6.6402867436409 | 8.337097837747251 |
| huginn | prefill | reserved_kv_bytes | 41.25 | 51.79072818450447 |
| huginn | prefill | peak_allocated_bytes | 49.557697772979736 | 62.221315268126446 |
| huginn | decode | weight_bytes | 6.6402867436409 | 8.337097837747251 |
| huginn | decode | reserved_kv_bytes | 67.353515625 | 84.5645483637612 |
| huginn | decode | peak_allocated_bytes | 74.29233503341675 | 93.27646375534123 |
| ouro | prefill | weight_bytes | 2.6722488421946764 | 3.35509608308997 |
| ouro | prefill | reserved_kv_bytes | 48.0 | 60.265574614696106 |
| ouro | prefill | peak_allocated_bytes | 52.437626361846924 | 65.83716008598836 |
| ouro | decode | weight_bytes | 2.6722488421946764 | 3.35509608308997 |
| ouro | decode | reserved_kv_bytes | 48.0 | 60.265574614696106 |
| ouro | decode | peak_allocated_bytes | 51.3661904335022 | 64.49193713006075 |

The denominator is the measured NVML capacity: 81,559 MiB per GPU.
Allocator peak totals include component allocations. Do not stack them with weights and KV pool bytes.
A receiving request already reserves its D KV lease before P starts.
Its continuation must not require a second full-request lease.
D compute slots and KV lease capacity are different limits.
Recorded ACK protects source lifetime. No optimization releases an in-flight source buffer early.
No proof of a future SLO-safe workspace or KV budget follows from average free HBM.

## 12. Failures, missing experiments, and attribution

The complete configuration inventory retains every failed and unrun cell.
Interrupted controller ledgers can retain RUNNING status. Dispatch receipts and the closure receipt establish their actual final state.
Misrouted early collector jobs completed Native serving rather than their intended synthetic self-test.
Only the revised `_v2` self-tests establish their stated collector plumbing results.
An empty or partial trace never becomes an idle GPU interval.
The superseded invalid derived gap table remains named and preserved.
Final-export failures remain distinct from model-state errors and serving-body progress stalls.

The untested method arms are Static Token/Loop clean serving under this campaign, Dynamic Idle-gated Token, and Dynamic Idle-gated Loop.
Also untested are an adaptive mixed-cut policy, delayed-ACK fault injection, stale-credit rejection, and N512 persistence.
P exclusive kernel union, SM active cycles, bandwidth saturation, and an online budget validation are NOT_MEASURED.
No current same-run P compute-saturation plus accepted adaptive D-work proof is reported.
The campaign does not change earlier accepted operational evidence into a new same-run result.
It also does not reinterpret earlier Dynamic zero-offload campaigns as successful adaptive work redistribution.

Observed costs establish a coarse full-prompt continuation quantum and nonzero state-transfer cost.
They do not establish that granularity or transfer alone makes the method impossible.
The relative contributions of CPU scheduling, recurrent structure, transfer progress, and tail jitter remain unresolved.
The result is INCONCLUSIVE, not NO_GO_IDLE_SLACK or NO_GO_NET_BENEFIT.
A future campaign must first qualify a low-overhead same-run observer or validate actual conservative admission directly.
It must then demonstrate genuinely variable online accepted cuts before claiming an Adaptive Loop-Split result.

## 13. Figures, reproducibility, and resource closure

| Figure | Evidence class |
| --- | --- |
| [figure_1_native_activity_idle_bounds](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_1_native_activity_idle_bounds.png) | Measured graph/eager activity and host pressure. Graph complements bound physical idle. Future deadline-safe slack remains unmeasured. Diagnostic traces remain labeled. |
| [figure_2_offline_budget_loop_fit](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_2_offline_budget_loop_fit.png) | Clairvoyant relaxed cost-fit diagnostics only. No point represents an actual accepted adaptive serving handoff. |
| [figure_3_actual_serial_continuation](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_3_actual_serial_continuation.png) | Actual first measured serial requests. Event enclosures include waits. Native transfer precedes D entry. No fabricated per-kernel or adaptive schedule. |
| [figure_4_native_load_controls_and_missing_arms](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_4_native_load_controls_and_missing_arms.png) | Measured finite Native cohorts. No Dynamic Loop/Token performance is invented. Primary marker-free Service TTFT remains unmeasured. |
| [figure_5_equal_quantum_token_loop_cost](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_5_equal_quantum_token_loop_cost.png) | Measured state-qualified serial costs when available. Same recurrent rows are not equal FLOPs. All costs include the existing numerical fixes. |
| [figure_6_serial_cost_components](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_6_serial_cost_components.png) | P reductions use matched-request serial enclosure differences. D and handoff show separate distributions. Do not add component percentiles or infer online TTFT gains. |
| [figure_7_cost_fit_not_safe_region](ouro_huginn_adaptive_idle_gated_loop_continuation/figures/figure_7_cost_fit_not_safe_region.png) | Measured best-observed serial costs compared with hypothetical budgets. Handoff and safety are absent. Blue cells do not establish feasible or SLO-safe offloading. |

The figures label diagnostic traces, serial enclosures, optimistic cost fits, and missing online arms.
No figure shows an invented adaptive execution timeline or measured speedup.
The operational restoration receipt records actual GPU engines, ownership, and utilization on all eight GPUs.
It is separate from scientific GPU evidence.

Analysis commands from the repository root:

```bash
PY=/home/esjung/.venvs/looped-pd-abc/bin/python
$PY -m poc_flashvep.adaptive_idle_loop_continuation.cpu_regression
$PY -m poc_flashvep.adaptive_idle_loop_continuation.analyze
$PY -m poc_flashvep.adaptive_idle_loop_continuation.primitive_analysis
$PY -m poc_flashvep.adaptive_idle_loop_continuation.decode_service_diagnosis
$PY -m poc_flashvep.adaptive_idle_loop_continuation.opportunity_oracle
$PY -m poc_flashvep.adaptive_idle_loop_continuation.inventory
$PY -m poc_flashvep.adaptive_idle_loop_continuation.figures
```

Existing immutable namespaces must not start new GPU jobs after closure.
A scientific GPU rerun requires a new authorized namespace and explicit new clock.
The archived per-run source, configuration hash, model revision, GPU mapping, and arrival offsets define each original run.
Lossless archive manifests retain oversized originals without deleting local files.
The ordinary push receipt records the evidence commit and matching remote SHA. No force push is used.

## 14. Answers to the four research questions

1. A validated same-run deadline-safe recurrent execution budget is not established. Diagnostic idle bounds alone are insufficient.
2. Different forced Loop counts are exact, but actual online Safe Slack-based variable Cut selection is not demonstrated.
3. Adaptive TTFT, E2E, goodput, and token-throughput gains with Native-like Decode tails remain unmeasured.
4. The serial Token/Loop comparison exposes a total-cost versus execution-quantum trade-off. No fair Dynamic Serving comparison establishes a winner.
