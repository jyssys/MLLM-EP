# Adaptive idle-time Loop continuation: online intervention and Token comparison

Scientific verdict: **NO_GO_NET_BENEFIT on the tested implementation and workloads, for both models.**

Actual variable-cut continuation passes N32 state qualification in both models.
All twenty planned N128 confirmation starts complete, totaling 2560 requests.
Decode protection does not produce a useful repeated improvement over Native PD.
Ouro remains approximately unchanged. Huginn loses TTFT, E2E, throughput, and goodput.
This is not proof that physical D slack is absent or that every possible Loop scheduler must fail.

## 1. Scope and clock

This campaign evaluates actual online boundary negotiation in two fixed-depth Looped Transformers.
It does not repeat the older completed Whole-Prompt Split or Cross-depth Batching campaigns.
The publication parent is `ddbf49ae3ef610fefd78248d765a27641b21b98f`.
The new branch is `flashvep/adaptive-idle-intervention-dynamic-loop-poc`.

The original interval starts at 2026-11-08 18:07:15 KST and ends at 2026-11-09 08:07:15 KST.
Its queued GPU work ends at 00:16:47 KST. This is not fourteen hours of continuous scientific GPU execution.
The user then requests completion of the unfinished PoC.
The separate resumption starts at 2026-11-09 13:00:54 KST.
Its GPU deadline is 16:00:54 KST. Its publication deadline is 17:00:54 KST.
The original clock remains frozen. The resumption does not reset it.

The original start cutoff is 15:40:54 KST.
A recorded completion-window amendment permits only already declared jobs whose startup, timeout, and 60-second cleanup margin fit before the GPU deadline.
It changes start scheduling, not either hard deadline or any scientific configuration.
The frozen configuration hashes, seeds, SLO, and inclusion rules remain unchanged.

The [specification](ouro_huginn_adaptive_idle_intervention_dynamic_loop_poc_spec.md) defines the fixed contracts.
The evidence directory is [the campaign namespace](ouro_huginn_adaptive_idle_intervention_dynamic_loop/).

## 2. Fixed models, workload, and evaluation contract

| Model | Depth | Input / output | P/D GPUs | P/D slots | Heavy offered QPS |
| --- | ---: | ---: | --- | --- | ---: |
| ByteDance/Ouro-1.4B | R4 | 4096 / 32 | 0/1 | 2/8 | 6.356 |
| tomg-group-umd/huginn-0125 | R32 | 2048 / 16 | 4/5 | 2/8 | 1.971 |

Both models retain original BF16 weights, exact Attention, qualified numerical fixes, and full recurrent work.
The Ouro revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The Huginn revision is `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
The qualified external runtime HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its recorded on-disk diff remains unchanged. New runtime hooks exist only in campaign worker processes.
There is no quantization, early exit, approximation, or reverse migration.
The 128 controlled ShareGPT/LongBench prompt IDs remain fixed.
Prefix caching is disabled. Output length uses a fixed cap with ignored EOS.
The sampling seed starts at 2026110362 and increments with the request index.

Calibration uses arrival seed 2026110910.
Held-out confirmation uses paired arrival seeds 2026110911, 2026110912, and 2026110913.
Only arrivals are held out. Reusing the same prompt pool does not establish unseen-text generalization.
All primary arrivals follow an open-loop Poisson process.
N32 simultaneous submissions mean concurrent submission, not 32 simultaneous GPU executions.

| Model | TTFT limit, ms | Request TPOT limit, ms | Additional-work TPOT p95, ms | ITL p99 limit, ms |
| --- | ---: | ---: | ---: | ---: |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

The additional target is ITL p99 at most 1.05 times paired Original Native PD.
This is a design target, not a guaranteed future deadline bound.
The absolute limits never change.
Heavy-load cohorts can violate the TTFT limit because requests accumulate.
Completed QPS and Token/s describe finite-cohort drain rates. They are not sustainable serving capacity.

Primary metrics use marker-free output-delivery timestamps and bounded host records.
ITL follows the frozen event-average delivery contract when an event contains multiple tokens.
Maximum ITL and threshold-exceedance counts use that same contract.
They are not exact individual GPU token-generation timestamps.
Primary Service TTFT, exact Queue latency, physical kernel union, SM activity, and memory bandwidth are NOT_MEASURED.

## 3. What the implementation actually does

The source completes a legal Native recurrent boundary before it offers state.
A non-null Native CUDA event must report completion.
The destination then evaluates current Decode state, resource credit, and the remaining-work estimate.
Prepare, commit, import, and ACK use transfer IDs, epochs, and bounded leases.
The source cannot resume duplicated work while commit is unresolved.
The destination owns the accepted remainder after commit.
Native ACK precedes source retirement. Final KV allocations and both coordinator credits must retire.

The bounded source checkpoints are Ouro R1/R2/R3 and Huginn R24/R28/R30/R31.
They are negotiation opportunities, not preassigned export cuts.
The online decision uses current state and past observations only.
No policy reads future arrivals or realized future Decode durations.

Admission and execution use different budgets.
Admission estimates whether the remaining request can finish across future predicted opportunities.
Execution checks the next indivisible Native scheduling quantum now.
The controller can yield between complete quanta. It cannot preempt a running GPU kernel.

The tiled arm transfers the entire remaining Loop suffix once.
It then selects dependency-valid token widths and grouped Loop counts on D.
It is **Adaptive Loop continuation with D execution tiling**.
It is not arbitrary adaptive two-dimensional state ownership.
It does not return state to P.
The CSV name `dynamic_tile` means adaptive Loop continuation with D execution tiling.

The policy protects a predicted deadline margin rather than strict Decode-ready-first ordering.
It can place one admitted quantum before ready Decode when the current budget permits that quantum.
The next boundary checks the budget again.
This distinction matters when interpreting Decode interference.

Unarmed requests retain grouped Native execution.
Armed requests can pay checkpoint and negotiation costs even if all offers fail.
The forced-rejection control measures this branch separately.
A zero-method control alone does not establish zero overhead for rejected dynamic requests.

## 4. Correctness: real online cuts, not forced mixed cuts

All listed state tests compare complete recorded valid prompt KV, hidden state, output tokens, work counts, and native lifetime.
The wire observer compares source and destination export bytes in the same valid-byte order.
These tests establish internal Native-reference parity, not original-HF fidelity or downstream accuracy.

| Model / method | N8 evidence | N32 evidence | Actual N32 offloads | Actual N32 choices | Result |
| --- | --- | --- | ---: | --- | --- |
| Ouro Loop with D tiling | `r9_ouro_actual_mixed_state8` | `r10_ouro_actual_mixed_state32` | 7/32 | 1 and 3 remaining Loops | Exact state, output, wire, work, and lifetime PASS |
| Huginn Loop with D tiling | `r9_huginn_actual_mixed_state8` | `r9_huginn_actual_mixed_state32` | 2/32 | 4 and 8 remaining Loops | Exact state, output, wire, work, and lifetime PASS |
| Ouro adaptive Token | `r4_ouro_actual_token_state8` | `r4_ouro_actual_token_state32` | 1/32 | 3072-token suffix | Exact state, output, wire, work, and lifetime PASS |
| Huginn adaptive Token | `r3_huginn_actual_token_state8` | `r4_huginn_actual_token_state32` | 7/32 | 32-, 64-, and 128-token suffixes | Exact state, output, wire, work, and lifetime PASS |

Ouro's N32 stress uses four clusters of eight requests, six seconds apart.
Huginn's N32 stress submits all requests together.
Neither stress assigns a Cut to a request.
Both contain Native fallback and multiple nonzero online Loop choices.
Their full-byte observers change execution timing. Their latency is not performance evidence.

The Ouro N32 observer checks 528,256 total active recurrent rows, including Decode.
It records 53,248 Prefill rows on D.
The Huginn N32 observer checks 2,112,512 total active recurrent rows, including Decode.
It records 24,576 Prefill rows on D.
These counts are not padded GEMM rows, measured FLOPs, or saved critical-path time.

Thirteen CPU protocol regressions and one evidence-bundle test pass.
The tests cover stale epochs, commit ownership, legal dependency frontiers, and the fixed scheduler cases.
Recorded GPU lifetime checks cover ACK ordering and final credit release.
They do not establish safety under every possible device or transport failure.

## 5. Calibration and Safe Compute Budget

The final common estimator uses matched-memory N128 Native and zero controls.
The absolute ITL target is capped by the frozen SLO and 1.05 times calibration Native ITL p99.
Conditional Decode reserves use observed batch-specific p95 intervals plus 1 ms.
Execution candidates retain their separate p99 guard of at least 1 ms or 20%.
These empirical guards are not worst-case execution bounds.

The no-active-Decode cap is 200 ms.
That cap is a policy limit, not a measured idle duration or a prediction of future arrivals.
A first Decode request without an observed output deadline prevents further immediate borrowing.
The predictive admission horizon is 500 ms.
Execution still requires a current budget after import.

Suspension prevents new borrowing after repeated excess intervals.
Already committed requests still complete exactly.
When active Decode remains, a suspended continuation can wait until a later opportunity.
This preserves work but can produce large Prefill completion delays.
There is no implemented hard continuation-deadline guarantee or reverse-state fallback after commit.

The [completion audit](ouro_huginn_adaptive_idle_intervention_dynamic_loop/completion_model_audit.csv) separates serial estimates from observed host waiting.
The [interpretation audit](ouro_huginn_adaptive_idle_intervention_dynamic_loop/interpretation_audit.csv) checks host enclosures and selection budgets.
Host enclosures include submission and polling. They are not exclusive GPU occupancy.
An enclosure that exceeds a modeled cost is not itself a measured GPU deadline miss.
The admission model remains UNVALIDATED as an online completion predictor.
No Oracle speedup follows from that model.

The frozen calibration targets are 42.46520500816404 ms for Ouro and 128.0371113633737 ms for Huginn.
Huginn's matched-zero reserve already reaches 128.26825876161456 ms at Decode batch two.
Its batch-three reserve reaches 139.771284930408 ms.
Both exceed the target before token age and the additional prediction guard.
The estimator therefore predicts no active-batch borrowing in those states.
This calculation explains a controller constraint. It does not prove that physical GPU idle time is absent.

The same calibration records Huginn ITL p99 of 121.94010606035589 ms for Native and 141.86470398679376 ms for zero.
No Prefill migration occurs in either control.
This is a runtime or workload-composition confound, not an effect of offloaded Loop work.
The campaign does not establish its unique cause.
The held-out controls check whether that difference repeats.
The held-out pair does not repeat this difference.
At seed 2026110911, zero ITL p99 (ms) is 125.661.
The Native value (ms) is 127.127.
The original calibration difference must not be treated as a fixed 16% wire overhead.

Historical N24 Native A/A ITL p99 values are 40.671/40.393/38.751 ms for Ouro.
They are 121.874/119.948/123.589 ms for Huginn.
Those starts reuse the same arrival trace within each model.
The Huginn A/A controls precede the pool change from 26,752 to 23,552 blocks.
They therefore do not establish a formal current-pool A/A bound.

### Serial primitive measurements

The following values use eight disjoint calibration states and CUDA Event enclosures.
They do not measure online Decode-safe windows.

| Full-prompt continuation | D total enclosure p50, ms | Largest one-Loop enclosure p99, ms |
| --- | ---: | ---: |
| Ouro, one Loop | 46.161 | 46.381 |
| Ouro, two Loops | 91.127 | 46.291 |
| Ouro, three Loops | 137.229 | 48.109 |
| Huginn, one Loop | 27.238 | 28.453 |
| Huginn, two Loops | 44.906 | 27.929 |
| Huginn, four Loops | 79.814 | 27.956 |
| Huginn, eight Loops | 150.306 | 28.032 |

These totals use one complete Loop per submission.
They do not imply one non-preemptive kernel with the total duration.
Terminal costs include required boundary work and differ from nonterminal core costs.

| Model / equal-row comparison | D serial total p50, ms | Largest grouped enclosure p99, ms |
| --- | ---: | ---: |
| Ouro: 4096 tokens × 3 Loops, one Loop per submission | 137.229 | 48.109 |
| Ouro: 3072-token suffix, 3072-token width, one Loop per submission | 136.087 | 35.033 |
| Ouro: same suffix, 512-token width, one Loop per submission | 214.603 | 9.698 |
| Huginn: 2048 tokens × 8 Loops, one Loop per submission | 150.306 | 28.032 |
| Huginn: 512-token suffix, 512-token width, one Loop per submission | 177.481 | 18.952 |
| Huginn: same suffix, 256-token width, two Loops per submission | 211.287 | 10.352 |
| Huginn: same suffix, 32-token width, one Loop per submission | 1261.230 | 7.058 |

Ouro's wide Token configuration matches the total Loop cost in this comparison.
Restricting Token to small chunks would create a weak baseline.
Huginn shows a total-cost advantage for Loop execution in this serial equal-row comparison.
Token execution offers a smaller non-yielding quantum.
Neither observation alone establishes an online serving advantage.

Equal recurrent rows do not mean equal Attention work.
The Ouro comparison has 25,171,968 versus 31,463,424 logical causal pairs per core Attention layer.
The Huginn comparison has 16,785,408 versus 29,368,320 pairs.
Prelude, Coda, reinjection, context lengths, and padding also differ.
These structural counts are not measured FLOPs.

## 6. Primary held-out confirmation

### Ouro: three completed paired starts

Each row uses N128 with one of the three fixed arrival seeds.
The seed suffix identifies 2026110911, 2026110912, or 2026110913.

| Method / seed suffix | TTFT p95, ms | ITL p99, ms | Joint SLO-goodput, requests/s | Actual offloads |
| --- | ---: | ---: | ---: | ---: |
| Native / 11 | 2797.039 | 43.114 | 2.27795 | 0/128 |
| Native / 12 | 4530.543 | 42.526 | 1.88908 | 0/128 |
| Native / 13 | 5712.592 | 40.672 | 0.97459 | 0/128 |
| Dynamic Token / 11 | 4189.484 | 41.315 | 0.14883 | 3/128 |
| Dynamic Token / 12 | 4451.078 | 40.679 | 2.05821 | 1/128 |
| Dynamic Token / 13 | 5983.458 | 41.668 | 0.85147 | 1/128 |
| Adaptive Loop with D tiling / 11 | 2792.197 | 41.199 | 2.20068 | 1/128, two Loops |
| Adaptive Loop with D tiling / 12 | 4532.386 | 43.958 | 2.09034 | 1/128, one Loop |
| Adaptive Loop with D tiling / 13 | 5870.037 | 41.835 | 0.81952 | 1/128, two Loops |

Loop continuation completes 3/384 offloaded requests, or 0.78125%.
It moves 20,480/6,291,456 Prefill recurrent rows, or 0.32552%.
Token continuation completes 5/384 offloaded requests and moves 32,896 recurrent rows.
The state stress demonstrates variable cuts. None of these three primary Loop cohorts contains two nonzero accepted Cut choices.
Pooling their one- and two-Loop choices does not establish same-run primary adaptation.

Ratios below use geometric means of paired independent starts.
The 95% intervals use 10,000 start-level bootstrap samples.
Three starts give limited uncertainty coverage.
The analysis does not treat individual tokens as independent starts.

| Ouro comparison | TTFT p95 ratio [95% CI] | E2E p95 ratio [95% CI] | ITL p99 ratio [95% CI] | Goodput ratio [95% CI] |
| --- | --- | --- | --- | --- |
| Loop / Native | 1.00866 [0.99827, 1.02756] | 1.00492 [0.99437, 1.01937] | 1.00530 [0.95558, 1.03367] | 0.96510 [0.84089, 1.10654] |
| Token / Native | 1.15513 [0.98246, 1.49783] | 1.10645 [0.97584, 1.34397] | 0.97927 [0.95657, 1.02449] | 0.39619 [0.06533, 1.08953] |
| Loop / Token | 0.87320 [0.66648, 1.01827] | 0.90824 [0.73988, 1.02597] | 1.02658 [0.99719, 1.08060] | 2.43594 [0.96248, 14.78696] |

Loop passes the absolute Decode gate in all three starts.
Its empirical ITL non-inferiority screen also passes the 1.05 target with these limited data.
Its TTFT, E2E, and goodput do not show a repeated benefit over Native.
The Loop/Token TTFT and goodput intervals include no difference.
Loop's finite-cohort completed-QPS ratio over Token is 1.03291 [1.00831, 1.05621].
Its ratio over Native is only 0.97624 [0.96101, 1.00215].
This does not establish useful additional serving capacity.

Native and Loop record zero intervals above the frozen ITL threshold across 11,904 intervals each.
Their maximum ITLs (ms) are 50.844 and 48.432.
Token records five exceedances across 11,904 intervals.
Its maximum ITL (ms) is 146.858.
Token's p99 gate still passes. A passing p99 does not erase those rare delays.
All three arms have zero coalesced non-first delivery events in these Ouro cohorts.

### Huginn

| Method / seed suffix | TTFT p95, ms | ITL p99, ms | Joint SLO-goodput, requests/s | Actual offloads |
| --- | ---: | ---: | ---: | ---: |
| Native / 11 | 10138.930 | 127.127 | 0.26688 | 0/128 |
| Native / 12 | 15271.979 | 123.250 | 0.28521 | 0/128 |
| Native / 13 | 19991.840 | 132.568 | 0.17672 | 0/128 |
| Dynamic Token / 11 | 10408.513 | 141.954 | 0.27797 | 0/128 |
| Dynamic Token / 12 | 14947.584 | 145.297 | 0.27361 | 0/128 |
| Dynamic Token / 13 | 35858.703 | 125.302 | 0.12357 | 4/128 |
| Adaptive Loop with D tiling / 11 | 20092.003 | 127.814 | 0.03304 | 1/128, one Loop |
| Adaptive Loop with D tiling / 12 | 22442.191 | 128.131 | 0.14440 | 1/128, one Loop |
| Adaptive Loop with D tiling / 13 | 21897.181 | 128.130 | 0.08616 | 3/128, one Loop each |

Loop continuation completes 5/384 offloaded requests, or 1.30208%.
It moves 10,240/25,165,824 Prefill recurrent rows, or 0.040690%.
Every primary accepted Loop request moves only the last Loop.
The four- and eight-Loop choices occur in state diagnostics, not these primary cohorts.
Dynamic Token completes four actual offloads in the third start and none in the first two.
It moves 8192 recurrent rows in total.

| Huginn comparison | TTFT p95 ratio [95% CI] | E2E p95 ratio [95% CI] | ITL p99 ratio [95% CI] | Goodput ratio [95% CI] |
| --- | --- | --- | --- | --- |
| Loop / Native | 1.47201 [1.09531, 1.98167] | 1.41791 [1.09214, 1.83668] | 1.00340 [0.96653, 1.03961] | 0.31264 [0.12379, 0.50631] |
| Token / Native | 1.21695 [0.97876, 1.79367] | 1.21138 [0.98246, 1.73342] | 1.07556 [0.94519, 1.17888] | 0.88735 [0.69924, 1.04156] |
| Loop / Token | 1.20960 [0.61065, 1.93034] | 1.17049 [0.63005, 1.75957] | 0.93291 [0.88186, 1.02257] | 0.35233 [0.11885, 0.69728] |

Loop protects Decode but loses net serving performance.
Its TTFT geometric ratio increases by 47.201% over Native.
Its goodput ratio decreases by 68.736%.
Its completed-QPS ratio is 0.91133 [0.86650, 0.97513].
All three absolute Decode gates pass.
The empirical Loop ITL non-inferiority screen also passes the 1.05 target with these limited starts.

Token provides higher goodput than Loop under the absolute SLO contract in this comparison.
However, Token does not establish the Native-relative ITL target.
The Loop/Token TTFT interval also crosses one.
This is a trade-off between two unsuccessful policies, not proof of an intrinsic Token or Loop advantage.

Native, Token, and Loop each record zero frozen-ITL threshold exceedances across 5760 intervals.
Their maximum ITLs (ms) are 172.118, 153.607, and 156.418, respectively.
The full table retains TPOT and E2E percentiles, offered and realized rates, token rates, and every maximum.

The frozen plan retains all failures and fixed seeds.
Later starts stop after an earlier absolute Decode gate or correctness failure for that method.
Original Native PD is the primary reference.
Matched-wire zero is a separate reference, not a replacement selected after results.
The limited completion budget includes one held-out zero control per model.
It does not support a three-start held-out zero confidence interval.
The [confirmation contract](ouro_huginn_adaptive_idle_intervention_dynamic_loop/confirmation_contract.json) checks fixed inputs and loaded core source hashes.

## 7. Runtime fixes and retained failures

The campaign retains failed attempts instead of replacing their seeds or changing their status.
The [failure inventory](ouro_huginn_adaptive_idle_intervention_dynamic_loop/failure_inventory.csv) lists incomplete and unqualified configurations.

- Short startup prompts now use their actual length when calculating a Token prefix.
- A zero-suffix Native fallback now uses Native reservation, acceptance, and export paths.
- A ready observation can expire before batch selection. The scheduler retries another valid phase or returns to Native policy.
- Pressure counts distinct ready and pending request IDs. It does not count one request twice.
- Mixed-depth grouped source batches can end before a request's planned offer boundary. Only the completed target creates an offer.
- Empty-D admission no longer invents a Batch-1 reserve. Both Token and Loop use the same existing policy cap.
- The full-byte observer uses pinned snapshots and deferred CPU hashes. It remains a diagnostic, not a clean timing path.
- Ouro's intended shutdown timeout now reaches the resolved Native configuration. Earlier top-level settings did not change the effective 30-second limit.
- New evidence records resolved configuration values and actual ModelRunner entry-to-completion host enclosures.

The readiness-race explanation is an inference for the earlier no-progress failure.
The failed run did not capture enough state to establish a unique cause.
The successful fixed Huginn N32 replay is evidence of recovery, not proof of that unique cause.

Earlier Ouro state runs saved all hashes but failed finalization.
They remain failed runs. The later completed N32 tests do not change those receipts.
Historical profiler-associated NIXL stalls do not receive a new root-cause verdict here.
This campaign does not use CUPTI or Nsight for primary results.

Huginn's original 26,752-block D pool exhausts the multi-shape startup envelope.
The compared configurations reserve 23,552 D blocks instead.
All compared methods use that same pool and a 64-shape graph budget.
The source retains 16,384 blocks and the same slot limits.
This reserves 3200 blocks of HBM for exact graph and continuation workspace.
No live KV plane is released early.

## 8. Memory, communication, and attribution limits

The following close snapshots belong to the matched-zero N128 calibration controls.
Each cell gives GiB and the percentage of total NVML HBM.
The pool remains allocated after individual request credits retire.

| Model / role | Model buffers | KV pool | Exposed workspace | Exposed NIXL staging | Allocator allocated total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ouro P | 2.6722 (3.355%) | 48.0000 (60.266%) | 1.3162 (1.653%) | 0.0312 (0.039%) | 52.1763 (65.509%) |
| Ouro D | 2.6722 (3.355%) | 48.0000 (60.266%) | 0.7677 (0.964%) | 0.6250 (0.785%) | 52.3393 (65.714%) |
| Huginn P | 6.6418 (8.339%) | 41.2500 (51.791%) | 0.5027 (0.631%) | 0.0806 (0.101%) | 48.7910 (61.259%) |
| Huginn D | 6.6418 (8.339%) | 59.2969 (74.449%) | 0.5321 (0.668%) | 1.6113 (2.023%) | 68.6372 (86.176%) |

Unattributed allocator storage is separate from exposed workspace.
Graph storage does not have a separately resolved total.
Zero exposed graph tensors must not be interpreted as zero graph memory.
Allocator allocation is not total physical memory use.
The [capacity summary](ouro_huginn_adaptive_idle_intervention_dynamic_loop/memory_capacity_summary.csv) reports sampled free memory, allocator peaks, and KV ownership separately.

The [HBM inventory](ouro_huginn_adaptive_idle_intervention_dynamic_loop/gpu_memory_inventory.csv) reports GiB and explicit capacity percentages.
Its NVML denominator is 81,559 MiB. CUDA-accessible capacity is recorded separately.
Model buffers are not mislabeled as weights alone.
Allocated totals are not added to their component categories.

The [migration payload table](ouro_huginn_adaptive_idle_intervention_dynamic_loop/actual_migration_payloads.csv) records actual state payloads.
The [handoff table](ouro_huginn_adaptive_idle_intervention_dynamic_loop/handoff_host_receipts.csv) records host commit, receipt, and release intervals.
These intervals are not exclusive DMA time.
Native PD also transfers KV. The entire continuation payload is not an additional cost over Native.
The report distinguishes total bytes from payload differences and exposed host delay.

Measured active rows establish that P performs less recurrent work for accepted requests.
They do not measure saved P critical-path milliseconds.
Sampled P-ready state, pending device work, and credit-clear coordinator waiting remain separate.
This campaign does not replace those distinctions with a new GPU compute-saturation claim.
Physical idle-window counts and their duration percentiles remain NOT_MEASURED.

### A measured Huginn completion failure

One held-out example is `confirm_huginn_tile_seed2026110911`, request seed 2026110364.
The real source commits at R31. D accepts one remaining Loop.
The admission forecast provides a 9.669 ms quantum and a 38.054 ms capacity estimate.
The serial remaining-cost model predicts 34.144 ms.
The request then needs eight actual D submissions.
Their guarded candidate costs sum to 72.786 ms.
This sum remains a model. It is not measured GPU occupancy.
Observed import-to-first-output host time (ms) reaches 86,524.743.
Observed first-to-last quantum submission time (ms) reaches 86,498.910.
This is direct evidence that the serial admission estimate does not predict online completion.

The Native coordinator receives transferred state 7.259 ms after its transport commit receipt.
P releases its state 562.983 ms after the earlier source Cut commit.
These are different host intervals. Neither is an exclusive transfer measurement.
P does not retain that source state for the entire 86-second D wait.
The long downstream wait therefore cannot be called an 86-second source-ACK stall.

The unfinished destination request retains 4128 KV blocks.
That storage equals 10.3931 GiB, or 13.049% of total HBM.
The D pool can hold only five reservations of this size, despite eight configured slots.
At five used reservations, 2912 blocks remain: 7.3315 GiB, or 9.205% of HBM.
Those blocks cannot hold another 4128-block request.
Native and zero controls also reach this occupancy boundary.
The continuation's unusually long residence adds an admission-lifetime cost that its serial compute estimate ignores.

The cohort saves only 2048/8,388,608 P Prefill rows, or 0.024414%.
P ModelRunner submissions increase from 128 in Native to 132 in the Loop arm.
All observed P graph selections hit existing graphs in both arms: 4096/4096 versus 4095/4095.
This example is not evidence of a graph-cache miss.
It combines sparse work relief, extra boundaries, extended continuation waiting, and constrained KV residence.
The measurements do not uniquely divide the total TTFT change among those causes.

## 9. Relation to prior work

The Token implementation is a **FluidToken-style reproduction**, not execution of official FluidPD.
FluidPD also considers partial Prefill migration and pressure-based decisions.
Our estimator and Native vLLM-RLT integration differ from its implementation. [FluidPD](https://arxiv.org/abs/2610.06917)

Prefill Deflection assigns a new request to D before P starts its Prefill.
The present method instead continues an already computed P frontier.
It therefore retains an exact state-transfer cost. [Load-Aware Prefill Deflection](https://arxiv.org/abs/2607.02043)

CDB schedules recurrent work across depth.
This campaign does not implement a new shared-weight cross-depth GEMM batch or early exit. [Continuous Depth Batching](https://arxiv.org/abs/2608.09444)

The complete ExactLoop optimization stack is not applied here.
No result establishes that an advantage survives that separate runtime stack. [ExactLoop](https://arxiv.org/abs/2611.05150)

A generic logical-layer scheduler can reproduce the same recurrent boundary and state protocol.
That case is EQUIVALENT_MECHANISM.
The Loop name alone does not establish a Loop-specific contribution.

## 10. Figures and evidence recovery

All figures distinguish real execution, serial calibration, structural counts, and modeled budgets.
No figure draws an unmeasured GPU-idle interval.
No figure replaces an unrun arm with zero performance.

1. [Controlled interventions](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_1_controlled_intervention.png)
2. [Primary budget and actual Loop decisions](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_2_budget_and_actual_loop_count.png). [Separate N32 mixed-state diagnostics](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_2b_actual_mixed_state_decisions.png).
3. [Online host protocol timeline](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_3_online_protocol_timeline.png)
4. [Marker-free serving comparison](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_4_actual_serving_comparison.png). [Paired effects and 95% intervals](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_4b_paired_native_effects.png).
5. [Work and payload accounting](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_5_offloading_accounting.png)
6. [Full-prompt costs and actual execution tiles](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_6_full_prompt_and_tiled_costs.png)
7. [Actual Cut distribution and refusal reasons](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_7_actual_decision_statistics.png)
8. [Strong Token versus Loop calibration](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figures/figure_8_matched_rows_efficiency_pareto.png)

The [figure manifest](ouro_huginn_adaptive_idle_intervention_dynamic_loop/figure_manifest.json) identifies source runs and evidence classes.
PDF versions accompany the PNG files.
Figure 4 contains one zero-control seed per model.
Its zero bar is not a three-start paired comparison.
Use the matching Native value from seed 11 for that control.
Source snapshots, configurations, arrival offsets, model revisions, GPU mappings, and loaded source hashes remain available per run.
The lossless evidence bundle deduplicates bytes without removing failed experiments.
Its manifest maps every archived path to SHA256 and size.
The active burner log uses its closure snapshot. Later operational output is outside the scientific interval.
The bundle contains 42,887 paths in 23 shards.
Its full reconstruction passes every recorded SHA256 check.
The [round-trip receipt](ouro_huginn_adaptive_idle_intervention_dynamic_loop/state/evidence_roundtrip_verification.json) records the independent reconstruction.

The following commands check and reconstruct the bundle.
Extraction requires a new destination path.

```bash
python -m poc_flashvep.idle_intervention_dynamic_loop.evidence_bundle --verify
python -m poc_flashvep.idle_intervention_dynamic_loop.evidence_bundle --extract /path/to/new/evidence_copy
```

## 11. Verdicts, unrun conditions, and closure

| Gate | Ouro | Huginn |
| --- | --- | --- |
| Exact N8/N32 online continuation | PASS | PASS |
| Multiple online Loop choices within one N32 state stress | PASS: 1/3 Loops | PASS: 4/8 Loops |
| Multiple nonzero Cut choices within each primary N128 cohort | NOT_DEMONSTRATED | NOT_DEMONSTRATED |
| Actual N128 Loop migration | 3/384 requests | 5/384 requests |
| Absolute Decode SLO, three starts | PASS | PASS |
| Native-relative ITL empirical screen | PASS, limited three-start CI | PASS, limited three-start CI |
| Repeated benefit over Native | NOT_CONFIRMED | FAIL: repeated net loss |
| Loop-specific advantage over Dynamic Token | NOT_CONFIRMED | NOT_CONFIRMED |
| Current implementation verdict | NO_GO_NET_BENEFIT | NO_GO_NET_BENEFIT |

The verdict applies to this implementation, estimator, and tested heavy workloads.
It does not establish NO_GO_SLACK because physical deadline-safe idle windows remain unmeasured.
It does not establish NO_GO_GRANULARITY because real tiled work executes under the Decode limits.
The main demonstrated problem is sparse useful redistribution plus an inadequate completion forecast.
Accepted continuations can retain KV credit while waiting for a later execution opportunity.
That waiting can dominate the small amount of P work saved.
Source boundary costs and different request timing also remain part of the runtime cost.
The evidence does not isolate every causal contribution.

Do not expand this result into a Loop-specific paper-method claim.
First establish useful accepted-request completion deadlines, resource-lifetime accounting, and matched fallback behavior.
Any future scheduler must preserve the frozen Decode contract instead of enlarging its SLO to obtain more work.
Identical logical-layer boundaries remain EQUIVALENT_MECHANISM.

Whole-prompt adaptive selection accepts no work in the latest clean N32 selection for either model.
It does not enter large confirmation as a successful adaptive arm.
N512 is not expanded because no repeated Native benefit appears.
No clairvoyant speedup appears in this report.
The online completion model remains UNVALIDATED.
Arbitrary adaptive two-dimensional state ownership and D-to-P state return remain unimplemented.
N128 compares output tokens, full work, and lifetime. It does not hash every intermediate state again.
Full byte-state coverage remains the declared N8/N32 qualification set.
The search does not establish a globally optimal KV pool, slot geometry, or completion policy.
Normal-load method comparisons and large Static comparisons cannot be inferred from heavy-load confirmation.

### Secondary load controls and unrun Static cells

The secondary Native controls use N16 and arrival seed 2026110914.
Each condition has one start only.
They establish context for normal latency, not normal-load equivalence across all methods.

| Model / offered load | Nominal QPS | TTFT p95, ms | TPOT p95, ms | ITL p99, ms |
| --- | ---: | ---: | ---: | ---: |
| Ouro low | 1.23460875 | 422.568 | 30.962 | 36.270 |
| Ouro medium | 3.4569045 | 506.187 | 28.837 | 32.105 |
| Huginn low | 0.37903355455 | 1142.740 | 101.802 | 110.943 |
| Huginn medium | 1.06129395274 | 1720.168 | 111.964 | 119.393 |

Huginn medium meets the TTFT threshold for 15/16 requests, not every request.
The heavy-load TTFT values therefore must not be described as tens of seconds of model computation.
Exact Service TTFT and its queue decomposition remain NOT_MEASURED.

All four secondary Static N32 serving cells receive NOT_RUN_BOUNDED_COMPLETION_WINDOW receipts.
They cannot fit startup, timeout, and cleanup limits before the unchanged GPU deadline.
Their latency is NOT_MEASURED, not zero.
The main three-start Dynamic Token/Loop comparison completes before those optional cells.
The forced-rejection N32 selection controls arm zero measured requests in both models.
They therefore do not quantify an armed-request rejection penalty, despite their configuration names.
Actual rejected offers and fallback remain available in the dynamic decision logs.

There are 293 immutable configurations in this namespace.
Of these, 248 complete, 25 have failed or incomplete receipts, four are budget-blocked, and sixteen never enter the queue.
Three older stopped-run receipts retain the word RUNNING with return code -15.
One early failure has return code 70 and no final status string.
The derived inventory identifies those cases without editing their original receipts.
No scientific worker remains active at closure.

### Preservation and resource restoration

The scientific queue closes at 2026-11-09 15:53:20 KST.
The declared queue contains 271 historical and resumed cells with retained outcome receipts.
All scientific owners, deadline guards, and the monitor stop before final resource verification.
No additional scientific campaign remains scheduled.

The original staged-index entry SHA256 remains `d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3`.
The external-runtime diff SHA256 remains `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
The actual checkout remains on `flashvep/looped-pd-kv-profile-headroom-recovery`.
Publication creates the new branch through a temporary index. It does not switch or reset that checkout.

GPU 0–7 each have a resident approved VLLM EngineCore with the expected worker SID and UID.
Every latest worker records actual generation and at least one completed round.
The final verification also samples nonzero utilization on every GPU.
A subsequent direct query shows 100% utilization on all eight devices.
GPU 0 is present. No partial seven-engine restoration remains.
The burners continue after scientific closure and publication.
This is a timestamped operational check, not a perpetual utilization guarantee.

The [restoration receipt](ouro_huginn_adaptive_idle_intervention_dynamic_loop/state/final_burner_verification.json) records engine ownership and generation counters.
The publication receipt records the verified evidence commit.
The final remote-SHA check records the metadata child separately.

Publication uses a temporary index over the declared parent.
The active dirty checkout, original staged index, and external runtime remain separate.
Their preservation signatures and the final remote verification accompany the evidence.
Final resource verification must identify eight resident approved engines, not eight wrapper processes alone.
