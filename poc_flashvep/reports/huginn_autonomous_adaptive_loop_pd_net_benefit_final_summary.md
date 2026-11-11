# Huginn R32 adaptive Loop PD: net-benefit campaign

## Outcome

The required TO-BE is not reproduced under the primary contract.
The campaign's practical verdict is `NO_GO_DECODE_INTERFERENCE` for the tested high-migration configurations.
The tested online admission policies also show `NO_GO_COMPLETION_ECONOMICS`.
Neither is a proof that exact Loop offloading is physically impossible.
Physical idle capacity remains `INCONCLUSIVE_MEASUREMENT`.

The strongest qualified Zero remains the recommended configuration.
Loop31 reduces TTFT p95 by 17.3% in the held-out confirmation.
Its ITL p99 increases by 10.3%, beyond the required 5% limit.
Loop28 reduces TTFT p95 by 51.1%, but increases ITL p99 by 70.9%.
Both comparisons use three paired independent Arrival seeds.

The late-transfer online Loop policy actually moves two requests and six Loops over 384 requests.
Its TTFT p95 increases by 7.5% relative to Zero.
Its two accepted requests also exceed their completion reservations.
The early-transfer online policy moves no measured request in held-out confirmation.

Loop28 does outperform the tested equal-recurrent-row Token256 baseline on TTFT.
That benefit does not satisfy the Native-relative Decode contract.
Different Coda work and internal graph geometry also prevent a pure Loop-specific attribution.
A generic logical-layer cut can reproduce the same frontier.
The campaign therefore does not support `GO_LOOP_SPECIFIC_NET_BENEFIT` or `GO_GENERIC_PD_BENEFIT`.

## 1. Comparison contract

The primary reference is Native-path Zero with the qualified Runtime optimizations.
It uses the Native packed handoff and executes the complete prompt on P.
It does not transfer any recurrent prompt computation to D.
Original Native is a control, not the denominator for success.

The primary workload uses Huginn-0125 at fixed R32.
It uses original BF16 weights, 2048 input tokens, and 16 output tokens.
The offered load is 1.971 requests/s.
The workload uses open-loop Poisson arrivals, 128 distinct prompt IDs, and 2/8 P/D logical slots.
The workload disables prefix caching.
Generation ignores EOS until the output cap.
The model revision is `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
The Huginn experiments use H100 80 GB GPUs 4/5 for P/D in a quiet GPU lane.
No result from the separate 2048/4 discovery workload is a primary result.

| Frozen limit | Value (ms) |
| --- | ---: |
| Request TTFT | 1963.9920 |
| Request-average TPOT | 186.5715 |
| ITL p99 | 192.8213 |
| Additional-work TPOT p95 | 167.9144 |

The relative TTFT p95 target is at most 0.95× Zero.
The E2E p95 target is at most 1.00×.
The ITL p99 and TPOT p95 targets are at most 1.05×.
The confirmation requires exact outputs and state, real migration, measured P critical-path reduction, and Decode tail protection.
TTFT includes arrival-to-admission waiting.
Service TTFT is `NOT_MEASURED` in marker-free trials.
No Host receipt substitutes for the first GPU Prefill execution.
Throughput is a finite-cohort drain rate, not sustainable capacity.

See the [experiment specification](huginn_autonomous_adaptive_loop_pd_net_benefit_optimization_spec.md).

## 2. Cost decomposition

### P checkpoints

The prior Recovery moved 142/384 requests and 853/12,288 recurrent prompt Loops.
Its Loop/Zero ratios were 1.359 for TTFT p95, 1.333 for E2E p95, 1.328 for ITL p99, and 0.911 for completed QPS.
Those ratios compare complete configurations.
They do not isolate checkpoint cost.

The Recovery configuration generator also changed the effective Native wire credit from 8 GiB to 256 MiB.
It changed the extra draining request limit from zero to eight.
The new tests retain both of the optimized Zero settings.
The all-checkpoint control still measures a TTFT p95 of 24.068 seconds with no migration.
The matched fixed-Decode Zero measures 9.676 seconds.
The selective no-migration control measures 9.824 seconds.
These are selection starts on the same Seed 2026110940, not independent confirmation.

In the initial event diagnostic, P issues 896 prompt submissions with checkpoints.
P issues 128 without checkpoints.
Their summed CUDA Event enclosures are 75.3971 s and 76.3246 s, respectively.
Those diagnostics have the transport confound.
They do not show a proportional increase in device work.
They also do not measure exclusive kernel time, CPU time, or time on the request critical path.
Nested Host counters overlap and must not be added into TTFT.

In that diagnostic, inclusive P submit CPU time decreases from 54.82 to 34.13 seconds.
Inclusive scheduler CPU time increases from 1.14 to 9.32 seconds.
These counters also have the wire configuration confound.
They rule out using submission count as a CPU time measurement.

The separate wire-matched paired Host audit measures a mean TTFT increase of 7998.59 milliseconds for all checkpoints.
The same 128 request pairs show 7752.06 milliseconds of added admission wait.
Their P start-to-release receipt adds 199.48 milliseconds.
The selective no-migration arm adds 71.01 milliseconds of mean TTFT.
The large queue penalty remains after wire matching.

The later matched priority control uses CUDA Event enclosures and samples Decode every 31st submission.
Zero measures a mean P prompt enclosure of 598.3531 milliseconds.
Forced Loop31 measures 569.2878 milliseconds on P.
The measured difference is 29.0654 milliseconds per request, or 4.86% of the Zero enclosure.
Its added D prompt enclosure has mean 32.2327 milliseconds.
These are observed device-stream enclosures, not exclusive kernel time.

| Matched event diagnostic, N128 | P prompt enclosure sum (s) | D prompt enclosure sum (s) | P prompt submissions | D prompt submissions |
| --- | ---: | ---: | ---: | ---: |
| Priority Zero | 76.5892 | 0 | 128 | 0 |
| Priority Loop31 | 72.8688 | 4.1258 | 128 | 128 |

The diagnostic's TTFT, E2E, TPOT p95, and ITL p99 are each within 5% of its own marker-free control.
The maximum ITL screen fails.
Loop's diagnostic maximum is 10.2% higher.
The Zero maximum varies in the other direction.
This single-start comparison does not identify a causal profiler effect.
The enclosures remain diagnostic evidence.
They do not satisfy the clean physical kernel or P critical-path gate.
Do not add P and D enclosure durations into TTFT.

### KV and handoff

The measured Native wire descriptor payload for one 2048-token prompt is 5,709,454,080 KV bytes and 21,120 hidden bytes.
Loop31 carries 5,449,973,760 KV bytes and 43,253,760 hidden bytes.
The combined Loop31 payload is 5,493,227,520 bytes, or 3.79% less than the Native payload of 5,709,475,200 bytes.
The extra hidden state is real.
The complete KV payload is not an incremental Loop cost.
The descriptor account measures bytes requested, not DMA time.
Reserved physical KV quota, resident HBM, and wire bytes are distinct.

The matched 2048/16 early-transfer Zero test queues half of its prompt KV at R16.
Early and late transfer both complete 730,812,825,600 payload bytes and 11,008 submissions over 128 measured requests.
Early transfer increases the descriptor count from 1,028,030 to 1,092,480.
Both paths record 4096 measured P graph hits, no new captures, and no fallback.
Early transfer increases TTFT p95 from 9762.66 to 12698.77 milliseconds.
The paired mean admission wait increases by 1943.08 milliseconds.
The paired mean release-to-wire-commit receipt interval increases by 120.30 milliseconds.
This test rejects a speedup from that early transfer policy.
It does not identify the isolated effect of DMA overlap, HBM traffic, or descriptor preparation.

The monotone-frontier protocol permits an exact later commit after early KV receipt.
It retains the final hidden-state transfer and Native ACK.
N32 verifies 16 commits after actual fragment receipt.
Each commit moves four complete Loops.
The protocol passes the frozen prompt state, output, full-work, and lifetime checks.

The primary N128 early Loop arm moves one request and four Loops.
It commits that request after actual fragment receipt.
All 128 outputs match the early Zero control.
TTFT p95 is 12143.07 milliseconds. ITL p99 is 90.01 milliseconds.
The early Loop is faster than its early Zero control on TTFT.
It is slower than the strongest late-transfer Zero.
This is a working continuation protocol, not a net benefit over the required baseline.

The Source slot release, fragment submission, D import, activation, and ACK receipts show event order.
They also show Host delays.
The initial Native diagnostic measures the Host interval from first POST to last DONE.
Its p50 is 105.85 milliseconds.
Its p95 is 121.69 milliseconds.
This interval includes transport progress and Host observation.
It is not a copy-engine duration.
The experiments do not separate pure DMA and CPU submission time on the request critical path.

### D completion and Decode pressure

The conservative primary policy reserves 93.1062 ms for one complete Decode cycle.
Its design target is 97.7615 ms.
The nominal remainder is 4.6553 ms before the guard.
The smallest calibrated terminal quantum is 6.1148 ms.
This mismatch explains why the serial estimator can reject an offload.
It does not measure physical GPU idle or prove that offload is impossible.

The matched dynamic Token selection moves one 256-token suffix.
| Completion observation | Value (ms) |
| --- | ---: |
| Commit to first output, prediction | 253.7948 |
| Commit to first output, Host observation | 318.8645 |
| D import to first output, Host observation | 129.4879 |
| Maximum observed inter-quantum wait | 8.0655 |

This request does not repeat the earlier multi-second starvation.
Its cohort still has worse TTFT than Zero.
One request does not qualify a completion predictor.

The later preparation-forecast Token run also moves one 256-token suffix.
Its commit-to-first-output interval is 379.5272 milliseconds.
It exceeds the 350-millisecond reservation.
The matched previous Loop run moves two Loops on one delayed request.
The model predicts 155.9524 milliseconds from commit to first output.
The Host interval is 1441.2124 milliseconds.
The largest observed inter-quantum wait is 1140.1350 milliseconds.
Do not attribute that interval to GPU kernel time or to the wire.

The [completion audit](huginn_autonomous_adaptive_loop_pd_net_benefit/audit/completion_model_audit.csv) contains the individual predictions and observations.
A finite reservation does not guarantee bounded service after ownership commits.
The campaign does not check a same-decision Native completion counterfactual for these individual requests.

### P/D backlog and timeline

In the full-device Zero selection, 193/193 central samples contain a P-only admission block.
None contains a D-only KV block.
The central window spans 20% to 90% of the arrival period.
The coordinator admission wait has p95 of 8.390 seconds.
The same interval in forced Loop31 has p95 of 6.126 seconds.
These are Host observations of Prefill-side queue pressure, not physical GPU saturation.

The next table uses 128 paired requests in that selection.

| Paired Host interval | Mean Loop31 minus Zero (ms) |
| --- | ---: |
| TTFT | −1558.64 |
| Arrival to admission | −1541.62 |
| P start to release receipt | −106.31 |
| D receipt to first output | +39.56 |

The [Host attribution evidence](huginn_autonomous_adaptive_loop_pd_net_benefit/paired_host_attribution/) contains the other intervals for each pair.
These means share the same matched requests.
Do not sum component percentiles or treat the P receipt change as GPU compute saved.

ITL also increases within the descriptive stratum with two live Host-visible Decode requests.
Its p99 is 88.78 ms for Zero and 107.77 ms for Loop31.
The stratum has 1718 and 1456 ITL intervals, respectively.
These counts use output delivery, not a physical Decode batch or ready queue.
The comparison does not isolate causal interference.
It does show why a larger Decode cohort is not a sufficient explanation by itself.

The [interactive timeline](huginn_autonomous_adaptive_loop_pd_net_benefit/figures/diagnostic_selection_observed_timeline.html) compares the same 128 requests.
The timeline labels Host submission, queueing, handoff, continuation, and output observations.
The blank regions do not represent measured GPU idle.

In the forced [Decode-priority timeline](huginn_autonomous_adaptive_loop_pd_net_benefit/figures/priority_selection_observed_timeline.html), 120/128 moved requests show later P work.
P submits a later request after D receives the selected request and before its first output.
This verifies the intended progression at the Host level.
It does not prove physical GPU overlap.
This intervention fixes the cut at reservation and does not establish adaptive scheduling.

## 3. Optimization and validation paths

The [optimization journal](huginn_autonomous_adaptive_loop_pd_net_benefit/optimization_journal.md) records each decision.
It lists the hypothesis, code change, cost, and result.
The [root-cause audit](huginn_autonomous_adaptive_loop_pd_net_benefit/root_cause_decomposition.md) links the frozen evidence.

Fixed R32 permits grouped Decode submission without a Host decision on an unused Gate result.
The implementation retains the Gate computation, attention, recurrent work, and real token collection.
The Native worker still executes every loop and commits the actual outputs.
It does not use speculative output, early exit, or approximate arithmetic.
The grouping and unused-readback change is a generic Runtime improvement.
The improved Zero uses it too.

The separate CODA collector initially changed cohort retirement and regressed ITL.
The retained version preserves cohort retirement.
It can submit independent prompt work while it waits for the actual CODA result.
The source progression extension has one control owner.
It only consumes safe control messages for other request IDs.
It retains FIFO order and uses Native producer events.
Its marker-free Zero selection does not show a speedup.

The preparation forecast estimates the drain of already admitted Decode work.
It uses past observed cycle times and known remaining output counts.
It does not use future arrivals.
It changes source checkpoint preparation, not actual commit or D execution admission.
Controlled N8 and N32 runs exercise real online boundary negotiations.
The selected 2048/16 Loop run transfers no measured prompt work.
A warm-up commit is not a measured offload.

Some prompt-only Green-context trials completed the cohort but failed at graph destruction.
The campaign excludes those trials from qualified serving comparisons.
The graph-lifetime fix drains and resets graph executables in their owner contexts before closing the transport.
The dispatcher records both Native child exits.
Successful process shutdown is part of qualification.

The separate resource experiment uses one documented CUDA SM-resource split.
It provisions 112 SMs for Decode and eight SMs for prompt work.
Twelve device SMs remain unassigned by this split.
Both paths pass N8/N32 exact state and worker exit checks.
The N128 provisioned Zero measures TTFT p95 of 9789.06 milliseconds and ITL p99 of 96.53 milliseconds.
The provisioned Loop31 moves 128 full-prompt Loops.
It measures TTFT p95 of 8356.11 milliseconds and ITL p99 of 292.00 milliseconds.
Its ITL p99 exceeds the frozen absolute limit.
The campaign rejects this configuration without repeated serving.
Disjoint provisioning does not establish simultaneous kernels or isolated HBM and hardware queues.
These results do not identify which shared resource causes the regression.

## 4. Primary selection controls

Each row is one measured N128 start at 2048/16 and offered load 1.971 requests/s.
Only rows with the same Arrival seed share a paired trace.
Execution policies also differ between some rows.
This table does not give confidence intervals.

| Method | Arrival seed | TTFT p95 (ms) | E2E p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Max ITL (ms) | Completed requests/s | SLO-goodput (requests/s) | Moved requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Native | 2026110931 | 28059.20 | 30118.97 | 160.59 | 147.47 | 168.42 | 1.4188 | 0.0222 | 0 |
| Native-path Zero, original execution policy | 2026110931 | 17058.49 | 18803.05 | 123.81 | 117.31 | 314.10 | 1.6178 | 0.2528 | 0 |
| Native-path Zero, fixed R32 Decode | 2026110940 | 9676.22 | 10867.81 | 91.64 | 80.64 | 324.52 | 1.6320 | 0.2167 | 0 |
| All checkpoints, no migration | 2026110940 | 24067.62 | 25253.11 | 85.83 | 79.44 | 118.53 | 1.3549 | 0.0741 | 0 |
| Selective checkpoints, no migration | 2026110940 | 9824.40 | 11080.60 | 98.44 | 84.36 | 104.90 | 1.6300 | 0.2037 | 0 |
| Previous Dynamic Loop, matched wire | 2026110940 | 28118.41 | 29323.61 | 88.26 | 80.91 | 2967.80 | 1.2945 | 0.0708 | 3 |
| Native-path Zero, deferred CODA v2 | 2026110940 | 9584.16 | 10768.96 | 88.48 | 80.04 | 393.27 | 1.6357 | 0.2428 | 0 |
| Adaptive Loop, preparation forecast | 2026110940 | 10034.14 | 11240.13 | 89.76 | 80.58 | 95.49 | 1.6258 | 0.1905 | 0 |
| Adaptive Token, preparation forecast | 2026110940 | 10112.17 | 11291.72 | 88.84 | 80.10 | 396.32 | 1.6223 | 0.1901 | 1 |
| Decode-priority Zero | 2026110940 | 10638.66 | 11857.48 | 92.87 | 83.28 | 1259.99 | 1.6138 | 0.2017 | 0 |
| Forced Decode-priority Loop31 | 2026110940 | 7238.13 | 8498.49 | 104.68 | 84.96 | 107.30 | 1.7115 | 0.3476 | 128 |
| Forced SM32 Token64 | 2026110940 | 9142.36 | 10450.90 | 101.65 | 88.57 | 116.52 | 1.6531 | 0.2195 | 128 |
| Token256, effective eight-Loop cap | 2026110940 | 4680.19 | 6097.22 | 166.43 | 98.48 | 176.14 | 1.7830 | 0.5572 | 128 |

The fixed-cut Decode-priority arm moves 128 complete Loops and 262,144 recurrent rows.
All 128 output sequences match its Zero control.
Its ITL p99 is 12.7% higher than the matched priority Zero.
It is 18.3% higher than the stronger full-device deferred CODA Zero.
The result therefore fails the relative Decode target.
The priority Zero's maximum ITL includes a 1.26-second delivery gap.
The [tail audit](huginn_autonomous_adaptive_loop_pd_net_benefit/audit/tail/) retains this observation.
It shows 38 D submissions in the fully contained half-second bins.
This does not identify the cause of the gap or justify removing the start.

The adaptive forecast arm transfers no measured Loop in the selected primary start.
It evaluates all 128 requests and exposes an intermediate checkpoint on three.
For 123 requests, the only recorded selective-checkpoint reason is `NO_EXECUTION_BUDGET`.
The other two requests record only `NO_P_PRESSURE`.
These counts exclude Warm-up events, even when Warm-up reuses a sampling seed.
They describe the policy, not physical D capacity.
The previous matched-wire arm moves 10 Loops on three requests in this start.
Do not pool these starts with the separate 384-request Recovery cohort.
The better forced TTFT does not establish an adaptive net benefit.

The first Token64 quantum control requests a one-Loop cap.
All 128 D prompt spans instead submit 32 Loops from the staged Prelude.
The SM32 Token64 row also submits 32 Loops per request despite a requested eight-Loop cap.
These are real Token transfers, but they are not bounded-quantum measurements.
The [quantum audit](huginn_autonomous_adaptive_loop_pd_net_benefit/audit/quantum/) records the actual spans.
The new implementation caps both staged prompt phases.
Its qualification checks the effective bound before a performance run.

Full-device Loop31 and Token64 each move 262,144 semantic recurrent rows in the selection trace.
The Token suffix attends to a longer context per row and executes different Prelude and CODA work.
The [structural work account](huginn_autonomous_adaptive_loop_pd_net_benefit/work_accounting/) counts these differences.
It also counts the qualified 256-row padding.
These calculations are not hardware FLOPs or GPU execution times.

The Token256 arm with an effective cap moves 1,048,576 semantic recurrent rows.
It passes N8 and N32 state qualification and the actual eight-Loop cap.
All 128 serving outputs match Zero.
It has substantially lower TTFT than the tested one-Loop frontier.
It moves four times the recurrent rows, so this is not an equal-work advantage.
Its ITL p99 is 1.881× the repeated full-device Zero.
Its absolute Decode SLO passes, but the relative gate fails.
The higher joint TTFT+TPOT goodput does not waive that gate.

### Equal recurrent rows do not equal identical work

Loop28 and Token256 each move 1,048,576 recurrent prompt rows in N128.
The Loop arm runs four full-prompt Loops.
The Token arm runs 32 Loops on a 256-token suffix.
Both issue 512 D prompt submissions in this selection.
Both pass N8/N32 exact state and every N128 output comparison.

| Selection observation | Full-prompt Loop28 | Token256, eight-Loop groups |
| --- | ---: | ---: |
| TTFT p95 (ms) | 3422.70 | 4680.19 |
| ITL p99 (ms) | 162.89 | 166.43 |
| D recurrent prompt rows | 1,048,576 | 1,048,576 |
| D Prelude rows | 0 | 32,768 |
| D Coda rows | 262,016 | 32,640 |
| Wire payload per request (bytes) | 4,974,182,400 | 4,998,533,760 |
| Measured P prompt graph-cache selections | 3584 | 4096 |
| Measured D prompt graph-cache selections | 512 | 4096 |

Loop28 has lower TTFT in this one paired selection.
It also moves substantially more Coda work.
Its structural padded dense plus ideal causal Attention estimate is about 6.4% greater on D.
These estimates exclude normalization, activation, softmax, and first-output logits.
The comparison therefore does not isolate a pure recurrent-boundary advantage.
Every listed graph-cache selection finds an existing entry.
Neither method captures a new prompt graph or falls back during measurement.
Their 512 D prompt submissions therefore contain different numbers of internal graph selections.
A selection is not a CUDA kernel count or a device execution time.
The observations exclude a missing prompt graph as the explanation for this selection difference.
Both arms still fail the relative Decode target.
The held-out confirmation includes both cuts.

The separate Loop28 tile control uses 256-token tiles and one Loop per submission.
It completes all four transferred Loops on every measured request.
Its 4096 D prompt spans pass the actual quantum audit.
All 128 outputs match Zero after N8 and N32 state qualification.
TTFT p95 is 3563.80 milliseconds. ITL p99 is 142.91 milliseconds.
Tiling reduces the selected ITL p99 from 162.89 milliseconds for the full-prompt Loop28 control.
It also increases TTFT p95 from 3422.70 milliseconds.
The tiled arm still fails the relative Decode target.
This is one diagnostic selection, not an independent-start result.

## 5. Short-output discovery

The discovery changes the requested output cap from 16 to four.
It retains the 2048-token prompt, fixed R32, Arrival trace, and 1.971 requests/s.
Each generated token still uses full R32 execution.

| Matched Zero observation | 2048/16 | 2048/4 |
| --- | ---: | ---: |
| Central samples with a P-only block | 191/192 | 192/192 |
| Central samples with a D-only KV block | 0/192 | 0/192 |
| Mean sampled D NVML utilization | 91.96% | 35.15% |
| Median sampled D NVML utilization | 93.0% | 4.5% |
| Central D recurrent submission counts by request count | B1: 13, B2: 522, B3: 44 | B1: 238 |

The shorter output changes the observed Decode workload without removing the P admission backlog.
NVML utilization samples do not measure continuous physical idle windows.

The forecasted Loop arm passes N32 state qualification.
Fourteen requests move four Loops each.
One request moves two Loops.
This controlled cohort uses spaced pairs.
It does not measure clean serving performance.

The clean N128 Loop selection moves nine requests and 36 complete Loops.
All nine measured cuts move four Loops.
All 128 output sequences match Zero.
TTFT p95 increases from 9662.43 to 10088.92 milliseconds.
ITL p99 increases from 77.42 to 81.08 milliseconds.
The 4.7% ITL increase is a single-start result, not a non-inferiority finding.
The selection does not meet the TTFT target.

The matched Token arm moves five 256-token suffixes and 40,960 recurrent rows.
The Loop arm moves 73,728 recurrent rows.
The workers complete these prompt spans.
All 128 Token outputs also match Zero.

| 2048/4 method, Seed 2026110940 | TTFT p95 (ms) | E2E p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Max ITL (ms) | Completed requests/s | SLO-goodput (requests/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero | 9662.43 | 9886.02 | 77.42 | 75.11 | 79.22 | 1.6533 | 0.2454 |
| Adaptive Loop forecast | 10088.92 | 10312.58 | 81.08 | 75.77 | 94.62 | 1.6453 | 0.2442 |
| Adaptive Token forecast | 9850.43 | 10080.78 | 82.39 | 78.00 | 83.70 | 1.6495 | 0.1546 |

Token has lower TTFT p95 than Loop in this start.
Loop has lower ITL p99 and higher goodput than Token.
Neither shows a net benefit over the matched Zero.
One paired start does not establish an ordering across Arrival seeds.

All nine moves occur among the first 16 measured requests.
One commit-to-first-output observation reaches 466.28 milliseconds.
The completion model predicts 199.57 milliseconds for that request.
The reservation miss occurs at 350.05 milliseconds with 4096 recurrent rows left.
The inherited suspension policy permits recovery from a Decode tail latch only.
It does not permit recovery from this completion reservation miss.
After the miss, the source records destination unavailability for most later requests.
This is a policy limitation, not a measured shortage of physical D idle.

The early-wire 2048/4 intervention shows two different nonzero cuts in a clean N128 run.
Seven requests move four Loops. Two requests move two Loops.
Five of the nine commits occur after actual early fragment receipt.
All 128 outputs match the early Zero control.
The arm moves 32 complete Loops and 65,536 recurrent rows.

Its ITL p99 is 79.01 milliseconds versus 78.17 for early Zero.
Its TTFT p95 is 10616.14 milliseconds versus 9644.23 for early Zero.
This is a 1.1% ITL increase and a 10.1% TTFT increase in one start.
The paired mean admission wait increases by 850.76 milliseconds.
The mean P start-to-release receipt decreases by 77.92 milliseconds.
The mean release-to-wire-commit receipt increases by 96.67 milliseconds.
These Host intervals do not establish a physical compute saving.
Actual adaptive Loop selection does not imply a serving benefit.

The drained-epoch retry diagnostic changes the selected Loop amount after a missed reservation.
It moves 25 complete Loops on 11 requests.
The distribution is four Loops on two requests, two Loops on eight, and one Loop on one.
The first four-Loop miss reduces the allowed suffix to two Loops.
One retry occurs after 700 milliseconds with no live Decode, committed continuation, offer lease, or pending quantum.
A later one-Loop miss stops new retries.
No SLO or reservation deadline changes.

All 128 outputs match Zero, and observed full recurrent work passes.
TTFT p95 is 9929.42 milliseconds.
ITL p99 is 82.47 milliseconds.
Both are worse than the matched 2048/4 Zero.
N32 state qualification does not exercise the recovery trigger.
The diagnostic therefore does not establish post-retry intermediate-state equality.
The campaign does not adopt this policy.

## 6. Interpretation of the Oracle and the prior work

The short Nsight experiment did not pass the 5% interference gate.
A separate collector-start control stalled in the Native NIXL transfer call.
The underlying cause of that stall is inconclusive.
Graph enclosures and sampled NVML utilization are not a physical kernel union.
They do not establish a D idle-window distribution.
The physical, Decode-safe, and completion-feasible Oracle capacities are `NOT_MEASURED`.
Actual forced continuations and clean serving comparisons remain valid evidence of the tested operating points.

The [source and prior-work audit](huginn_autonomous_adaptive_loop_pd_net_benefit/source_and_prior_work_audit.md) records the archive check and primary sources.
[FluidPD](https://arxiv.org/html/2610.06917v1) uses Token prefix/suffix stealing and also studies role changes.
The campaign's Token suffix arm is a local FluidToken-style control, not that implementation.
[DuetServe](https://proceedings.mlr.press/v306/gao26m.html) studies spatial multiplexing.
Its AR motivation and main system use different configurations.
[Prefill Deflection](https://arxiv.org/html/2607.02043v1) can route an unstarted request to D, without a mid-request handoff.
That is not the same mechanism as resuming a recurrent frontier.

Earlier stock AR results for Qwen3-8B at 8000/200 showed a P kernel union of 93.6353% in one valid profile.
That start did not establish D headroom.
Another start passed the service and D margin checks without a valid matched P profile.
Do not combine those starts into a simultaneous result.
See the [original AR report](looped_pd_atomic_prefill_qps_crossover_final_summary.md).
No AR measurement substitutes for Huginn evidence.

The new stock AR diagnostic completes 32/32 Qwen3-8B requests at 8000/200.
It uses original BF16 weights, stock vLLM/NIXL, and GPUs 0/1 in the exclusive quiet lane.
The nominal offered load is 4.0 requests/s.
The realized arrival rate is 4.4365 requests/s.
The finite-cohort completed rate is 2.7316 requests/s.
Input and output rates are 21,852.77 and 546.32 tokens/s, respectively.
TTFT p95 is 2022.82 milliseconds. TPOT p95 is 22.63 milliseconds.
ITL p99 is 157.96 milliseconds. Maximum ITL is 236.95 milliseconds.
All 6368 non-first delivery intervals contain one token.

The central stock scheduler samples show no waiting request in 11 P samples or 18 D samples.
D has mean running count 9.78 and mean KV utilization 25.20% in those samples.
Neither stage records a KV allocation refusal.
These observations do not establish P compute saturation or D deadline-safe slack.
Physical kernel busy and idle remain `NOT_MEASURED`.
The QPS 2/6 cells remain `NOT_RUN` because the long-cell start cutoff follows this diagnostic.
The campaign does not execute AR Token stealing, FluidPD, or DuetServe.
See the [new AR evidence](huginn_autonomous_adaptive_loop_pd_net_benefit/audit/ar_diagnostic_summary.json).

The qualified Huginn numerical path uses fixed 256-row Down and adapter projection tiles.
Loop and Token cuts with equal recurrent token rows can have different padded work, attention context, CODA work, and submission count.
A generic logical-layer cut with the same operations and state frontier is an equivalent mechanism.
No architecture-specific novelty follows from changing the boundary name.

The qualified 256-row projection path also applies to small Decode batches.
It pads the selected Down and adapter inputs before the original BF16 projection.
The campaign does not qualify a smaller exact replacement kernel.
The measured Decode costs therefore belong to this exact Runtime.
They are not irreducible costs of the model architecture.

## 7. Limits of the investigation

The campaign does not establish a clean physical D idle percentage or an Oracle maximum.
It does not establish that all exact Loop offloading is physically impossible.
It also does not check an individual request's same-decision Native completion counterfactual.
An empirical completion reservation is not a future guarantee.

The source graph path still has a Host loop over prepared forwards.
The campaign does not implement a persistent GPU-resident R32 controller.
It does not implement one captured whole-model graph that supports arbitrary safe boundary extraction.
Those changes need a stable exact-state and KV-addressing ABI.
The local reference does not provide a qualified drop-in implementation for this Runtime.
The report therefore makes no structural impossibility claim about those unimplemented designs.

The static comparisons do not implement FluidPD's full role controller or DuetServe.
A faster Token or Loop selection is not evidence that it recreates those systems.
The stock AR diagnostic has a separate model, KV layout, and execution path.
It cannot prove a Huginn benefit.

The primary serving rate is a finite N128 arrival-to-drain measurement.
No sustainable capacity follows from it.
The held-out Arrival seeds reuse the same 128 prompt texts.
They do not establish generalization to a new text distribution.
State observers and hashing are separate correctness diagnostics.
They do not provide marker-free Service TTFT.

One held-out Loop28 start overlaps a source search that takes more than ten seconds.
The possible CPU and I/O interference is not measured.
The campaign records an isolation replacement before inspecting that start's performance values.
The original result file already exists at that declaration.
The replacement uses the same frozen configuration and Arrival trace.
The analysis must use the replacement regardless of its result.
The original remains an isolation sensitivity observation.
The [isolation receipt](huginn_autonomous_adaptive_loop_pd_net_benefit/state/primaryconfirmv2_isolation_replacements.json) records the reason and timing limitation.

## 8. Held-out confirmation

The selection freezes before any held-out trial starts.
Arrival seeds are 2026110951, 2026110952, and 2026110953.
Each method completes N128 at every seed.
The order rotates across seeds.
Each seed also has a second independent Zero process start.
The primary analysis contains 27 starts and 3456 requests.
The isolation replacement adds one actual N128 execution, not another independent seed.
Its original and replacement TTFT p95 values are 7879.51 and 7832.21 milliseconds.
Their ITL p99 values are 159.30 and 158.88 milliseconds.
Both remain visible in the [isolation sensitivity account](huginn_autonomous_adaptive_loop_pd_net_benefit/confirmation/primaryconfirmv2_isolation_sensitivity.csv).

All 24 non-reference output comparisons pass for every input and output token.
The required N8/N32 state and lifetime gates pass.
Every included Native P and D worker exits normally.
The marker-free cohorts do not contain intermediate-state hashes.
Their state qualification comes from the separate N8/N32 diagnostics.
No confirmation arm coalesces non-first token deliveries.

### Latency ratios

Values below are paired geometric ratios to the first Zero at the same seed.
Brackets contain start-level bootstrap 95% confidence intervals from 10,000 samples.
The table rounds to four decimals.
The linked data retain full precision.
Three independent Arrival seeds provide limited confidence coverage.
These are not token-level confidence intervals.

| Method | TTFT p95 | E2E p95 | ITL p99 | TPOT p95 |
| --- | --- | --- | --- | --- |
| Loop31, 512-token tiles | 0.8272 [0.7893, 0.8619] | 0.8426 [0.8100, 0.8724] | 1.1027 [1.1025, 1.1028] | 1.0241 [1.0228, 1.0254] |
| Loop28, full prompt | 0.4885 [0.4185, 0.5662] | 0.5386 [0.4772, 0.6059] | 1.7088 [1.6853, 1.7359] | 1.2096 [1.1935, 1.2322] |
| Token256, eight-Loop groups | 0.5971 [0.5199, 0.6702] | 0.6377 [0.5711, 0.6991] | 1.7720 [1.7587, 1.7896] | 1.1826 [1.1721, 1.1908] |
| Token64, SM32, uncapped | 0.9801 [0.9552, 1.0236] | 0.9869 [0.9641, 1.0266] | 1.0892 [1.0844, 1.0928] | 1.0673 [1.0652, 1.0704] |
| Adaptive Loop, early KV | 1.1128 [1.0613, 1.1448] | 1.1041 [1.0576, 1.1316] | 0.9748 [0.9563, 1.0117] | 0.9830 [0.9678, 1.0082] |
| Adaptive Loop, late KV | 1.0754 [1.0144, 1.1769] | 1.0681 [1.0117, 1.1609] | 0.9513 [0.9470, 0.9554] | 0.9707 [0.9693, 0.9730] |
| Adaptive Token | 1.0341 [1.0189, 1.0582] | 1.0300 [1.0164, 1.0519] | 0.9569 [0.9504, 0.9655] | 0.9722 [0.9663, 0.9766] |

The static methods pass the absolute frozen Decode screen in all three confirmation starts.
They fail the stronger Native-relative tail gate.
The adaptive methods preserve the measured Decode tail, but do not improve TTFT or E2E.
No candidate passes all latency gates.

### Actual work and serving rates

Every method below completes 384 requests.
The denominator for full-prompt Prefill Loops is 12,288.
The denominator for recurrent prompt rows is 25,165,824.
Token suffix rows are not complete full-prompt Loops.

| Method | Actually moved requests | Complete full-prompt Loops | D recurrent prompt rows | Completed QPS ratio [95% CI] | SLO-goodput ratio [95% CI] |
| --- | ---: | ---: | ---: | --- | --- |
| Loop31, 512-token tiles | 384 | 384 | 786,432 | 1.0395 [1.0371, 1.0414] | 1.2375 [1.0399, 1.5621] |
| Loop28, full prompt | 384 | 1536 | 3,145,728 | 1.1312 [1.1267, 1.1372] | 2.6459 [1.2793, 7.8867] |
| Token256, eight-Loop groups | 384 | NOT_APPLICABLE | 3,145,728 | 1.0995 [1.0958, 1.1019] | 1.7430 [1.1708, 3.3022] |
| Token64, SM32, uncapped | 384 | NOT_APPLICABLE | 786,432 | 1.0043 [0.9952, 1.0094] | 0.8113 [0.7464, 0.9463] |
| Adaptive Loop, early KV | 0 | 0 | 0 | 0.9774 [0.9688, 0.9871] | 0.6221 [0.4844, 0.7324] |
| Adaptive Loop, late KV | 2 | 6 | 12,288 | 0.9858 [0.9690, 0.9970] | 0.7494 [0.6197, 0.9347] |
| Adaptive Token | 2 | NOT_APPLICABLE | 16,384 | 0.9933 [0.9901, 0.9952] | 0.8256 [0.7426, 0.8708] |

Loop28 moves 12.5% of recurrent prompt rows.
Loop31 moves 3.125%.
Higher goodput in these forced interventions does not waive the failed Decode contract.
These rates do not establish sustainable capacity.

The late online Loop policy moves two Loops in Seed 2026110951.
It moves four Loops in Seed 2026110953.
It moves none in Seed 2026110952.
The resulting moved-request fraction is 2/384, or 0.521%.
The transferred full-prompt Loop fraction is 6/12,288, or 0.0488%.
No primary confirmation run contains two different nonzero online cuts.
Same-run Primary adaptive diversity is therefore `NOT_DEMONSTRATED`.
The separate 2048/4 discovery does demonstrate same-run variable cuts, but fails the net-benefit target.

### Completion and tail checks

| Actual online continuation | Predicted commit-to-output (ms) | Observed Host interval (ms) | Reservation (ms) |
| --- | ---: | ---: | ---: |
| Two Loops, Seed 2026110951 | 155.9524 | 619.3925 | 350 |
| Four Loops, Seed 2026110953 | 199.5699 | 615.1849 | 350 |
| Token256, Seed 2026110951 | 253.7948 | 370.7540 | 350 |
| Token256, Seed 2026110952 | 253.7948 | 348.9265 | 350 |

Both online Loop continuations miss the reservation.
One of the two Token continuations also misses it.
The largest inter-quantum Host waits are 70.97 and 22.29 milliseconds for the Loop cases.
They are not kernel execution times.
These observations reject the current completion model as a future completion guarantee.
They do not identify the physical maximum safe migration count.

All 51,840 non-first confirmation ITL intervals remain below the frozen 192.8213-millisecond threshold.
This does not imply Native-relative non-inferiority.
The first Zero starts have maximum ITLs of 96.98, 97.38, and 97.78 milliseconds.
Loop31 maxima are 106.39, 105.32, and 105.53 milliseconds.
Loop28 maxima are 164.77, 164.25, and 159.40 milliseconds.
Token256 maxima are 171.33, 172.30, and 171.08 milliseconds.

The repeated Zero TTFT ratio is 1.0079 [1.0032, 1.0141].
Its ITL ratio is 0.9514 [0.9450, 0.9629].
Its maximum ITL varies more, with ratio 1.1462 [0.9433, 1.6711].
One repeated Zero maximum reaches 162.07 milliseconds.
The report retains this start and does not infer causal interference from that maximum alone.
The full [paired results](huginn_autonomous_adaptive_loop_pd_net_benefit/confirmation/primaryconfirmv2_paired_start_ratios.csv) also compare against the repeated Zero.

### Loop against the strong Token control

Loop28 and Token256 move the same recurrent rows, not the same complete operation set.
Their paired TTFT ratio is 0.8182 [0.8051, 0.8448].
Their E2E ratio is 0.8446 [0.8317, 0.8668].
Their ITL p99 ratio is 0.9643 [0.9530, 0.9870].
Loop28 therefore has a reproducible advantage over this Token cut on those metrics.
That result is real, but does not establish the requested Loop-specific net benefit.
Both methods fail the Native-relative Decode contract.
The Coda placement and graph-selection differences remain part of the comparison.
The generic logical-layer frontier remains an equivalent mechanism.

Per-seed p50/p95/p99 latency, maximum ITL, actual arrival rate, token rates, and SLO attainment are in the [serving CSV](huginn_autonomous_adaptive_loop_pd_net_benefit/confirmation/primaryconfirmv2_serving_results.csv).
The [gate audit](huginn_autonomous_adaptive_loop_pd_net_benefit/confirmation/primaryconfirmv2_scientific_gate_checks.json) retains each independent run's checks.

## 9. Implementation and research decision

The useful retained result is a qualified Native Runtime improvement, not a new successful Loop scheduler.
Fixed-depth grouped submission and the unused Gate readback change retain the complete model calculation.
Deferred CODA collection preserves Native cohort retirement.
Those changes also apply to Zero.
The source preparation and exact continuation paths remain optional experiment flags.

| Component | Source files | Preserved condition |
| --- | --- | --- |
| Fixed-depth execution | `fixed_decode.py`, `fixed_submit.py` | All R32 model operations and actual generated tokens |
| Independent submission and output collection | `stream_route.py`, `deferred_coda.py` | Separate buffers, Native events, cohort retirement |
| Selective source preparation | `planner.py`, `partial_prepare.py`, `predictive_prepare.py` | Current observations, exact Native fallback before commit |
| Early immutable KV and later cut commit | `early_tap.py`, `monotone_early_protocol.py` | Proven frontier, fragment counts, hidden state, Native ACK |
| D work ordering and quantum cap | `decode_priority.py`, `prefill_group.py` | Decode priority, complete recurrent work, no kernel preemption |
| Lifetime and deployment checks | `graph_lifetime.py`, `exit_integrity.py` | Owning context teardown and actual child exits |

The [journal](huginn_autonomous_adaptive_loop_pd_net_benefit/optimization_journal.md) records adopted and rejected hypotheses.
The campaign does not recommend another full dynamic Loop campaign on the unchanged primary Runtime and contract.
First qualify any generic exact projection or execution improvement against the same Native state contract.
Then repeat the Zero and offload controls together.
Do not assume such an unimplemented improvement will create D slack.

| Question | Final scope-limited decision |
| --- | --- |
| High-migration Loop service with Native-like Decode | `NO_GO_DECODE_INTERFERENCE` in the tested configurations |
| Tested online adaptive completion economics | `NO_GO_COMPLETION_ECONOMICS` |
| Loop-specific novelty beyond generic logical-layer scheduling | Not established |
| Physical idle maximum and clean packing Oracle | `INCONCLUSIVE_MEASUREMENT` |
| Universal impossibility of exact Loop PD | Not established |

The investigation does not exhaust every exact GPU-resident controller or numerical kernel design.
It does not execute a new Input/QPS/slot grid or N512 persistence.
It does not establish a physical capacity Oracle with measured prediction error.
No candidate satisfies the joint confirmation gate that would justify persistence expansion.
These missing tests remain limitations, not zero-valued measurements.

## 10. Reproduction and closure

The publication parent is `6cadf1011a0b596efe39ccccefc9898dd8ef2925`.
The external Runtime HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Every actual run retains its configuration hash, command, worker mapping, and frozen source archive.
The final source hash is not substituted for an older trial's frozen source.
The [inventory](huginn_autonomous_adaptive_loop_pd_net_benefit/audit/experiment_inventory.csv) includes failed and unexecuted configurations.
It does not treat missing execution as a zero result.

The final source check passes 112 new CPU tests, 26 inherited completion-aware tests, and 13 protocol tests.
Compilation also passes.
The read-only inherited-source audit matches 662 source blobs against the publication parent.
The existing staged-entry hash and external Runtime diff remain unchanged at this check.

The [confirmation figure](huginn_autonomous_adaptive_loop_pd_net_benefit/figures/figure_3_held_out_confirmation.svg) shows independent-start uncertainty.
The [figure manifest](huginn_autonomous_adaptive_loop_pd_net_benefit/figure_manifest.json) distinguishes Host observations and diagnostic device enclosures.
The [held-out interactive timeline](huginn_autonomous_adaptive_loop_pd_net_benefit/figures/held_out_seed51_observed_timeline.html) compares matched Seed 2026110951 requests.
Loop31 and Loop28 each show later P submission during 124/128 moved-request D receipt-to-output intervals.
The actual adaptive Loop request also shows later P submission during that interval.
This is observed Host progression, not a physical GPU overlap measurement or a clean idle-window Oracle.

The campaign starts at 2026-11-10 13:25:48 UTC.
Its last scientific GPU worker finishes by 2026-11-11 03:24:41 UTC.
The elapsed interval is 13 hours 58 minutes 53 seconds.
That interval includes design, implementation, analysis, and GPU work.
It is not continuous GPU kernel time.
Report preparation, packaging, and publication follow that interval.

Final Burner checks pass at 2026-11-11 03:32:01 UTC.
All GPUs 0–7 have an approved resident engine with checked process ownership.
Every engine starts generation and completes at least one round.
Both utilization samples show 100% on each GPU.
No unknown GPU process appears in the verification.
See the [restoration receipt](huginn_autonomous_adaptive_loop_pd_net_benefit/state/final_burner_verification.json).

The publication branch is `flashvep/huginn-adaptive-loop-net-benefit`.
The [publication receipt](huginn_autonomous_adaptive_loop_pd_net_benefit/state/publication_receipt.json) records the verified evidence commit and remote SHA.
The metadata child records that receipt.
The final handoff gives the metadata child's independently checked remote SHA.
The lossless [bundle manifest](huginn_autonomous_adaptive_loop_pd_net_benefit/evidence_bundle/manifest.json) identifies recoverable evidence by SHA256.
The original raw files remain unchanged.
The local Notion response cache is not part of the publication.
