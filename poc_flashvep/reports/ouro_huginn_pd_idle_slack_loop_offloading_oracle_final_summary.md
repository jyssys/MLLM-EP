# Native PD idle capacity and Loop offloading Oracle

## 1. Result

The requested clean Native idle capacity remains `NOT_MEASURED`.
The full D-stage collector completed, but it changed Decode tails beyond the 5% interference gate.
Both-stage captures also failed progress.
These failures do not prove absent physical headroom.
They prevent a reliable physical-capacity conclusion.

The campaign therefore does not establish that Ouro can transfer 500 Loops or Huginn can transfer 4000 Loops safely.
It also does not establish that either target is physically impossible.
The historical Huginn cuts alone cannot reach 4000 Loops.
Their structural ceiling is 3072 Loops across 384 requests.

The overall Oracle verdict is `INCONCLUSIVE_MEASUREMENT` for both models.
The separate adaptive Scheduler study continues beyond this measurement gate.
Its actual offloading results are not Oracle capacity.

## 2. Fixed configuration

| Field | Ouro | Huginn |
| --- | --- | --- |
| Model | ByteDance/Ouro-1.4B | tomg-group-umd/huginn-0125 |
| Recurrent depth | R4 | R32 |
| Input / output cap | 4096 / 32 | 2048 / 16 |
| Offered QPS | 6.356 | 1.971 |
| P/D slots | 2/8 | 2/8 |
| Scientific GPU pair | 0/1 | 4/5 |
| Precision | Original BF16 | Original BF16 |
| Prompt pool | Preserved ShareGPT/LongBench controlled prefixes | Same preserved source |
| Prefix caching | Disabled | Disabled |
| Output policy | Fixed cap, ignore EOS | Fixed cap, ignore EOS |

The Oracle calibration arrival seed is 2026110910.
The requested confirmation seeds are 2026110911, 2026110912, and 2026110913.
The same preserved 128 distinct prompt IDs appear in each seed.
Three starts do not contain 384 independent texts.

The scientific lane runs one pair at a time.
All eight approved Burners stop during clean timing and profiling.
The resource guard restores them after scientific GPU work.
The final restoration receipt is `state/final_burner_verification.json` in the evidence directory.

## 3. Evidence definitions

Kernel busy time is the union of clipped CUDA kernel intervals across recorded streams.
The analysis does not add overlapping stream durations.
Kernel idle time is the complement of that union within the recorded window.
This quantity is not SM occupancy or memory-bandwidth headroom.

The kernel-only collector does not measure H2D/D2H copies separately.
Its records include graph-node identity, device, context, and stream.
They do not identify every kernel's semantic category.
Separate NCCL timing remains `NOT_MEASURED`.
These runs use Native NIXL handoff, not an added NCCL data path.

Scheduler-visible empty queues are host observations.
CUDA Event durations are stream enclosures.
Neither quantity is a substitute for a physical kernel gap.
Decode-safe slack additionally requires deadlines, request readiness, state ownership, and resource admission.

## 4. Collector qualification and failure analysis

The campaign tested the following capture changes:

1. A legacy concurrent-kernel collector.
2. Deferred buffer parsing after serving.
3. Capture after graph warm-up, with kernel records only.
4. Periodic non-forced buffer flush.
5. A collector built against the installed CUDA 12.9 CUPTI library.
6. D-only capture, without P capture.

The collector self-tests recovered eager and graph kernels without dropped or invalid records.
Those self-tests do not establish low overhead in this serving workload.
Successful D-only serving traces reported zero dropped and invalid records.
Their Decode tails still changed substantially.

| Same-arrival N128 calibration | Ouro ITL p99, ms | Huginn ITL p99, ms |
| --- | ---: | ---: |
| Initial marker-free Native control | 41.733 | 124.434 |
| Legacy D-only full capture | 54.796 | 158.069 |
| Change from that initial control | +31.3% | +27.0% |

The analysis also uses two additional Native A/A starts.
The machine-readable gate compares the instrumented run with their clean median.
It checks throughput, TTFT p95, TPOT p95, and ITL p99.
All complete full D captures failed the 5% gate.
Maximum ITL also receives a separate comparison with observed Native A/A variation.
Shorter captures did not remove the problem.
The 50 ms capture controls measured ITL p99 of 45.169 ms for Ouro and 133.642 ms for Huginn.
Their Maximum ITL values were 92.618 ms and 377.418 ms.
The controls do not establish an interference-safe full-cohort idle distribution.

Both-stage captures produced progress failures, including Native NIXL/CUDA kernel submission errors.
Changing buffer parsing and the CUPTI library did not remove the failure.
The exact interaction remains `ROOT_CAUSE_INCONCLUSIVE`.
The evidence does not isolate a unique lock, driver, NIXL, or graph defect.
An older CUPTI version alone is not an established cause.

The analysis stops expansion of this intrusive collector to the three confirmation seeds.
It does not fill their missing observations with calibration values.
The complete attempt inventory is `idle_slack_oracle/capture_attempt_inventory.json`.

## 5. Instrumented D-stage measurements: diagnostic only

These values describe one perturbed N128 execution per model.
They do not describe qualified clean Native capacity.
They must not be multiplied by three to represent the requested 384 requests.

| Diagnostic metric | Ouro | Huginn |
| --- | ---: | ---: |
| Recorded measurement window | 26.258836 s | 114.761104 s |
| D-stage kernel busy time | 17.726857 s | 49.244725 s |
| D-stage kernel busy | 67.508% | 42.911% |
| D-stage kernel idle | 32.492% | 57.089% |
| Recorded idle time | 8.531979 s | 65.516379 s |
| Idle window count | 4,268,774 | 6,014,921 |
| Mean idle window | 0.001999 ms | 0.010892 ms |
| p50 | 0.000416 ms | 0.000448 ms |
| p90 | 0.000544 ms | 0.000640 ms |
| p95 | 0.000608 ms | 0.001632 ms |
| p99 | 0.000768 ms | 0.158080 ms |
| Maximum | 295.256428 ms | 1949.721530 ms |

Most counted gaps are much shorter than one recurrent quantum.
Total idle duration therefore cannot establish executable Loop capacity.
A long initial gap also precedes relevant source states.
Packing work into that gap without request readiness would overstate capacity.

| Minimum gap | Ouro count / total seconds | Huginn count / total seconds |
| --- | ---: | ---: |
| 1 ms | 2038 / 5.017466 | 706 / 35.566065 |
| 2 ms | 1902 / 4.799726 | 85 / 34.712080 |
| 5 ms | 8 / 0.358943 | 33 / 34.597399 |
| 10 ms | 4 / 0.333743 | 27 / 34.539216 |
| 20 ms | 1 / 0.295256 | 26 / 34.524148 |
| 30 ms | 1 / 0.295256 | 26 / 34.524148 |
| 50 ms | 1 / 0.295256 | 26 / 34.524148 |

The corresponding interval arrays and clock-drift checks remain in `idle_slack_oracle/`.
The histogram is `figures/instrumented_idle_gap_histogram.png`.
Its title marks the failed interference gate.
Decode-batch/KV-conditioned clean physical distributions remain `NOT_MEASURED`.
Host batch and KV records alone cannot establish those distributions.

## 6. Refreshed full-prompt continuation costs

The campaign completed 15 serial calibration cells with eight requests per cell.
All 120 outputs matched their preserved reference outputs.
Work accounting also passed.
These measurements use CUDA Events, not kernel unions.

| Model | D remaining Loops | Median sum of operation enclosures, ms | Median complete recurrent-suffix enclosure, ms |
| --- | ---: | ---: | ---: |
| Ouro | 1 | 46.234 | 46.272 |
| Ouro | 2 | 92.214 | 99.874 |
| Ouro | 3 | 138.334 | 153.356 |
| Huginn | 1 | 27.369 | 27.408 |
| Huginn | 2 | 45.492 | 49.490 |
| Huginn | 4 | 81.188 | 92.580 |
| Huginn | 8 | 153.062 | 179.541 |

The complete enclosure includes submission gaps between operations.
Neither column guarantees online execution time or worst-case execution time.
Coda sampling does not cover every request.
Missing Coda measurements remain null in the raw table.
The always-available GPU-clock-to-first-output field is not renamed Service TTFT.

The campaign also refreshed 128/256/512/1024-token execution geometry.
Each complete Loop requires every token tile and its valid dependencies.
The packing code divides completed tile counts by tiles per Loop.
It does not count an incomplete tile set as one completed Loop.

## 7. Oracle classes and current status

| Requested metric, three confirmation seeds | Ouro | Huginn |
| --- | --- | --- |
| Qualified Native D kernel busy / idle | NOT_MEASURED | NOT_MEASURED |
| Qualified idle count / distribution | NOT_MEASURED | NOT_MEASURED |
| Decode-safe slack, seconds | NOT_MEASURED | NOT_MEASURED |
| Qualified windows that fit one full Loop | NOT_MEASURED | NOT_MEASURED |
| No-tiling physical Oracle Loops | NOT_MEASURED | NOT_MEASURED |
| Tiled physical Oracle Loops | NOT_MEASURED | NOT_MEASURED |
| Decode-safe Oracle Loops | NOT_MEASURED | NOT_MEASURED |
| Completion-feasible Oracle Loops | NOT_MEASURED | NOT_MEASURED |
| Oracle offloaded requests / work fraction | NOT_MEASURED | NOT_MEASURED |
| Seed mean and variability | NOT_MEASURED | NOT_MEASURED |
| Target 500 / 4000 established | No | No |

`requested_three_seed_status.csv` lists each missing seed separately.
Unknown values are not zero.

The diagnostic packing code also calculates a relaxed cost-model bound.
It uses the fastest measured median core cost and each observed gap's duration.
It excludes source readiness, transfer, ACK, Coda, CPU scheduling, and completion deadlines.
It is not the requested physical, Decode-safe, or completion-feasible Oracle.

For the perturbed Ouro N128 trace, full-prompt packing finds one gap and six core Loop slots.
Dividing total idle time by the same cost instead gives 185 slots.
This difference demonstrates fragmentation, not qualified serving capacity.
The 128/256/512/1024-token relaxed results are 1/3/4/6 complete Loop equivalents.
Tiling therefore does not automatically increase useful capacity.

For the perturbed Huginn trace, full-prompt packing reaches the historical N128 structural ceiling of 1024 Loops.
The 128-token relaxed result is 822 Loops.
The 256/512/1024-token relaxed results also reach that ceiling.
These counts ignore request-ready times and new handoff costs.
They do not establish that 128 actual requests can complete those continuations.

The expanded R16 ceiling produces the following conditional Huginn packing estimates.
They use the same intrusive N128 trace and the same ready-free median-cost model.
The full-prompt result is 1961 Loop equivalents.
The 128/256/512/1024-token results are 822/1393/1767/1886 equivalents.
These are not additional measured serving results.
They do not establish a 384-request maximum.

An unchanged-gap packing bound also omits rescheduled Decode work.
It is not a global maximum over every legal counterfactual schedule.
The campaign does not infer fundamental physical failure from it.

## 8. Target scale and state frontiers

Ouro's historical R1/R2/R3 cuts permit at most 3 suffix Loops per request.
Their structural ceiling is 1152 Loops across 384 requests.
The target 500/1536 is 32.552% of all prompt recurrent Loops.
The ceiling permits that count but does not establish available time.

Huginn's historical R24/R28/R30/R31 cuts permit at most 8 suffix Loops per request.
Their structural ceiling is 3072/12288 Loops, or 25%.
The target 4000/12288 is 32.552%.
That target is impossible under the historical cut set alone.

The campaign separately qualified R16 and R21 frontiers in N8 GPU state tests.
Both matched output, valid prompt KV, hidden state, and full recurrent work.
An actual online mixed-cut N32 test also passed.
It committed 13 requests and completed 157 full-prompt Loop equivalents.
The selected suffix counts were 4, 8, 11, and 16 Loops.
N32 means simultaneous request submission, not 32 simultaneous resident GPU executions.
This result establishes more than a hypothetical frontier.
It does not establish their Decode-safe online capacity.
The expanded R16 ceiling is 6144/12288 Loops.
The main historical-cut Oracle remains separate.

The refreshed fastest-median cost projection is about 23.015 s for 500 Ouro Loops.
It is about 74.175 s for 4000 Huginn Loops under the expanded candidate assumption.
These projections include an optimistic terminal-cost allowance.
They exclude transfer, scheduling gaps, and waiting for Decode-safe opportunities.
They are not guaranteed minimum times or measured capacity.
Reliable clean safe-slack totals are missing, so a valid capacity ratio cannot be calculated.

The separate real intervention completed 853 Huginn Loops across 384 requests.
All three runs passed the frozen absolute Decode screens.
Its ITL p99 geometric ratio to original Native was 1.1773.
This result does not prove that those Loops fit unchanged Native physical idle windows.
The Scheduler can move Decode execution within the absolute deadline margin.
The actual Ouro intervention completed 6 Loops across 384 requests.
Neither result is an Oracle maximum or a completion-feasible upper bound.

## 9. Five direct answers

1. Clean Native D idle percentage is `NOT_MEASURED` for both models.
   Diagnostic instrumented idle is 32.492% for Ouro and 57.089% for Huginn.
   These percentages fail the required interference qualification.
2. Diagnostic counts are 4,268,774 and 6,014,921 gaps.
   Their means are 0.001999 ms and 0.010892 ms.
   Clean counts and means remain `NOT_MEASURED`.
3. Valid request-ready, transfer-constrained full-Loop completion capacity remains `NOT_MEASURED`.
   Relaxed packing values are diagnostic model results only.
   They do not prove that tiling increases online capacity.
4. The campaign does not establish SLO-safe transfer of 500 Ouro or 4000 Huginn Loops.
   Historical Huginn cuts additionally impose a 3072-Loop ceiling.
5. Measurement interference is the current capacity-verification blocker.
   Fragmentation, transfer, dependency, and scheduling granularity remain distinct candidate limits.
   The evidence does not identify one of them as the sole physical limit.

## 10. Reproduction and preservation

The source is `poc_flashvep/completion_aware_loop_pd/`.
The main analysis commands are `idle_oracle`, `cost_audit`, and `oracle_summary` in that package.
The source records per-run configuration hashes, source hashes, GPU mapping, and arrival seeds.
The evidence directory is `ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation/`.

Raw files remain unchanged.
The lossless evidence bundle preserves ordered SHA256 chunks and file hashes.
`python -m poc_flashvep.completion_aware_loop_pd.evidence_bundle --verify` checks every reconstructed file hash.
The publication uses a temporary Git index.
It preserves the active checkout, existing staged entries, and external runtime diff.
The final publication and Burner receipts record those checks.
