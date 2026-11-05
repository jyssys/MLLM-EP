# Native-path Loop Continuation and 2D Token × Loop: final results

## 1. Decision

**Overall decision: INCONCLUSIVE. Do not start a full Dynamic 2D Scheduler from these results.**

Native-path continuation is possible for the qualified cuts in both models.
It removes much of the earlier staged/snapshot cost.
Huginn Loop31 improves finite-cohort TTFT in the tested native path.
Ouro does not show a Decode-safe serving improvement over Native in the tested actions.
Neither model establishes additional 2D headroom over a qualified Token-only oracle.

| Question | Ouro R4 | Huginn R32 |
|---|---|---|
| Exact native-path continuation | PASS for tested Loop, Token, and 2D cuts at N8/N32 | PASS for Loop31 and 2D(c31,k1024) at N8/N32 |
| Zero-offload throughput within ±5% | Supported by three-start ratio interval | Supported by three-start ratio interval |
| Zero-offload latency equivalence within ±5% | NOT_ESTABLISHED | NOT_ESTABLISHED |
| Useful serving result | No Decode-safe gain over Native on tested points | Loop31 finite-cohort TTFT improvement |
| Partial Token baseline | Two cuts qualified | BLOCKED by effective D Wavefront scheduler |
| Validated dynamic Feasible Oracle | NOT_IMPLEMENTED_OR_VALIDATED | NOT_IMPLEMENTED_OR_VALIDATED |
| Additional 2D advantage over Token-only | NOT_ESTABLISHED | NOT_ESTABLISHED |
| Phase A | CONDITIONAL | CONDITIONAL |
| Final 2D research decision | INCONCLUSIVE | INCONCLUSIVE |

This is not `GO_2D_LOOP_SPECIFIC`.
It is also not a structural `NO_GO_LOOP_PD`.
Huginn Token qualification and a validated contention-aware oracle remain missing.
Ouro 2D repetitions contain a graph-cache coverage problem.

## 2. Execution, budget, and source

The campaign started on 2026-11-05 at 00:47:27 Asia/Seoul.
The first dispatch started at 00:59:00.381108.
The last dispatch ended at 05:17:52.787796.
The GPU phase lasted 4.5072 hours from campaign start.
The sum of bounded job durations was 4.2096 hours, including startup and diagnostics.
These durations are not kernel execution times.

The GPU deadline was 05:47:27. The overall deadline was 06:47:27.
GPU experiments ended before their deadline.
An assistant capacity interruption delayed analysis and publication beyond the overall deadline.
CPU closeout resumed at 09:42:31. No post-deadline GPU experiment ran.
The analysis closure timestamp and elapsed hours are in `state/CAMPAIGN_CLOSED.json`.
Publication follows that closure. The local final remote-verification receipt records its completion timestamp.

The inventory contains 120 immutable configurations and 113 dispatches.
There were 93 completed dispatches, 20 failed dispatches, and 7 configurations without a dispatch.
Completed dispatches contain 6,480 offered requests, excluding warm-up.
The main serving matrix contains 47 runs and 5,344 requests.
It includes 40 N128 runs and 7 low-load N32 runs.
Four runtime-control N128 runs bring the total N128 count to 44 runs and 5,632 requests.
Completion does not mean that every scientific gate passed.
One early completed diagnostic produced exact outputs but transferred no partial work. Its migration gate failed.

| Item | Value |
|---|---|
| Publication parent | `5a4ff4a8a3f13119951b4e2ff202fb3541ac1d0a` |
| Publication branch | `flashvep/native-loop-continuation-2d-oracle` |
| Active checkout, preserved | `flashvep/looped-pd-kv-profile-headroom-recovery`, HEAD `f4ecea59f36f620c69c4fef09c590bf790802cd9` |
| External runtime HEAD | `6b164d3844d35ab8e753df1864f0c87dc180401a` |
| External runtime diff SHA256 | `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7` |
| Real index signature | `d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3` |
| Runtime | Python 3.14.6, PyTorch 2.11.0+cu129, NIXL 1.4.1, Transformers 5.16.1, FlashAttention 4.0.0b30 |
| GPU | H100 80GB, driver 570.211.01 |

Process-local hooks implement the candidate. They do not modify the external runtime.
Each dispatch preserves its configuration, source hashes, source archive, GPU mapping, and logs.
The alternate publication index preserves existing staged entries and the active checkout.
`state/publication_receipt.json` records the verified evidence commit and remote SHA.
A later metadata commit contains that receipt. The final remote verification records the metadata SHA locally.

## 3. Models, workload, and evaluation contract

| Model | Checkpoint revision | Fixed work | Physical pair | Nominal QPS: low / medium / heavy |
|---|---|---|---|---|
| ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | BF16, R4, input 4096, output 32 | 0/1 | 1.23460875 / 3.4569045 / 6.356 |
| tomg-group-umd/huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | BF16, R32, input 2048, output 16 | 4/5 | 0.37903355455 / 1.06129395274 / 1.971 |

Each N128 cohort uses 128 unique prompt IDs: 20 ShareGPT and 108 LongBench prompts.
The workload preserves previous model-tokenized controlled prefixes.
Generation uses fixed output caps with EOS ignored. It is not natural-EOS generation.
Prefix caching is disabled. No prefix reuse explains the results.
The first-start arrival seed is 2026110402. The generation seed base is 2026110362.
Paired methods share request order, token IDs, generation seeds, and open-loop Poisson arrivals.
Serial primitive calibration uses eight disjoint prompts.
Huginn remains within its 4096-token context limit.
Both models use two P compute slots and eight D compute slots.
Ouro P/D token budgets are 8192/4096. Huginn P/D token budgets are 4096/2048.
The primary logical prompt chunks are 4096 for Ouro and 2048 for Huginn.
Partial Token boundaries remain separate from those chunk budgets.

| Frozen limit, ms | Ouro | Huginn |
|---|---:|---:|
| Request TTFT | 1112.9925646819174 | 1963.9920117333531 |
| Request-average TPOT | 53.70702898374411 | 186.5715059141318 |
| Token ITL p99 | 65.08287642151119 | 192.8212538175284 |

The additional-work screen requires TPOT p95 ≤90% of its limit and ITL p99 ≤its limit.
Joint goodput counts requests that satisfy TTFT and request-average TPOT together.
The ITL screen remains separate. Failed requests cannot improve goodput.
Finite cohorts and overloaded heavy runs do not establish sustainable capacity.

TTFT includes arrival-to-first-token queueing.
Service TTFT starts at the recorded first Prefill execution boundary and includes handoff, D continuation, and first-token generation.
The boundary is a host execution/submission event. It is not the first physical GPU instruction.
All percentiles use per-request intervals. Component percentiles are not added.
CUDA event enclosures include stream waits and launch gaps. They are not exclusive kernel union.
Kernel union, SM activity, achieved FLOPs, and memory bandwidth are NOT_MEASURED in this campaign.
This report does not claim a new same-run proof of P compute saturation.

## 4. Implementation and qualification

Native Atomic Prefill uses one outer submission with multiple recurrent graph replays.
It is not one indivisible whole-depth CUDA graph.
A partial P request retains all hidden rows at its cut.
Huginn retains both recurrent state and fixed injection state.
The candidate prevents premature Huginn Coda-KV work at an incomplete cut.
D executes the remaining recurrent work and required boundary work.
Source buffers remain valid until native ACK permits release.

The candidate packs arbitrary valid KV depth-plane prefixes through the same native descriptor path.
Token, Loop, and 2D actions share that packing implementation.
Zero-offload uses native full-depth arithmetic and the feature's registered continuation storage.
It transfers only the valid final hidden row, not the full registered hidden buffer.
Partial handoff still transfers an exact state snapshot. A new incremental-streaming protocol is not implemented.
Per-request cut selection exists, but mixed-cut GPU lifecycle qualification is NOT_RUN.
No new online dynamic scheduler is implemented.

| N8/N32 qualification | Ouro | Huginn |
|---|---|---|
| Native and zero-offload | PASS | PASS |
| Loop cuts | 3, 2: PASS | 31: PASS. 28: BLOCKED |
| Token prefix cuts | 3072, 3840: PASS | 1984, 2016: BLOCKED |
| 2D frontiers | (2,2048), (3,3072): PASS | (31,1024): PASS. (28,1536): BLOCKED |

Qualification compares exact generated sequences and all valid prompt KV planes at the first Coda boundary.
It also compares the final prompt-token hidden state before Coda.
Source/destination fingerprints compare all transferred valid KV and hidden bytes.
Full active recurrent-row accounting and source/ACK lifetime checks pass for qualified actions.
The gate checks both coordinator credits and empty final native KV allocations.
It does not compare every intermediate hidden value after every operation.
Original-HF equivalence and downstream benchmark accuracy are NOT_MEASURED.
All 47 completed main serving runs pass their recorded output, work, and lifetime gates.

### Huginn blocker

The requested scheduler flag was non-wavefront. The actual D contract was Wavefront.
The inherited `D_local_probe_output_pacing=True` hook forces that change, even without an active local probe.
Huginn's synchronous callback then rejects grouped remaining depths:
`wavefront prefill requires one recurrent depth per scheduling step`.
This blocks Token suffixes, Loop28, and the deeper 2D action before performance qualification.
It is not evidence that those actions are inherently slow or numerically incorrect.

The CPU regression calls the actual probe installer with fake stage objects.
It reproduces the forced scheduler change. Disabling the unused hook preserves the requested setting.
The disabled configuration has no GPU qualification in this campaign.
The guard remains intact. No failed path expands into serving measurements.

## 5. Native-path cost and zero-offload gate

Medium-load first start, N128, event-marked. Values are Service TTFT p95 in ms.

| Path | Ouro | Huginn |
|---|---:|---:|
| Native Atomic PD | 239.928 | 975.565 |
| Staged Native PD, zero migration | 493.458 | 1979.929 |
| Full Snapshot PD, zero migration | 762.145 | 1991.622 |
| Native-path zero-offload | 251.128 | 943.872 |
| Native-path Loop3 / Loop31 | 380.621 | 828.793 |

The staged and snapshot controls use full depth and match the current Native output reference.
These controls are instrumented diagnostics. They have no separate marker-free control.
The earlier NO-GO therefore included substantial avoidable runtime overhead.
Native-path continuation does not automatically make every cut Decode-safe.

Three paired medium starts give the following zero/Native ratios.
Intervals use Student t with two degrees of freedom on start-level log ratios.
They assume normally distributed log ratios. They do not include selection uncertainty.

| Ratio, geometric mean [95% CI] | Ouro | Huginn |
|---|---|---|
| TTFT p95 | 1.1136 [0.7650, 1.6212] | 0.9980 [0.8611, 1.1566] |
| Service TTFT p95 | 0.9671 [0.6914, 1.3528] | 0.9694 [0.8285, 1.1343] |
| Completed throughput | 0.9982 [0.9931, 1.0033] | 0.9991 [0.9982, 1.0000] |

Throughput satisfies the ±5% equivalence screen. Latency equivalence remains unestablished.
The full zero-offload gate must not be marked PASS from throughput alone.

## 6. Measured serving results

Medium load, first independent start, N128. Latencies are ms. Goodput is requests/s.
The Decode column applies the frozen additional-work screen, not the joint TTFT gate.

| Model / method | TTFT p95 | Service p95 | TPOT p95 | ITL p99 | Joint goodput | Decode |
|---|---:|---:|---:|---:|---:|---|
| Ouro Native | 607.112 | 239.928 | 35.626 | 38.338 | 2.956 | PASS |
| Ouro zero | 802.731 | 251.128 | 36.909 | 39.277 | 2.903 | PASS |
| Ouro Loop3 | 521.835 | 380.621 | 51.736 | 101.920 | 2.858 | FAIL |
| Ouro Loop2 | 780.821 | 713.701 | 63.472 | 291.236 | 1.013 | FAIL |
| Ouro Token3072 | 512.157 | 413.813 | 50.004 | 98.315 | 2.959 | FAIL |
| Ouro Token3840 | 515.108 | 255.268 | 42.095 | 68.993 | 2.956 | FAIL |
| Ouro 2D(c2,k2048) | 541.161 | 338.048 | 48.887 | 131.183 | 2.951 | FAIL |
| Ouro 2D(c3,k3072) | 626.094 | 384.633 | 37.890 | 54.315 | 2.954 | PASS |
| Huginn Native | 1884.766 | 975.565 | 114.743 | 120.942 | 0.888 | PASS |
| Huginn zero | 1892.829 | 943.872 | 116.586 | 121.836 | 0.887 | PASS |
| Huginn Loop31 | 1696.883 | 828.793 | 125.337 | 156.890 | 0.902 | PASS |
| Huginn 2D(c31,k1024) | 2755.470 | 1472.040 | 111.987 | 152.384 | 0.680 | PASS |

### Huginn repetition and heavy-load control

Three medium starts give a Loop31/Native TTFT ratio of 0.9000, with 95% CI [0.8640, 0.9375].
The Service TTFT ratio is 0.9398 [0.7137, 1.2375].
The E2E ratio is 0.9722 [0.9520, 0.9928].
The completed-throughput ratio is 1.0003 [0.9983, 1.0023].
The goodput ratio is 1.0538 [0.9732, 1.1410]. Goodput improvement is not established.
All three Loop31 runs pass the Decode screen.
The intervals exclude cut-selection uncertainty. Start 1 also informed the selected comparison.

At heavy load, nominal QPS is 1.971 and realized arrival QPS is about 1.718.
Native TTFT p95 is 9788.222 ms. Loop31 gives 7365.133 ms, a 24.755% reduction.

Queue p95 changes from 8965.366 to 6569.082 ms.

Service p95 changes from 963.387 to 904.279 ms.

E2E p95 changes from 11519.022 to 9261.857 ms.

Goodput changes from 0.307254 to 0.419820 requests/s, a 36.636% increase.
Loop31 TPOT p95 is 127.356 ms. ITL p99 is 159.307 ms. Maximum ITL is 173.446 ms.
This is one heavy start. It does not establish repeatable capacity or superiority over Token Split.

### Ouro repetition and heavy-load control

The Decode-safe fine 2D action has no measured advantage over Native.
Its three-start TTFT ratio is 2.1427 [0.3811, 12.047].
Its Service TTFT ratio is 1.9819 [1.2473, 3.1489].
Its goodput ratio is 0.5150 [0.1215, 2.1833].
These runs expose graph-cache coverage differences described below.

At heavy load, nominal QPS is 6.356 and realized arrival QPS is about 5.539.
Native TTFT p95 is 2572.970 ms. Fine 2D gives 2778.898 ms.

Native Service p95 is 245.407 ms. Fine 2D gives 385.830 ms.

Goodput changes from 2.094 to 1.839 requests/s. This is not an improvement.
Full per-request percentiles, E2E, maximum ITL, and arrival/completion rates remain in `serving_results.csv`.

## 7. Instrumentation limits and graph-cache diagnosis

Seven matched medium-load marker controls pass the ≤5% global response-span interference screen.
Their global wall changes are below 0.2%.
That screen does not bound local request-tail perturbation.
Marker-free Huginn Native TTFT p95 is 1779.668 ms. Loop31 gives 1707.036 ms, a 4.08% reduction.
Marker-free Ouro Native gives 609.462 ms. Fine 2D gives 1023.953 ms.
The Ouro 2D marker comparison changes TTFT p95 substantially despite a small global-wall change.
Do not call marked local tails profiler-independent.
Do not apply a medium first-start screen automatically to heavy load or other starts.

The packed recurrent graph cache keys exact query and context geometry.
It does not force every Token prefix into a power-of-two bucket.
The campaign gives each method the same four-entry cache budget.
Startup shape order can consume entries before the primary single-request shape appears.
Missing shapes use eager fallback.

| Ouro fine 2D run | Captured P query rows | Measured successful P graph calls |
|---|---|---:|
| Serial primitive N8 | 8192, 6144, 1024, 384 | 0 |
| Medium start 1 | 4096, 3072, 8192, 6144 | 512 |
| Medium start 2 | 8192, 6144, 3072, 1024 | 119 |
| Medium start 3 | 8192, 3072, 6144, 1024 | 119 |

Starts 2 and 3 miss the 4096-row shape. Their Service TTFT p95 is 564.426 and 622.734 ms.
Start 1 gives 384.633 ms.
The serial primitive misses both primary shapes. Its timing is not a clean intrinsic 2D cost.
Deterministic primary-shape warm-up is a proposed correction. No GPU result tests that correction.
The graph-cache result prevents a structural NO-GO from the current Ouro 2D slowdown.

## 8. Actual redistributed work and migration

The table describes N128 medium cohorts. Values are measured active recurrent token rows and transferred bytes.
They are not FLOP fractions. Attention lengths and Huginn boundary work differ across actions.

| Method | P prompt rows | D prompt rows | KV + hidden payload bytes, cohort |
|---|---:|---:|---:|
| Ouro Native | 2097152 | 0 | 412317384704 |
| Ouro Loop3 | 1572864 | 524288 | 311385128960 |
| Ouro fine 2D | 1966080 | 131072 | 388694540288 |
| Huginn Native | 8388608 | 0 | 730812825600 |
| Huginn Loop31 | 8126464 | 262144 | 703133122560 |
| Huginn fine 2D | 8257536 | 131072 | 719742566400 |

Every request in a successful static partial cohort performs actual P→D continuation.
These are not independent D-local 16-token probes.
Huginn Loop31 moves 3.125% of recurrent prompt rows and required full-prompt Coda-KV work.
Huginn fine 2D moves 1.5625% of recurrent prompt rows.
Ouro Token3840 and fine 2D both move 131072 rows per N128 cohort.
Equal rows do not imply equal Attention execution time or hidden-state payload.

Ouro Loop3 transfers 309237645312 KV bytes and 2147483648 hidden bytes per N128 cohort.
Huginn Loop31 transfers 697596641280 KV bytes and 5536481280 hidden bytes.
The current nonzero-depth 2D connector transfers the full retained hidden buffer, including redundant prefix rows.
Accounting includes those rows. It does not substitute a theoretical minimum.
Payload bytes exclude control metadata. Native transfer, descriptor, commit, ACK, and source-retention records remain in each raw run.

Serial disjoint calibration gives Ouro Native median P stream enclosure 181.938 ms.
Loop3 gives 137.293 ms on P and 47.255 ms on D.
Token3072 gives 133.878 ms on P and 53.747 ms on D.
Huginn Native gives 583.410 ms on P. Loop31 gives 555.459 ms on P and 27.240 ms on D.
Huginn median handoff increases from 61.454 to 132.420 ms in this diagnostic.
The complete eight-request records retain capture outliers and scheduling waits.
These enclosures are not exclusive GPU compute. Their medians must not be added to predict E2E.

## 9. HBM and admission

NVML reports 79.647461 GiB physical HBM per GPU. Percentages below use that denominator.
CUDA reports a different usable-byte denominator. The CSV retains both values.
Allocator totals overlap weights, KV, and workspace. They are not additional additive components.

| Initial stage component | Ouro P GiB (%) | Ouro D GiB (%) | Huginn P GiB (%) | Huginn D GiB (%) |
|---|---:|---:|---:|---:|
| Weights | 2.672 (3.36%) | 2.672 (3.36%) | 6.642 (8.34%) | 6.642 (8.34%) |
| KV pool | 48.000 (60.27%) | 48.000 (60.27%) | 41.250 (51.79%) | 67.354 (84.56%) |
| Workspace | 1.253 (1.57%) | 0.627 (0.79%) | 0.338 (0.42%) | 0.532 (0.67%) |

After zero-offload connector registration, allocated HBM is 52.019/52.065 GiB on Ouro P/D, or 65.31%/65.37%.
Huginn P/D allocated HBM is 48.482/76.151 GiB, or 60.87%/95.61%.
The registered hidden buffer is 32/640 MiB on Ouro P/D and 82.5/1650 MiB on Huginn P/D.
Initial pre-connector hidden-storage fields can be zero. They do not prove zero feature footprint.

The full-KV reservation policy remains unchanged.
Huginn D has about six full-request KV credits at this geometry, despite eight configured compute slots.
Logical free KV blocks are not free physical HBM.
The campaign does not claim that KV admission is eliminated.
Per-run reserved/free blocks, refusals, HBM records, and source-release events remain in raw evidence.

## 10. Dependency and oracle result

The implemented one-handoff staircase has a valid dependency order:

1. P computes all prompt tokens through depth c.
2. P computes prefix k through remaining depths.
3. D receives valid prefix KV and suffix state, then computes the remaining suffix depths.

Each tile needs previous-depth hidden state and same-depth causal prefix KV.
Huginn also needs Prelude, injection, and Coda state.
An arbitrary P-owned tile that depends on a D-owned predecessor cannot execute before one handoff.
CPU tests reject such invalid ownership partitions.
The tested GPU frontiers establish feasibility only for their qualified subsets.

`structural_frontiers_not_gpu_costs.csv` enumerates structural rows and state requirements.
It is not a latency model.
`measured_fixed_action_envelope.csv` selects the best executed Decode-safe fixed point from each family.
Native endpoints remain available in every family.
The best tested point is Native for Ouro and Loop31 for Huginn.
The Huginn 2D family includes the Loop31 endpoint. It has no measured incremental 2D advantage.
The Huginn Token family contains only its Native endpoint because partial Token qualification failed.
It is not a competitive Token-only oracle.

No event-driven model validates queueing, batching, transfer overlap, KV admission, and Decode interference against these runs.
Therefore Feasible Oracle prediction error is NOT_AVAILABLE.
Dynamic Token-only, Loop-only, and 2D oracle results are NOT_IMPLEMENTED_OR_VALIDATED.
A rigorous optimistic compute-only upper bound is NOT_ESTABLISHED.
No modeled speedup substitutes for these missing results.

An unrolled generic logical-layer split can reproduce each Loop boundary with the same weights, state, and costs.
Both workers already hold the full model. No weight-residency saving is measured.
The current evidence establishes no exclusive Loop-specific advantage.

## 11. Figures and raw evidence

The evidence root is `poc_flashvep/reports/ouro_huginn_native_loop_continuation_2d_oracle/`.
`figure_manifest.json` records sources and limitations for every PNG/SVG pair.

| Figure | File stem | Scope |
|---|---|---|
| 1 | `figure_1_native_staged_continuation` | Measured runtime costs and serving latency |
| 2 | `figure_2_zero_offload_uncertainty` | Three-start ratios and uncertainty |
| 3 | `figure_3_measured_actions_and_decode_tail` | Measured actions, transfer, and Decode gates |
| 4 | `figure_4_measured_envelope_oracle_unavailable` | Fixed-action envelope. Missing oracle remains explicit |
| 5 | `figure_5_dependency_frontier_and_measured_points` | Structural dependency diagram and qualified measured points |
| 6 | `figure_6_graph_cache_root_cause` | Measured cache coverage and latency variation |

These figures do not invent kernel-idle timelines or unmeasured oracle values.
Configurations, failures, request outputs, work accounting, transfer checks, and source hashes remain in `configs/`, `logs/`, and `raw/`.
Analysis tables include `serving_results.csv`, `primitive_costs.json`, `paired_start_ratios.json`, and `graph_cache_observations.csv`.
The full failed-job inventory remains in `failure_inventory.csv` and `failures.json`.
No failed startup job becomes a latency datapoint.

## 12. Reproduction and next decision

Run the following CPU-only checks from the repository root:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest discover -s poc_flashvep/native_loop_2d_oracle -t . -p 'test_*.py' -v
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_loop_2d_oracle.analyze
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_loop_2d_oracle.figures
```

The dispatcher commands and exact immutable configurations remain in `commands.jsonl`.
The original GPU deadline remains enforced. Re-running GPU jobs requires a new approved campaign namespace and time budget.
Do not edit the expired deadline to present later runs as original-budget evidence.

The next useful work is limited and conditional:

1. Disable the unused D probe scheduler hook consistently across all arms. Qualify Huginn Token/deeper cuts at N8→N32.
2. Warm required primary graph shapes deterministically within the same graph-cache budget. Repeat the Ouro controls.
3. Repeat marker-free matched comparisons. Separate local-tail perturbation from global-wall interference.
4. Build and validate a contention-aware oracle only after those costs and baselines are qualified.

The current evidence supports a native-path runtime correction and a promising Huginn static result.
It does not support a new online Dynamic 2D method or a common Ouro/Huginn Loop-specific GO.
GPU 0–7 restoration and process ownership are recorded in `state/CAMPAIGN_CLOSED.json`.
