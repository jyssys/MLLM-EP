# Native PD Bottleneck Crossover

Status: FINAL_FINITE_CHARACTERIZATION.
The resumable queue drained at 15:25:01.896 UTC on 2026-10-31.
All planned successful conditions and the preserved failed startup have terminal dispatcher results.
The final evidence review did not authorize Phase C.
See [the final summary](looped_pd_prefill_decode_bottleneck_final_summary.md) for the scope-limited decision.

## Protocol

The primary Looped layout is one Prefill GPU and one Decode GPU.
Both GPUs contain the complete checkpoint.
This is PD disaggregation, not TP, EP, or permanent layer PP.
Full-Prefill PD describes full P-side recurrent ownership, not an unchunked Prompt.
The primary screens use C256.
At 3,500 input tokens, the recorded native boundaries contain 14 chunks.
The C2048 control contains two chunks.
Whole-prompt atomic PD was not measured.
The Looped comparisons disable shared-GEMM Prefill/Decode Mixed batching.
Decode requests can still batch with other Decode requests through the native recurrent path.

Clean timing uses one isolated lane.
The dispatcher stops only approved owned burners.
It refuses unrelated active GPU jobs.
Each run records physical GPU IDs and source hashes.
Selected repeats cross the physical GPU pairs.
The study does not claim a measured eight-worker contention penalty.

All primary lengths use explicit ShareGPT tokenizer prefixes.
Every request uses greedy controlled generation with EOS ignored.
The primary input/output cells are G1–G6 and C1/C2 from the frozen specification.
Huginn never exceeds its 4,096-token context limit.
See `looped_pd_bottleneck_crossover/configs/` for exact arrival offsets and Prompt IDs.

## Frozen SLOs

Eight separate serial calibration prompts determine each model's thresholds.
The input contains 3,072 tokens and the output cap is 128.
Each model has 1,016 measured post-first-token intervals in calibration.
No calibration event combines several tokens.
The thresholds remain fixed during candidate evaluation.

| Model | TTFT limit | Request-average TPOT limit | Secondary ITL limit |
|---|---:|---:|---:|
| Ouro R4 | 2,576.074 ms | 53.233 ms | 66.031 ms |
| Huginn R32 | 7,567.570 ms | 204.961 ms | 243.072 ms |
| Qwen3-1.7B | 184.982 ms | 8.631 ms | 11.490 ms |

The primary criterion combines TTFT and request-average TPOT.
The secondary ITL criterion is separate.
A passing average TPOT does not guarantee that every token interval passes.
`slo_sensitivity.csv` also uses the predeclared multiplier and absolute-ITL grids.
It does not replace the frozen primary criterion.

Goodput equals passing offered requests divided by the arrival-to-drain observation window.
Missing or failed requests do not pass.
The completed-request throughput uses the same observation window.
These finite values are not maximum sustainable capacity.

## Rates and queue classification

Initial N24 screens use 0.8 times the smaller measured independent service rate.
Selected G2 cases use N64 and factors 0.4, 0.6, 0.8, 1.0, and 1.2.
The central factor has three independent starts where the queue completes.
`independent_start_summary.csv` reports the resulting start-level intervals.

Target Poisson rate and realized cohort rate are different quantities.
For seed 2026103111, the N24 realized rate is 1.3423 times the target rate.
The N64 multiplier is 1.1451.
These are fixed finite traces, not a later rate adjustment.
All matched controls retain the same offsets.

The analysis separates these intervals:

1. Harness delivery and request admission.
2. KV or slot reservation before Prefill submission.
3. Actual P submission and recurrent queue observations.
4. Prefill service observed by the coordinator.
5. Exposed handoff and Decode activation.
6. First-token delivery and subsequent generation.

Reservation waiting is not automatically a Prefill compute queue.
Admission reasons can occur together.
Their sampled occupancy is not an exclusive causal time decomposition.
`bottleneck_screening.csv` therefore labels finite pressure, not proven infinite-horizon saturation.
A candidate still needs GPU, KV-credit, and Decode-SLO evidence before Phase C.

## Generic chunk-budget controls

The following pairs use the same native NIXL protocol, inputs, output caps, seeds, and arrival offsets.
They retain the default pool and eight compute slots.
The tuned setting changes both the chunk and token budget from 256 to 2,048.
It does not execute Loop Split or Token migration.
Each row uses one independent start with 24 measured requests.

| Model / cell | Chunk / budget | Joint SLO passed/N | TTFT p95 | TPOT p95 | E2E p95 | Finite SLO-goodput |
|---|---:|---:|---:|---:|---:|---:|
| Ouro G1 | 256 | 4/24 | 6,535.8 ms | 38.29 ms | 11,049.2 ms | 0.1450 requests/s |
| Ouro G1 | 2,048 | 23/24 | 2,239.8 ms | 40.82 ms | 6,428.0 ms | 1.0270 requests/s |
| Ouro G2 | 256 | 0/24 | 18,608.2 ms | 44.81 ms | 23,194.5 ms | 0.0000 requests/s |
| Ouro G2 | 2,048 | 24/24 | 1,398.5 ms | 34.80 ms | 5,652.5 ms | 0.8249 requests/s |
| Ouro G3 | 256 | 0/24 | 23,771.6 ms | 47.12 ms | 27,546.2 ms | 0.0000 requests/s |
| Ouro G3 | 2,048 | 24/24 | 1,596.3 ms | 33.20 ms | 5,616.9 ms | 0.7516 requests/s |
| Huginn G2 | 256 | 8/24 | 44,262.7 ms | 103.25 ms | 57,374.1 ms | 0.0415 requests/s |
| Huginn G2 | 2,048 | 11/24 | 36,486.5 ms | 104.32 ms | 49,727.0 ms | 0.0589 requests/s |

These comparisons show a generic chunk-budget effect.
They do not establish a Dynamic A gain.
Their empirical p95 values have limited N24 tail evidence.
One start per variant cannot establish a between-start improvement interval.
The original C256 capacity values remain the rate source for these identical-arrival controls.
They must not be presented as independently measured C2048 capacity.

## Enlarging the engine does not automatically provide SLO-safe spare capacity

The enlarged Ouro setting uses 32 compute slots and 22,528 KV blocks per worker.
Its complete 128-token and state qualification passed.
Its independent local capacity suggested a possible P/D crossover at G3.
Its prepared native-PD Decode rate was 3.706 requests/s.

The actual N64 online cohorts did not preserve the assumed Decode margin.
At target 2.5206 requests/s, the three starts passed 0/64, 1/64, and 5/64 primary SLOs.
Their request TPOT p95 values were 99.06, 101.91, and 104.61 ms.
All exceed the frozen 53.23 ms limit.
Sampled D-credit refusals also occurred.
An independent capacity ratio alone therefore cannot authorize continuation offloading.

These results do not compare identical workloads across the default and enlarged engine.
Their cohort sizes and offered rates differ.
They are a resource-control diagnosis, not a same-work speedup estimate.

## Prepared Decode: simultaneous versus staggered readiness

This diagnostic removes all Prefill computation and incoming KV transfer from the timed Decode interval.
Every request has valid imported Prompt KV and hidden state before timing starts.
The checkpoint, output cap, seed, backend, and memory geometry stay unchanged.
The release pattern is the experimental variable.

The table uses Ouro R4, 3,500 input tokens, 128 output tokens, and 24 requests.
It uses C2048, 32 slots, and 22,528 KV blocks.
The staggered condition uses a fixed Poisson release trace with target 2.5206 requests/s.
The zero-offset condition exercises the release gate without intentional delay.
The variant order is zero-offset, staggered, then simultaneous.

| Release condition | Decode observation window | Request TPOT p50 | Mean valid recurrent rows/submission | Mean valid Coda rows/submission |
|---|---:|---:|---:|---:|
| Simultaneous | 6.386 s | 48.776 ms | 23.828 | 23.303 |
| Zero-offset gate | 6.528 s | 48.339 ms | 23.828 | 23.303 |
| Staggered | 16.102 s | 86.469 ms | 12.577 | 5.532 |

The row averages include the two-request warm-up in process-lifetime counters.
The TPOT statistics exclude warm-up.
All measured output sequences match the native reference.
The zero-offset control does not explain the large observed TPOT difference.
The row counts describe active logical rows, not padded GEMM dimensions.
The recurrent graph capacity remains 32.
The qualified down projection still uses its fixed numerical geometry.

The staggered observation window includes deliberate arrival gaps.
Its ratio to the simultaneous window is not a valid Serving speedup.
However, per-request TPOT increases after the first token despite the absence of timed Prefill and transfer work.
The effective recurrent and Coda batches also become smaller.
The readiness pattern changes effective packing and runtime control work.
The diagnostic does not isolate those two costs.
Waiting-gate overhead is not separately measured.
This is evidence against using simultaneous cohort capacity as an online capacity guarantee.
It is not proof that recurrent model mathematics inherently requires that loss.
It also does not establish a pure GEMM throughput loss.

Do not subtract these diagnostic latencies from a different N64 Serving run to assign communication cost.
Those cohorts have different arrival horizons and request populations.
The result identifies a mechanism, not an additive E2E breakdown.

## E2E percentages and GPU traces

`e2e_component_shares_aggregate.csv` reports exclusive host intervals as percentages of summed request E2E.
The per-request percentage validator checks a 100% total.
This is not a pure GPU-computation or DMA decomposition.

`profile_device_summary.csv` reports separately measured CUPTI device unions.
Heavy profiling can change latency substantially.
The clean comparison excludes those runs.
NVML utilization is also not SM occupancy or exclusive kernel-busy time.

## Complete screens and rate sweeps

All eight original cells completed for both Looped models, with 24 requests per cell.
All eight native AR cells completed with 64 requests per cell.
The [complete screen table](looped_pd_bottleneck_crossover/completed_screen_tables.md) includes every model and cell.
The underlying CSV includes mean, p50, p95, and p99 request metrics.
No missing requests were removed from the offered denominator.

The C256 G2 sweep completed five target rates per model.
The central factor has three independent starts.
Each start contains 64 requests.
The [rate table](looped_pd_bottleneck_crossover/completed_rate_sweep_tables.md) preserves all fourteen runs.
No measured C256 rate reaches 90% joint SLO attainment.
This does not establish zero capacity or a tuned-engine capacity limit.
The lowest rates are finite points, not a complete search down to zero load.

Three-start mean finite goodput at the central factor is 0.0268 requests/s for Ouro and 0.0110 for Huginn.
The corresponding 95% intervals are [0.0000, 0.0857] and [0.0000, 0.0262].
These intervals describe independent starts, not paired algorithmic improvement.
The [interval table](looped_pd_bottleneck_crossover/independent_start_intervals.md) includes all repeated settings.
Tail evidence remains limited even with three starts.

## DP2 controls at the same G2 arrival trace

Each condition below uses 24 requests and the same two-GPU budget.
DP2 contains two independent replicas.
DP2 chunking keeps Prefill and Decode in separate core executions.
It is not a production-quality Native Mixed baseline.

| Model | Baseline | Chunk | Passed/N | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Finite goodput requests/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Ouro | PD_G1 | 256 | 0/24 | 18,608.22 | 44.81 | 23,194.48 | 0.0000 |
| Ouro | PD_G1 tuned | 2,048 | 24/24 | 1,398.53 | 34.80 | 5,652.53 | 0.8249 |
| Ouro | DP2 | 4,096 | 24/24 | 321.43 | 35.33 | 4,719.53 | 0.8727 |
| Ouro | DP2 + Chunked | 2,048 | 24/24 | 515.73 | 43.34 | 5,836.35 | 0.8503 |
| Huginn | PD_G1 | 256 | 8/24 | 44,262.67 | 103.25 | 57,374.07 | 0.0415 |
| Huginn | PD_G1 tuned | 2,048 | 11/24 | 36,486.53 | 104.32 | 49,726.97 | 0.0589 |
| Huginn | DP2 | 4,096 | 20/24 | 10,343.59 | 115.99 | 24,058.71 | 0.1268 |
| Huginn | DP2 + Chunked | 2,048 | 20/24 | 11,030.25 | 114.22 | 24,665.92 | 0.1263 |

These single-start controls do not establish sustainable capacity or method-level confidence intervals.
Their different chunk configurations remain explicit.
The study does not attribute a generic chunk or replica benefit to Dynamic A.

## Average TPOT and tail ITL can move in opposite directions

Ouro G2 C2048 passes the primary joint SLO in 24/24 requests.
Its population ITL p99 is nevertheless 292.36 ms, compared with 75.54 ms at C256.
Only 7/24 tuned requests keep every observed ITL within 100 ms.
The frozen primary criterion uses request-average TPOT.
It is not an every-token guarantee.
The report retains this negative tail result instead of replacing the criterion after measurement.

## Short-output diagnostic

The secondary 3,500/32 condition completed 64 requests per model.
Ouro passed 3/64 joint SLOs, with TPOT p95 130.01 milliseconds.
Huginn passed 15/64, with TPOT p95 120.08 milliseconds.
D-credit-refusal observation fractions were 49.43% and 94.56%, respectively.
These are sampled admission-reason fractions, not time shares.
The diagnostic does not establish safe D spare or replace the primary output grid.

Exact 4K/100 and Ouro 8K conditions were not measured.
Huginn's Prompt plus output cannot exceed 4,096 tokens.
The lack of these controls limits generalization beyond this engine configuration and grid.

## Native dense AR diagnostic

Qwen3-1.7B completed 512 native-vLLM NIXL requests.
Every G2 request reports 3,072 cached Prompt tokens on D.
Actual completed transfer telemetry and remote descriptors substantiate the native handoff.
Process-lifetime transfer counters include warm-up.
Empty P counters mean unavailable telemetry, not zero traffic.

Several Qwen cells also have Decode-SLO pressure despite short total execution.
The Qwen scatter plot combines different input/output cells.
It is not a same-workload rate sweep.
Different kernels, engines, model sizes, and KV geometries prevent an architecture-only inference.
The study does not reproduce DuetServe's Qwen3-8B 8,000/200 setting.

## Final interpretation

Ouro's C256 G1 screen is a finite P-pressure candidate.
Its same-arrival C2048 control passes 23/24 primary SLOs.
The candidate therefore lacks verified partition-specific headroom beyond generic tuning.
Enlarged-pool Ouro runs lose Decode SLO or admission reserve.
All 22 Huginn classified conditions satisfy the declared D-KV-admission criterion.
Neither result proves that Prefill-heavy Looped Serving is impossible.

The decision is `NO_P_HEADROOM_ON_TESTED_GRID`, with architectural causality `INCONCLUSIVE`.
Phase C remains `NOT_RUN_GATE_NOT_MET`.
This is not a measured `NO_GO_PERFORMANCE` result for a new online Dynamic A policy.
