# Loop width versus Token chunk after GEMM tuning — final summary

## Outcome

Huginn 2048/4: `SERVING_LOOP_BENEFIT`, conditional on this tested workload.
Loop28 passes the tested p99/TPOT gates. Its maximum ITL still increases.

Loop31 provides the smaller TTFT gain with a more stable observed maximum ITL.
The seed tables retain both effects. A p99 pass is not a guarantee for every Decode interval.

Huginn 2048/16: the tested Loop28 reduces TTFT but fails Native-relative Decode protection.
Ouro 4096/32: both tested split arms failed the N64 Decode screen. Unsafe split arms did not receive N128 repetition.

A practical width advantage survives the tested GEMM search in Huginn.
This is not universal Loop dominance or a globally optimal-kernel proof.

It is also not proof of exclusive recurrent-model novelty.
An equivalent logical-layer frontier remains an equivalent mechanism.

## Contract and experiment scope

The campaign preserves the original BF16 checkpoints and fixed R32/R4.
It preserves Attention, full recurrent work, numerical normalization, Native handoff, ACK, and source lifetime.

It does not use quantization, early exit, approximate Attention, or changed SLOs.
All compared serving arms use the same frozen prompt-projection policy.

Small Decode projections keep the qualified original implementation.
Huginn uses GPUs 4/5. Ouro uses GPUs 0/1. Timing uses one isolated pair.

The serving methods are static splits under online Poisson arrivals.
This campaign does not implement a new adaptive online scheduler.

The preserved pool contains 128 distinct controlled ShareGPT/LongBench prompt IDs.
N128 uses those 128 IDs. Arrival seeds and process starts are independent.

The prompt texts are not a new dataset relative to earlier campaigns.
Prefix caching remains disabled. Output uses a fixed cap and ignores EOS.

Huginn uses QPS 1.971. Ouro uses QPS 6.356.
Output4 is a separate discovery workload. It does not replace the Output16 anchor.

The frozen TTFT/TPOT/ITL thresholds remain unchanged.

ITL uses the frozen output-delivery contract.
Coalesced delivery intervals do not reveal individual GPU token-generation times.

The seed-level evidence records coalesced events and threshold exceedances.

Campaign start: 2026-11-11T07:38:55+00:00.
GPU work closed: 2026-11-11T12:22:24.813052+00:00.

Elapsed time to GPU closure: 4.725 hours.

| Model | Pinned revision | P slots | D slots | P KV blocks | D KV blocks | Graph-shape budget | Initial warm-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | bb6621b65e90b6a4b9b29ef88dc83866d450470c | 2 | 8 | 16384 | 24832 | 64 | 3 requests, 2 outputs |
| ouro | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | 2 | 8 | 16384 | 16384 | 64 | 2 requests, 2 outputs |

The common policy reserves the primary graph shape before measurement.
Warm-up requests do not enter serving statistics.

The same model-specific KV pool and graph budget apply to every compared serving arm.
The runtime also primes configured recurrent graph geometries. Raw configurations retain those details.

| Model | Arm label | P frontier | D prompt work |
| --- | --- | --- | --- |
| Huginn R32, input 2048 | zero | All 32 Loops | No prompt continuation |
| Huginn R32, input 2048 | loop | Loop31 | Full prompt × 1 Loop |
| Huginn R32, input 2048 | loop28 | Loop28 | Full prompt × 4 Loops |
| Huginn R32, input 2048 | token / token_group32 | Prefix 1984, all depths | Suffix 64 × 32 Loops |
| Huginn R32, input 2048 | token1792 | Prefix 1792, all depths | Suffix 256 × 32 Loops |
| Ouro R4, input 4096 | loop | Loop3 | Full prompt × 1 Loop |
| Ouro R4, input 4096 | token | Prefix 3072, all depths | Suffix 1024 × 4 Loops |

Token suffixes of 64 and 256 are the measured Huginn serving candidates.
This finite serving search does not prove the globally best Token cut.

The projection experiment covers more row geometries than the serving experiment.

## Kernel search and state qualification

The search contains 441 projection cells across 12 independent GPU processes.
Each geometry has a 64 MiB workspace cap and up to 32 algorithm configurations.

The candidates include library heuristics, valid no-split-K variants, and canonical-shape algorithm reuse.
Canonical reuse executes the real unpadded row count.

Each map selects kernels from start 1 only. Starts 2 and 3 do not reselect kernels.
Additional search covers actual batched P shapes, including Huginn M3968 and Ouro M6144/M8192.

Existing geometry choices remain unchanged in the v2 map.
The loaded library is cuBLASLt 120901. Torch is 2.11.0+cu129.

The bridge uses CUDA 12.8 headers. The evidence retains loaded libraries and algorithm attributes.
cuBLASLt supports reusable geometry-specific choices. See the [NVIDIA API documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic).

The recorded state qualification contains 720 test requests.
All recorded N8/N32 qualification gates: PASS.

Those counts include repeated prompts across methods and kernel policies.
The checks compare exact output IDs, every recorded valid KV-plane hash, and prompt Hidden State.

They also check full recurrent work, Native wire identity, ACK order, credit retirement, and normal worker exits.
This is tested internal Native parity. It does not prove equality for every possible input.

[Qualification and source inventory](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/campaign_inventory.json)

## Width after kernel tuning

The table uses the fixed start-1 policy across three independent starts.
It sums isolated projection timings with structural layer multiplicities.

It is not an executed recurrent core or serving speedup.

| Model | Token batch geometry | Token / Loop dense-time sum | Range across starts |
| --- | --- | --- | --- |
| huginn | 1 | 4.155 | 4.076–4.239 |
| huginn | 2 | 2.207 | 2.164–2.253 |
| huginn | 4 | 1.200 | 1.174–1.227 |
| huginn | 8 | 1.017 | 0.984–1.054 |
| ouro | 1 | 1.126 | 1.124–1.129 |
| ouro | 2 | 1.028 | 1.025–1.030 |
| ouro | 4 | 1.000 | 1.000–1.000 |

Compatible Token batch shapes nearly remove the isolated dense gap.
That experiment does not provide ready requests, Attention state, or free KV credit.

A single full-prompt Loop obtains its width from one request.
The 64-token suffix requires 32 dependent passes in Huginn.

A different GEMM cannot make later nonlinear recurrent inputs ready in advance.
Batching or a different fused execution design can change the frontier.

[Fixed-policy projection confirmation](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/fixed_policy_confirmation.json)

## Actual legal recurrent core

These controls measure real Native-path continuation with valid Attention and KV state.
The core intervals exclude separate Prelude and prompt Coda calls.

They include graph output clones and possible host submission gaps.
They are CUDA Event enclosures, not exclusive kernel unions.

| Control | Core sum p50 ms | Full D prompt sum p50 ms | Output/work/lifetime |
| --- | --- | --- | --- |
| loop_original | 17.460 | 27.330 | PASS |
| loop_tuned | 15.996 | 26.780 | PASS |
| token_original | 72.988 | 78.005 | PASS |
| token_tuned | 69.440 | 74.288 | PASS |
| token_group32_original | 72.935 | 77.912 | PASS |
| token_group32_tuned | 69.435 | 74.166 | PASS |

The primary geometry is 2048 tokens × one Loop versus 64 tokens × 32 Loops.
Both contain 2048 recurrent token rows.

Their recurrent dense FLOPs match at 6,706,062,950,400 operations.
Their Attention and boundary work do not match.

Per core layer, causal-pair counts are 2,098,176 for Loop and 4,129,792 for Token.
D performs 2047 prompt Coda tokens for Loop and 63 for Token.

Token also performs 64 Prelude tokens on D.
Do not infer equal total FLOPs or equal P saving from equal recurrent rows.

[Structural accounting](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/structural_accounting.json)

The separate cold-matrix roofline is a modeled explanation, not measured HBM traffic.
It assumes one HBM read per GEMM matrix and one output write.

It uses theoretical H100 SXM rates, not measured sustained rates.
Cache reuse, fusion, Attention, and submission costs remain outside that model.

It does not prove an optimal-kernel lower bound.
See [NVIDIA GEMM fundamentals](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

[Conditional roofline](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/conditional_cold_matrix_roofline.json)

## Realized v2 quantum and serial redistribution

P saving is a paired difference against the same-policy serial Native control.
Do not add percentile components to reconstruct E2E latency.

A model-submission enclosure is not one non-preemptive GPU kernel.

| Arm | P saved enclosure p50 ms | D prompt sum p50 ms | Largest D submission ms | D submissions/request | Wire GB/request |
| --- | --- | --- | --- | --- | --- |
| zero | 0.000 | NOT_MEASURED | 0.000 | 0.000 | 5.709 |
| loop | 24.431 | 25.906 | 27.162 | 1.000 | 5.493 |
| token | 7.164 | 76.894 | 5.743 | 32.000 | 5.532 |
| token_group32 | 7.415 | 74.171 | 74.931 | 1.000 | 5.532 |
| token1792 | 52.539 | 98.088 | 13.207 | 32.000 | 4.999 |

Native also transfers KV. The complete split payload is not all incremental transfer cost.
Serial handoff fields retain observed host boundaries. They are not isolated DMA durations.

[All serial costs and per-request values](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/serial_path_costs.json)

## N64 discovery: retain every arm

All fifteen N64 cells passed exact output, full work, and lifetime checks.
Every measured packed-prefill graph selection hit an existing graph.

Measured graph captures and eager fallbacks were zero in those cells.
The table shows marker-free coordinator delivery latency.

Clean online Service TTFT is NOT_MEASURED. Host receipts do not replace it.

| Model | Output | Arm | TTFT p95 s | E2E p95 s | ITL p99 ms | TPOT p95 ms | ITL misses | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 16 | loop28 | 6.330 | 7.688 | 153.847 | 99.593 | 0 | PASS |
| huginn | 4 | loop28 | 6.426 | 6.648 | 77.388 | 75.377 | 0 | PASS |
| huginn | 16 | loop | 9.278 | 10.598 | 105.329 | 89.214 | 0 | PASS |
| huginn | 4 | loop | 9.228 | 9.451 | 76.032 | 74.548 | 0 | PASS |
| huginn | 16 | token1792 | 8.606 | 9.968 | 136.250 | 97.333 | 0 | PASS |
| huginn | 4 | token1792 | 7.563 | 7.785 | 82.523 | 75.548 | 0 | PASS |
| huginn | 16 | token | 10.503 | 11.778 | 121.592 | 89.021 | 0 | PASS |
| huginn | 4 | token | 10.442 | 10.665 | 76.197 | 74.614 | 0 | PASS |
| huginn | 16 | token_group32 | 10.486 | 11.740 | 115.470 | 88.840 | 0 | PASS |
| huginn | 4 | token_group32 | 10.460 | 10.685 | 76.695 | 75.212 | 0 | PASS |
| huginn | 16 | zero | 10.419 | 11.620 | 89.826 | 81.602 | 0 | PASS |
| huginn | 4 | zero | 10.615 | 10.838 | 76.460 | 74.695 | 0 | PASS |
| ouro | 32 | loop | 4.582 | 5.792 | 88.210 | 49.266 | 405 | FAIL |
| ouro | 32 | token | 5.240 | 6.441 | 81.103 | 50.681 | 488 | FAIL |
| ouro | 32 | zero | 2.929 | 3.842 | 42.123 | 38.789 | 0 | PASS |

## Held-out N128 confirmation

The campaign defined the selection rule before N64 results.
The campaign added two Output4 challengers before any N128 result.

They test equal recurrent rows and the strongest absolute-SLO Token throughput candidate.
All added challengers remain in the report, regardless of outcome.

Arrival seeds are 2026111111, 2026111112, and 2026111113.
Run order rotates between starts. The independent start is the bootstrap resampling unit.

The small number of starts limits population-level uncertainty claims.
These intervals describe the observed trace/start pairs. They do not establish coverage for untested workloads.

The following ratios use Candidate / matched tuned Native-path Zero.
Intervals are paired percentile-bootstrap 95% intervals. They are not simultaneous multi-arm confidence intervals.

| Output | Arm | Starts | TTFT ratio [CI] | E2E ratio [CI] | ITL ratio [CI] | TPOT ratio [CI] | Relative tail gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | loop | 3 | 0.840 [0.802, 0.886] | 0.844 [0.807, 0.888] | 0.979 [0.955, 1.000] | 0.987 [0.975, 0.997] | PASS |
| 4 | loop28 | 3 | 0.641 [0.615, 0.673] | 0.650 [0.625, 0.680] | 0.995 [0.974, 1.016] | 0.987 [0.975, 1.001] | PASS |
| 4 | token | 3 | 1.072 [0.978, 1.206] | 1.070 [0.978, 1.200] | 0.985 [0.966, 1.004] | 0.987 [0.979, 0.998] | PASS |
| 4 | token1792 | 3 | 0.816 [0.699, 0.953] | 0.819 [0.704, 0.954] | 1.059 [1.038, 1.077] | 0.992 [0.984, 0.999] | FAIL |
| 16 | loop28 | 3 | 0.592 [0.570, 0.619] | 0.662 [0.636, 0.689] | 1.688 [1.661, 1.709] | 1.219 [1.215, 1.223] | FAIL |
| 16 | token1792 | 3 | 0.712 [0.643, 0.810] | 0.766 [0.699, 0.851] | 1.491 [1.450, 1.524] | 1.195 [1.162, 1.227] | FAIL |

The next table retains finite-cohort rates and maximum delivery intervals.
A ratio above one means higher goodput or completion rate. It means worse maximum ITL.

| Output | Arm | Goodput ratio [CI] | Completed QPS ratio [CI] | Maximum ITL ratio [CI] | Candidate ITL misses | Native ITL misses |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | loop | 1.598 [1.487, 1.755] | 1.029 [1.022, 1.035] | 0.972 [0.931, 1.006] | 0 | 0 |
| 4 | loop28 | 3.387 [3.168, 3.619] | 1.069 [1.046, 1.089] | 1.295 [1.083, 1.595] | 0 | 0 |
| 4 | token | 0.767 [0.483, 1.139] | 0.994 [0.975, 1.010] | 1.125 [0.941, 1.405] | 0 | 0 |
| 4 | token1792 | 1.445 [0.817, 2.371] | 1.035 [1.009, 1.065] | 1.478 [1.399, 1.580] | 0 | 0 |
| 16 | loop28 | 5.672 [4.269, 7.634] | 1.087 [1.072, 1.102] | 1.606 [1.568, 1.648] | 0 | 0 |
| 16 | token1792 | 2.684 [2.504, 2.931] | 1.064 [1.045, 1.080] | 1.435 [1.386, 1.504] | 0 | 0 |

Seed-level confirmation follows. Each row is one independent server start.

| Model | Output | Seed | Arm | TTFT p95 s | E2E p95 s | ITL p99 ms | TPOT p95 ms | Maximum ITL ms | Goodput req/s | Completed req/s | Offloaded requests | D recurrent rows | Exactness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 16 | 2026111111 | loop28 | 4.762 | 6.246 | 154.032 | 99.751 | 160.652 | 0.944 | 1.830 | 128 | 1048576 | PASS |
| huginn | 16 | 2026111112 | loop28 | 6.773 | 8.254 | 153.439 | 99.500 | 155.213 | 0.909 | 1.908 | 128 | 1048576 | PASS |
| huginn | 16 | 2026111113 | loop28 | 6.242 | 7.764 | 155.645 | 102.437 | 157.706 | 0.558 | 1.830 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111111 | loop28 | 4.633 | 4.856 | 78.135 | 75.267 | 80.796 | 1.088 | 1.857 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111112 | loop28 | 7.060 | 7.285 | 79.827 | 75.223 | 143.698 | 0.859 | 1.928 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111113 | loop28 | 6.391 | 6.614 | 76.105 | 74.599 | 94.510 | 0.523 | 1.859 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111111 | loop | 6.031 | 6.254 | 75.762 | 74.325 | 76.595 | 0.541 | 1.775 | 128 | 262144 | PASS |
| huginn | 4 | 2026111112 | loop | 9.265 | 9.488 | 75.595 | 74.347 | 76.030 | 0.415 | 1.830 | 128 | 262144 | PASS |
| huginn | 4 | 2026111113 | loop | 8.422 | 8.649 | 78.949 | 76.391 | 79.852 | 0.229 | 1.829 | 128 | 262144 | PASS |
| huginn | 16 | 2026111111 | token1792 | 5.393 | 6.885 | 137.806 | 101.196 | 142.815 | 0.520 | 1.800 | 128 | 1048576 | PASS |
| huginn | 16 | 2026111112 | token1792 | 7.620 | 9.071 | 136.496 | 97.542 | 143.496 | 0.584 | 1.868 | 128 | 1048576 | PASS |
| huginn | 16 | 2026111113 | token1792 | 8.491 | 9.891 | 135.021 | 97.025 | 136.964 | 0.167 | 1.780 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111111 | token1792 | 7.532 | 7.755 | 78.834 | 74.860 | 124.491 | 0.189 | 1.727 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111112 | token1792 | 7.493 | 7.716 | 85.285 | 76.022 | 118.888 | 0.699 | 1.904 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111113 | token1792 | 7.623 | 7.846 | 85.022 | 75.195 | 110.238 | 0.287 | 1.838 | 128 | 1048576 | PASS |
| huginn | 4 | 2026111111 | token | 9.638 | 9.862 | 77.204 | 74.989 | 117.977 | 0.117 | 1.660 | 128 | 262144 | PASS |
| huginn | 4 | 2026111112 | token | 10.911 | 11.134 | 76.824 | 75.151 | 77.218 | 0.264 | 1.781 | 128 | 262144 | PASS |
| huginn | 4 | 2026111113 | token | 9.285 | 9.511 | 77.563 | 74.936 | 78.941 | 0.184 | 1.813 | 128 | 262144 | PASS |
| huginn | 16 | 2026111111 | zero | 8.186 | 9.398 | 90.424 | 82.002 | 102.949 | 0.171 | 1.683 | 0 | 0 | PASS |
| huginn | 16 | 2026111112 | zero | 11.939 | 13.137 | 89.692 | 81.230 | 93.351 | 0.229 | 1.725 | 0 | 0 | PASS |
| huginn | 16 | 2026111113 | zero | 9.911 | 11.155 | 94.382 | 84.302 | 98.874 | 0.067 | 1.714 | 0 | 0 | PASS |
| huginn | 4 | 2026111111 | zero | 7.583 | 7.806 | 76.503 | 74.740 | 76.624 | 0.295 | 1.718 | 0 | 0 | PASS |
| huginn | 4 | 2026111112 | zero | 11.205 | 11.434 | 79.944 | 76.652 | 83.326 | 0.276 | 1.768 | 0 | 0 | PASS |
| huginn | 4 | 2026111113 | zero | 9.331 | 9.560 | 78.725 | 76.600 | 79.209 | 0.154 | 1.794 | 0 | 0 | PASS |
| ouro | 32 | 2026111111 | zero | 1.672 | 2.860 | 42.062 | 38.852 | 219.213 | 3.532 | 5.581 | 0 | 0 | PASS |

These heavy finite cohorts do not represent an all-request SLO-compliant steady state.
The following table retains arrival rates, token rates, and attainment denominators.

| Model | Output | Seed | Arm | Offered QPS | Scheduled QPS | Submitted QPS | Completed QPS | Input token/s | Output token/s | TTFT attainment % of 128 | TPOT attainment % of 128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 16 | 2026111111 | loop28 | 1.971 | 1.927 | 1.927 | 1.830 | 3747.593 | 29.278 | 51.562 | 100.000 |
| huginn | 16 | 2026111112 | loop28 | 1.971 | 1.972 | 1.972 | 1.908 | 3906.964 | 30.523 | 47.656 | 100.000 |
| huginn | 16 | 2026111113 | loop28 | 1.971 | 1.959 | 1.959 | 1.830 | 3748.101 | 29.282 | 30.469 | 100.000 |
| huginn | 4 | 2026111111 | loop28 | 1.971 | 1.927 | 1.927 | 1.857 | 3803.168 | 7.428 | 58.594 | 100.000 |
| huginn | 4 | 2026111112 | loop28 | 1.971 | 1.972 | 1.972 | 1.928 | 3948.954 | 7.713 | 44.531 | 100.000 |
| huginn | 4 | 2026111113 | loop28 | 1.971 | 1.959 | 1.959 | 1.859 | 3808.136 | 7.438 | 28.125 | 100.000 |
| huginn | 4 | 2026111111 | loop | 1.971 | 1.927 | 1.927 | 1.775 | 3634.485 | 7.099 | 30.469 | 100.000 |
| huginn | 4 | 2026111112 | loop | 1.971 | 1.972 | 1.972 | 1.830 | 3748.819 | 7.322 | 22.656 | 100.000 |
| huginn | 4 | 2026111113 | loop | 1.971 | 1.959 | 1.959 | 1.829 | 3746.181 | 7.317 | 12.500 | 100.000 |
| huginn | 16 | 2026111111 | token1792 | 1.971 | 1.927 | 1.927 | 1.800 | 3687.344 | 28.807 | 28.906 | 100.000 |
| huginn | 16 | 2026111112 | token1792 | 1.971 | 1.972 | 1.972 | 1.868 | 3825.089 | 29.884 | 31.250 | 100.000 |
| huginn | 16 | 2026111113 | token1792 | 1.971 | 1.959 | 1.959 | 1.780 | 3644.824 | 28.475 | 9.375 | 100.000 |
| huginn | 4 | 2026111111 | token1792 | 1.971 | 1.927 | 1.927 | 1.727 | 3536.394 | 6.907 | 10.938 | 100.000 |
| huginn | 4 | 2026111112 | token1792 | 1.971 | 1.972 | 1.972 | 1.904 | 3899.088 | 7.615 | 36.719 | 100.000 |
| huginn | 4 | 2026111113 | token1792 | 1.971 | 1.959 | 1.959 | 1.838 | 3763.928 | 7.351 | 15.625 | 100.000 |
| huginn | 4 | 2026111111 | token | 1.971 | 1.927 | 1.927 | 1.660 | 3398.719 | 6.638 | 7.031 | 100.000 |
| huginn | 4 | 2026111112 | token | 1.971 | 1.972 | 1.972 | 1.781 | 3647.617 | 7.124 | 14.844 | 100.000 |
| huginn | 4 | 2026111113 | token | 1.971 | 1.959 | 1.959 | 1.813 | 3713.643 | 7.253 | 10.156 | 100.000 |
| huginn | 16 | 2026111111 | zero | 1.971 | 1.927 | 1.927 | 1.683 | 3446.575 | 26.926 | 10.156 | 100.000 |
| huginn | 16 | 2026111112 | zero | 1.971 | 1.972 | 1.972 | 1.725 | 3532.030 | 27.594 | 13.281 | 100.000 |
| huginn | 16 | 2026111113 | zero | 1.971 | 1.959 | 1.959 | 1.714 | 3510.773 | 27.428 | 3.906 | 100.000 |
| huginn | 4 | 2026111111 | zero | 1.971 | 1.927 | 1.927 | 1.718 | 3519.371 | 6.874 | 17.188 | 100.000 |
| huginn | 4 | 2026111112 | zero | 1.971 | 1.972 | 1.972 | 1.768 | 3620.239 | 7.071 | 15.625 | 100.000 |
| huginn | 4 | 2026111113 | zero | 1.971 | 1.959 | 1.959 | 1.794 | 3673.877 | 7.176 | 8.594 | 100.000 |
| ouro | 32 | 2026111111 | zero | 6.356 | 6.215 | 6.215 | 5.581 | 22861.742 | 178.607 | 63.281 | 100.000 |

Direct Loop / Token comparisons follow. Recurrent-row matching is not full-FLOP matching.

| Output4 comparison | TTFT ratio [CI] | E2E ratio [CI] | ITL ratio [CI] | TPOT ratio [CI] |
| --- | --- | --- | --- | --- |
| loop / token | 0.784 [0.669, 0.894] | 0.789 [0.677, 0.897] | 0.994 [0.982, 1.010] | 1.000 [0.990, 1.013] |
| loop28 / token1792 | 0.786 [0.659, 0.919] | 0.793 [0.669, 0.921] | 0.940 [0.904, 0.979] | 0.996 [0.990, 1.003] |
| loop28 / token | 0.598 [0.513, 0.679] | 0.607 [0.524, 0.686] | 1.011 [0.988, 1.033] | 1.000 [0.997, 1.003] |

[Seed-level serving metrics](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/serving_results.json)
[Native-relative paired statistics](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/paired_start_confirmation.json)

[Direct method statistics](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/paired_method_comparisons.json)

All rates describe finite-cohort completion and drain.
They do not establish sustainable capacity.

Queueing is included in Request TTFT.
No component-percentile sum substitutes for a request-level percentile.

## Completion of the frozen confirmation plan

The frozen plan contains 27 cells. Completed cells: 25.
A missing cell is NOT_RUN or FAILED. It is not a zero-valued performance result.

- `serve_v2_ouro_zero_O32_N128_a2026111112`: NOT_RUN_TIME_BUDGET.
- `serve_v2_ouro_zero_O32_N128_a2026111113`: NOT_RUN_AFTER_BUDGET_CUTOFF.

All 24 Huginn confirmation cells completed. One of three secondary Ouro Native controls completed.
The new-job cutoff prevented the remaining two controls. This is not a measured model failure.

The inherited budget field says 11 hours. Its configured timestamps enforce the six-hour operational bound.
[Budget interpretation](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/state/budget_interpretation.json)

## Retained failures and measurement limits

The first core observer missed the BulkWrittenGraphs override.
Its absent recurrent records are NOT_MEASURED, not zero time.

Fresh controls observe the actual graph instance.
One Ouro observer startup assumed prepared Prelude/Coda methods that Ouro does not expose.

That failed before measurement. Its log remains intact. Fresh s2 controls passed.
Initial nominal Token cap-1 controls did not cap Prelude.

They actually grouped all 32 Loops. Their total-throughput costs remain valid.
They do not provide small-quantum Decode-safety evidence.

V2 caps Prelude and recurrent phases. Actual submission audits are mandatory.
The report does not hide those configuration aliases.

No intrusive CUPTI or Nsight collector supplies primary serving latency in this campaign.
Physical kernel idle, HBM traffic, Tensor Core occupancy, and exclusive kernel time are NOT_MEASURED.

Exact percentages of full-core cost caused by memory traffic versus submission overhead are NOT_IDENTIFIED.
The best tested GEMM is not the globally optimal GEMM.

The single Ouro N128 Native control records six ITL threshold exceedances.
Its maximum ITL is 219.213 ms, although its ITL p99 passes.

Two planned Native repetitions did not run. This control cannot establish Native maximum-tail stability.
[Failure, coverage, and alias inventory](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/analysis/campaign_inventory.json)

## Research decision and follow-up boundary

The wide-token hypothesis has practical evidence after finite kernel tuning.
Small Token suffixes preserve all recurrent dependencies and matrix invocations.

Larger Token suffixes and compatible batches can reduce their efficiency disadvantage.
Shorter Token quanta can still protect Decode better in a constrained execution budget.

Actual serving must determine the useful region. Dense throughput alone cannot decide it.
No result establishes universal Loop superiority or exclusive Loop-specific novelty.

The follow-up requires minimum Output16 and qualified physical-idle measurements.
Current Output4 findings cannot satisfy that follow-up gate.

Its ten-hour budget starts only after this campaign is reviewed, published, and resource-verified.

## Evidence and preservation

Publication parent: `4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd`.
Publication branch: `flashvep/loop-width-gemm-validation-20261111`.

Every run retains configuration hashes, source snapshots, model revisions, GPU mapping, commands, and seeds.
The lossless evidence bundle contains raw logs and repeated source snapshots.

The publication uses a private Git index. It does not replace the active checkout.
The final verification records the preserved staged index and external-runtime diff.

[Figures and evidence classification](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/figures/figure_manifest.json)
[Final source and preservation checks](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/state/final_source_verification.json)

[Burner process and generation verification](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/state/final_burner_verification.json)
[Evidence bundle manifest](ouro_huginn_loop_width_vs_token_chunk_gemm_validation/evidence_bundle/manifest.json)
