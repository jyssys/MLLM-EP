# RCC-PD fixed-cut co-batching results

## Result scope

Evidence class: `MEASURED` for the quiet tables below.
The comparison uses two physical GPUs, 0 and 1.
Each arm uses eight frozen 128-token ShareGPT prefixes and an output cap of 32.
EOS remains enabled.
Each arm runs after a new server start, with a separate warmup.
Each model has three independent starts per arm.
Arm order reverses in the second repetition.

These are small mechanism cohorts, not representative long-generation serving tests.
They do not measure sustainable capacity or SLO goodput.
They do not establish benchmark answer accuracy.
Ouro generates 241 output tokens per start.
Huginn generates 256 output tokens per start.
The same model produces exactly the same outputs across all three arms and starts.

The numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
Every arm includes its qualified numerical repair.
No B frontier-streaming code runs in this comparison.
All work uses fixed R4 or R32.

## Main mechanism table

Latency values are arithmetic means over three independent starts.
Each per-start request latency is a mean over eight requests.
Throughput uses cohort completion, including its drain interval.
It is not a sustainable open-loop service rate.
Speedup uses the geometric mean of paired start ratios.
Its denominator is the same model's two-GPU PD_G1 cohort makespan.

| Model | Arm | Mean makespan, ms | Mean TTFT, ms | Mean TPOT, ms | Mean E2E, ms | Requests/s | Output tokens/s | Versus PD_G1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ouro R4 | PD_G1 | 2072.686 | 358.189 | 50.674 | 1838.445 | 3.860 | 116.279 | 1.00× |
| Ouro R4 | A_sep, cut 2 | 2301.229 | 394.393 | 55.255 | 2009.720 | 3.501 | 105.457 | 0.90× |
| Ouro R4 | A_native_mixed, cut 2 | 1892.877 | 349.158 | 45.575 | 1677.152 | 4.240 | 127.722 | 1.10× |
| Huginn R32 | PD_G1 | 5394.360 | 411.198 | 138.048 | 4690.699 | 1.486 | 47.552 | 1.00× |
| Huginn R32 | A_sep, cut 16 | 5406.167 | 404.652 | 141.550 | 4792.717 | 1.480 | 47.367 | 1.00× |
| Huginn R32 | A_native_mixed, cut 16 | 5465.167 | 394.524 | 137.532 | 4658.020 | 1.467 | 46.934 | 0.99× |

Huginn A_sep's unrounded speedup is 0.996965×.
The displayed 1.00× is rounding, not evidence of a gain.

## Paired uncertainty

The bootstrap resamples independent start pairs, not output tokens.
With only three pairs, the intervals are coarse.
They do not generalize to different prompts, lengths, arrivals, or SLO thresholds.

| Model | Reference → candidate | Makespan speedup | Paired 95% bootstrap interval |
|---|---|---:|---:|
| Ouro | A_sep → A_native_mixed | 1.21× | 1.09×–1.39× |
| Ouro | PD_G1 → A_sep | 0.90× | 0.84×–0.96× |
| Ouro | PD_G1 → A_native_mixed | 1.10× | 1.04×–1.17× |
| Huginn | A_sep → A_native_mixed | 0.99× | 0.96×–1.02× |
| Huginn | PD_G1 → A_sep | 1.00× | 0.94×–1.05× |
| Huginn | PD_G1 → A_native_mixed | 0.99× | 0.91×–1.05× |

The Ouro result supports a useful existing-Mixed operating point.
The Huginn result does not show a repeatable makespan gain.
Neither result establishes performance beyond Native Mixed.
The candidate is Native Mixed itself, applied to the existing A import path.

## Latency distributions

Each cell contains `P50 / P95 / P99`, in milliseconds.
Values are means of the three per-start quantiles.
TTFT, TPOT, and E2E use request distributions.
ITL uses individual output delivery intervals within a start.
The harness does not use HTTP or network token streaming.
It records delivery after `engine.step()` returns.

| Model | Arm | TTFT | TPOT | ITL | E2E |
|---|---|---|---|---|---|
| Ouro | PD_G1 | 375.958 / 459.204 / 469.641 | 51.914 / 57.750 / 59.477 | 29.630 / 256.036 / 273.337 | 1920.646 / 2045.632 / 2056.485 |
| Ouro | A_sep | 405.307 / 707.512 / 727.109 | 54.823 / 66.046 / 66.052 | 29.187 / 248.398 / 562.429 | 2094.900 / 2220.280 / 2231.314 |
| Ouro | A_native_mixed | 366.071 / 448.836 / 456.340 | 45.331 / 52.655 / 54.873 | 28.883 / 189.625 / 366.889 | 1749.737 / 1864.801 / 1872.477 |
| Huginn | PD_G1 | 361.324 / 638.024 / 668.106 | 137.491 / 145.753 / 146.020 | 121.669 / 279.024 / 416.658 | 4749.042 / 4926.397 / 4953.746 |
| Huginn | A_sep | 417.617 / 523.468 / 529.272 | 140.641 / 156.398 / 157.362 | 124.149 / 197.298 / 623.296 | 4814.572 / 5134.083 / 5149.811 |
| Huginn | A_native_mixed | 376.080 / 536.098 / 553.607 | 136.361 / 145.204 / 146.392 | 117.742 / 311.309 / 450.844 | 4693.450 / 4801.862 / 4810.171 |

Ouro Mixed improves mean completion but has higher ITL P99 than PD_G1 in this probe.
Huginn Mixed has higher ITL P95 than A_sep.
These differences prevent an automatic SLO-safety claim.
No SLO threshold was selected after these results.

## Handoff and memory

Bytes and fragments cover the measured eight-request cohort only.
They exclude warmup.
Fragments are completed bounded transport units, not CUDA-copy or descriptor counts.
Peak memory is the mean of each start's maximum sampled physical-device HBM.
NVML samples every 0.2 seconds, so this is not an exact continuous peak.

| Model | Arm | Payload bytes | Fragments | Reserved KV per worker, GiB | Weight per worker, GiB | Sampled peak HBM, GiB |
|---|---|---:|---:|---:|---:|---:|
| Ouro | PD_G1 | 805339136 | 16 | 12.00000 | 2.67225 | 16.61572 |
| Ouro | A_sep | 406847488 | 8 | 12.00000 | 2.67225 | 16.71077 |
| Ouro | A_native_mixed | 406847488 | 8 | 12.00000 | 2.67225 | 16.68604 |
| Huginn | PD_G1 | 2854579200 | 48 | 30.93750 | 6.64029 | 40.48161 |
| Huginn | A_sep | 1449000960 | 24 | 30.93750 | 6.64029 | 40.47119 |
| Huginn | A_native_mixed | 1449000960 | 24 | 30.93750 | 6.64029 | 40.47119 |

A_sep and A_native_mixed move exactly the same payload.
Their comparison does not attribute A's transfer reduction to co-batching.
The total recurrent row-work also matches exactly.
Ouro executes 5028 loop-rows per measured cohort.
Huginn executes 40704 loop-rows per measured cohort.
These counts include full-depth prompt work and all post-prompt decode-input work.

## Physical fusion evidence

Evidence class: `MEASURED_DIAGNOSTIC` and `MEASURED_HEAVY_PROFILE`.
These probes use 256-token prefixes and common eager execution.
Their timings must not replace the quiet table.

Both models execute a batch with 1793 rows: one decode row plus seven 256-row continuations.
The rows belong to different requests and logical depths.
The selected Ouro example contains decode depth 0 and continuation depth 3.
The Huginn probe also records later 257-row batches with one decode and one continuation.
The raw descriptors retain each row range and request seed.

Ouro's shared Q/K/V/O projections receive `[1793, 2048]` matrices.
Its Gate/Up projections receive the same combined row count.
Down uses 256-row tiles under the qualified numerical repair.
The selected core contains 337 physical GEMM/GEMV kernels and 1745 total GPU kernels.

Huginn's shared projections also receive combined matrices.
Its adapter and Down projections retain their qualified 256-row tiles.
The selected core contains 52 physical GEMM kernels and 309 total GPU kernels.

The first trace includes diagnostic warmup work.
It proves physical execution, not clean serving latency.
`operator_fusion_evidence.json` links the execution descriptors, matrix operands, and actual kernel traces.
The profiler emits duplicate nested `aten::linear` annotations under the dispatch mode.
The physical counts above use GPU kernel events, not that duplicated annotation count.

This is genuine shared-core execution, not only a scheduler label.
It is also an existing Native Mixed execution path.
The study adds measurement hooks, not a new fused operator.

## What the comparison does not isolate

Native Mixed changes phase selection and packing together.
It can alter when decode advances and when graphs capture or replay.
The present study does not run a paired, same-operand CUDA microbenchmark of separate and fused full cores.
It therefore does not assign the entire makespan change to GEMM efficiency.

Large mixed batches exceed the common eight-row decode graph limit.
They use the native eager fallback.
Separate decode can use a graph while continuation runs eager.
The graph configuration is equal, but graph availability differs by batch shape.
The common-eager diagnostic controls this distinction only for operator proof.
It does not establish a production-optimal graph comparison.

Full GPU idle time, exclusive E2E kernel shares, and graph workspace bytes are `NOT_MEASURED`.
Selected-core kernel timings and serial host E2E shares are reported separately.
No sum of overlapping transfer and kernel durations is presented as E2E speedup.

## Decision

Physical cross-depth co-batching is feasible for both pinned models in these probes.
Ouro has a positive existing-Mixed cohort result.
Huginn does not show a repeatable makespan benefit.
The fixed-cut fused mechanism is `ALREADY_PRESENT` in A plus Native Mixed.
It is not a new RCC implementation or a demonstrated architecture advantage over strong DP_Mixed.
The later admission and scaling stages remain gated.
