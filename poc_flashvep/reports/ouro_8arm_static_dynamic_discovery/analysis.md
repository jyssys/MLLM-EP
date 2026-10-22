# Ouro eight-arm discovery: final analysis

## Decision

**Verdict: NO-GO for a demonstrated Loop-PP serving advantage over optimized native TP.**

All eight primary configurations completed actual free-running evaluation.
Each configuration used the same 400 questions: 300 GSM8K and 100 MATH.
Both native Static controls also completed those 400 questions.
The official CDB Dynamic control contains 114 questions, not 400.
The original 1,819-question full-suite evaluation is USER_REDUCED_NOT_RUN.

This decision concerns the tested implementations and workloads.
It does not prove that every optimized Loop-PP implementation must lose.
No optimized native-attention PP engine was implemented.
The current evidence does not justify a new architecture or adaptive scheduler.

## Model, devices, and policy

The checkpoint is base ByteDance/Ouro-1.4B, not Ouro-1.4B-Thinking.
Its revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
It has 24 shared physical layers and 96 depth-indexed KV slots at R4.
Weights remain BF16.
The complete checkpoint hash, tokenizer identity, libraries, UUIDs, and topology are in [environment.json](environment.json).

Single uses physical GPU4.
Four-rank jobs use `CUDA_VISIBLE_DEVICES=4,5,6,7`.
Logical ranks 0,1,2,3 map to physical GPUs4,5,6,7.
No other physical GPU runs this study.

Static always executes R4.
Dynamic uses the original CDB cumulative-hazard gate, q=0.5, minimum R2, maximum R4, and last_exited KV.
Prefill remains full R4.
Later decode loops actually stop after exit.
All custom Dynamic arms use the same synchronous no-refill successor rule.

The custom backend uses B1 stage calls.
The official CDB control packs up to four rows and batches coda differently.
These implementations share the selected policy, but they do not have full natural-generation behavioral parity.
See [dynamic_reference_validation.md](dynamic_reference_validation.md).

## Actual answer accuracy

| Parallelism | Static GSM8K | Static MATH | Dynamic GSM8K | Dynamic MATH |
|---|---|---|---|---|
| Single | 232/300, 77.333% | 61/100, 61.000% | 232/300, 77.333% | 59/100, 59.000% |
| Custom TP4 | 233/300, 77.667% | 61/100, 61.000% | 231/300, 77.000% | 61/100, 61.000% |
| Layer PP4 | 232/300, 77.333% | 61/100, 61.000% | 232/300, 77.333% | 59/100, 59.000% |
| Loop PP4 | 232/300, 77.333% | 61/100, 61.000% | 232/300, 77.333% | 59/100, 59.000% |

TP4 and Dynamic TP4 remain NUMERICALLY_UNQUALIFIED_EXPLORATORY.
Their measured answer scores do not pass the historical numerical gate.

Custom TP4 minus S4 is +0.333 pp on GSM8K, with paired 95% CI [-1.667, 2.333].
The MATH difference is 0.000 pp, with CI [-5.000, 6.000].
These samples do not show a uniform accuracy decrease.
They also do not establish noninferiority or population equivalence.

SD minus S4 is 0.000 pp on GSM8K, with CI [-2.667, 2.667].
The MATH difference is -2.000 pp, with CI [-10.000, 6.000].
Dynamic algorithm quality and placement quality are different comparisons.

S4, L4, and O4 match all 400 complete output-token sequences.
SD, LD, and OD match all 400 output-token and exit-depth sequences.
This is observed same-backend placement parity, not cross-backend equivalence.

The cap is 4096 generated tokens, with greedy feedback and normal EOS.
S4 reaches that cap on 16/100 MATH questions.
Its answer score is 61/100, while completed-correct is 58/100.
All arms expose truncation and parsing failures in [quality_results.md](quality_results.md).
The frozen cohort retains the earlier pilot and adds metadata-only random questions.
It is not a full-suite or official Thinking-model accuracy reproduction.

## Natural serving versus identical-work replay

| Arm | GPUs | Actual task service-cohort sum, s | Speedup vs same-mode custom Single | Clean C4 replay makespan, s | Clean C4 speedup |
|---|---|---|---|---|---|
| S4 | 1 | 6074.192 | 1.000x | 2.602 | 1.000x |
| T4 | 4 | 6219.233 | 0.977x | 2.794 | 0.931x |
| L4 | 4 | 3707.747 | 1.638x | 0.921 | 2.824x |
| O4 | 4 | 3311.532 | 1.834x | 1.159 | 2.245x |
| SD | 1 | 4373.494 | 1.000x | 2.171 | 1.000x |
| TD | 4 | 4267.465 | 1.025x | 2.306 | 0.941x |
| LD | 4 | 3326.628 | 1.315x | 0.944 | 2.298x |
| OD | 4 | 3777.943 | 1.158x | 1.465 | 1.482x |

The natural table sums 101 resident-model service windows.
It includes task observers and excludes request-bound graph preparation.
It is not whole-command makespan or model-cold E2E.
Static PP uses asynchronous FIFO, while custom Single/TP use waves.
The natural Static table is therefore not fully scheduler-matched.

The clean C4 example uses the same frozen prompt and output IDs for all eight arms.
It generates 32 replay tokens per request.
Dynamic replay additionally locks observed SD exit depths.
The gate and required cache repair still execute.
This is measured hardware replay, not benchmark accuracy or an online future-aware policy.

All 72 selected replay comparisons completed one warm-up and five measured repetitions.
They cover 40 Static comparisons and 32 observed-depth comparisons.
Ten of the legacy 21 shape cases were prospectively selected.
The study does not claim a full 21-case rerun.
[replay_summary.csv](replay_summary.csv) records ranges and paired intervals.

For the clean C1/P128/O32 Static case, Single takes 0.650 s.
Custom TP4 takes 0.690 s, Layer PP4 takes 0.886 s, and Loop PP4 takes 0.677 s.
No qualified custom PP B1 latency win appears there.

Loop PP is 1.120x faster than Layer PP on the larger Static natural cohort.
That advantage is not universal.
Layer PP wins the clean C4 example in the table.
The matched-natural eight-request control also preserves workload-dependent differences.
Scheduling, output-length heterogeneity, allocation shape, and stage granularity can all affect the ranking.
A single causal contribution was not isolated.

Graph preparation is substantial for short replay cohorts.
For the same clean C4 example, preparation-inclusive resident times are 3.530 s for L4 and 2.065 s for O4.
Those values reverse the prepared-service ordering.
The prototype recaptures request-bound graphs.
This is a runtime lifecycle effect, not a deployable steady-state architecture result.
[extension286_serving.csv](extension286_serving.csv) supplies the fully instrumented longer-task preparation control.

## Why Dynamic Loop PP loses to Dynamic Layer PP

OD takes 3777.943 s versus LD at 3326.628 s on identical natural outputs and exit traces.
LD is 1.136x faster.
The independent stage counters and copy calls identify two physical costs.

First, last_exited KV placement adds remote publication.
OD copies 37,006,934,016 logical KV bytes through 188,227 remote receive calls.
Its total explicit remote tensor payload is 39,656,782,256 bytes.
LD uses local owner repair and sends 14,014,785,968 remote tensor bytes.
These are observed tensor payloads, not network wire counters or bandwidth estimates.

Second, early exits distribute loop work unevenly.
OD layer-call CV is 0.5311, with max/mean 1.3822.
Its token-layer-work CV is 0.2828, with max/mean 1.2035.
LD has equal counted layer work across its four owners.
Work proxies are not GPU-time fractions.

The heavy trace directly records packing, remote import, local repair, and acknowledgement ranges.
An OD R3 acknowledgement median is 3.658 ms in the bounded short-allocation trace.
Its export median is 1.557 ms.
The matching LD acknowledgement median is 0.908 ms.
These are host protocol intervals under instrumentation.
Nested intervals overlap and cannot form an additive clean E2E decomposition.

Fewer hidden-state handoffs do not guarantee less total communication.
Static O4 sends one fifth of L4's hidden-state payload.
Dynamic OD introduces a much larger exit-KV payload.
This is a measured placement trade-off, not a new optimization method.

## Optimized native controls

| Native Static engine | GSM8K | MATH | Task service-cohort sum, s | Output tok/s | Output tok/s/GPU |
|---|---|---|---|---|---|
| TP1 | 232/300 | 61/100 | 948.674 | 171.812 | 171.812 |
| TP4 | 232/300 | 60/100 | 780.024 | 229.060 | 57.265 |

Native TP4 is 1.216x faster than native TP1 by the summed task-service clock.
Their generated token totals differ: 178,672 versus 162,994.
Native TP4 minus native TP1 accuracy is 0.000 pp on GSM8K, with CI [-2.000, 2.000].
The MATH difference is -1.000 pp, with CI [-7.000, 5.000].

Even native TP1 completes this task cohort sooner than custom Loop PP4.
The engines differ in attention, batching, KV layout, projection fusion, graph reuse, and sampling overhead.
This is a production-oriented observation, not an isolated architectural effect.

Custom TP additionally retains FP32 GEMM outputs/row partials before BF16 casts.
It uses separate projections and replicated embedding/head weights.
Native TP uses fused and vocabulary-parallel operators.
Do not attribute custom TP delay entirely to networking or recurrence.

## Numerical and physical validity

The historical P512 C2 gate remains FAIL.
The five-engine study finds internal differences even between HF Single and native TP1.
All BF16 candidates match 112/112 argmax positions in the bounded FP32-reference study.
That result does not extend to long natural outputs.
The investigation supports numerical sensitivity, but it does not identify one unique full-model defect.

All four custom TP ranks execute real NCCL kernels with owner-local sharded attention and FFN weights.
The representative trace records 24,704 NCCL kernels on each rank.
The native TP4 trace also records 6,369 custom reduction kernels and 33 NCCL kernels per rank.
Kernel counts are not automatically logical collective-call counts.
Kernel duration can include readiness waits.

Both PP placements show actual different-request GPU kernel overlap on four ranks.
The complete timeline and activity tables are in [profiling_analysis.md](profiling_analysis.md).
All 18 bounded observer comparisons preserve output IDs.
Custom comparisons also preserve recorded exit traces and computed argmax IDs.
Heavy wall overhead ranges from 24.9% to 405.0%.
Heavy timing is UNVALIDATED for headline latency.

No additive non-overlapped full-request component decomposition is claimed.
No kernel-service sum becomes an E2E percentage.
Native task HBM peak and graph-only scratch separation remain PARTIAL.
Optional DP4 capacity, C32/C64 expansion, and optimized PP controls are NOT_RUN.

## Research answers and next action

1. TP hidden divergence did not imply a uniform benchmark accuracy drop. Numerical qualification remains unresolved.
2. Static custom Loop PP leads the larger task cohort, but not every matched replay or B1 case.
3. Dynamic changes the ranking. Layer PP outperforms Loop PP on identical Dynamic outputs.
4. Weight replication and fewer handoffs offer a trade-off. Last_exited KV ownership can reverse its benefit.
5. A gain surviving capable generic PP/runtime controls was not demonstrated.
6. A new Loop-aware multi-GPU architecture is not justified by this evidence.

The smallest next experiment is an optimized-attention, pooled-graph PP control on a few frozen requests.
It must use the same backend and cache lifecycle as the native control where feasible.
Do not build a new scheduler or repeat full quality evaluation before that bounded control shows a measured crossover.

## Evidence and reproduction

- [Full quality and paired intervals](quality_results.md).
- [Serving metrics and clock definitions](serving_summary.md).
- [Eight-arm matrix](eight_arm_matrix.md).
- [Numerical diagnosis](numerical_tp_diagnosis.md).
- [Correctness scope](correctness.md).
- [Memory and communication](memory_communication.md).
- [Profiling and observer validity](profiling_analysis.md).
- [Commands and environments](reproduce.md).
- [Preserved failures](failures.md).

The background controller completed quality and stopped during the first native profiler initialization.
The failure was an executable PATH error, not lost quality results.
Recovery skipped completed captures and preserved the failed attempt.
All original protected files and raw failures remain unchanged.
