# Track 1 — Modality-Phase Resource Co-Scheduling

**Decision: NO-GO.** No stable pair approached the required median `eta >= 0.5`,
and vision/text differences were small relative to run variance.

## Fresh 4×H100 result

The statistic is `eta=(T_attention+T_stage-T_concurrent)/min(T_attention,T_stage)`.
All values below are rank-critical medians across 12 post-warmup randomized
repetitions. Maximum-only observations are deliberately not used.

Clean TTFT and the instrumented stage atlas give the following Amdahl context.
Shares use clean TTFT in the denominator but instrumented event sums in the
numerator, so they are attribution proxies (with the measured observer tax),
not an additive decomposition:

| Modality | clean TTFT | Attention | Dispatch | Expert | Combine | full MoE |
|---|---:|---:|---:|---:|---:|---:|
| Text-heavy | 129.014 ms | 41.83 ms (32.4%) | 34.20 ms (26.5%) | 40.42 ms (31.3%) | 20.14 ms (15.6%) | 101.83 ms (78.9%) |
| Vision-heavy | 264.979 ms | 41.24 ms (15.6%) | 32.18 ms (12.1%) | 39.61 ms (14.9%) | 18.96 ms (7.2%) | 98.56 ms (37.2%) |

Vision preprocessing/encoder cost explains the larger vision TTFT but is not
part of the matched language-attention concurrency unit.

| Attention input | Concurrent EP unit | Attention ms | EP unit ms | Concurrent ms | median eta | p10–p90 eta | median wall saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text-heavy | Dispatch | 0.772 | 0.441 | 1.206 | 0.014 | -0.032–0.346 | 0.44% |
| Text-heavy | Combine | 0.767 | 0.462 | 1.209 | 0.011 | -0.193–0.163 | 0.46% |
| Text-heavy | Expert | 0.766 | 1.284 | 2.078 | -0.017 | -0.101–0.062 | -0.65% |
| Text-heavy | whole MoE | 0.773 | 1.685 | 2.468 | -0.027 | -0.165–0.037 | -0.83% |
| Vision-heavy | Dispatch | 0.758 | 0.407 | 1.174 | -0.010 | -0.288–0.217 | -0.33% |
| Vision-heavy | Combine | 0.776 | 0.460 | 1.236 | 0.047 | -0.386–0.165 | 1.74% |
| Vision-heavy | Expert | 0.742 | 1.288 | 2.039 | 0.001 | -0.109–0.184 | 0.04% |
| Vision-heavy | whole MoE | 0.738 | 1.678 | 2.425 | 0.012 | -0.080–0.158 | 0.37% |

Vision-minus-text median eta was -0.024 for Dispatch, +0.036 for Combine,
+0.018 for Expert, and +0.040 for whole MoE. These are neither large nor
stable: all p10 intervals cross zero and absolute-eta CVs are high.

## Interpretation

The hidden assumption behind this pivot was that a vision-content attention
invocation at matched shape occupies a materially different H100 resource
regime. It did not: the stock Qwen language-attention kernel geometry is mostly
shape-driven, and matched 2,363-token text and vision inputs produced nearly
identical standalone layer attention (sum across 48 layers: 41.83 vs 41.24 ms).
Running full-sized units concurrently exposed resource contention rather than
useful complementarity.

This cleanly differs from the previously closed token-fragmentation direction:
no stage or token group was subdivided. It also agrees with earlier whole
vision-encoder overlap evidence, where encoder+Dispatch/Combine/Expert slowed
rather than accelerated.

## Prior-art risk

Even if a positive pair existed, the broad overlap space is crowded by
[NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan),
[COMET](https://seed.bytedance.com/en/public_papers/comet-fine-grained-computation-communication-overlapping-for-mixture-of-experts),
[FLUX](https://arxiv.org/abs/2406.06858), and
[Lancet](https://proceedings.mlsys.org/paper_files/paper/2024/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html).
A strong modality-specific causal difference would have been needed. The live
control falsified that prerequisite.

## Gate

- Stable median eta >=0.5: **FAIL** (best 0.047).
- Large vision/text eta separation: **FAIL** (largest 0.040).
- Correctness: **PASS**.
- Recommendation: do not implement a modality-phase co-scheduler for these
  stock full-size units.
