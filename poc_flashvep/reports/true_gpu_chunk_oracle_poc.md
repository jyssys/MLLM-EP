# True GPU-cost chunk-oracle PoC

## Scope and frozen protocol

This is a bounded operator-replay validation of the existing Qwen3-VL route
artifacts. It does not change routing, token order, expert placement, model
weights, or a serving scheduler. The base was the fair decomposition result
(`1857ca0aa28985c804ed76ef07b9ccc17a9ce76d`); the measurement code was
recorded before the Stage-B validation. GPU execution used
`CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1--4), TP=2, DP=2, EP=4,
PP=1, BF16, vLLM 0.20.0, DeepEP high-throughput and the TritonExperts
backend. The replay uses the validated layer-24 activation capture cycled to
the immutable route lengths, as in the preceding fair replay; it is therefore
an exact-route grouped-MoE operator test, not a live 48-layer forward.

Commands used:

```text
python poc_flashvep/true_gpu_chunk_oracle_poc/stage_a.py --result <result> --make-candidates
WARMUPS=5 ITERATIONS=20 poc_flashvep/true_gpu_chunk_oracle_poc/run_gpu.sh <result>
python poc_flashvep/true_gpu_chunk_oracle_poc/analyze.py --result <result>
python poc_flashvep/true_gpu_chunk_oracle_poc/stage_b_prepare.py --result <result>
TRUE_B_MODE=cost ... poc_flashvep/true_gpu_chunk_oracle_poc/run_gpu.sh <result>
python poc_flashvep/true_gpu_chunk_oracle_poc/stage_b_analyze.py --result <result>
TRUE_B_MODE=validate ... poc_flashvep/true_gpu_chunk_oracle_poc/run_gpu.sh <result>
python poc_flashvep/true_gpu_chunk_oracle_poc/stage_b_report.py --result <result>
```

## Stage A — same-M routing-shape sensitivity

Pair selection was latency-blind: token windows were on an 8-token grid,
non-overlapping windows in the same request/layer were required, and the
largest composite routing-shape distance (histogram L1, active experts, HHI,
maximum load and rank-CV) was selected. A fixed cap of 16, 16 and 8 pairs for
M=128, 256 and 512 was applied before timing. This produced 40 pairs (80
windows) across natural, fine-grained, chart/document and multi-image
requests, with short and long route artifacts.

Primary metric is the absolute paired difference of the four-rank maximum
expert CUDA median. Results:

| M | pairs | expert gap median | p25 | p75 | wall gap median | dispatch gap median | combine gap median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 3.96% | 2.88% | 19.48% | 2.12% | 1.81% | 17.54% |
| 256 | 16 | 8.69% | 1.51% | 10.09% | 1.40% | 2.20% | 14.18% |
| 512 | 8 | 8.78% | 7.57% | 10.20% | 0.38% | 1.78% | 16.88% |

Across all 40 pairs the expert gap median was 7.80% (p25 2.71%, p75
10.09%). By source, short-route medians were 3.90%, 1.29% and 4.47% for
M=128/256/512; long-route medians were 7.15%, 9.28% and 8.85%. Thus the
promising aggregate signal is driven largely by long/high-scale windows and
has substantial dispersion. Wall latency did not show the same magnitude,
and combine had a much larger gap than dispatch, so this is not evidence of a
stable end-to-end shape staircase by itself.

The candidate distributions and feature relationship plots are:

* [plot1_same_m_shape_vs_latency.png](../deepep_revalidation/results/true_gpu_chunk_oracle_poc_20260831_221410/figures/plot1_same_m_shape_vs_latency.png)
* [plot2_same_m_routing_features.png](../deepep_revalidation/results/true_gpu_chunk_oracle_poc_20260831_221410/figures/plot2_same_m_routing_features.png)

**Stage-A classification: PROMISING (7.80% primary median), but variable.**
The user gate's 5--10% band is used without adding a post-hoc threshold;
Stage B is therefore run, while the low M=128 short-route result remains a
counter-evidence.

## Stage B — measured GPU-cost same-count oracle

The bounded subset was fixed before validation: `coffee_rocket`, `model_card`,
`retina` and `method`, layers 0/24/47, budgets 128/256. Candidate boundaries
were fixed-cut +/-32 tokens on a 16-token grid. Each interval was prewarmed
and measured in deterministic shuffled batches (2 warmups + 5 measurements for
the cost table). The final selected partitions were then remeasured in an
independent 5-warmup/20-measurement validation run. The DP objective was the
max-rank measured expert-CUDA median, with contiguous order, exact Fixed
chunk count and strict `chunk_size <= B`.

The measured interval cost table contains 20,388 rank observations. Its
expert-CUDA CV median was 4.40%, p95 19.88%, and 14.19% of intervals had CV
over 10%; these values are retained as the noise limitation rather than
filtered. Six of 24 tasks selected Fixed exactly (no false oracle gain).

Independent validation (12 tasks per budget) is summarized below; times are
max-rank expert medians in milliseconds.

| Budget | Fixed | Balanced | Tile same-count | True GPU-cost | Fixed→Balanced | Balanced→True |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.39099 | 0.38884 | 0.39194 | 0.38935 | 0.91% | **-0.21%** |
| 256 | 0.43574 | 0.42694 | 0.42980 | 0.43108 | 0.40% | **+0.04%** |

Positive fraction for Balanced→True was 41.7% (128) and 50.0% (256). The
True-GPU partition's reduction versus the Tile same-count partition was only
0.52% and 0.65% at 128/256. All validation outputs were finite, route
identity was true, and token partition identity was true. Chunk counts were
exactly the same across Fixed/Balanced/Tile/True within each task; Balanced
reduced size CV as expected, but the routing-aware cuts did not provide a
reliable additional reduction.

Figures:

* [plot3_true_gpu_oracle_comparison.png](../deepep_revalidation/results/true_gpu_chunk_oracle_poc_20260831_221410/figures/plot3_true_gpu_oracle_comparison.png)
* [plot4_interval_cost_distribution.png](../deepep_revalidation/results/true_gpu_chunk_oracle_poc_20260831_221410/figures/plot4_interval_cost_distribution.png)

**Stage-B gate: NO-GO.** The final remeasurement is below 0.5% relative to
Balanced at both requested budgets, far below the 5% minimum. The measured
cost objective therefore does not rescue the tile-count oracle in this
bounded real-route replay. Stage-B does not justify a serving-trace or
scheduler implementation.

## Comparison with the preceding fair decomposition

The corrected fair replay (same prewarm/order protocol) measured the
Balanced→Same-count component at approximately -0.057%, +0.332%, +0.097% and
-0.046% for short budgets 128/256/512/1024, and +0.039%, +0.258%, +0.072% and
-0.518% for long routes. The present true-GPU DP gives +0.04% at 256 and
-0.21% at 128 relative to Balanced, consistent with a negligible
routing-boundary component. The large earlier Relaxed-oracle reductions are
therefore not evidence for routing-aware boundaries: they are dominated by
chunk-size/count relaxation and warm-cache/order effects identified in the
fair correction.

## Final decision

`FINAL STATUS: NO-GO` for the routing-aware chunk-boundary direction under
this exact-route, GPU-cost objective. Stage A shows a conditional same-M
expert-time sensitivity (PROMISING overall, strongest in long windows), but
it does not translate into a robust partitioning opportunity; Balanced often
matches or beats the true GPU oracle, and the end-to-end wall gap is much
smaller than the expert/interval tails. No real serving trace or scheduler
implementation is recommended from this result.

The complete result directory is
`poc_flashvep/deepep_revalidation/results/true_gpu_chunk_oracle_poc_20260831_221410/`.
It includes candidate manifests, raw rank timings, interval costs, selected
cuts, validation timings, CSV summaries, gate JSON and figures.

## Reproducibility / limitations

The GPU replay uses existing layer-24 hidden-state capture and actual Qwen3-VL
weights/kernels with exact route IDs; it is not a live per-layer hidden-state
capture. Stage-B cost candidates were intentionally bounded around fixed cuts
and not a full all-boundary search. The independent validation and explicit
CV report prevent treating a noisy interval table as a learned predictor.
No model, router, placement, communication primitive or production scheduler
was modified.
