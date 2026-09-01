# MLLM-Specific MoE/EP Bottleneck Localization

## Executive result

`FINAL STATUS: NO-GO` for a strong MLLM-specific GPU bottleneck in this
bounded study.  Vision routing has a repeatable shape difference, but the
shape difference does not produce a consistent matched-token expert,
dispatch, or combine latency penalty.  Image count increases the amount of
work, as expected, but the measured critical-path expert cost is sublinear
and communication is non-monotonic.  The evidence therefore does not justify
an MLLM-specific optimization or scheduler.

## Environment and provenance

* Model: Qwen3-VL-30B-A3B-Instruct, BF16
* vLLM 0.20.0, eager mode, DBO off, prefix cache off
* TP2 / DP2 / EP4 / PP1, DeepEP high-throughput, TritonExperts
* expert placement: linear, global expert `e` on EP rank `e // 32`
* physical GPU mapping: `CUDA_VISIBLE_DEVICES=1,2,3,4` (logical EP ranks 0–3)
* repository branch: `flashvep/mllm-specific-bottleneck-localization`
* live command:

  ```text
  CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=1 ITERATIONS=2 \
    bash poc_flashvep/mllm_specific_bottleneck_localization/run_gpu.sh \
    poc_flashvep/deepep_revalidation/results/mllm_specific_bottleneck_localization_20260901_124106
  ```

The live hook is the existing read-only CUDA-event instrumentation around
`_prepare` (dispatch), `_fused_experts` (expert), and `_finalize` (combine).
One bounded synchronization resolves events after the run; no route,
placement, weight, or communication decision is changed.  Backend proof files
show `DeepEPHTPrepareAndFinalize`, `DeepEPHTAll2AllManager`, EP world size 4,
and `TritonExperts`.  All four rank proofs report `deep_ep_collective_overlap=0`.

## Workload and matching quality

The live extension has 11 deterministic requests.  Image rows use the local
skimage/MODE assets and a fixed short comparison prompt; text rows reuse the
validated local documentation prompts.  The actual processor token counts
and image-token counts are recorded in `workload_metadata.json`.

| condition | requests | image counts | prompt-token counts (range) | notes |
|---|---:|---:|---:|---|
| Text-only | 3 | 0 | 128, 277, 1,589 | exact validated text route lengths |
| Single image | 2 | 1 | 123, 271 | coins and grass |
| Repeated image | 3 | 2, 4, 8 | 238, 458, 898 | same coins image repeated |
| Diverse multi-image | 3 | 2, 4, 8 | 256, 772, 1,672 | distinct local images |

The only repeated/diverse pair inside a fixed 10% token range is the 2-image
pair (238 vs 256, 7.6%).  The 4-image and 8-image pairs are retained as
scale diagnostics, but are explicitly not treated as matched causal pairs
(token ratios 1.69 and 1.86).  The previous 24-request exact route suite is
used for the equal-token modality-shape control; the four prior long
multi-image manifests (6/8/10/12 images, 3,203–9,065 tokens) are included as
offline scale context without inventing live timing.

## Routing shape: primary equal-token control

Using the previous 24 real-image requests, equal-token (64-token) within-
request subsampling gives the following request/layer medians:

| source token type | active experts | effective experts | HHI | top-4 share | rank imbalance |
|---|---:|---:|---:|---:|---:|
| Vision | 96.0 | 67.11 | 0.01942 | 0.1660 | 1.2188 |
| Text | 81.0 | 50.48 | 0.02828 | 0.2344 | 1.2969 |

This reproduces the prior routing observation: Vision uses a broader and less
concentrated expert working set at matched token count.  The prior discovery
report also shows this direction across natural, chart/document, and
fine-grained categories.  This is a routing phenomenon, not yet a GPU
bottleneck.

## Live shape and execution measurements

The new live run contains 1,056 complete request/layer invocations (11
requests × 2 measured repetitions × 48 layers).  Values below are medians of
the critical path (maximum of the four EP ranks) per request, then summarized
by request.  Expert/dispatch/combine are CUDA-event spans; prefill wall time
is a coarse driver interval and is shown only for completeness.

| condition/request | tokens | active experts | effective experts | rank imbalance | expert ms | dispatch ms | combine ms | prefill wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text control (small) | 128 | 95 | 55.85 | 1.275 | 0.563 | 1.315 | 0.404 | 2,894 |
| Single coins | 123 | 105 | 64.97 | 1.232 | 0.419 | 0.442 | 0.113 | 2,930 |
| Text control (medium) | 277 | 105 | 55.00 | 1.259 | 0.433 | 0.509 | 0.133 | 118 |
| Single grass | 271 | 122 | 84.73 | 1.167 | 0.429 | 0.381 | 0.124 | 132 |
| Repeated 2 | 238 | 113 | 67.77 | 1.232 | 0.423 | 0.424 | 0.119 | 2,843 |
| Diverse 2 | 256 | 122 | 84.80 | 1.176 | 0.425 | 0.366 | 0.119 | 125 |
| Repeated 4 | 458 | 117 | 66.74 | 1.213 | 0.439 | 0.363 | 0.124 | 134 |
| Diverse 4 | 772 | 127 | 90.71 | 1.143 | 0.470 | 0.415 | 0.105 | 2,973 |
| Repeated 8 | 898 | 119 | 65.58 | 1.219 | 0.467 | 0.426 | 0.110 | 3,036 |
| Diverse 8 | 1,672 | 128 | 96.42 | 1.136 | 0.523 | 0.449 | 0.132 | 226 |

The very large alternating driver wall intervals (roughly 0.1–0.2 s versus
2.8–3.1 s) occur in this single-request-per-wave vLLM/DP harness and are not
used as a modality causal claim.  The CUDA-event stage timings are the
primary execution comparison.

Layer-band medians are stable in direction for shape but not for a latency
penalty: diverse image requests have effective experts 88.3/92.5/91.4 in
early/middle/late bands and expert spans 0.475/0.480/0.471 ms; repeated image
requests have 54.3/69.1/77.9 and 0.451/0.443/0.455 ms.  Text has 59.7/51.0/56.7
effective experts but higher expert spans 0.507/0.512/0.515 ms.  Thus the
shape expansion is not co-located with a Vision-specific slowdown.

## Matched-token comparisons

The fixed pairs were selected by token count before looking at latency:

| pair | token counts | expert CUDA difference (Vision − Text) | dispatch difference | combine difference |
|---|---:|---:|---:|---:|
| coins vs small text | 123 vs 128 | −0.144 ms (−25.6%) | −0.872 ms | −0.292 ms |
| grass vs medium text | 271 vs 277 | −0.004 ms (−0.9%) | −0.127 ms | −0.010 ms |

The first pair is a small-shape regime with unstable communication spans; the
second is the more informative medium pair and is effectively equal in
expert cost.  Across the full live request/layer sample, a descriptive
Spearman correlation between total assignments and expert span is 0.445, but
effective-expert count is only 0.074.  Dispatch and combine correlations with
shape features are below |0.16|.  These values do not support H2.

For repeated versus diverse, only the 2-image comparison is token-eligible:
expert span differs by +0.0024 ms (+0.6%) for diverse, dispatch is −13.6%, and
combine is +0.2%.  The 4/8 pairs show larger raw differences but also 69–86%
more tokens for diverse and are therefore not causal evidence.  After image
count rises, expert span grows only from 0.419 ms (single coins) to 0.467 ms
(repeated 8) and 0.523 ms (diverse 8), while assignment volume grows 7.1× and
13.3× respectively.  Communication is non-monotonic.  The prior long
6/8/10/12-image artifacts similarly establish scale coverage but have no
live timing in this bounded run.

## Hypothesis decisions

| hypothesis | verdict | evidence |
|---|---|---|
| H1: matched-token Vision uses a wider expert working set | **ACCEPT** | 24-request equal-token control: active 96 vs 81 and effective 67.1 vs 50.5; direction repeats across categories in the prior artifact. |
| H2: working-set expansion increases expert/EP GPU cost | **REJECT** | Medium matched pair expert cost is −0.9% for Vision; dispatch/combine are not consistently higher. Shape/latency correlations are weak. |
| H3: image count increases cost beyond token count | **REJECT** | Shape expands with count, but stage cost is sublinear and communication is non-monotonic; no matched extra-token effect. |
| H4: diverse multi-image costs more than repeated at the same count | **REJECT** | The only token-eligible 2-image pair has a 0.6% expert difference; 4/8 comparisons are token-confounded. |
| H5: high-resolution single image differs from diverse at matched tokens | **PROMISING for routing only** | Existing route shapes are image-conditioned, but this run lacks a live, token-matched high-resolution single-image control; no GPU claim is made. |

## Routing phenomenon versus GPU bottleneck

The strongest positive evidence is the equal-token, source-image-internal
working-set expansion (higher active/effective experts and lower HHI for
Vision), reproduced by the new live aggregate shape metrics.  The strongest
counter-evidence is that this expansion does not translate into a consistent
expert or communication latency penalty: the informative medium matched pair
has essentially identical expert CUDA time, while Text-only controls are often
slower despite their narrower working set.

No dense MLLM control was run, so absence from Dense MLLM is not empirically
established.  The result also does not claim that generic serving/runtime
effects do not exist; it says that this bounded MLLM-specific modality shape
is not a demonstrated optimization bottleneck.

## Final interpretation and next action

There is no strong MLLM-specific bottleneck candidate in this study.  The
best-supported phenomenon remains **image-conditioned expert working-set
expansion without a corresponding per-token GPU penalty**, not a method target.
The repeated/diverse control does not justify an image-count-aware optimizer,
and the route-shape difference alone should not be treated as a latency
opportunity.

**Next single recommended action:** close the MLLM-specific optimization line;
if the unresolved H5 question is still important, run one preregistered
equal-token live comparison of a high-resolution single image and a diverse
multi-image request before considering any method work.

## Artifact index

Result directory:
`poc_flashvep/deepep_revalidation/results/mllm_specific_bottleneck_localization_20260901_124106/`

Key artifacts:

* `workload_rows.json`, `workload_metadata.json`, `run_metadata.json`
* `raw_live/rank{0,1,2,3}.jsonl` and rank proof files
* `analysis/live_invocation_metrics.csv`
* `analysis/request_summary.csv`
* `analysis/exact_route_shape.csv`, `analysis/exact_route_shape_summary.csv`
* `analysis/layer_band_summary.csv`, `analysis/long_multimodal_offline_shape.csv`
* `analysis/routing_latency_correlations.csv`, `analysis/matched_token_pairs.csv`
* `analysis/repeated_diverse_pairs.csv`, `analysis/hypothesis_summary.csv`
* `figures/plot1_routing_shape_by_condition.png`
* `figures/plot2_latency_by_condition.png`
* `figures/plot3_image_count_scaling.png`
* `figures/plot4_routing_shape_vs_latency.png`
* `figures/plot5_matched_token_latency.png`

The historical exact-token modality and category robustness source is
`poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`
and the long offline scale source is
`poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000/`.
