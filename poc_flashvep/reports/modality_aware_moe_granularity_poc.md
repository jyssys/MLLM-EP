# Modality-aware MoE execution granularity PoC

## Result first

`MODALITY_AWARE_MOE_GRANULARITY: NO_GO`.

Both real-route populations selected the same best granularity (`M=512`).
Vision had more active/effective experts at large M, but this did not create a
different latency curve or a useful modality-specific choice.  The large
benefit of increasing M was a generic batching effect shared by both
modalities, not evidence for a Vision-specific fused/persistent policy.

## Registration and environment

- Branch: `flashvep/modality-aware-moe-granularity-poc`
- Base: `57d97d8` (previous visual-streaming PoC)
- Model: Qwen3-VL-30B-A3B-Instruct, BF16
- Runtime: vLLM 0.20 V1, eager, DBO off, prefix cache off
- Parallelism: TP2 / DP2 / EP4 / PP1, DeepEP high-throughput, TritonExperts
- Placement: linear `expert_id // 32`, 128 global experts, top-8
- GPU mapping: `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1, 2, 3, 4)
- Route source: `live_prefill_execution_regime_20260821_111609` (24 real-image
  requests, 48 layers, `image_token_id=151655`)
- Selected requests: `model_card` (chart/document), `deep_field` (natural),
  `retina` (fine-grained); layers 4/24/44.
- M values: 32/64/128/256/512.  For each modality, the first M positions in
  the original request order were selected; no token reorder was performed.
- Timing: three warmups and 30 measured repetitions per case, CUDA events in
  the existing vLLM worker hook, four EP ranks.  Route IDs are exact real
  captures.  The activation input is the validated BF16 layer-24 capture
  (`layer24_capture.pt`) cycled only as an operator-replay input; therefore
  these are real-weight/real-route operator timings, not a new end-to-end
  serving benchmark.

Preparation and execution command:

```text
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=3 ITERATIONS=20 \
  poc_flashvep/modality_aware_moe_granularity/run_gpu.sh \
  poc_flashvep/deepep_revalidation/results/modality_aware_moe_granularity_poc_20260903_2000
```

## GPU results

Critical-path values are the maximum of the four EP-rank medians for each
request/layer case.  The table is the median over nine matched
request/layer cases per modality and M.

| modality | M | total ms/token | dispatch ms/token | expert ms/token | combine ms/token | active experts | effective experts | p50 local expert M | rank CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 32 | 0.026663 | 0.005643 | 0.012604 | 0.002840 | 67 | 43.05 | 2 | 0.165 |
| Text | 64 | 0.013371 | 0.002906 | 0.006343 | 0.001489 | 82 | 50.12 | 4 | 0.155 |
| Text | 128 | 0.006768 | 0.001488 | 0.003250 | 0.000818 | 92 | 54.93 | 6 | 0.161 |
| Text | 256 | 0.003406 | 0.000805 | 0.001791 | 0.000392 | 102 | 56.83 | 12 | 0.184 |
| Text | 512 | **0.002655** | 0.000450 | 0.001230 | 0.001003 | 109 | 56.93 | 19 | 0.173 |
| Vision | 32 | 0.026649 | 0.005623 | 0.012615 | 0.002983 | 47 | 30.19 | 3 | 0.218 |
| Vision | 64 | 0.013447 | 0.002869 | 0.006365 | 0.001547 | 77 | 44.68 | 4 | 0.199 |
| Vision | 128 | 0.006750 | 0.001500 | 0.003272 | 0.000799 | 97 | 59.91 | 6.5 | 0.178 |
| Vision | 256 | 0.003376 | 0.000786 | 0.001819 | 0.000392 | 110 | 70.62 | 12 | 0.155 |
| Vision | 512 | **0.002620** | 0.000455 | 0.001206 | 0.000754 | 118 | 78.50 | 24 | 0.149 |

At M=512, Vision's route working set is broader (118 vs 109 active experts,
78.50 vs 56.93 effective experts), but critical total latency/token differs by
only +0.04% (Vision minus Text) in the paired median.  Paired phase deltas are:
dispatch +0.09%, expert +0.65%, and combine -0.06% (Vision minus Text).
These are far below the preregistered 5%/8% decision levels.

The M curve decreases for both populations: the common fixed M=128 to M=512
reduction is 61.18% for Vision and 60.77% for Text.  This is a shared
granularity/batching effect, not a modality-aware gain.  Every one of the nine
request/layer groups selected M=512 as its per-case minimum for both modalities.

## Route and phase interpretation

The route statistics do show modality-conditioned geometry: at M=512 Vision
has larger active/effective working sets and larger p50 local expert M.  At
small M, Text happens to have more active experts (67 vs 47 at M=32), so the
working-set difference is not a monotonic “Vision is always wider” effect.
Rank-load CV remains comparable (0.149–0.218), and no phase shows a repeated
modality-specific latency penalty.  Expert execution is the largest measured
phase at M=32–256; dispatch and combine scale similarly and do not produce a
different optimum.  Layout preparation is small (about 0.026 ms per case) and
was measured separately; it is not included in the modality decision.

No route-remap/shuffle GPU control was run: the main question was falsified by
the paired real-route curves themselves, and adding a remapped route would no
longer be a model-correctness path.  Route identity and token-partition
identity were asserted for all 90 cases and all four ranks; all output tensors
had the expected `[M, 2048]` shape.  This is operator-output/route correctness,
not a claim of end-to-end logits equivalence for the replay activation.

## Gate

```json
{
  "status": "NO_GO",
  "vision_optimal_M": 512,
  "text_optimal_M": 512,
  "modality_specific_gain": "0% (same optimum)",
  "common_M": 128,
  "common_to_own_opt_gain_pct": {"vision": 61.18, "text": 60.77},
  "repeated_optimum": {"vision": "9/9 at 512", "text": "9/9 at 512"},
  "routing_causality": "rejected for a modality-specific optimum",
  "correctness": "route and token partition identity pass"
}
```

`NO_GO` is required because the two curves and optima are effectively the
same; selecting M=512 for both is not a modality-aware policy.  The routing
working-set difference therefore does not support implementing a
fused/persistent or modality-conditioned execution method in this branch.

## Artifacts

Result directory:
[`modality_aware_moe_granularity_poc_20260903_2000`](../deepep_revalidation/results/modality_aware_moe_granularity_poc_20260903_2000/)

- `workload_manifest.json`, `selection_policy.json`, `cases.json`
- `route_statistics.csv`, `rank_timing_raw.csv`, `granularity_results.csv`
- `matched_pair_results.csv`, `modality_curves.csv`, `gate_summary.json`
- `latency_per_token_vs_granularity.png`
- `phase_breakdown_vs_granularity.png`
- `local_expert_m_distribution.png`
- rank/layer replay records under `replay/`

## Final interpretation

- `VISION_OPTIMAL_GRANULARITY`: 512
- `TEXT_OPTIMAL_GRANULARITY`: 512
- `MODALITY_SPECIFIC_GAIN`: none; both prefer the largest tested M
- `PHASE_RESPONSIBLE`: shared expert/batching scaling, not a modality-specific
  dispatch/expert/combine phase
- `ROUTING_CAUSALITY`: REJECTED for the proposed modality-specific granularity
- `MATCHED_PAIR_CONSISTENCY`: strong; median total delta -0.08%, with all nine
  request/layer pairs measured
- `NEXT`: do not implement fused/persistent modality-aware kernels.  If the
  direction is revisited, first obtain live hidden-state per-token captures and
  a much larger, latency-stable workload; no such follow-up is justified by
  this bounded result.
