# Phase 2-B1 Tail-Aware Layer-Wise Placement Report

Date: 2026-06-25

## Scope

This run replaces the previous placement heuristic with a batch-tail objective over calibration expert loads. The model, routing, precision, vLLM EP mode, and input order are unchanged between As-Is and To-Be. Only the layer-wise expert-to-rank placement map is changed.

Old Phase 2-B1 placement outputs were removed before this run. Current artifacts are under:

- `outputs/placement_tail/`
- `outputs/asis_tobe_tail/`

## Objective

For each MoE layer `l`, choose a layer-specific map `m_l(e)` from expert id to EP rank.

```text
R[b,l,r] = sum_e token_count[b,l,e] * 1[m_l(e)=r]
imbalance[b,l] = max_r R[b,l,r] / mean_r R[b,l,r]

objective =
  mean_batch_imbalance
  + 0.5 * p95_batch_imbalance
  + 0.2 * max_batch_imbalance
  + capacity_penalty
```

Implementation detail: the capacity constraint is hard-enforced as exactly 16 experts per rank for every layer. A small total-rank balance term is also tracked in the optimizer summary to avoid solving only local batch tails while creating global rank skew.

## Calibration

Calibration source: `data/sharegpt4v_512/`, 512 samples, batch size 8.

Captured tensor:

- `outputs/placement_tail/sharegpt4v_batch_expert_counts.npz`
- shape: `[64, 48, 128]` = `[batch, layer, expert]`

Placement outputs:

- Linear baseline: `outputs/placement_tail/linear_map_perlayer.json`
- Tail-aware map: `outputs/placement_tail/tail_optimized_map_perlayer.json`

Validation:

- 48 layer maps exist.
- All 48 To-Be layer maps are unique.
- Every layer assigns exactly 16 experts to each of 8 ranks.
- Layer 9 and layer 20 maps are different.

## Offline Calibration Result

Batch-layer imbalance on ShareGPT4V calibration:

| Metric | As-Is linear | Tail-aware To-Be |
|---|---:|---:|
| mean max/mean | 1.3275 | 1.0471 |
| std max/mean | 0.1528 | 0.0154 |
| p90 max/mean | 1.5279 | 1.0681 |
| p95 max/mean | 1.6390 | 1.0742 |
| max max/mean | 2.0382 | 1.1088 |
| rank-total max/mean | 1.0631 | 1.0023 |

Figures:

- `outputs/placement_tail/offline_layer_imbalance.png`
- `outputs/placement_tail/offline_layer_imbalance_box.png`

Interpretation: on the calibration distribution, the new objective does exactly what was intended. It strongly suppresses layer-specific tail imbalance and greatly reduces variance across layer/batch cases.

## Main Profile

Main profile data:

- ChartQA: 64 samples
- MMMU: 64 samples
- MMStar: 64 samples
- batch size: 8
- profile-only, prefill-dominant `max_tokens=1`
- vLLM 8-way EP, TRITON MoE backend

Custom placement audit:

- As-Is audit: `outputs/asis_tobe_tail/asis_audit.jsonl`
- To-Be audit: `outputs/asis_tobe_tail/tobe_audit.jsonl`

The To-Be audit confirms the real vLLM EP loader used non-linear, layer-specific local expert sets. For example, one EP rank had different expert sets for layers 0, 1, 2, etc., instead of a fixed contiguous block.

### Batch-Layer Straggler

| Metric | As-Is linear | Tail-aware To-Be | Change |
|---|---:|---:|---:|
| mean max/mean | 1.2919 | 1.2503 | -3.22% |
| p90 max/mean | 1.4607 | 1.4204 | -2.76% |
| p95 max/mean | 1.5290 | 1.4762 | -3.45% |
| max max/mean | 1.8239 | 1.6678 | -8.56% |

### Layer-Total Rank Imbalance

| Metric | As-Is linear | Tail-aware To-Be | Change |
|---|---:|---:|---:|
| rank-total max/mean | 1.0519 | 1.0341 | -1.69% |
| layer mean max/mean | 1.1857 | 1.1425 | -3.64% |
| layer p95 max/mean | 1.2917 | 1.2764 | -1.19% |
| layer max max/mean | 1.3943 | 1.3606 | -2.42% |

### Prefill Latency

| Metric | As-Is linear | Tail-aware To-Be | Change |
|---|---:|---:|---:|
| total elapsed seconds | 45.8646 | 45.0943 | -1.68% |
| prefill tokens/s | 6834.03 | 6950.76 | +1.71% |

Figures:

- `outputs/asis_tobe_tail/summary_tail.png`
- `outputs/asis_tobe_tail/rank_imbalance_by_layer.png`
- `outputs/asis_tobe_tail/rank_load_profile.png`
- `outputs/asis_tobe_tail/hot_layer_rank_load.png`
- `outputs/asis_tobe_tail/expert_load_profile.png`

Full metrics:

- `outputs/asis_tobe_tail/summary_metrics.json`
- `outputs/asis_tobe_tail/load_profile.json`
- `outputs/asis_tobe_tail/asis.json`
- `outputs/asis_tobe_tail/tobe.json`

## Takeaway

The new objective fixes the previous mismatch in intent: it optimizes the actual batch/layer tail imbalance rather than only balancing average modality preference. On ShareGPT4V calibration, the improvement is very large. On ChartQA/MMMU/MMStar, the improvement transfers but is more modest: p95 batch-layer straggler drops by 3.45%, worst batch-layer straggler drops by 8.56%, and prefill throughput improves by 1.71%.

This suggests the layer-wise placement mechanism is working and the objective is pointed in the right direction. The smaller main-set gain is consistent with calibration-set mismatch, since ShareGPT4V calibration does not perfectly match the main measurement mix. The next diagnostic should be the planned calibration=main ablation: build the same objective from ChartQA/MMMU/MMStar expert counts and check whether the main-set tail and latency gains become much stronger.

## Not Run In This Pass

Accuracy was not re-run here; this pass was profile-only to test the new objective. Because placement only changes expert-to-rank ownership and does not change router choices, token values, precision, or expert math, accuracy should be functionally unchanged, but the formal accuracy comparison should be run once the placement objective is selected.
