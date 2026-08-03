# Phase 2-B1 Main-Calibration Timing Report

Date: 2026-06-25

## Scope

This run tests the calibration-mismatch hypothesis by using the same main profile set as the placement calibration source. The model remains vanilla Qwen3-VL MoE under vLLM 8-way EP. The only changed variable between As-Is and To-Be is the layer-wise expert-to-rank placement map.

Accuracy is intentionally not measured in this pass because generation uses `max_tokens=1`; this is a prefill and load-profile ablation.

## Setup

- vLLM EP: `tensor_parallel_size=8`, `enable_expert_parallel=True`
- Expert placement: As-Is linear vs To-Be layer-wise tail-optimized
- Samples: ChartQA 64, MMMU 64, MMStar 64
- Batch size: 8
- Prefix cache: disabled
- Generation: `max_tokens=1`
- Profile source: real vLLM routed experts, no single-GPU simulation

The To-Be map uses 48 independent layer maps. Every layer assigns exactly 16 experts to each rank, and all 48 maps are unique.

## Objective

For each layer `l`, choose `m_l(e)` to reduce batch-tail rank imbalance:

```text
R[b,l,r] = sum_e token_count[b,l,e] * 1[m_l(e)=r]
imbalance[b,l] = max_r R[b,l,r] / mean_r R[b,l,r]

objective =
  mean_batch_imbalance
  + 0.5 * p95_batch_imbalance
  + 0.2 * max_batch_imbalance
  + capacity_penalty
```

The capacity constraint is hard-enforced as exactly 16 experts per rank per layer.

## Offline Main-Calibration Result

Using the main-set routed expert counts as calibration:

| Metric | As-Is linear | To-Be tail-optimized | Change |
|---|---:|---:|---:|
| mean batch-layer max/mean | 1.2920 | 1.0561 | -18.26% |
| std batch-layer max/mean | 0.1206 | 0.0203 | -83.17% |
| p90 batch-layer max/mean | 1.4653 | 1.0845 | -25.99% |
| p95 batch-layer max/mean | 1.5281 | 1.0932 | -28.46% |
| max batch-layer max/mean | 1.8239 | 1.1620 | -36.29% |

Figures:

- `outputs/main_calib_tail/offline_layer_imbalance.png`
- `outputs/main_calib_tail/offline_layer_imbalance_box.png`

Interpretation: when calibration matches the measured workload, the objective strongly suppresses per-batch, per-layer straggler tails and variance.

## Actual vLLM EP Profile

The real vLLM EP run closely matches the offline prediction:

| Metric | As-Is linear | To-Be tail-optimized | Change |
|---|---:|---:|---:|
| mean batch-layer max/mean | 1.2920 | 1.0573 | -18.17% |
| p90 batch-layer max/mean | 1.4653 | 1.0855 | -25.92% |
| p95 batch-layer max/mean | 1.5281 | 1.0945 | -28.38% |
| max batch-layer max/mean | 1.8239 | 1.1643 | -36.17% |
| variance | 0.01455 | 0.00042 | -97.12% |

Layer-total rank imbalance also improves:

| Metric | As-Is linear | To-Be tail-optimized | Change |
|---|---:|---:|---:|
| mean layer rank max/mean | 1.1856 | 1.0185 | -14.10% |
| p95 layer rank max/mean | 1.2916 | 1.0306 | -20.21% |
| max layer rank max/mean | 1.3943 | 1.0345 | -25.80% |

Figures:

- `outputs/asis_tobe_maincalib/rank_imbalance_by_layer.png`
- `outputs/asis_tobe_maincalib/rank_load_profile.png`
- `outputs/asis_tobe_maincalib/hot_layer_rank_load.png`
- `outputs/asis_tobe_maincalib/expert_load_profile.png`
- `outputs/asis_tobe_maincalib/batch_layer_imbalance_hist.png`

## Timing

Request-level timing is split into TTFT, a scheduled-to-first-token prefill proxy, decode, and E2E where available. In this vLLM run with `max_tokens=1`, request-level `last_token_ts` is not populated, so request-level E2E appears as zero. Batch wall-clock time is used as the E2E proxy.

| Metric | As-Is linear | To-Be tail-optimized | Change |
|---|---:|---:|---:|
| total batch wall time | 44.2174 s | 43.8728 s | -0.78% |
| prefill tokens/s | 7088.61 | 7144.29 | +0.79% |
| mean TTFT | 1.0203 s | 1.0031 s | -1.69% |
| mean scheduled-to-first-token | 0.4563 s | 0.4338 s | -4.94% |
| p95 scheduled-to-first-token | 1.4586 s | 1.4163 s | -2.90% |
| decode time | 0.0000 s | 0.0000 s | n/a |

Figures:

- `outputs/asis_tobe_maincalib/summary_maincalib.png`
- `outputs/asis_tobe_maincalib/timing_breakdown.png`
- `outputs/asis_tobe_maincalib/timing_p95.png`

## Takeaway

The main-as-calibration ablation confirms that the layer-wise placement mechanism and tail objective are doing the intended thing: batch-layer straggler p95 drops by 28.38%, worst-case batch-layer imbalance drops by 36.17%, and straggler variance drops by 97.12%.

Latency improves, but less dramatically than load balance. The cleanest timing signal is the scheduled-to-first-token prefill proxy, which improves by 4.94% on average. Full batch wall-clock improves by 0.78%, likely because it includes non-MoE work such as image processing, vision encoding, scheduling, all-to-all overheads, and residual decode/generation plumbing. This supports the mismatch diagnosis: placement can strongly flatten the MoE load when calibrated to the target distribution, but end-to-end speedup is bounded by the fraction of runtime actually dominated by MoE straggler work.

## Artifacts

- Summary metrics: `outputs/asis_tobe_maincalib/summary_maincalib.json`
- As-Is profile: `outputs/asis_tobe_maincalib/asis.json`
- To-Be profile: `outputs/asis_tobe_maincalib/tobe.json`
- Load profile: `outputs/asis_tobe_maincalib/load_profile.json`
- Optimized map: `outputs/main_calib_tail/tail_optimized_map_perlayer.json`
- Linear map: `outputs/main_calib_tail/linear_map_perlayer.json`
