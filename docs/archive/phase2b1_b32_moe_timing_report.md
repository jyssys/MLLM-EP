# Phase 2-B1 Batch-32 MoE Timing Report

Date: 2026-06-25

## Scope

This run increases the profile batch size to 32 and adds MoE-only CUDA event timing so the latency result is closer to the ReaLB/MACS measurement style. Accuracy is not measured because generation uses `max_tokens=1`.

Two timing paths are separated:

- Clean wall-clock/TTFT: non-instrumented vLLM runs under `outputs/asis_tobe_b32_wall/`
- MoE-only CUDA event timing: instrumented `FusedMoE.forward` runs under `outputs/asis_tobe_b32_moetiming/`

The instrumented run wraps vLLM `FusedMoE.forward` with CUDA events only. It does not modify fused kernels, routing, precision, token values, all-to-all, or expert math.

## Setup

- Model: Qwen3-VL-30B-A3B-Instruct
- Runtime: vLLM 8-way EP
- Placement: As-Is linear vs To-Be layer-wise tail-optimized
- Batch size: 32
- `max_num_seqs`: 32
- `max_num_batched_tokens`: 131072
- KV cache: 8 GiB per GPU
- Samples: ChartQA 64, MMMU 64, MMStar 64
- Total profile samples: 192
- Total prefill tokens: 313,440
- Number of batches: 6
- Generation: `max_tokens=1`
- Prefix cache: disabled

Batch 32 did not OOM. vLLM reported a GPU KV cache capacity of 349,520 tokens and maximum concurrency of 42.67x for 8,192-token requests.

## Placement Objective

The batch-32 placement was rebuilt from the batch-32 main expert counts:

```text
R[b,l,r] = sum_e token_count[b,l,e] * 1[m_l(e)=r]
imbalance[b,l] = max_r R[b,l,r] / mean_r R[b,l,r]

objective =
  mean_batch_imbalance
  + 0.5 * p95_batch_imbalance
  + 0.2 * max_batch_imbalance
  + capacity_penalty
```

Each layer still assigns exactly 16 experts to each EP rank.

## Offline Placement Result

| Metric | As-Is linear | To-Be tail-optimized | Change |
|---|---:|---:|---:|
| mean batch-layer max/mean | 1.2674 | 1.0186 | -19.63% |
| std batch-layer max/mean | 0.1149 | 0.0081 | -92.94% |
| p90 batch-layer max/mean | 1.4292 | 1.0264 | -28.18% |
| p95 batch-layer max/mean | 1.5006 | 1.0345 | -31.06% |
| max batch-layer max/mean | 1.6309 | 1.0531 | -35.43% |

Figures:

- `outputs/main_calib_tail_b32/offline_layer_imbalance.png`
- `outputs/main_calib_tail_b32/offline_layer_imbalance_box.png`

## Clean Wall-Clock Result

These numbers come from non-instrumented runs.

| Metric | As-Is linear | To-Be layer-wise | Change |
|---|---:|---:|---:|
| total batch wall time | 42.3955 s | 42.8201 s | +1.00% slower |
| prefill tokens/s | 7393.24 | 7319.92 | -0.99% |
| mean TTFT | 3.1873 s | 3.3593 s | +5.40% slower |
| mean scheduled-to-first-token | 1.1335 s | 1.2558 s | +10.78% slower |
| p95 TTFT | 7.2187 s | 7.5171 s | +4.13% slower |
| p95 scheduled-to-first-token | 2.8452 s | 3.5456 s | +24.62% slower |
| decode time | 0.0000 s | 0.0000 s | n/a |

`max_tokens=1` means decode is intentionally near zero. vLLM request-level `last_token_ts` is unavailable/zero in this path, so batch wall-clock is the E2E proxy.

## Load Balance

The placement strongly flattens the actual routed load:

| Metric | As-Is linear | To-Be layer-wise | Change |
|---|---:|---:|---:|
| mean batch-layer max/mean | 1.2673 | 1.0196 | -19.55% |
| p95 batch-layer max/mean | 1.5033 | 1.0352 | -31.14% |
| max batch-layer max/mean | 1.6309 | 1.0536 | -35.40% |
| mean layer-total rank max/mean | 1.1856 | 1.0101 | -14.80% |
| p95 layer-total rank max/mean | 1.2917 | 1.0161 | -21.33% |
| max layer-total rank max/mean | 1.3947 | 1.0278 | -26.30% |

Figures:

- `outputs/asis_tobe_b32_wall/rank_imbalance_by_layer.png`
- `outputs/asis_tobe_b32_wall/rank_load_profile.png`
- `outputs/asis_tobe_b32_wall/hot_layer_rank_load.png`
- `outputs/asis_tobe_b32_final/batch_layer_imbalance_hist_b32.png`

## MoE-Only CUDA Timing

These numbers come from the instrumented CUDA event run. The first call per layer/rank is dropped as warmup. Because vLLM uses chunked prefill, each layer/rank has many recorded calls rather than one call per user batch.

| Metric | As-Is linear | To-Be layer-wise | Change |
|---|---:|---:|---:|
| MoE critical path total | 4588.74 ms | 4159.88 ms | -9.35% |
| mean critical layer-call | 1.2919 ms | 1.1555 ms | -10.55% |
| p95 critical layer-call | 2.9639 ms | 2.8616 ms | -3.45% |

Figures:

- `outputs/asis_tobe_b32_moetiming/moe_cuda_summary.png`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_by_layer.png`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_rank_total.png`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_asis_layer_rank.png`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_tobe_layer_rank.png`

## Interpretation

Batch 32 confirms the placement mechanism works: routed-load straggler p95 drops by 31.14%, worst-case imbalance drops by 35.40%, and the MoE-only CUDA critical path drops by 9.35%.

The clean wall-clock does not improve in this run; it is about 1.00% slower. This is not a contradiction. The measured MoE critical path is about 4.59 s out of 42.40 s As-Is wall time, roughly 10.8% of the clean run. A 9.35% improvement in that slice has an ideal E2E upper bound of about 1.0%, before considering vision encoder time, image preprocessing, attention, scheduler overhead, communication details, chunked-prefill scheduling, and run-to-run noise.

So the current result is:

- Comparable-to-ReaLB/MACS MoE-only timing: improved.
- Routed-load straggler: strongly improved.
- Full vLLM wall-clock/TTFT: not yet improved, likely because the optimized portion is a small fraction of total measured latency and the remaining runtime dominates.

This suggests the next measurement should either isolate MoE layer latency even more directly, or move to a workload/runtime setting where MoE expert compute is a larger fraction of total prefill time. Otherwise, placement gains will remain visible in MoE-only metrics but muted or noisy in full request latency.

## Artifacts

- Final summary: `outputs/asis_tobe_b32_final/summary_b32_final.json`
- Final summary figure: `outputs/asis_tobe_b32_final/summary_b32_final.png`
- Clean timing figure: `outputs/asis_tobe_b32_final/timing_b32_clean.png`
- Clean wall As-Is/To-Be: `outputs/asis_tobe_b32_wall/asis.json`, `outputs/asis_tobe_b32_wall/tobe.json`
- MoE CUDA timing summary: `outputs/asis_tobe_b32_moetiming/moe_cuda_timing_summary.json`
- MoE CUDA raw JSONL: `outputs/asis_tobe_b32_moetiming/asis_moe_cuda_timing.jsonl`, `outputs/asis_tobe_b32_moetiming/tobe_moe_cuda_timing.jsonl`
- Batch-32 placement map: `outputs/main_calib_tail_b32/tail_optimized_map_perlayer.json`
