# Bottleneck Reproduction

## Decision

**Gate A: GO.** The current four-H100 Qwen3-VL runtime contains a real
long-prefill regime in which expert compute is a large part of clean TTFT and
rank imbalance has a nontrivial, although not dominant, analytical tail.

This finding says that MACS/ReaLB-like work-altering methods have an economic
target on this machine. It does **not** say that heterogeneous top-k can safely
recover that target.

## Reproducibility contract

- Source checkout: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`.
- Model: Qwen3-VL-30B-A3B-Instruct, BF16.
- Runtime: vLLM 0.20.0, TP2 / DP2 / EP4, DeepEP high-throughput,
  Triton unquantized experts, DBO off, EPLB off, eager execution.
- Hardware: only physical H100 GPUs 4,5,6,7; full NVLink/NV18 connectivity.
- Clean runs: one warmup and three repetitions per prompt length.
- Stage timing: same-device CUDA events; critical rank selected per logical
  invocation; 48 sequential MoE layers summed once per DP request.
- Attention timing: a separate nonblocking-event run with three measured
  repetitions after warmup.

## Clean TTFT crossover

| Prompt tokens | Median TTFT (ms) |
|---:|---:|
| 256 | 102.759 |
| 512 | 98.414 |
| 1,024 | 100.266 |
| 2,048 | 103.370 |
| 4,096 | 104.583 |
| 8,192 | 185.613 |
| 16,384 | 384.937 |

The 8K and 16K cases are the economic regimes used for the policy gate. The
shape transition is visible in clean request time, not only in an operator
microbenchmark.

## Critical-path attribution

The table joins observer-heavy same-device stage durations to a separate clean
TTFT measurement. Percentages are consequently **observer-assisted Amdahl
attribution**, not a claim that the observer-heavy request itself was clean.

| Prompt | Local routed M | MoE (ms / TTFT) | Expert (ms / TTFT) | Dispatch (ms) | Combine (ms) | Rank max/mean | Perfect balance upper bound / TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 60.842 / 59.21% | 22.757 / 22.15% | 7.939 | 2.385 | 1.36 | 5.58% |
| 2,048 | 1,024 | 71.205 / 68.88% | 34.274 / 33.16% | 7.661 | 0.187 | 1.38 | 8.82% |
| 8,192 | 4,096 | 134.106 / 72.25% | 73.744 / 39.73% | 8.501 | 0.153 | 1.38 | 10.58% |
| 16,384 | 8,192 | 239.397 / 62.19% | 137.660 / 35.76% | 7.427 | 0.150 | 1.38 | 9.45% |

The perfect-balance column estimates the count-proportional expert time above
the mean rank load. It is an optimistic upper bound: kernel shape, padding,
communication, and selector cost prevent all of it from being removable.

Median natural-route statistics were stable with scale: max-rank/mean was
1.26--1.30 in the raw route rows, rank-load CV 0.23--0.25, and active experts
78 at M=128 and about 90--95 at larger M. The bottleneck is therefore not a
single pathological maximum.

## Attention control

| Prompt | 48-layer critical attention (ms) | Share of clean TTFT | Observer tax |
|---:|---:|---:|---:|
| 256 | 31.989 | 31.13% | +7.09% |
| 8,192 | 48.306 | 26.03% | -0.47% |
| 16,384 | 127.778 | 33.19% | ~0.00% |

At 8K, attention plus observer-assisted MoE numerically reaches about 98.3% of
clean TTFT; at 16K it reaches about 95.4%. This supports their dominance, but
it is not treated as a closed decomposition because the MoE observer itself
has substantial tax.

## Instrumentation sanity

The full stage observer raised request TTFT by 120--609%, so its own request
latency is rejected as primary evidence. It remains usable for stage ordering
and same-device duration attribution. The dedicated attention hook had near
zero tax in the two promoted long-context regimes.

Two earlier low-overhead hook attempts did not attach to the active attention
path; they are explicitly logged as instrumentation failures and excluded.

## Answer to the bottleneck question

The 4-GPU environment does reproduce the prerequisite regime:

1. long-prefill expert compute is 35.8--39.7% of clean TTFT;
2. route-count imbalance implies a 9.5--10.6% optimistic TTFT tail;
3. the result persists at both 8K and 16K and across all 48 layers;
4. therefore a work-reducing/load-aware method can matter here.

The bottleneck alone is not novel. MACS explicitly targets modality-aware
capacity and EP stragglers [1], while ReaLB changes per-rank precision for
overloaded vision-heavy ranks [4]. The successor gate is whether assignment
selection gives a better quality/critical-path Pareto than those adjacent
ideas; the companion report finds that it does not.

## Evidence

- `results/bottleneck_clean_20260911_0041/summary.json`
- `results/bottleneck_analysis_20260911_0051/summary.json`
- `results/bottleneck_attention_20260911_0138/summary.json`
- `results/final_analysis/layer_bottleneck.csv`
- `results/final_analysis/plots/stage_fraction_vs_scale.png`
- `results/final_analysis/plots/expert_and_tail_headroom.png`

## Sources

1. [MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference, ACL 2026](https://aclanthology.org/2026.acl-long.1012/)
2. [MoDES: Dynamic Expert Skipping, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.html)
3. [AnyExperts: On-Demand Expert Allocation, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_AnyExperts_On-Demand_Expert_Allocation_for_Multimodal_Language_Models_with_Mixture_CVPRF_2026_paper.html)
4. [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503)
