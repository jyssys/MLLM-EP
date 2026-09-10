# Vision-token Heterogeneous Top-K

## Decision

**Gate B: NO-GO. Gate C mechanism: positive but not quality-matched. Gate D:
NO-GO.** EP-tail-aware removal is causally better at reducing max-rank work at
the same assignment count, but the useful timing points incur very large local
MoE output error. At plausibly mild error, projected request gain is too small.

No full variable-k vLLM path or benchmark run was implemented because the
pre-specified replay/economic gate failed.

## Exact policy definitions

All policies preserve every text-token branch and each vision token's top-1.
For a fixed number of removable vision assignments:

- **P1 random:** uniformly selects eligible branches.
- **P2 semantic:** removes globally lowest router-weight branches.
- **P3 EP-tail-aware:** repeatedly selects the lowest-weight eligible branch
  that resides on the currently most-loaded EP rank. Rank load is updated
  after each removal.

Thus P1/P2/P3 have identical assignment budgets. P3 changes only the coupling
between semantic risk and the current EP critical rank.

## Fresh routing evidence

The analysis uses nine real images, three resolutions (100/196/441 vision
tokens), three content classes, and early/middle/late layers. The causal route
oracle has 54 image-layer observations.

| Vision assignment budget | Random max-rank reduction | Semantic | EP-tail-aware | P3/P2 |
|---:|---:|---:|---:|---:|
| 1% | 0.97% | 0.78% | 2.77% | 3.55x |
| 2% | 1.83% | 1.51% | 5.45% | 3.61x |
| 5% | 4.36% | 4.11% | 12.42% | 3.02x |
| 10% | 8.66% | 8.12% | 19.48% | 2.40x |
| 20% | 17.77% | 16.19% | 29.56% | 1.83x |
| 50% | 43.29% | 41.94% | 53.42% | 1.27x |

At 10%, P2 drops median router mass 5.88%, while P3 drops 6.47%. P3's large
load benefit is therefore not free: the overloaded rank does not necessarily
contain the globally least important branches.

The intended geometry hypothesis is nevertheless real. P3 exceeds the
pre-specified 1.5x max-rank-reduction target at budgets up to 20%, and the
effect is a median across fresh layers/images rather than one maximum.

## Surprise: vision routing is less concentrated

| Modality | Router entropy | Top-1 mass | Top-2 | Top-4 | Top-6 | Top-7 |
|---|---:|---:|---:|---:|---:|---:|
| Vision | 2.068 | 16.35% | 30.63% | 55.90% | 78.72% | 89.59% |
| Text | 1.985 | 22.64% | 40.66% | 66.59% | 85.23% | 92.83% |

This reverses a key premise behind cheap vision-only reduction on this model:
Qwen3-VL's top-8 vision mass is flatter than text after conditioning on the
same captured layers. Keeping only six vision experts discards a median 21.3%
of normalized route mass even before propagation through later layers.

## Exact GPU replay

The replay uses captured Qwen3-VL hidden vectors/routes, the actual layer
expert weights, DeepEP HT, and the Triton fused-expert implementation. Each
condition has five warmups and 30 randomized-order same-device-event
measurements. M=8192 is a tiled stress replay of the real captured distribution;
it is not claimed as a full-model variable-k request.

### Fixed k, camera, layer 24, M=8192

| Policy | MoE reduction | Worst-rank relative L2 | Projected clean TTFT gain |
|---|---:|---:|---:|
| Semantic k=6 | 9.69% | 22.20% | 6.03% |
| EP-tail k=6 | 13.79% | 22.62% | 8.58% |
| Semantic k=4 | 20.84% | 33.92% | 12.96% |
| EP-tail k=4 | 27.71% | 34.72% | 17.23% |
| Semantic k=2 | 40.52% | 48.86% | 25.20% |
| EP-tail k=2 | 45.83% | 49.03% | 28.50% |
| Semantic k=1 | 57.07% | 60.11% | 35.49% |
| EP-tail k=1 | 58.89% | 60.29% | 36.63% |

The attractive k=4 timing point is unusable under the local quality screen.
Renormalizing retained weights did not repair this frontier; it generally
increased error.

### Exact budgets across three contents, layer 24, M=8192

| Policy | Median MoE reduction | Median worst-rank relative L2 | Projected TTFT |
|---|---:|---:|---:|
| P2 semantic, 10% | 7.22% | 10.09% | 4.49% |
| P3 tail-aware, 10% | 11.17% | 13.16% | 6.94% |
| P2 semantic, 20% | 10.18% | 18.96% | 6.33% |
| P3 tail-aware, 20% | 13.34% | 19.69% | 8.30% |

P3 improves the median projected TTFT by about 2.46 percentage points over P2
at the 10% budget, but its local output error is worse. At 20%, it merely
touches the lower economic gate with about 19.7% local relative-L2. This is not
a quality-matched advantage.

### Mild-budget falsification

At M=8192/layer 24, 1% semantic removal yielded only 0.38% MoE gain with
1.55% local relative-L2; 1% P3 yielded 1.39% MoE gain with 4.08% relative-L2.
At 5%, P3 reached 5.75% MoE / 3.57% projected TTFT but already had 9.51%
relative-L2. Even the 1% policies failed the specified <=1% local relative-L2
screen.

The P3 timing direction reproduced at layers 4, 24, and 44 and on camera,
cell, and chessboard images. The failure is therefore not lack of mechanism
repeatability; it is the joint quality/economic Pareto.

### Full-model logit propagation

A separate correctness-only Hugging Face run masks the selected branch
weights in all 48 decoder layers while retaining all expert computations. It
uses the same Qwen3-VL checkpoint but is deliberately excluded from timing.

| Policy | Budget | First-token match | Four-token exact match | Median logit relative L2 | Median KL | Median max-rank reduction |
|---|---:|---:|---:|---:|---:|---:|
| P2 semantic | 10% | 9/9 | 9/9 | 4.23% | 0.00185 | 8.37% |
| P3 tail-aware | 10% | 9/9 | 6/9 | 6.57% | 0.00106 | 18.45% |
| P2 semantic | 20% | 9/9 | 7/9 | 5.39% | 0.00143 | 16.60% |
| P3 tail-aware | 20% | 9/9 | 6/9 | 4.38% | 0.00281 | 29.48% |

The first generated token alone is too weak a quality test—it stayed equal in
all conditions. By four tokens, P3 diverges on one third of the examples even
at 10%, while P2 remains exact at the same budget. This confirms the key
same-budget conflict in the requested end-to-end representation: P3 buys rank
balance by selecting branches that are not as semantically safe as P2's.

The nine examples are an early kill diagnostic, not benchmark accuracy. A
large evaluation would only be justified if the economic gate were strong;
the median 10% P3 projection is still 6.94%.

## Amdahl evidence boundary

Projected TTFT is calculated as measured replay MoE reduction multiplied by
the 16K observer-assisted MoE share (62.19%). It is intentionally optimistic:
selector construction, variable-k packing, metadata, and full-model error
propagation are absent. It must not be read as an observed E2E speedup.

Because the optimistic median is only 6.94% at 10% removal—and quality is
already less stable than P2—a runtime implementation cannot plausibly satisfy
the 8--10% minimum without crossing the quality boundary. The 20% point is a
weak 8.30% projection, not a 12--15% strong signal, and four-token exact match
is only 6/9. Per contract, Gate E was not entered.

## Prior-art attack

- **MoDES** already performs modality- and layer-aware per-token expert
  skipping using expert importance [2]. P2 therefore has no novel core.
- **AnyExperts** already allocates a variable number of real expert slots by
  token semantic importance under a budget [3]. A generic “vision can use
  fewer experts” claim directly overlaps it.
- **MACS** is the closest systems collision: it combines entropy-weighted
  visual-token importance with dynamic modality-aware expert capacity to
  address EP stragglers [1]. P3 differs narrowly by deleting individual
  low-weight branches on the current critical rank instead of applying its
  published capacity/rerouting policy. Since P3 does not establish a safe
  quality/economic frontier, that distinction is not sufficient for a paper.
- **ReaLB** attacks overloaded vision-heavy ranks with per-rank lower-precision
  expert execution rather than branch removal [4]. Its reported 1.29x layer
  speedup with bounded accuracy loss illustrates a more plausible way to keep
  route mass while reducing hot-rank cost.

## Final answer to the hypothesis

At a fixed compute budget, EP-tail awareness does reduce critical-rank work
more than semantic-only selection. But there is no demonstrated
**same-quality** budget where this becomes an economic request-level gain.
The exact requested contribution is therefore falsified on the present Qwen3-VL
runtime.

## Evidence

- `results/final_analysis/fresh_route_policy_rows.csv`
- `results/final_analysis/router_mass_distribution.csv`
- `results/final_analysis/replay_policy_summary.csv`
- `results/final_analysis/ttft_projections.csv`
- `results/final_analysis/plots/max_rank_reduction_vs_budget.png`
- `results/final_analysis/plots/router_cumulative_mass.png`
- `results/final_analysis/plots/projected_ttft_quality_pareto.png`
- `results/logit_correctness_nine_20260911_0202.summary.json`

## Sources

1. [MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference, ACL 2026](https://aclanthology.org/2026.acl-long.1012/)
2. [MoDES: Dynamic Expert Skipping, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.html)
3. [AnyExperts: On-Demand Expert Allocation, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_AnyExperts_On-Demand_Expert_Allocation_for_Multimodal_Language_Models_with_Mixture_CVPRF_2026_paper.html)
4. [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503)
