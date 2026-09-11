# Prior-art boundary

This is an adversarial boundary audit, not a novelty claim. Both candidates fail
the economic gate, so no paper contribution is asserted.

| Work | Problem and signal | Approximation / placement | Relationship to this PoC |
|---|---|---|---|
| [dInfer](https://arxiv.org/abs/2510.08666) | dLLM inference framework and block/threshold decoding | Base runtime | Substrate only. This PoC additionally proves that its stock four-GPU LLaDA-MoE command lacks a live A2A dispatch manager. |
| [TEAM](https://arxiv.org/abs/2602.08404) | Temporal and spatial consistency guides expert activation and decoding | Changes activated work; approximate with negligible reported loss | Directly owns temporal-routing consistency. 1A would only differ by exact immutable GPU replica lifetime/prefetch; 1B by request grouping. |
| [Epoch](https://arxiv.org/abs/2609.09748) | Compiles a diffusion block and executes fresh/live token-expert work | Removes redundant dead-position work | Stronger block-level systems framing. 1A/1B preserve required router and expert work, but their measured headroom is much smaller. |
| [DES](https://arxiv.org/abs/2602.00879) | Sequence-level expert coreset combats expert explosion | Changes selected expert set | Not exact and not physical replica placement/batching. It already captures the larger “reduce active expert work” opportunity. |
| [TIDE](https://arxiv.org/abs/2605.20179) | Temporal stability drives interval-based expert placement refresh for constrained devices | Lossless CPU↔GPU offload/placement | Closest dLLM temporal-placement collision for 1A. 1A's narrow residual gap is transient additional GPU↔GPU replicas, but the measured E2E ceiling is 2.63%. |
| [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html) | Staleness-centric synchronization/communication optimizations for parallel diffusion MoE | Consumes stale or selectively synchronized state | Covers cross-step overlap and synchronization, not immutable exact weight copies. It makes a generic “dLLM overlap” novelty claim unsafe. |
| [Predictive Prefetching and Expert Replication](https://arxiv.org/abs/2605.11537) | Predicts overloaded experts for upcoming batches and dynamically replicates them | Generic MoE replication; reported 90–95% task performance | Direct generic-method collision risk for 1A. dLLM multi-iteration lease amortization is a narrower distinction, not a clean blue ocean. |
| [XShare](https://arxiv.org/abs/2602.07265) | Batch-aware collaborative expert sharing | Changes in-batch expert selection/work | Adjacent to batch composition, but not exact rank-complementary grouping. It again targets a larger work-removal opportunity than 1B's 2.36% absolute bound. |

## Bottom line

The broad observations—temporal expert stability, diffusion-block reuse, expert
placement, dynamic replication, and batch-aware expert sharing—are already
crowded. The exact variants tested here remain technically distinguishable:

- 1A is exact, GPU↔GPU, additive transient replication with measured prefetch
  amortization.
- 1B preserves routing and only composes requests by predicted physical rank
  vectors.

Technical distinction is not enough. With absolute request-level ceilings of
2.63% and 2.36%, respectively, neither residual gap supports a paper-level
method in the tested regime.
