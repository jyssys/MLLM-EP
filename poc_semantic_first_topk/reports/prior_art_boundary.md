# Prior-art boundary

## Claim under test

The narrow candidate claim is not that visual tokens can use fewer experts.
It is that, after a semantic policy has fixed a quality-safe total assignment
budget, the omissions can be rearranged using the *current physical EP-rank
critical path* to lower real multi-GPU latency at the same compute and task
quality.

## Adversarial comparison

| Work | Already establishes | Residual distinction tested here | Collision risk |
|---|---|---|---|
| [AnyExperts](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_AnyExperts_On-Demand_Expert_Allocation_for_Multimodal_Language_Models_with_Mixture_CVPRF_2026_paper.html) | Per-token semantic importance can allocate a variable number of real expert slots; it reports comparable accuracy with 40% fewer real activations on general image/video tasks and 10% fewer on OCR/NLP. | Assignment-count-preserving redistribution of omissions against the current EP rank load. | High for Stage 1; lower for the narrow Stage 2 mechanism. |
| [MoDES](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.html) and its [official code](https://github.com/ModelTC/MoDES) | Training-free, modality- and layer-aware dynamic expert skipping; Qwen3-VL and Kimi evaluation; reported 2.16x prefill speedup. | Explicit physical-rank critical-path objective at a fixed retained-assignment budget. | High for semantic adaptive K and all generic speed claims. |
| [MACS](https://aclanthology.org/2026.acl-long.1012/) | Entropy-weighted visual-token semantic load, modality-adaptive expert capacity, local semantic rerouting, and fail-safe dropping specifically to mitigate MLLM EP stragglers; evaluated Qwen3-VL and Kimi on EP. | Prefix-valid per-token K swaps preserve the original experts and exact total assignment count, rather than capacity overflow/rerouting. | **Very high.** The problem statement and semantic-first/physical-load coupling are already central to MACS. A different primitive is not enough without a clearly superior frontier or new causal fact. |
| [ReaLB](https://arxiv.org/abs/2604.19503) | Makes overloaded EP-rank expert work cheaper using per-rank dynamic precision while retaining routes; reports 1.29x layer speedup. | This PoC omits low-risk suffix branches and leaves precision/placement unchanged. | Adjacent, not exact; it may dominate when route mass cannot safely be removed. |
| [ACE](https://arxiv.org/abs/2609.05228) | Uses offline expert-weight/response proxies plus router gates for calibration-free token-adaptive expert skipping, always retaining top-1. | Multimodal task-conditioned safety and a physical EP-rank objective. | High for the contribution-proxy/selector part, even though ACE is text-only. |

## Novelty gate

Stage 1 alone cannot support a new contribution.  The candidate survives only
if Stage 2 gives a robust same-quality, same-compute latency advantage that is
not recovered by MACS-style capacity handling.  Because MACS already couples
visual semantic weights to EP straggler mitigation, a count-only 1.5x
max-rank result is insufficient: the advantage must appear in real DeepEP
critical latency and materially improve the request-level Pareto frontier.

## Evidence timestamp

The primary papers, official project pages, and current ACE preprint were
checked on 2026-09-11.  This document treats the published mechanisms—not
paper titles or abstracts alone—as the novelty boundary.
