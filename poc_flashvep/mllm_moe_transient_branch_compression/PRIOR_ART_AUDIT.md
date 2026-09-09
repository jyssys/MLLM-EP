# Prior-art audit: transient visual expert-branch compression

Audit date: 2026-09-10.  The experiment's intended operation is unusually
specific: preserve every token, position, original top-k expert identity, and
router weight, but evaluate a selected same-expert visual branch once and reuse
that expert output for other tokens during the same forward pass.

| Work | Actual unit changed/reused | Exact collision? | Boundary for this PoC |
|---|---|---:|---|
| FastMMoE [1,2] | Prunes/merges visual tokens using routing-distribution similarity and reduces visual expert activation/intermediate width | No | It changes the attention sequence and/or activated expert work; this PoC retains attention tokens and the original top-k route. |
| MoDES [3] | Skips routed experts with layer-importance and modality-specific thresholds | No | It removes selected contributions rather than sharing one evaluation of the same expert across distinct inputs. |
| AnyExperts [4] | Trains variable real/virtual expert slots per token according to semantic importance | No | It changes the model/routing contract and may use zero-compute virtual experts. |
| XShare [5] | Selects a batch-wide expert subset and constrains each token's route to that subset | No, but close in economic objective | It shares expert *selection/weight access*, not `E_e(h)` across tokens while preserving expert identity. |
| SERE [6] | Reroutes secondary slots to functionally similar primary experts | No | The selected expert identity changes; no two distinct inputs share one same-expert evaluation. |
| ACE [7] | Uses offline expert-response proxies plus router weights to skip low-contribution slots | No; directly crowds the S2 child | Any contribution-weighted result that degenerates to skipping must beat this stronger 2026 baseline. |
| MoECa [8] | Reuses expert-branch features across diffusion timesteps in DiT-MoE | **Closest conceptual neighbor** | Same-forward cross-token reuse in autoregressive MLLM is a different reuse axis, but branch-level feature reuse itself is no longer a safe novelty claim. |
| FastV / SparseVLM [9,10] | Removes/recycles visual tokens in the transformer sequence | No | Both alter attention entities; the proposed operation explicitly does not. |
| SpecMoE [11] | Self-assisted speculative decoding for memory-constrained MoE | No | It speculates tokens and verifies them, rather than sharing same-forward expert branch computation. |
| vLLM Omni EP [12] | Standard exact expert dispatch, execute, and combine across GPUs | Baseline | It provides no documented same-forward branch-output reuse primitive. |

## Code-level checks

The official FastMMoE repository was inspected at commit
`b355b15a46601be24e440e1449df82de08e3684d`.  Its DeepSeek-VL2 implementation
truncates the visual expert intermediate width and its merge path deletes or
replaces visual tokens using mean/MLERP-like aggregation.  This is not the
token-identity-preserving operation tested here.

The official SERE repository was inspected at commit
`8512f39b108cd1fa04e9261a5b6846fb173989e4`: secondary expert IDs are mapped to
similar primary experts.  The current PoC never changes expert IDs.

The previous local MoDES audit used official code commit
`933b3ccac3c23e01c763771f1a7d60c4e7ed13a4`.  Its causal unit is dynamic expert
skipping, so contribution-weighted sharing must be reported as an expert-
skipping collision if reuse fails to improve over a zero-output control.

## Novelty gate

The only defensible incremental gap before measurement was:

> Same-forward MLLM visual tokens may preserve their independent attention
> identities while selected same-expert branch outputs are shared transiently.

MoECa means that “expert-branch reuse” alone is not novel.  XShare and ACE mean
that reducing branch rows or low-weight contributions alone is also not novel.
The gap survives only if real MLLM branch outputs exhibit a strong, deployable,
quality-safe cross-token geometry and the one-invocation implementation yields
material request-level benefit.

## Sources

1. [FastMMoE paper](https://arxiv.org/abs/2511.17885)
2. [FastMMoE official code](https://github.com/MindVLA-Team/FastMMoE)
3. [MoDES CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.html)
4. [AnyExperts](https://arxiv.org/abs/2511.18314)
5. [XShare](https://arxiv.org/abs/2602.07265)
6. [SERE](https://arxiv.org/abs/2602.07616)
7. [ACE](https://arxiv.org/abs/2609.05228)
8. [MoECa](https://arxiv.org/abs/2606.15615)
9. [FastV](https://arxiv.org/abs/2403.06764)
10. [SparseVLM](https://arxiv.org/abs/2410.04417)
11. [SpecMoE](https://arxiv.org/abs/2604.10152)
12. [vLLM Omni expert-parallel design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/expert_parallel/)
