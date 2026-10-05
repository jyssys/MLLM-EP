# Prior overlap audit

Read against the linked original methods on 2026-10-06, before implementation.
These are literature comparisons, not executed system reproductions.

| Source / inspected method | Regime and established mechanism | Boundary here |
|---|---|---|
| [Epoch v1, §4.3–4.4 / Algorithm2](https://arxiv.org/html/2609.09748v1) | Block diffusion: fresh live/newly decoded sparse work; stable decoded-output cache with refresh; merges a full logical shard before downstream consumers | No Epoch runtime available in the pinned bridge. No stale-live or decoded-output caching introduced. Old5.773× is structural accounting, **not Epoch**. Earlier prefix work must still obey full logical state dependencies. |
| [MPK v2, §3–5,6.4–6.5](https://arxiv.org/html/2512.22219v2) | Compiler-generated SM task graph; event-driven cross-operator scheduling; MoE routing/dispatch/expert/combine and distributed transfer tasks | Fine-grained dependency scheduling already exists. Porting a compiler is out of scope. Same B1 bidirectional diffusion/cache seam is not demonstrated by the inspected scope. |
| [TokenWeave v3, §3.1–3.2](https://arxiv.org/html/2505.11329v3) | TP inference, explicitly **B1 sequence splitting**; wave-aware split, causal prefix-first order, fused residual/norm collectives | B1 is not novelty. Its causal ordering is not valid for a fresh bidirectional current block. Correct fixed-group scheduling is a mandatory control. |
| [SGLang EP, TBO/SBO](https://docs.sglang.io/docs/advanced_features/expert_parallelism) | Cross-microbatch attention/communication; same-batch shared-expert/communication overlap | Pinned0.5.3 manual runner does not execute current serving TBO/SBO flag plumbing. Test a bounded shared-expert adaptation separately; never credit it to prefix overlap. |
| [COMET, §3.1–3.2](https://arxiv.org/html/2502.19811) | Shared-tensor decomposition, grouped expert GEMM/A2A overlap, adaptive GPU resource assignment inside MoE | Intra-MoE improvement is prior, not this cross-layer claim. |
| [Weave v1, §3](https://arxiv.org/html/2609.21483v1) | Routing-derived spatial SM partition and temporal chunk scheduling in MoE megakernel, distributed BF16 prefill | Strong generic resource/contention precedent; does not establish this native immutable-prefix/fresh-block seam. Public implementation unavailable in inspected paper; NOT_RUN. |
| [BPT, §3 / Algorithm1](https://arxiv.org/html/2305.19370) | Blockwise attention and per-query-block FFN; global normalization statistics across KV partitions | Decomposition itself is prior. Different training/long-context scope; no claimed superiority. |
| [FlashAttention, §3](https://arxiv.org/html/2205.14135) | Exact tiled attention with online softmax normalization and IO-aware layout | Stable partial-state merge is a known correctness primitive, not contribution. |

Status: substantial generic mechanism overlap; no verified exact duplicate of
this specific native block-diffusion EP/TP visibility experiment. This does not
establish novelty or performance superiority over MPK/TokenWeave/Epoch.
EPOCH_RUNTIME_NOT_RUN; MPK_RUNTIME_NOT_RUN; TOKENWEAVE_RUNTIME_NOT_RUN.
