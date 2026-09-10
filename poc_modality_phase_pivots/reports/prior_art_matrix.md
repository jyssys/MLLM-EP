# Prior-art matrix

| Pivot | Closest primary work | Collision/risk | What would have been distinct | Outcome |
|---|---|---|---|---|
| Full-unit resource co-scheduling | [NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan), [COMET](https://seed.bytedance.com/en/public_papers/comet-fine-grained-computation-communication-overlapping-for-mixture-of-experts), [FLUX](https://arxiv.org/abs/2406.06858), [Lancet](https://proceedings.mlsys.org/paper_files/paper/2024/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html) | High broad overlap | Content-conditioned vision/text resource complementarity without token/kernel fragmentation | Causal prerequisite absent |
| Modality-aware layer/phase scheduling | [Layered Prefill](https://arxiv.org/abs/2510.08055), [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [FastPP](https://github.com/Sys-KU/FastPP) | High | MLLM modality label changes full-unit compatibility and yields a material incremental oracle | Incremental oracle 0.13% TTFT |
| Phase-specific EP policy | [DeepEP](https://github.com/deepseek-ai/DeepEP/blob/main/README.md), [Analytical Resource Management](https://arxiv.org/abs/2609.07536), [COMET](https://arxiv.org/abs/2502.19811) | Very high for dynamic SM/resource selection | Vision/text/decode require orthogonal policy optima with >=10% best-static regret | TTFT oracle 0.71% |

This audit is adversarial rather than a novelty claim. Since every economic gate
failed, no exhaustive claim of absence from all literature is made.
