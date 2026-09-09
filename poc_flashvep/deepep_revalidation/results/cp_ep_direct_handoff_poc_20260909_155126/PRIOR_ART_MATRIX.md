# Prior-Art Matrix

| Work/runtime | Scope and broken assumption | Exact CP-output to EP-input block streaming? | Exact direct CP-to-expert transport? | Collision assessment |
|---|---|---:|---:|---|
| NanoCP, arXiv:2605.21100 | Dynamic request-level CP for data/expert-parallel long-context decoding | No | No | Adjacent decode scheduling; not this prefill boundary |
| Helix Parallelism / HOP-B, arXiv:2507.07120 | KV-parallel long-context decode and batchwise overlap with TP or TP×EP | No | No | Adjacent overlap, different phase and execution contract |
| MoE Parallel Folding, arXiv:2504.14960 | Decouples attention and MoE parallel dimensions in training | Not found | Not found | Adjacent topology composition, not cross-boundary handoff |
| DeepSpeed Ulysses | Sequence-to-head A2A, attention, return A2A to sequence/full-hidden layout | No | No | Supplies the baseline layout contract |
| Megatron CP + token dispatcher | Ring/A2A CP plus a separate MoE dispatcher | No | No | Supplies production-style CP/EP composition |
| SGLang CP+MoE | CP-local tokens are gathered before MoE and sliced after MoE | No | No | Direct evidence that a materialized bridge remains |
| vLLM PCP+MoE | PCP hidden/router all-gather then MoE, MoE output reduce-scatter | No | No | Direct evidence of a bridge, but standard PCP attention unsupported locally |
| TensorRT-LLM Ulysses/Helix | CP execution; Helix is decode-oriented and backend support varies | Not found | Not found | No direct collision found |

## Audit conclusion

No audited work/source directly implements the two exact long-prefill counterfactuals. Absence of a prior-art collision did not rescue the candidates: streaming failed the actual cost/correctness gates, while direct transport failed the exact algebraic lower bound.

Primary sources:

- NanoCP: <https://arxiv.org/abs/2605.21100>
- Helix Parallelism: <https://arxiv.org/abs/2507.07120>
- MoE Parallel Folding: <https://arxiv.org/abs/2504.14960>
- DeepSpeed Ulysses: <https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md>
- DeepSpeed sequence parallelism: <https://www.deepspeed.ai/tutorials/ds-sequence/>
- Megatron context parallelism: <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html>
- Megatron MoE: <https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html>
