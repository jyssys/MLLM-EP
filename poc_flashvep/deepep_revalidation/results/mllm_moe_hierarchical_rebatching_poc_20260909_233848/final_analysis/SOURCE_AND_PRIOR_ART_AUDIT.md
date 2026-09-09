# Source and prior-art audit

## Scope

This PoC is about **closed-set offline/asynchronous batch inference**.  Its
objective is batch completion time (BCT), the interval until every request in
the submitted job completes.  It is not an interactive TTFT/TPOT scheduler
study.  Offline execution permits request reordering and module-boundary
waiting that an interactive service may not tolerate.

## Audit result

| System | Audited mechanism | Relation to this PoC | Collision judgement |
|---|---|---|---|
| MoE-Gen | Module-based attention/MoE batching on one GPU; predecessor to the broader coroutine architecture | Establishes that attention and sparse MoE can prefer different batch sizes | Direct collision for the generic attention→MoE idea |
| BatchGen paper | Sequence coroutines, `YIELD`/`COMBINE`, static intra-forward yield points, attention→MoE regrouping, dynamic inter-forward migration/partition | Strongest baseline for LM-internal rebatching; explicitly names a post-vision-encoder yield as future work | A single vision yield or attention→MoE yield is not novel |
| BatchGen artifact | Paper implementation pin `df221143d0520ea37d2dd35b23f2915ae5f92678` | Reproduction reference | No Qwen3-VL path used here |
| BatchGen current main | Commit `ca2aaca8a24fa110bb48b86f1876fe25b5cf3267`; newer model paths including Kimi-K3 scaffolding | Current implementation was audited separately from the paper artifact | Kimi-K3 explicitly hard-fails on media placeholder tokens; no working Qwen3-VL hierarchical path was found |
| vLLM-Omni | Stage abstraction and asynchronous chunk transfer between Thinker, Talker and Code2Wav | Strong stage-wise MLLM reference | Does not implement an internal Qwen3-VL encoder→LM-attention→MoE regrouping hierarchy |
| ModServe | Disaggregates modality/stage pools and scales them independently | Shows that MLLM stages can merit independent resources in interactive serving | Adjacent, but different objective, deployment scale and boundary |
| ElasticMM | Modality groups, elastic stage partitions, non-blocking encoding and prefix caching | Strong stage-disaggregation baseline | Adjacent; a generic encoder/LM stage split is crowded |
| RPS-Serve | Interactive modality-aware prioritization and anti-starvation | Relevant workload heterogeneity reference | Different metric and no intra-forward MoE hierarchy |
| BlendServe | Offline request reordering balances resource overlap with prefix sharing | Closest offline scheduling reference outside BatchGen | Does not expose exact vision→attention→MoE module yields |
| HeteroServe | Cross-tier modality partitioning for MLLM serving | Confirms stage/modality specialization is crowded | Different cross-tier placement problem |
| M* | Walk-graph abstraction for heterogeneous composite models | Raises the prior-art bar for generic module graphs | A generic “MLLM as stages” framing is no longer a sufficient contribution |

BatchGen is explicitly an offline batch engine and reports up to 2.3× BCT
reduction at 128-GPU scale.[^1] Its sequence-coroutine abstraction already
supports pausing and regrouping work at module boundaries.[^2]  The paper
selects intra-forward yield points statically per model and adopts the
attention/MoE split; it explicitly lists vision-language models and a yield
after the encoder as future work.  Consequently, the only potentially new
claim here would have been **a measured, non-trivial three-way compatibility
conflict** whose adaptive hierarchy beats both an encoder/LM stage split and
BatchGen-style LM rebatching.

vLLM-Omni's current async-chunk mechanism is real, but its documented exemplar
streams Thinker→Talker→Code2Wav stages rather than the internal modules of a
Qwen3-VL Thinker.[^3] ModServe and ElasticMM already establish independent
modality/stage resource management,[^4][^5] while BlendServe already studies
offline reordering under competing resource and prefix objectives.[^6]

## Local source findings

- vLLM 0.20.0's `group_and_batch_mm_kwargs` groups **consecutive** compatible
  multimodal items.  `_execute_mm_encoder` calls this helper, and a local FIXME
  states that output reordering would be the proper general solution.
- Qwen3-VL uses packed variable-resolution vision inputs; there is no basis for
  assuming a simple padded-image tensor whose padding can be freely removed.
- BatchGen current main has a `VISION_MSG` guard in the Kimi-K3 path and raises
  `NotImplementedError` when a media placeholder is observed.
- The tested vLLM path was `DeepEPHTPrepareAndFinalize` plus `TritonExperts`, not
  a TP-only path mislabeled as EP.

## Novelty consequence

There is a genuine implementation gap: current BatchGen does not provide a
working Qwen3-VL hierarchy, and vLLM-Omni's async-chunk abstraction targets
inter-stage composite pipelines.  The measured oracle, however, shows that
this gap is not economically material on the tested four-H100 Qwen workload.
An unimplemented feature is not automatically a research opportunity.

[^1]: [BatchGen, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)
[^2]: [BatchGen official repository](https://github.com/batchgen-project/batchgen)
[^3]: [vLLM-Omni async chunk design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk.md)
[^4]: [ModServe](https://arxiv.org/abs/2502.00937)
[^5]: [ElasticMM](https://arxiv.org/abs/2507.10069)
[^6]: [BlendServe](https://arxiv.org/abs/2411.16102)
[^7]: [RPS-Serve](https://arxiv.org/abs/2603.26498)
[^8]: [HeteroServe](https://arxiv.org/abs/2603.12707)
[^9]: [M*, modular serving for multimodal models](https://arxiv.org/abs/2606.12688)
