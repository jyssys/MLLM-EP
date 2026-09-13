# Stage 2 prior-art audit

| work | relevance | boundary |
|---|---|---|
| [DICE](https://arxiv.org/abs/2411.16786) | interweaved diffusion-MoE parallelism, selective synchronization, conditional communication | explicitly permits activation staleness; this PoC requires exact semantics |
| [Sangam](https://arxiv.org/abs/2607.04206) | dLLM serving schedules repeated block-sized prefill/decode work and studies interference | adjacent phase multiplexing; not MoE sub-operation overlap |
| [Serving Masked Diffusion LLMs](https://arxiv.org/abs/2608.23807) | real-hardware dLLM batching/step characterization | reinforces that batching and denoising-step scheduling are existing serving dimensions |
| [SGLang TBO](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/batch_overlap/operations_strategy.py) | operation graphs yield between dispatch/combine halves and independent microbatches | direct collision for resource-blind cross-wave overlap |
| [X-Stage](https://arxiv.org/abs/2607.23264) | diffusion inference plus post-issue pipeline modeling | direct adjacency for phase/resource-aware issue scheduling |

A novel Stage 2 direction would require refinement state to predict a different
overlap policy that materially beats a generic resource-aware scheduler. The
measured phase fractions and cross-state replay do not meet that standard.
