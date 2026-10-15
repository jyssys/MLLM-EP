# Prior overlap and novelty boundary

| Prior | Boundary |
|---|---|
| [Ouro](https://arxiv.org/abs/2510.25741) | Supplies the pretrained shared core. We do not change training, recurrence, or inference quality settings. |
| [LoopSpec](https://arxiv.org/abs/2609.17184) | Uses intermediate-depth drafts and full-depth verification. This experiment has no speculative future token. |
| [WaveFront](https://arxiv.org/abs/2609.23033) | Batches mixed-depth speculative token states and correction. Our chunks contain known prompt tokens with exact causal KV. |
| [Generic PP](https://arxiv.org/abs/1811.06965) | Stage/microbatch scheduling already supplies the same depth×chunk graph after logical unrolling. GPipe is a training reference, not a serving benchmark here. |
| [Chunked prefill](https://arxiv.org/abs/2403.02310) | Existing serving systems split prompt work and schedule it with other work. Native TP4 chunking is a required physical control. |
| [vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt/tree/9d601ceaf9af63121c742df5f584467e23630a0b) | Earlier audit pins this separate loop-serving engine. This PoC does not assume its scheduler exists in native vLLM. |
| [FlashLoop](https://arxiv.org/abs/2609.29812) | Lazy/sparse updates and residual compression are outside this exact dense-execution study. |
| [DistriFusion](https://arxiv.org/abs/2402.19481) | Image-diffusion feature reuse is an analogy. It does not license stale causal AR activations. |
| [PipeFusion](https://arxiv.org/abs/2405.14430) | Patch/denoising pipelines target diffusion Transformers. Their semantics are not an exact Ouro prefill baseline. |

The exact macro DAG is isomorphic to generic PP on the tied-unrolled model.
Shared immutable weights do not remove a causal attention edge.
Stage-local KV does not introduce a remote-KV transfer that this method can hide.
No recurrent-specific mechanism survives this graph comparison yet.
Any calibrated physical improvement therefore remains generic engineering headroom, not a new loop-specific contribution.
