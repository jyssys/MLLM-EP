# Prior art and novelty

Current [DeepEP V2](https://github.com/deepseek-ai/DeepEP) has replaced the
legacy Normal/LL split with a unified `ElasticBuffer`, switched to NCCL Gin,
and reports up to 1.3x V1 performance with fewer SMs. The legacy
[DeepEP documentation](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
already exposes two-micro-batch overlap hooks and explicitly discusses fixed
maximum-capacity buffers. These facts make variable capacity or double-buffer
cleanup alone weak novelty.

[StreamEP](https://github.com/evolutionaryscale/StreamEP) overlaps expert-major
tile arrival, GEMM, and combine with release/acquire signaling. It occupies a
different training-oriented integration point but collides with generic claims
about streaming EP transactions. [FlashInfer's current EP architecture](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md)
also provides LL/HT split transports and mega-kernel integration, while
[NCCL EP](https://arxiv.org/abs/2603.13606) exposes LL and HT dispatch/combine
through NCCL Device API.

The proposed differentiator would have to be a reproducible dLLM-refinement
serving failure that these general systems do not address. This PoC did not
find that failure: non-stationary M and Q did not reverse the strongest valid
legacy LL path, and the structural oracle is too small. Therefore there is no
defensible paper-level RefineStreamEP kernel target on this evidence.
