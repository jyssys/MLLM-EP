# Dependency graph

- `VISION_ENCODER->LLM_PREFILL`: **HARD_DEPENDENCY**
- `LLM_PREFILL->DEEPEP_DISPATCH`: **HARD_DEPENDENCY**
- `DEEPEP_DISPATCH->EXPERT_GEMM`: **HARD_DEPENDENCY**
- `EXPERT_GEMM->DEEPEP_COMBINE`: **HARD_DEPENDENCY**
- `pending_request_VISION_ENCODER->current_request_DEEPEP_COMM`: **CROSS_REQUEST_INDEPENDENT**
- `image_i_encoder->image_j_encoder`: **CROSS_IMAGE_POSSIBLE**

Classes reflect source audit and cross-request semantics; absent full-serving NVTX prevents timing-derived dependency proof.
