# Bounded original-Libra MLLM bridge — not yet validated

The supplied SGLang0.4.10 predates its Qwen3-VL model entry. Do not interpret
that missing entry as a Libra algorithm failure. A bounded numerical bridge is
being prepared without altering author planner/expert/communication code.

1. Source Qwen3-VL BF16 text-stack weights are translated by name and split from
   packed tensors to native individual-expert loader inputs, without arithmetic.
2. Captured exact real-image input embeddings, MRoPE cos/sin and the three
   DeepStack tensors are injected at their original semantic boundaries.
3. Only exact equal-length natural request groups are eligible for the initial
   fixed-shape author benchmark. Do not introduce fake image tokens or hidden
   routing padding to manufacture source symmetry.
4. Compare native vanilla logits to the original full HF Qwen-VL forward, then
   native Libra to native vanilla. Reject the port if it changes model semantics.
5. Native timing is post-vision request prefill/first-token latency. Captured
   visual encoding is NOT free in a full-serving claim. No request-E2E speedup
   may be inferred by relabeling a cached-embedding LM-only benchmark.
6. This is baseline adaptation, not a successor method. If it becomes a large
   framework rewrite, stop the port and explicitly leave full-serving evidence
   unavailable rather than attributing port limitations to Libra.
