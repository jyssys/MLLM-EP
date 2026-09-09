# Isolated official dev-h100 compatibility work

- Native CMake explicitly pins the isolated Python 3.10 interpreter. Automatic discovery initially selected the host Python 3.14 interpreter; those binaries were never used as research evidence.
- Official Qwen1.5 MoE entry retained the pre-PP `DistKVPool(num_layers, ...)` call, while the shared class now requires `(start_layer_idx, end_layer_idx, ...)`. All four workers failed at the same argument binding before model execution. Add `start_layer_idx=0` at this single call site. The layer range remains the full 24 layers; this is API compatibility, not scheduling or model math modification.
- Official model/tokenizer paths point at another host's `/code/hf`. The harness substitutes our pinned local snapshot and verified complete official weight-packing cache only.
- BF16 is not supported by this version's native NCCL wrapper. The supported official smoke uses FP16 and is not a BF16 cross-system latency claim.
- Qwen's obsolete `is_cuda_graph_enabled`/`is_auto_search_enabled` attributes are mapped to the current base class's `cuda_graph_enabled`/`auto_search_enabled`; the KV-cache update keyword is likewise renamed. Both features remain disabled in correctness smoke. Failure logs preserve the original exceptions.
- Runtime FlashInfer JIT requires the isolated environment's `ninja` and CUDA 12.8 compiler on PATH, not merely launching its Python by absolute path. The native launcher pins both explicitly.

## Native nano-split regression (18:55–18:59 KST)

The unsplit eager, CUDA graph and separate-stream one-part plans all pass the
same four-request/82-token HF reference. The first two-part native split failed
at the shared-expert sigmoid activation. `Activation.copy_nano` constructed a
new Activation without forwarding `act_fn`, silently selecting `silu_mul`.
The gate tensor has last dimension one, making that incorrect operation fail.

`test_activation_copy.py` executes the actual method's AST with a light factory
fixture: RED is `('sigmoid', 'silu_mul')`. Forwarding `act_fn=self.act_fn` makes
sigmoid/SiLU/SiLU-multiply property-preservation tests pass. Fresh native workers
then pass 82/82 HF tokens for both two-part eager and two-part CUDA graph plans.

This is a one-line compatibility/correctness repair, not a novel successor
failure or a performance contribution. Arbitrary shapes and searched plans
remain unverified. The original failure log is retained.
