# Runtime and source audit

## Repository baseline

- Remote: `https://github.com/jyssys/MLLM-EP.git`
- Inspected HEAD: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`
- HEAD subject: `research: evaluate transient visual expert branch sharing`
- The original checkout was dirty, so the PoC ran in the isolated worktree
  `/home/esjung/MLLM-EP-attention-wavefront` on branch
  `flashvep/attention-moe-wavefront-poc`.
- No file below the existing `poc_flashvep/` tree was edited.

## Installed runtime

- Qwen3-VL-30B-A3B-Instruct snapshot:
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- BF16, TP2 / DP2 / EP4, linear expert placement.
- vLLM `0.20.0+cu129`; PyTorch `2.11.0+cu129`; CUDA `12.9`;
  NCCL `2.28.9`; DeepEP `1.2.1+73b6ea4`.
- FlashAttention 3; Triton unquantized experts; DeepEP high-throughput
  prepare/finalize; DBO disabled for stock and sequential measurements.
- Eager execution and CUDA graphs disabled. This avoids graph-capture and
  replay-state ambiguity in Phase 0.

## Full-query dependency proof

The installed Qwen3 MoE decoder is monolithic at the Python/runtime boundary:

1. `qwen3_moe.py:343-361` computes QKV, calls the attention backend for the
   complete `hidden_states` tensor, applies the output projection, and returns
   a complete output tensor.
2. `qwen3_moe.py:416-436` calls `self.self_attn(...)`, then
   `post_attention_layernorm(...)`, then `self.mlp(...)` in strict order.
3. No token-range readiness object, producer event per query block, or partial
   output contract is exposed between those calls.
4. The MoE implementation's `_prepare`, `_fused_experts`, and `_finalize`
   boundaries correspond to dispatch, expert execution, and combine in the
   instrumentation.

Because all calls are enqueued on the same default stream in stock execution,
MoE prepare cannot begin before the full attention output projection completes.
This verifies the barrier being tested; it does not establish that breaking it
will be profitable.

## EP path proof

The current `cuda_communicator.py:117-170` selects
`DeepEPHTAll2AllManager` for `deepep_high_throughput`. Fresh worker logs state:

```text
Using DeepEPHTAll2AllManager all2all manager.
Using DeepEPHTPrepareAndFinalize
Using TRITON Unquantized MoE backend
```

The DeepEP manager uses intra-node high-throughput mode and a default maximum
of 20 communication SMs. The four physical H100s are fully connected by NV18.

## Timing semantics

- All stage durations are `start.elapsed_time(end)` from CUDA events on the
  same device.
- A logical invocation takes the maximum same-device duration across EP ranks.
- Absolute timestamps from different GPUs are never subtracted.
- Detailed instrumentation had a median paired request overhead of 7.05%, above
  the 3% target. Consequently, instrumented runs are used only for attribution;
  uninstrumented request runs supply TTFT denominators.
- The N2 diagnostic is a bounded two-thread/two-stream runtime intervention,
  not a production wavefront implementation.

