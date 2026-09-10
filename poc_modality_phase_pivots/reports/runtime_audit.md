# Runtime and measurement audit

## Repository baseline

- Active `jyssys/MLLM-EP` research HEAD used to create this isolated worktree:
  `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` (`research: evaluate
  transient visual expert branch sharing`, 2026-09-10 04:15:31 +09:00).
- Worktree/branch: `/home/esjung/MLLM-EP-modality-phase-pivots`,
  `flashvep/modality-phase-pivots-poc`.
- `origin/main` currently resolves to the divergent older artifact commit
  `e228b44cec1f5ffa32953e52540086423f84f33f`; it was not silently substituted
  for the active local research HEAD.
- Prior `poc_flashvep/` artifacts were read-only. All new code/results live under
  `poc_modality_phase_pivots/`.

## Hardware and runtime

- Physical devices only: `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Four NVIDIA H100 80GB HBM3, driver 570.211.01, all-to-all NV18 links.
- UUIDs: GPU4 `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139`; GPU5
  `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730`; GPU6
  `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28`; GPU7
  `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b`.
- Qwen3-VL-30B-A3B-Instruct BF16, vLLM 0.20.0, torch 2.11.0+cu129.
- Verified live engine configuration: TP2 / DP2 / EP4, DeepEP
  high-throughput, Triton fused experts, FlashAttention-3, DBO off,
  eager execution, 48 sparse decoder layers.
- CPU offloading, backend switching, token-axis splitting, routing changes, and
  model changes were not used.

## Source contract

- `Qwen3MoeAttention.forward` is the stock full invocation: QKV projection,
  Q/K norm and RoPE, attention, then output projection
  (`vllm/model_executor/models/qwen3_moe.py:343`).
- `Qwen3MoeDecoderLayer.forward` enforces attention, post-attention RMSNorm, then
  MoE (`qwen3_moe.py:416`). Diagnostic concurrency replays the captured *whole*
  attention invocation; it does not split tokens or kernels.
- DeepEP HT defaults to 20 communication SMs and exposes only a reducing
  `set_num_sms` path (`vllm/distributed/device_communicators/all2all.py:156` and
  `:250`). The live sweep used 4/8/12/16/20.
- DeepEP HT obtains dispatch/combine configurations from the installed DeepEP
  library and owns its internal communication stream. No safe vLLM runtime API
  for changing that stream's priority was found; this individual axis is
  `UNSUPPORTED_IN_CURRENT_PATH`, not a method failure.

## Timing trust

- Stage durations use same-device CUDA events. Logical observations are joined
  by phase/iteration/layer and reduced with the maximum rank-local duration;
  cross-GPU absolute timestamps are never subtracted.
- Concurrent runs use two explicit CUDA streams, start events after a common
  readiness event, and a final stream wait before reading elapsed times.
- Every concurrent pair used the same captured inputs/routes as its standalone
  controls. There were four warmups and 12 measured randomized iterations.
- Exact full logits and greedy tokens were identical between clean,
  instrumented, pairwise and policy modes. Attention replay minimum cosine was
  >0.999999; policy replay minimum cosine was >0.999999.
- Clean/instrumented TTFT observer tax was 17.39% for text and 18.91% for
  vision. Therefore clean TTFT is the only request baseline; detailed stage
  traces are used only for composition/oracle analysis.

## Workload control

- Text-heavy and vision-heavy prompts both had exactly 2,363 actual LM input
  tokens. The vision case contained 2,340 vision tokens; the text case used the
  matched `text_23_method` control.
- Pairwise EP work was fixed across modalities: the same actual routed capture,
  repeated as whole units three times, produced 2,397 input tokens per rank.
- Full-policy sweeps used each modality's own actual captured route/input.
