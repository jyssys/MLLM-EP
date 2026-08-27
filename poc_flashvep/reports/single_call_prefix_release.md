# Single-call Prefix Early-Release Feasibility

## Final result

`SINGLE_CALL_PREFIX_RELEASE: NO-GO`

The stock Qwen3-VL/vLLM 0.20 unsplit path does not expose a safe prefix-row
completion boundary.  A prefix store that happens to appear early in a GPU
trace is not sufficient: a consumer in the next layer needs a CUDA dependency
whose contract covers all writes needed by that prefix.  The available
Attention and DeepEP events cover the whole operation only.

## Scope and environment

* Base commit: `6d8aad0a02c8da69c9955a8b6098495f3fb85885`
* Branch: `flashvep/single-call-prefix-release`
* Model target: Qwen3-VL-30B-A3B, BF16
* Intended device visibility: physical GPUs 1,2,3,4 (`CUDA_VISIBLE_DEVICES=1,2,3,4`)
* Intended backend: vLLM 0.20 + DeepEP high-throughput, DBO OFF
* Installed vLLM: `0.20.0+cu129`; PyTorch `2.11.0+cu129`; Triton `3.6.0`
* DeepEP source revision: `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`

Stage 1 was not launched because the Stage-0 safety gate failed.  Therefore
there are no fabricated histology/layer-24 GPU timings, correctness outputs,
or result directory.  The audit used source inspection only and made no
model/operator call.

## Stage 0 — API and completion audit

### Unsplit MoE

The `DeepEPHTPrepareAndFinalize` path performs one high-level dispatch, one
expert invocation, and one combine per MoE layer:

* `deepep_ht.py:_do_dispatch` calls `get_dispatch_layout` and then
  `buffer.dispatch` once (`:114-162`).
* `modular_kernel.py:_fused_experts` calls `TritonExperts.apply` once
  (`:1246-1262`).  In the stock Triton implementation this invocation
  contains the W13 grouped-MoE kernel, activation, W2 grouped-MoE kernel, and
  final `moe_sum` (`fused_moe.py:2075-2186`).
* `deepep_ht.py:_finalize` calls `buffer.combine` once (`:362-376`) and its
  receiver waits the event before copying the complete result (`:380-399`).

At the DeepEP CUDA level, dispatch has one layout launch and one intranode
dispatch launch (`deep_ep.cpp:304-322`, `:576-603`) plus coordination; combine
has one queue/barrier reset and one intranode combine launch
(`deep_ep.cpp:719-771`).  These are whole-tensor operations.  The stock event
is recorded only after the corresponding launch (`deep_ep.cpp:605-635`,
`:773-788`) and `EventHandle` only supports a whole-event stream wait
(`event.hpp:9-40`).

The returned tensors are full received/output tensors.  No prefix/tail row
range, partial completion event, or callback is present.  Thus dispatch and
combine cannot safely release only Prefix rows to another CUDA consumer.

### Unsplit Attention

`attention.py:493-549` allocates a full output, optionally queues a complete KV
cache update, and invokes one `unified_attention_with_output` operation.  The
custom-op registration at `attention.py:754-801` has no completion handle.  The
FlashAttention backend passes all active query rows to one
`flash_attn_varlen_func` call (`flash_attn.py:682-820`), while the Triton
implementation launches a single whole-call grid
(`triton_unified_attention.py:653-675`).  Neither path exposes a row-level
event; KV-cache writes are also a complete operation (`flash_attn.py:851-882`).

Consequently Prefix attention output is safe only at full-call completion.

## Stage-1 fields and gate interpretation

Because no safe event exists, the only contractually valid values are:

| Measurement | Safe value/status |
|---|---|
| Attention prefix-ready | `N/A` (no prefix event); conservatively equals full-ready |
| Attention full-ready | no separate measurement run |
| Attention safe slack | `0 ms` by contract, not an observed timing |
| MoE prefix-ready | `N/A` (no prefix event); conservatively equals full-ready |
| MoE full-ready | no separate measurement run |
| MoE safe slack | `0 ms` by contract, not an observed timing |
| Additional kernel/collective count | none; Stage 1 was gated off |
| Processed tokens/assignments | unchanged/not run |
| Correctness | unchanged/not run |

This is a source/API NO-GO rather than an inference that prefix writes never
occur physically before tail writes.  Proving usefulness from such writes
would require a new row-granular kernel/runtime completion primitive (and a
consumer-side dependency), or a second operator/collective.  Both are outside
the requested bounded PoC.

## DeepEP completion granularity summary

| Component | Whole-call work | Safe Prefix release in current API |
|---|---|---|
| Dispatch | layout + intranode dispatch | No |
| Expert | W13 + activation + W2 + reduce kernels | No |
| Combine | queue/barrier + intranode combine | No |
| Attention | KV update (when enabled) + full attention op | No |

## Conclusions

* **Attention prefix-ready/full-ready/slack:** prefix completion cannot be
  safely observed before the full custom-op completion; safe slack is 0 ms by
  contract.
* **MoE prefix-ready/full-ready/slack:** DeepEP exposes one event after the
  full dispatch/combine operation; safe slack is 0 ms by contract.
* **Stock call counts:** one unsplit dispatch/layout path, one expert apply,
  and one combine per MoE layer; no additional calls were introduced.
* **Current API:** safe early release is not possible.
* **Required modification:** a row-granular completion contract in Attention
  and/or DeepEP plus a downstream CUDA wait; merely adding an event after the
  existing full launch does not provide it.
* **Strongest positive evidence:** DeepEP already has asynchronous whole-call
  stream events, so it has a foundation for overlap, but those events are
  explicitly whole-operation handles.
* **Strongest counter-evidence:** output tensors are allocated and consumed as
  complete tensors, and both `EventHandle` and `EventOverlap` implement only a
  full event wait with no row-range semantics.
* **Causal-wavefront branch:** do not continue on a single-invocation
  prefix-release assumption.  It can be revisited only after a separately
  specified row-granular completion API is designed and validated.
* **Next single action:** design a minimal, contractually safe row-granular
  completion primitive (including lifetime and downstream wait semantics) in a
  separate design/feasibility task; do not implement it in this PoC.

Detailed source evidence is in
[`poc_flashvep/single_call_prefix_release/source_audit.md`](../single_call_prefix_release/source_audit.md).
