# Single-call prefix early-release: source/API audit

Audit date: 2026-08-27<br>
Repository base: `6d8aad0a02c8da69c9955a8b6098495f3fb85885`<br>
vLLM package: `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm` (`0.20.0+cu129`)<br>
DeepEP source: `/home/esjung/.cache/flashvep-deepep-v020/DeepEP` (`73b6ea4a439ba03a695563f9fd242c8e4b02b37c`)

## Decision

`SINGLE_CALL_PREFIX_RELEASE = NO-GO` at the source/API gate.  The stock
unsplit path has whole-operation completion events only.  Neither the
Attention output nor the DeepEP combine output has a contract that permits a
next-layer CUDA consumer to wait on a prefix-row event.  A write observed
before the event is not treated as a safe release.

## Unsplit MoE call graph

For `DeepEPHTPrepareAndFinalize`:

1. `deepep_ht.py:_do_dispatch` (lines 114-162) obtains one dispatch layout,
   then calls `buffer.dispatch` once.  DBO is disabled for this experiment;
   the `async_prepare` flag only changes whether the one whole dispatch event
   is returned.
2. `modular_kernel.py:_fused_experts` (lines 1246-1262) calls
   `TritonExperts.apply` once.
3. `deepep_ht.py:_finalize` (lines 362-376) calls `buffer.combine` once.
   The receiver waits the returned event before copying the complete combined
   tensor to the layer output (lines 380-399).

The high-level stock per-layer granularity is therefore one dispatch, one
expert invocation, and one combine; there is no second prefix call.

### DeepEP dispatch

The high-throughput intranode path contains:

* one `get_dispatch_layout` launch (`deep_ep.cpp:304-322`) and one
  `intranode::dispatch` launch (`deep_ep.cpp:576-603`), plus CPU counter/barrier
  coordination (`deep_ep.cpp:517-537`); and
* one `EventHandle(comm_stream)` recorded only after the dispatch launch
  (`deep_ep.cpp:605-635`).

The Python API describes this as one `Buffer.dispatch` result and defines the
event as “the event after executing the kernel” (`buffer.py:319-401`).  The
returned `recv_x` is a single tensor covering all received rows.  No row range,
prefix sub-event, or callback is returned.

### DeepEP expert stage

The stock `TritonExperts.apply` is one Python invocation but is not one device
kernel: it launches the W13 grouped-MoE kernel (`fused_moe.py:2075-2096`), the
activation operation (`fused_moe.py:2125-2127` → `activation.py:94-134`), the
W2 grouped-MoE kernel (`fused_moe.py:2140-2161`), and `moe_sum`
(`fused_moe.py:2182-2186`).  Exact low-level count can vary with quantization
and backend, but all consume the complete dispatched tensor; none publishes a
prefix-ready event.

### DeepEP combine

The intranode combine path launches a queue/barrier reset
(`cached_notify_combine`, `deep_ep.cpp:719-729`; kernel definition in
`intranode.cu:613-703`) followed by one `intranode::combine` launch
(`deep_ep.cpp:743-771`).  It then records one whole-operation `EventHandle`
(`deep_ep.cpp:773-788`).  The output is allocated as one full
`[num_recv_tokens, hidden]` tensor (`deep_ep.cpp:743-749`).  The event wrapper
only supports waiting for the complete event (`event.hpp:9-40`; `buffer.py:10-61`).
There is no contract for the order in which rows are written and no per-row
event.  Consequently an externally observed early write cannot be consumed by
the next layer safely.

## Unsplit Attention call graph

`attention.py:493-549` allocates one output for all query rows, optionally
queues one KV-cache update (`unified_kv_cache_update`), and invokes one
`unified_attention_with_output` custom op.  The op registration
(`attention.py:754-801`) mutates the complete output tensor and returns no
event or row-level completion handle.

For the FlashAttention backend, `flash_attn.py:682-820` passes the full
`query[:num_actual_tokens]` and `output[:num_actual_tokens]` to one
`flash_attn_varlen_func` call (or the full cascade path).  The Triton unified
implementation launches one whole-call grid (`triton_unified_attention.py:653-675`)
whose blocks may cover different query rows, but it exposes no row-completion
event.  KV-cache writes are likewise a separate full operation
(`flash_attn.py:851-882`) without a prefix-ready dependency handle.

Thus, even if a GPU trace happens to show some prefix stores before tail
stores, the current API only makes the output safe after the full attention
call (and its KV update) has completed on the stream.

## Safety assessment

| Stage | Stock unsplit granularity | Prefix-row event/API | Safe external prefix release |
|---|---|---|---|
| Attention | full output custom op (plus optional full KV update) | none | No |
| DeepEP dispatch | layout + one dispatch kernel | one whole-dispatch event only | No |
| Expert | one `TritonExperts.apply` with full dispatched tensor | none | No |
| DeepEP combine | queue/barrier + one combine kernel | one whole-combine event only | No |

The safe interpretation is `prefix_ready = full_ready`, so the exposed safe
slack is exactly `0 ms`, not a timing estimate.

## Stage-1 decision

The specification permits measurement only when Stage 0 finds a safe release
opportunity.  Since neither Attention nor DeepEP provides one, no model/GPU
measurement was run, no operator was duplicated, and no vLLM/DeepEP source was
modified.  Any such measurement would either rely on an unsafe write-order
observation or require adding a kernel/runtime completion primitive, which is
outside this bounded PoC.
