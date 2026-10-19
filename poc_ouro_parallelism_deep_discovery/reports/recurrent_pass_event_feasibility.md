# Eight-event recurrent-pass diagnostic contract

## Decision

The installed CUDA event interface supports explicit event-record nodes inside a captured graph.
The proposed diagnostic is conditionally feasible.
Source inspection does not establish eight correctly placed nodes with unchanged native lowering and negligible overhead.
An opaque marker alone is not a sufficient compiler barrier.
The scope is the native R4 path, not adaptive early exit.

This audit implements no GPU code and executes no GPU work.
It does not establish an instrumentation result or a candidate method.

## Supported interfaces

Installed Torch is `2.11.0+cu129`.
`torch.cuda.Event(enable_timing=True, external=True)` is available.
Its documentation distinguishes explicit record/wait nodes from internal graph dependencies.
See `torch/cuda/streams.py:159–209`.

During capture, the installed C++ implementation selects `cudaEventRecordExternal`.
Outside capture, it selects the default record flag.
See `torch/include/c10/cuda/CUDAEvent.h:145–153`.
Both events must exist, enable timing, and complete before `elapsed_time` succeeds.
See the same header at lines 194–214.

Torch also traces an existing event's `record` method into `streams::record_event`.
See `torch/_dynamo/variables/streams.py:461–470`.
That operator has a fake implementation and an explicit side-effect mark at lines 124–139.
Inductor provides its fallback lowering at `torch/_inductor/lowering.py:2472`.
This is a source-backed alternative to a campaign-specific opaque operator.
It still requires the ordering and graph checks below.

A campaign-specific opaque operator can use `torch.library.custom_op`, a CUDA implementation, and `register_fake`.
The real implementation would only record an existing event on the current native stream.
The fake implementation must not touch CUDA events.
`CustomOpDef.register_effect(EffectType.ORDERED)` exists in this installed version.
The associated registration API remains version-specific.

## Compiler ordering is the main risk

A pure operator with no used output can disappear or move during compilation.
Explicit effect registration can retain it.
However, ordered effects constrain operations with the same effect type.
They do not automatically fence all intervening pure tensor operations.
See `torch/library.py:1099–1108` and `torch/_inductor/lowering.py:7732–7746`.

A read-only hidden-state argument provides a dependency on its producer.
It does not prove that the marker precedes every later consumer.
Declaring a model tensor mutable can introduce functionalization copies or different fusion.
An alias-returning marker also needs separate compiler support.
Do not assume any of these choices adds only an event node.

A Python pass boundary is not necessarily an existing kernel boundary.
A job-associated R4 graph places the loop norm and next-pass input norm in the same compiled piece.
Its next QKV GEMM also belongs to that piece.
See the saved `submod_48` at graph lines 5537, 5544, and 5547.
This does not prove kernel fusion across the boundary.
It requires a check before instrumentation can claim unchanged lowering.

If a native kernel spans a requested boundary, an external event cannot divide that kernel's execution.
Splitting the kernel would measure a modified graph.
If exact placement requires extra tensor kernels, copies, collectives, waits, or graph splits, reject the eight-node premise.

## Boundary and capture contract

Use four start/end pairs per rank for one selected captured graph.
These are eight added record nodes per rank, not eight nodes across all ranks.
Preserve the original loop in `vllm/model_executor/models/ouro.py:367–374`.

1. Record start `r` immediately before pass `r`'s first layer.
2. Execute the original physical layers with unchanged depth and residual rules.
3. Execute the original loop norm.
4. Record end `r` before the next pass starts.

The first start excludes embedding work.
The last end excludes LM-head and sampling work outside the core.
The four intervals describe rank-local stream spans, not a kernel-category breakdown.
They include waits already present on that stream.
They do not measure unrelated auxiliary-stream work unless native dependencies join that work before the boundary.
Do not add synchronization to create that join.

Create and retain all timing events before warmup and capture.
Materialize their device handles outside capture.
Keep strong references until every graph and pending measurement using them finishes.
Do not allocate events, query results, synchronize, or calculate elapsed times inside the marker.

The V2 native manager warms and captures PIECEWISE and FULL descriptors separately.
See `vllm/v1/worker/gpu/cudagraph_utils.py:295–358`.
Its FULL replay uses the selected graph at lines 388–401.
The event nodes execute on replay without Python marker calls for each replay.
The capture-time backend call must nevertheless execute and insert all eight nodes.

Use `CUDAGraph.enable_debug_mode` and `debug_dump` for a separate structural check where available.
See `torch/cuda/graphs.py:153–165`.
Check the added record nodes and their dependencies, not only the number of FX marker calls.
Also check the lowered arithmetic kernels, collective sequence, and graph partitioning against the clean comparator.

## Event reuse and replay identity

An event pair stores its most recent recording, not a history of replay durations.
Reusing one bank across multiple replays overwrites earlier timestamps.
Reading the final replay and multiplying by the replay count is invalid.

A two-generated-token B1 request is a suitable bounded target if it performs one selected cached decode replay.
Count that replay and retain its graph descriptor, request identity, rank, stream, and ordinal.
Do not assume the count from output length alone.
Prefill, warmup, other descriptors, or another request must not overwrite the selected bank afterward.

For the smallest test, permit one active request and one selected replay before reading the events.
Wait for completion only after that diagnostic request.
Keep the readout outside the measured serving path.
Do not overlap graph instances that share a bank.
More than one observed replay requires separate graph-bound banks or an explicit per-replay collection protocol.
That extension is outside the eight-event single-step test.

CUDA event clocks are rank-local.
Do not subtract timestamps from different ranks or add rank durations as elapsed model time.

## Acceptance and failure conditions

Use fresh per-job vLLM compile directories.
Also isolate Inductor and Triton caches when the control requires separate lower-level storage.
Retain a clean, unmodified native comparator with the same model, P, shapes, scheduler, attention, collectives, and graph settings.

Accept the diagnostic only after these checks:

- All eight record nodes exist exactly once in the selected graph and have the intended kernel dependencies.
- The original operation order, parameter ownership, KV slots, and collective order remain unchanged.
- No marker creates a graph break, host replay callback, new tensor kernel, or tensor copy.
- Matched logits, hidden state, and sampled KV meet the frozen numerical-parity criterion.
- Repeated clean and instrumented requests establish observer overhead against a predeclared absolute and relative budget.
- The readout identifies one actual replay and no later overwrite.

A retained no-record marker lane can separate compiler-barrier effects from event-record costs.
That lane is another diagnostic, not the clean comparator.
Measure overhead with external replay envelopes and wall time, not the eight internal events alone.
Do not relax the numerical criterion after the observation.
Reject causal timing attribution if the marker changes lowering or exceeds the overhead budget.
Report failure if event retention, boundary placement, capture safety, or parity remains unresolved.

## Source and time ledger

Package root: `/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages`.
The saved graph is `/home/esjung/.cache/vllm/torch_compile_cache/37ba5182b0/rank_0_0/backbone/computation_graph.py`.
It belongs to the earlier campaign alias job, not the clean comparator proposed here.
Its SHA256 is `7720e433f11c7ea1068ce7d7570ccab4f164130ade0678654f68c43172ae47ab`.

| Installed source | SHA256 |
|---|---|
| `torch/cuda/streams.py` | `3b34ed08b67cf2ec411e1a483b31835bbff8652c915a08f2a0e4c140e342dd9c` |
| `torch/include/c10/cuda/CUDAEvent.h` | `c054cfba735c9653df248563c01bcb56c5c75d70828eb10d591d9ed470450f3f` |
| `torch/_dynamo/variables/streams.py` | `36c3fc66be9dfb9693af35e5364ae967d89681d4dcf7e175f3ccf742c201f1f3` |
| `torch/library.py` | `bd76a792649fc14a86f32f73d5464d42f07ecb71da41fd882d8a2736d72cd004` |
| `torch/_inductor/lowering.py` | `0b1e1ac3585aa5e8916ad05c8426d05f9bb1ea758899a3f661c9bc0d47cdebb6` |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `5b328faa0c29363815eb112f03bb1e8772f5b08e9f158e154aa3c09d676d2da8` |
| `vllm/model_executor/models/ouro.py` | `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff` |

CPU introspection checked the event signature, inheritance, effect-registration method, and graph-debug interfaces.
It constructed no CUDA object.
No GPU implementation, graph capture, parity test, or overhead test ran in this audit.

Own segment start: `2026-10-19 12:44:00 UTC`.
Recorded end after checks: `2026-10-19 12:50:15 UTC`.
The recorded segment lasts 375 seconds.
All seven source hashes matched, and the report linter found zero violations.
