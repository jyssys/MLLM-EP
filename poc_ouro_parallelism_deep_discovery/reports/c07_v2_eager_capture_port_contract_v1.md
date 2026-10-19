# C07 V2 eager capture port

## Status and scope

`CPU_ONLY_PORT_READY_FOR_REVIEW` applies to the bounded eager hook adaptation.
No GPU command or GPU execution occurred.
Graph execution remains `INFRASTRUCTURE_BLOCKED` under the [graph audit](c07_graph_metadata_feasibility_audit_v1.md).
The port does not construct a graph, change the serving computation, or force V1.
It does not change cache layout, physical service, trial order, or bitwise parity.
No runner or archive payload code changes.

The pre-port worker assumed V1 `_model_forward` and a persistent V1 input batch.
The installed V2 runner has neither interface.
The graph audit retains that failure fact and both pre-port source hashes.
This port addresses only the eager observation entry points and actual padding metadata.
It does not prove runtime fixture correctness.

## Native observation boundaries

[V2CaptureHooks](../c07_fixture.py:166) observes two existing instance methods.
It first calls native `runner.prepare_inputs` without changing its arguments.
The observer copies CPU fields from the returned actual V2 `InputBatch`.
It then returns that same batch object.
The wrapper does not sort rows, change admission, or prepare new work.

The second hook observes native `runner.model.forward`.
V2 [calls that model](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/model_runner.py:1339) inside the actual ForwardContext in eager mode.
The hook reads that context and calls the original bound forward exactly once.
The original physical-layer and recurrent-norm hooks remain the reference capture points.
The complete R4 reference computation remains intact.

The hook lease restores the exact prior instance attributes.
If a method originally came from its class, restoration removes the instance override.
Restoration is idempotent.
Failure cleanup also restores the hook lease.
The export method changes only that cleanup call.
Its archived payload logic remains unchanged.

## Exact V2 admission

[snapshot_v2_batch](../c07_fixture.py:105) copies native CPU arrays into owned Python lists.
[v2_batch_rejection](../c07_fixture.py:118) requires these properties:

- Four distinct native request IDs and four distinct request-state indices.
- Four real requests and four requests after padding.
- Four real tokens and four tokens after padding.
- One scheduled token per request and no draft tokens.
- CPU query boundaries `[0, 1, 2, 3, 4]`.
- Prefill lengths, computed lengths, and computed-prefill lengths equal the required position.
- No request remains in prefill.

The hook consumes each pending batch observation once.
It does not substitute the submitted request order for native row order.
After synchronization, the existing checks still require exact GPU positions and sequence/query metadata.
The current-token write addresses and all KV parity checks remain mandatory.
The CPU computed-length mirror alone is not proof of the GPU position.

[select_decode_metadata](../c07_fixture.py:155) also sets selected-row decode counters explicitly.
TP1 has one decode request and one decode token.
TP4 has four decode requests and four decode tokens.
Both retain zero prefill requests and zero prefill tokens.
The selection preserves other native FA3 fields, including scheduler ownership.
The existing TP1 scheduler rebuild still follows that selection.

## Padding refinement before GPU execution

The first proposal required every captured padding flag to be false.
Source inspection shows that assumption is stronger than unpadded dense-Ouro admission.
V2 [dummy preparation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/input_batch.py:143) writes true flags.
Its [real preparation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/model_runner.py:881) refreshes them only when `VLLM_MOE_SKIP_PADDING` equals true.
That flag defaults to false.
A true value can therefore be stale, unused metadata in this exact dense route.
This is not evidence that a real padded row is valid fixture work.

The relevant [MoE consumer](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py:1154) reads padding under that environment flag.
Native [OuroMLP](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:68) contains dense projections and `SiluAndMul`, not MoE.
The runtime checks require original native Ouro core/layer/MLP classes and unquantized projections.
The [unquantized CUDA route](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/utils.py:92) calls `torch.nn.functional.linear`.
The port rejects `VLLM_BATCH_INVARIANT` so that proof does not silently expand to another GEMM route.
Native attention [context lookup](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:753) does not read padding.
The inspected Ouro, RMSNorm, activation, dense-linear, rotary, embedding, Attention, and FA3 modules contain no padding reader.

The observer checks the actual padding tensor's identity, shape, and Boolean dtype without reading its contents.
It then clones that tensor with the other context metadata.
[observe_dense_padding](../c07_fixture.py:139) reads those flags only after synchronization.
It records `ALL_FALSE` or `DENSE_UNUSED_TRUE_FLAGS` without changing any value.
The explicit V2 admission checks remain mandatory in either case.

[select_v2_padding](../c07_fixture.py:150) selects the actual native flags for TP1 or TP4 into private storage.
Preparation checks those selected values after synchronization.
It does not zero, synthesize, or discard flags.
This refinement changes neither numerical tolerance nor model semantics.
The runtime checks restrict it to the unquantized dense-Ouro route.

## CPU evidence and limits

The suite passes 33 tests in both interpreters.
It covers V2 hook order, original-call preservation, restoration after errors, owned CPU snapshots, and rejection of mixed progress.
It also covers padding selection, flag preservation, and the previous exact-word and collective-gate regressions.
A source test finds no padding reader in nine inspected dense-route modules.
Its MoE positive control does find that reader.
An actual Torch CPU probe also checks selected flag values and private storage ownership.
Syntax checks pass.

These checks do not execute the native V2 worker or FA3 on a GPU.
Unexpected context fields, native cache packing, page capacity, or numerical differences can still block the fixture.
The original layout and lease checks remain unchanged.
Any bitwise difference remains `UNRESOLVED`.
No eager or graph timing conclusion follows from this port.

## Source and actual time

```text
eb4449b8a0f9b403455f54d27a34bb7f84232af5dc3da9d1bf41f5ed4eb08ab6  c07_fixture.py
e228ca70190e564e404fad40e3525f298e4f4b72c6c7c4c6a35704c8eac84a58  c07_fixture_probe.py
0d923f742a46f3cabf44194766e6813c87704828f235c16b49b3cedf44eca73c  tests/test_c07_fixture_cpu.py
```

The source and CPU port segment started at `2026-10-19 15:25:58 UTC`.
It ended after the metadata review at `2026-10-19 15:35:42 UTC`.
That interval is 584 seconds.
Final contract updates and lint follow that interval.
