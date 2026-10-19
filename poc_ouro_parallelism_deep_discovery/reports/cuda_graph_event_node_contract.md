# CUDA graph event-node diagnostic

Revision 1. Implementation and CPU checks only. GPU regression: `NOT_RUN`.
This diagnostic supplies no model latency, quality, or speedup result yet.

## Supported mechanism

CUDA 12.9 provides `cudaGraphAddEventRecordNode` and dependency APIs.
Each graph launch records the event after its dependencies.
Timing requires events without `cudaEventDisableTiming`.
The installed header specifies approximately 0.5-microsecond elapsed-time resolution.
Other device work can affect elapsed intervals. [NVIDIA graph API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__GRAPH.html), [event API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html)

The installed Torch interface supports retained `raw_cuda_graph` access and re-instantiation after changes.
The implementation keeps Torch's original replay method.
It does not launch a cloned executable through cudart.
That choice preserves Torch's replay bookkeeping, including any generator prologue.
No Torch or vLLM source file changes.

The observer uses cudart handles directly. It does not pass a Torch Event object into a compiled fixture.
It resolves exactly one already-loaded cudart library and checks runtime version 12.9.
It rejects unsupported node types, branching, and non-default dependency data.
An explicit environment variable enables the worker's process-local wrappers.

## Topology and insertion

The supplied P1 DOT contains 1,154 nodes and 1,153 edges in one chain.
It contains 1,153 kernels and one memcpy.
The parser finds cuts at `[0, 288, 576, 864, 1154]`.
The first four cuts precede observed fused normalization kernels.
These cuts separate fused kernel groups. They do not mark exact semantic pass boundaries.
A boundary kernel can combine preceding-pass normalization and following-pass input work.
The final group includes the remaining tail. The memcpy purpose remains unresolved.

For a cut between original nodes A and B, the observer keeps `A -> B`.
It adds `A -> event -> B`.
The root event precedes the first node. The final event follows the last node.
Thus the supported graph gains five event nodes and eight edges.
Its resulting counts are 1,159 nodes and 1,161 edges.
No original model node or edge disappears.
The code checks the exact added edge set and the total model-node order before replay.

The actual captured order can include stream-order constraints stronger than mathematical dependencies.
This diagnostic does not establish that all model operations must execute serially in every legal implementation.

## Execution and restoration

Only the coordinator may execute these probes inside the existing GPU guard.
The new constant-tensor regression must precede the model diagnostic.
It changes the source tensor between replays to detect a nonexecuting or stale graph.
It checks exact output values and topology restoration.
Those checks remain `NOT_RUN` on CUDA in this task.

The worker retains raw graphs before engine capture.
After warmup, one RPC arms one controlled P1 request.
The observer checks the live DOT rather than reusing node handles from an old process.
It inserts timed event-record nodes and calls Torch's `instantiate()`.
Each observed replay still uses Torch's original replay function.
After synchronization, the observer removes only its event nodes and instantiates the original topology again.
It checks restoration before reporting success.

Use a fresh worker for each armed request.
Do not update raw graph nodes, executable parameters, or graph ownership during an armed observation.
The supported scope excludes concurrent requests and multiple replay streams.
An API or restoration failure invalidates the diagnostic. Do not continue with a partial timing result.

## Timing contract

Each event handle records again on every replay.
After all launches complete, its timestamps describe only the last replay of that graph.
The output therefore records the observed replay count and the timed replay ordinal separately.
`request_total_ms` remains null. `request_extrapolation` is `FORBIDDEN`.

Do not multiply the final replay by the number of request replays.
Do not sum final-replay records from different graph objects into a request total.
Do not label the four intervals as exact Python pass durations.
Do not label event-instrumented execution as clean native latency.

The intervals include event overhead and any intervening device interference.
The worker also changes graph retention and performs diagnostic setup.
The driver captures controlled teacher outputs and selected state for a separate comparison.
These observations cannot establish natural task quality or an E2E architecture gain.

## Entry points and verification

| Entry point | Scope |
|---|---|
| `scripts/cuda_graph_event_api_regression.py --run-cuda` | Guarded constant tensors only. Requires one permitted GPU. |
| `scripts/native_graph_event_nodes.py --size 1.4B --expected-nodes 1154` | Guarded P1 model diagnostic. Requires `OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC=1`. |
| `tests/test_cuda_graph_event_nodes.py` | CPU topology, fake ABI, rollback, and last-replay contract. |

Eighteen CPU tests pass: thirteen new tests and five existing parser tests.
The actual-DOT test checks every original node and the complete insertion order.
Two negative tests reject stale node handles and a missing repeated kernel pattern.
The fake-ABI tests check restoration after an instantiation failure.
Four new Python modules pass syntax compilation.
The inspected library exports the required event and graph symbols.
Symbol presence and fake-ABI tests do not validate a CUDA driver call.

Actual segment start: 2026-10-19 13:20:05 UTC.
Validated evidence cutoff: 2026-10-19 13:33:53 UTC. Recorded interval: 828 seconds.
Only final metadata publication follows this cutoff.
This worker interval is not additional campaign elapsed time.
