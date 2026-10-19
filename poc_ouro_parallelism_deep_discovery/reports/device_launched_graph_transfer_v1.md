# Device-launched graphs after whole-R4 capture

Worker: `/root/independent_axes`.
Segment began at 2026-10-19 13:17:24 UTC.
Evidence class: primary-source and read-only code audit.
This segment contains no GPU execution or new model-quality result.

## Outcome

No recurrence-specific cost reduction survives this source screen.
Device launch can change who submits work.
It does not make a later Ouro hidden state ready.
Persistent fusion can reduce costs inside an existing graph.
That mechanism has strong generic inference priors and unmeasured Ouro benefit.
The verdict is `GENERIC_ONLY_WITH_UNMEASURED_INTERNAL_HEADROOM`.
No new candidate card or engine follows from this audit.

This verdict does not reject all persistent implementations.
It rejects a novelty argument based only on device launch, repeated descriptors, or the word “recurrent”.

## 1. The stronger baseline removes the proposed host loop

The baseline uses native asynchronous scheduling and one whole-R4 model graph per cached decode.
Its installed Ouro forward contains the recurrent loop inside the model call.
It executes each physical layer, applies the loop norm, and then advances the recurrent depth.
The fixed path defines an exit gate but does not consume it.
See `ouro.py:348–375` in the structured source record.

The installed graph wrapper captures `self.runnable(*args, **kwargs)` and then replays the captured graph.
See `cuda_graph.py:313–360`.
The inspected Python Torch wrapper delegates replay to its C++ parent.
This audit did not inspect the installed binary's graph implementation.

Consequently, fixed R4 does not require four host graph submissions per token in this baseline.
Replacing four imagined host submissions with one device loop saves a cost that the baseline does not contain.
The 1.4B path still needs 96 layer visits per token.
The 2.6B path still needs 192 layer visits per token.
These counts equal four passes through 24 or 48 physical layers.
They are structural counts, not kernel counts or timing measurements.

One physical weight allocation already serves all four passes.
An aliased logical unroll can preserve the same pointers.
The earlier [persistent-execution audit](persistent_execution_redteam.md) separately rejects complete on-chip BF16 core residence under its stated representation.
Device graph launch does not increase register, shared-memory, or L2 capacity.

## 2. Three execution mechanisms have different effects

| Mechanism | Cost it can change | Cost it does not remove by itself |
|---|---|---|
| Device launch of an unchanged whole-R4 graph | Host submission or host-dependent control | Internal kernel boundaries, GEMMs, attention, collectives |
| Repeated launch of a one-pass device graph | Program representation and loop control | Four pass executions and inter-pass dependencies |
| Persistent fused task runtime | Internal boundaries, selected transfers, legal task overlap | Required data dependencies and communication |

Splitting one whole-R4 graph into four device-launched graphs can add dispatch and control boundaries.
Descriptor reduction is not arithmetic reduction.
A persistent task runtime is not simply the same graph with a device-side launch call.

CUDA 12.9.1 requires a device-launch graph to contain nodes on one GPU.
Its allowed node types exclude host callbacks and conditional nodes.
The graph needs host instantiation and device upload.
Its structure cannot change through a device-side update.
Device launch must originate inside another graph.
Tail launch supplies ordered continuation and self-relaunch.
It does not remove dependencies between consecutive recurrent passes.
Peer-accessible copies can target another GPU's memory, but their issuing graph remains on one GPU.
The current guide retains the same single-device requirement.
[CUDA 12.9.1, sections 6.2.8.7.7.1–6.2.8.7.7.2](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#device-graph-launch),
[current CUDA guide, section 4.2.6](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-launch).

Conditional WHILE nodes are a separate generic option.
Their bodies are also single-device and cannot contain device graph launch.
The audit does not assume that these two control mechanisms compose freely.
[CUDA 12.9.1, section 6.2.8.7.8.2](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#condtional-node-body-graph-requirements).

## 3. Cross-GPU persistence still needs a communication protocol

NVSHMEM device APIs can issue remote memory operations from running kernels.
They do not create a remote CUDA worker automatically.
The current collective-launch documentation says that each call launches only on the calling PE's GPU.
The required PEs must launch and participate.
Synchronizing device APIs require the documented launch path and a supported grid size.
This is not evidence that one CUDA device graph spans P2/P4/P8.
[NVSHMEM kernel launch routines, description and grid-size query](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html).

A persistent design can launch a worker on each GPU and exchange data and readiness signals.
It must preserve request, position, recurrent depth, physical layer, and buffer-version identities.
An early producer signal does not establish that a consumer's actual tensor is ready.
Signal ordering and buffer lifetime therefore remain part of the cost.

NVSHMEM documents signal operations and separate ordering/completion rules.
A fence orders delivery but does not establish completion.
Quiet completes the covered operations but does not itself notify another PE.
The protocol must use the appropriate destination-side wait or synchronization.
CUDA and NVSHMEM dependencies can also create deadlock when the runtime cannot see the complete dependency graph.
[NVSHMEM signaling](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html),
[memory ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html),
[CUDA interactions](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html).

These requirements add polling, signal traffic, queue state, and progress constraints.
Worker and scheduler resources can reduce resources available to GEMMs or attention.
No API statement establishes a favorable net cost for Ouro.

## 4. Strong generic priors cover internal graph savings

MPK uses SM-level tasks and an in-kernel runtime for computation and communication.
Its paper explicitly compares this representation with CUDA Graphs.
It describes cross-task pipelining and finer compute–communication overlap.
Its multi-GPU path decomposes TP collectives into communication and local reduction tasks.
These are generic dense-model mechanisms, not a recurrent Ouro result.
The reported offline evaluation uses 64-token prompts and 1024-token outputs.
This audit transfers no reported performance ratio to the current campaign.
[MPK v2, sections 3, 5, 6.2, 6.5–6.6](https://arxiv.org/html/2512.22219v2).

The pinned MPK code supplies a concrete implementation boundary.
Its launcher prepares queues and records an event before it launches persistent workers.
The single-kernel NVSHMEM path calls `nvshmemx_collective_launch`.
The same function then synchronizes after that persistent workload.
The split worker/scheduler path explicitly excludes NVSHMEM in its source comment.
These details do not establish a drop-in server implementation or local compatibility.
[MPK runtime, revision ca58a02, lines 1959–2045](https://github.com/mirage-project/mirage/blob/ca58a02d110a278bffaed503be5e66494db2597d/include/mirage/persistent_kernel/persistent_kernel.cuh#L1959).

Hazy Research's Llama-1B megakernel is another direct prior.
It uses an on-GPU interpreter, paged shared memory, and readiness counters.
Its author report already targets residual costs after CUDA Graphs.
Its model, one-GPU scope, and workload differ from Ouro.
The audit transfers neither its timing nor its bandwidth claim.
[Author report, interpreter, shared-memory, and synchronization sections](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

## 5. What existing timing can and cannot bound

The completed [native graph observation](native_graph_observation.md) covers fixed-R4 BF16 TP2 on GPU4/GPU5.
It reports both model sizes and contexts 1024, 4096, and 8192.
Each context has three observed repeats of one frozen shape and 128 forced output tokens.
The graph fractions span 97.7986–98.9896% of the defined rank-local cached window.
Prefill and the first and final sampling intervals are outside this denominator.
The intervals do not align clocks across ranks.

For a pure submission change, hold each graph envelope unchanged.
Even deleting the entire complement gives only a conditional local-window screen.
If `f` is the graph fraction, the optimistic ratio is `1/f`.
The observed fraction range gives approximately 1.0102–1.0225 times for that artificial screen.
This is not measured speedup, E2E headroom, or proof that the complement is removable.
The complement can contain useful kernels, communication, readiness waits, and sampling.

Internal persistent fusion changes the graph envelope itself.
The outside-graph screen therefore does not bound that different mechanism.
Its useful cost test must separate removed internal costs from added scheduler and kernel costs.
It must not add nested timing intervals or independent per-rank maxima.
No current evidence supplies that decomposition for Ouro.

## 6. Fixed and original-adaptive paths need different controls

Fixed R4 contains no host decision between its four passes in the stated baseline.
Original adaptive CDB does contain a possible device-to-host gate wait.
The inspected policy calls `.item()` after sigmoid and updates Python survival/CDF state.
This cost belongs to that policy path, not the measured fixed-R4 graph denominator.

Device control could remove that wait while preserving the actual gate dependency.
That is a changed control path, not a new independent ready frontier.
The matched generic control is device-resident conditional execution with the same gate policy.
It must preserve minimum R2, maximum R4, selected outputs, and required missing-depth KV handling.
Every participating GPU must follow compatible collective and routing decisions.

An FP32 device CDF does not automatically match Python binary64 arithmetic near the threshold.
Matching the current policy may require binary64 state and the same separate multiplication/addition order.
Matching the sigmoid result also needs qualification.
A new implementation must declare any numerical difference before reporting model quality or speed.
The source audit proves neither bitwise equivalence nor a quality regression.

## 7. Smallest discriminator and stopping rule

Before an engine, identify one exposed cost absent from the matched generic control.
The discriminator needs these conditions:

1. Retain same-P TP, native async scheduling, whole-R4 capture, weight aliases, and the exact KV policy.
2. For launch-only work, keep internal kernels and dependencies unchanged.
3. For fusion, compare against generic persistent fusion with the same task granularity and cache policy.
4. For adaptive control, retain the original gate arithmetic and output-selection contract.
5. Attribute only exclusive critical-path time to removed launch, boundary, communication, or host-control costs.

Stop the recurrence-specific claim if aliased unrolling receives the same benefit under the same generic runtime.
Stop engine work if no removable cost can cover added control, synchronization, and kernel-efficiency costs.
P4/P8 receive no projected speedup from the P2 local-window fraction.
The API analysis applies to P2/P4/P8, but performance and progress need separate qualification.

## 8. Source and time limits

The [structured observations](../artifacts/device_launched_graph_observations_v1.json) record source locators, hashes, searches, and timing checkpoints.
Source inspection reached the 13:19:59 UTC checkpoint.
Report synthesis and validation followed.
The evidence and validation checkpoint was 13:22:54 UTC.
This segment lasted 330 seconds to that checkpoint.
This audit adds no executable engine or GPU probe.
The separate PT/BFS segment ended before this segment began.
Concurrent campaign work does not multiply this worker's elapsed time.
