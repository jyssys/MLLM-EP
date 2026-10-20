# Timing contract

## Evidence classes

- LIVE_MEASURED: actual inference with pretrained weights on physical GPUs 4,5,6,7.
- GPU_BRACKET: CUDA events around a stage or peer-copy call. A bracket can contain host launch gaps.
- NSIGHT_ACTIVITY: kernel or copy intervals on the profiler clock. These are heavy-lane measurements.
- STRUCTURAL: bytes or counts from tensor dimensions and ownership checks.
- NOT_RUN: an experiment that did not execute.

Never add four ranks' service times to obtain a request wall time.
Never present a CUDA bracket as exclusive kernel execution time.
Never present a replay or a structural transfer count as measured throughput.

## Workload and stopping

Each stage consumes one entire prompt or one eligible decode token from one request.
There is no prompt chunking and no per-stage request batching.
Concurrency is the number of admitted independent requests, not a tensor batch dimension.
Each request has a distinct cache namespace.

The primary workload forces exactly 32 or 128 generated tokens.
EOS does not terminate those hardware comparisons.
The tokenizer and frozen token IDs are identical across placements.
The text originates from preserved requests. Repetition and truncation control its shape.
These runs do not measure benchmark accuracy.

## Clean serving clock

The serving clock excludes model setup, common warm-up, and graph preparation.
The workload makespan includes prefill, all token forwards, peer copies, head work, token return, and host control.
TTFT ends when the first token ID is available to the controller.
Request E2E ends when its final token ID is available.
TPOT is `(last_token_time - first_token_time) / (generated_tokens - 1)`.
Tokens/s divides all emitted tokens by the workload makespan.
Requests/s divides completed requests by that same makespan.

Graph construction is outside the primary serving clock.
Every raw result records its graph-preparation wall time.
The graph-capture count during each headline run must be zero.
This is warm-runtime serving latency, not process initialization or graph-compilation latency.

Clean mode records one local completion event per stage for safe publication.
It does not record per-layer events or run a heavy profiler.
No global barrier occurs between stages or loops.
Workload boundaries synchronize the four devices.

## Common graph implementation

The two arms use the same official BF16 decoder and SDPA implementation.
Both prefill and decode use whole-stage CUDA Graphs in the final runtime.
The decode graph uses the exact current attention length.
It does not include masked future KV in SDPA.
Owner-local KV moves through a graph scratch buffer on the same GPU.
These extra copies and graph-input copies count toward serving latency.
They are implementation overhead, not remote activation traffic or new attention work.

Graph scratch has separate storage for each stage shape.
It can consume much more memory than the logical KV state.
Raw records distinguish weights, owner-local KV, allocated memory, reserved memory, and peak allocated memory.
The allocation peak is a PyTorch peak, not an instantaneous NVML peak for every driver allocation.
Graph buffers from earlier shapes can remain resident within a worker process.

## Trace clocks and GPU activity

Trace mode records transfer, compute, and head event brackets.
Each GPU event clock has a host-bounded anchor.
The trace also checks the ending anchor when the workload finishes.
Cross-rank overlap must exceed the recorded clock uncertainty.
Nsight provides an independent common clock for physical kernel-overlap evidence.

Queue waiting is time from ready-queue insertion to worker dispatch.
A stage's host interval includes its local publication wait.
The complement of stage brackets is not a measured whole-GPU idle fraction.
Nsight activity unions provide the separate kernel/copy activity measure.
That measure does not imply full SM utilization while a kernel runs.

## Communication accounting

Activation tensors use direct GPU peer copies.
The controller never receives a hidden-state tensor in host memory.
Initial token IDs use H2D copies. Terminal scalar token IDs use D2H control copies.
The returned token tensor travels directly from its terminal GPU to its entry GPU.
Raw records keep hidden, token-return, and input-copy payload counts separate.
Payload bytes exclude driver and protocol overhead.

A CUDA-event peer-copy bracket can include enqueue and allocator delays.
Its bytes/time ratio is apparent copy throughput, not pure link bandwidth.
Only an isolated profiler data-movement interval can support effective bandwidth.

## Repeats and uncertainty

The primary plan uses three measured repeats after warm-up.
Arm order alternates between repeats.
The speedup is median Layer-PP makespan divided by median Loop-PP makespan.
Confidence intervals use 2000 paired whole-workload bootstrap resamples.
Tokens, layers, and ranks are not independent serving samples.
Three repeats provide limited uncertainty evidence.

## Dynamic depth boundary

Fixed R2/R3 execute exactly two/three loops for both prefill and decode.
This is not the earlier full-R4-prefill adaptive CDB workload.
The official HF exit selector can select a state after all configured loops execute.
That selector does not establish faithful online dynamic-depth pipeline execution.
True dynamic exit requires separate KV and quality validation.
