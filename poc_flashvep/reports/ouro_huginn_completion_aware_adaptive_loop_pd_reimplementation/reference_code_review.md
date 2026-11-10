# Reference code review

## FluidPD

[FluidPD](https://arxiv.org/html/2610.06917v1) defines PPI from queued work and remaining TTFT deadlines.
DPI includes active Decode load and reserved KV capacity.
FluidToken transfers a computed token prefix and completes its suffix on D.
Its policy includes suffix completion in the TTFT path.
The suffix executes at the next eligible Prefill iteration after import.
Its stealing budget derives from predicted Decode iteration cost and TPOT slack.
The paper permits a multiplier above one to borrow average stream-level slack.
Our stricter ITL-tail contract does not inherit that permission automatically.

The inspected paper and repository search did not establish an official public implementation.
The local Token method remains a FluidToken-style reproduction, not official FluidPD.
We reuse the pressure, KV-credit, and completion-admission concepts.
We do not copy its token dependency model into recurrent state ownership.

## SGLang

The inspected source is official [decode.py at fe30d1e0fd4ede090b48a4147d23e23571b327f2](https://github.com/sgl-project/sglang/blob/fe30d1e0fd4ede090b48a4147d23e23571b327f2/python/sglang/srt/disaggregation/decode.py).
`DecodeReqToTokenPool` separates running slots from preallocated and transfer slots.
`DecodePreallocQueue` reserves resources before publication.
Pending publication can wait for a CUDA allocation event.
`DecodeTransferQueue` separates received KV from runnable Decode requests.
This separation supports asynchronous handoff without treating transfer-pending requests as executable work.

We reuse the lifecycle separation as a design principle.
We retain the qualified runtime's Native reservation and ACK implementation.
We do not transplant SGLang request objects or allocators into vLLM-RLT.
The two runtimes use different state, scheduler, and KV interfaces.

## Local ref_loopserve

The local README identifies a vLLM Grouped-EE elevator runtime.
Its fixed-depth path and Early Exit path share parts of the infrastructure.
This PoC does not enable Early Exit.

The inspected sources include:

- `source/vllm/v1/core/sched/scheduler.py`
- `source/vllm/v1/worker/ouro_runner/loop_spec.py`
- `source/vllm/v1/worker/ouro_runner/loop_planner.py`
- `source/vllm/v1/worker/ouro_runner/runtime_owner.py`
- `source/vllm/v1/worker/ouro_runner/grouped_prefill_mixin.py`
- `source/vllm/v1/worker/ouro_runner/grouped_service_mixin.py`
- `source/csrc/ouro_lockstep/graph_runtime.cu`

The scheduler distinguishes KV-transfer pending state from request execution.
Its KV reservation horizon can differ from the execution window.
`loop_spec.py` assigns entry, recurrent, exit, and boundary operations explicitly.
Huginn Coda KV must become valid before dependent attention executes.
This requirement also constrains our continuation frontier.

`runtime_owner.py` records payload identity, generation, token position, and KV generation separately.
It protects mapped command publication during shutdown.
The graph controller uses resident state and generation checks.
These mechanisms motivate explicit leases, epochs, and ownership in this PoC.

The cost planner anchors predictions to current measured wave costs.
It retains the current policy when predicted improvement does not exceed its configured gain threshold.
This motivates measured calibration rather than recurrent-row scaling alone.

Direct binary reuse is not safe.
The reference uses vLLM request objects, pooled depth KV, mapped queues, and a resident GPU graph controller.
The qualified PD runtime uses `vllm_rlt` requests, explicit execution batches, and Native NIXL ownership.
A direct controller transplant would require a new numeric and lifecycle qualification campaign.

## Qualified vLLM-RLT reuse

The new implementation retains the existing Native-path connector and deterministic graph warm-up.
It reuses exact token and recurrent frontiers, existing Attention kernels, and Native source ACK.
No new CUDA model kernel is required.

The optional campaign Scheduler replaces the old admission and execution hooks only in its worker processes.
The original runtime checkout remains unchanged.
The disabled feature path remains Native Atomic PD.
Checkpoint-capable requests still pay boundary costs, even when D later rejects them.
The evaluation must measure that cost rather than claim a free zero-offload path.
