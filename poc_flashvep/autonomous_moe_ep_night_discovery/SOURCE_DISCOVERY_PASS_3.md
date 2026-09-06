# Source discovery pass 3 — runtime state and accounting

This pass was deliberately observational rather than aimed at proving a
specific optimization. The installed vLLM 0.20.0 worker path exposes
`vllm.v1.worker.dp_utils.coordinate_batch_across_dp`; it selects the CPU
process group when `disable_nccl_for_dp_synchronization` is true and otherwise
uses the DP device group. The API server accepts both
`--async-scheduling/--no-async-scheduling` and
`--disable-nccl-for-dp-synchronization/--no-disable-nccl-for-dp-synchronization`,
so these are genuine bounded controls rather than invented flags.

The DeepEP high-throughput prepare path calls
`ExpertTokensMetadata.make_from_list` for every MoE invocation. The installed
implementation constructs a fresh pageable CPU `int32` tensor and performs a
nonblocking `.to(device)` copy. This is explicitly marked TODO in the local
source. The observation motivates H33/H38, but the copy is a host/allocation
tax candidate, not evidence of a critical-path improvement by itself.

The worker's asynchronous scheduling path can also submit a dummy batch through
`GPUWorker.execute_dummy_batch`, a separate RPC from the normal
`GPUModelRunner.execute_model` path. The observer now records this RPC when it
occurs, allowing H36 to separate idle-rank participation from a missing trace
row. No dummy marker appeared in the current v4 trace because the first
wrapper imported a non-existent `GPUWorker` symbol; this was corrected to the
installed `Worker` class in the resumed fresh-worker run. H36 then recorded
6,399 dummy calls on each DP1 worker and 286 on each DP0 worker, while the
matched request-wave effect remained only -0.26%. Dummy participation is
observable but not a material latency driver in this control.
