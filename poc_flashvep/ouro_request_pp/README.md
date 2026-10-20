# Ouro request pipelines

This discovery compares two exact placements on four physical GPUs.
Layer-PP partitions 24 physical layers into four groups of six.
Loop-PP replicates the 24 layers and assigns one recurrent iteration to each GPU.
Both placements use a common FIFO runtime and exact owner-local KV.

The implementation uses one process with four CUDA worker threads.
It uses direct CUDA peer copies, not host activation staging or NCCL.
This is a bounded implementation choice, not a production serving engine.

Read the frozen specification and timing contract before interpreting the results.
The runtime does not implement true token-level dynamic exit.
It does not split prompts or batch several requests inside a stage.

## Main modules

- `model_adapter.py`: pinned official decoder, norms, RoPE, and ownership.
- `graph_cache.py`: exact-length graph buffers and owner-local KV staging.
- `runtime.py`: four independent FIFO workers and request state transitions.
- `reference.py`: single-GPU fixed-depth reference and numerical comparisons.
- `benchmark.py`: correctness and clean/trace workload execution.
- `campaign.py`: frozen measurement plans, commands, and durable logs.
- `analyze.py`: serving metrics, paired comparisons, and timelines.
- `gpu_lease.py`: scoped project-burner control.
- `profile_capture.py` and `parse_profile.py`: actual kernel overlap and physical peer-copy evidence.
- `receive_control_campaign.py`: common receive-buffer and RoPE controls after correctness checks.
- `test_evidence.py`: raw-route, cache, token-parity, and measured-overlap regressions.

The detailed reproduction instructions are in the report directory.
The tests and diagnostic probes retain negative results from implementation checks.
Do not use those failed implementations as performance baselines.

The default runtime preserves the measured v5 path.
`--preallocated-recv` removes per-handoff destination allocation.
`--precompute-rotary` reuses position-only constants outside graph replay.
Both optional paths need their own reference checks. They do not change placement or model weights.
