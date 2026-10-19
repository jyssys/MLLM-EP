# Hardware counter access attempt

The targeted hardware-counter lane did not collect a valid metric.
Nsight Compute returned `ERR_NVGPUCTRPERM` on logical CUDA device 0.
The guard mapped that device to physical GPU5 and retained its UUID.
The experiment did not use another physical GPU.

The installed CLI is `/usr/local/cuda-12.8/bin/ncu`, version 2025.1.1.0.
It was absent from the shell PATH but present in the local CUDA installation.
The command requested only DRAM read/write byte counters.
It used an actual pretrained MLP-shard fixture and a bounded NVTX range.
This was a single-GPU component experiment, not native TP2 execution.

The profiler could not access the device counters under the current user permissions.
The experiment changed no driver setting, permission, clock, environment package, or foreign process.
The guard restored the owned GPU5 burner after the failed command.
No counter retry with elevated authority will occur in this campaign.

## Profiling controls

The command selected `--clock-control none` and `--cache-control none`.
It did not change GPU clocks or intentionally flush cache state between profiler passes.
Application replay and profiler serialization would still require separate interpretation if counters became available.
[Nsight Compute profiling guide, replay and cache control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

The profiler also emitted a Python `sitecustomize` warning before connecting to the target process.
The preserved log records both that warning and the explicit counter-permission error.
No result depends on assigning one cause to every startup message.

## Evidence boundary

`outputs/counter_MLP_alias_base14_v1/console.log` preserves the failed attempt.
`run_contract.json` preserves the command and physical GPU mask.
`guard_status.json` records exit code 1 and burner restoration.
The fixture produced no validated DRAM byte result.

Weight tensor capacity remains structural accounting.
Alias/intervening-access timings remain controlled component measurements.
Neither becomes measured HBM traffic because a counter experiment was attempted.
