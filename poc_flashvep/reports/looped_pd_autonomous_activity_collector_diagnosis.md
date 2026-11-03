# GPU activity collector diagnosis

## Scope

This report separates collector failures from model execution failures.
No collector diagnostic establishes serving headroom or model accuracy.

## Native collector self-test

The test executes 100 elementwise GPU operations per case.
Graph cases replay a previously captured CUDA Graph.
The test uses the qualified software environment and an isolated GPU lane.
It does not load a proxy model or replace any model experiment.

| Activities | Execution | Dynamic activity toggle | Recorded kernel events |
|---|---|---|---:|
| CUDA | Direct | No | 100 |
| CUDA | Graph replay | No | 100 |
| CUDA | Direct | Yes | 0 |
| CUDA | Graph replay | Yes | 0 |
| CUDA + CPU | Direct | No | 100 |
| CUDA + CPU | Graph replay | No | 100 |
| CUDA + CPU | Direct | Yes | 0 |
| CUDA + CPU | Graph replay | Yes | 0 |

These are measured collector-event counts, not throughput measurements.
The tested toggle sequence disables collection, enables collection, executes work, and disables collection before export.
The test does not isolate which toggle transition causes event loss.

Raw evidence: `looped_pd_autonomous_prefill_compute_decode_headroom/raw/collector_activity_toggle_selftest/`.

## Huginn observations

The native bounded collector completed `huginn_native_cuda_bounded_graph_control24`.
Its exported trace contained no kernel events.
The trace contained 54 metadata events and one trace-level event.
Therefore, the result is `NOT_MEASURED_GPU_ACTIVITY`, not zero GPU activity.

The self-test reproduces event loss without NIXL, recurrent execution, or model operations.
This narrows the empty-trace problem to the tested collector path.
It does not explain every previous profiling failure.

## Separate Nsight failure

Full-interval Huginn Nsight captures stopped making response progress in the native NIXL transfer call.
The post-warmup stats handshake completed, but the serving run still stalled after ten measured responses.
Thus, collection during graph warm-up is not a sufficient explanation for that stall.

Two additional controls used a zero CUDA flush interval.
Both node-level and graph-level collection failed to complete.
The failures remain in their original run directories.

Installed `nsys profile --help` states that graph-level tracing omits node activities.
Graph-level durations must not be labeled kernel-union measurements.

## Next controlled test

The new native path starts the profiler during a predefined serving interval.
It stops the profiler after a short capture window.
It does not call the activity-toggle API.

Model operations, BF16, fixed depth, and native transfers remain unchanged.
The test must retain kernel events, preserve output parity, and pass the existing perturbation criterion.
Until those checks pass, the collector remains unqualified for the final headroom claim.

## Reproduction

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.run collector_activity_toggle_selftest
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.native_start_stop_control
nsys profile --help
```

Each run records its configuration, source hashes, logs, and process status.
