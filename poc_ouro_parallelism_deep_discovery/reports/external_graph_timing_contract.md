# External graph timing contract

This diagnostic records every observed replay in a request.
It does not multiply the final replay duration by the number of tokens.
It does not expose graph-internal component times.

## Stream boundary

PyTorch v2.11.0 launches a replay on the current CUDA stream.
The implementation calls `cudaGraphLaunch` from `CUDAGraph::replay`.
The inspected release source is lines 174–190.
[PyTorch release source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CUDAGraph.cpp#L174).

CUDA orders the launch with other work in that stream.
An event after the launch measures a complete graph envelope, not one internal kernel.
[CUDA 12.9.2 graph API](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-runtime-api/group__CUDART__GRAPH.html).

The installed Python wrapper delegates `replay()` to its C++ implementation.
The installed package reports PyTorch 2.11.0+cu129.
The release-source inspection is not a binary identity check.
The environment manifest retains the installed package metadata.

## Measurement procedure

The worker creates and materializes an event pool before the observed request.
It places one event pair around each actual Python replay call.
It also records outer `execute_model` and `sample_tokens` calls.
Each pair retains its stream identity and rank-local timestamps.
The worker reads completed events only after generation returns.
It adds no synchronization inside a replay or between model layers.

The outer intervals can contain graph intervals.
The analysis must not sum nested intervals.
An empty cleanup call is not a full model forward.
Sampling and staging outside the graph require their own interpretation.

## Critical-path limits

GPU event clocks are rank-local.
This diagnostic does not align clocks across GPUs.
It may select one rank by cumulative complete-graph duration.
It must then retain that rank for the reported cumulative graph interval.
It must not add independent per-component rank maxima.

An interval between two graph calls is unattributed.
Other kernels, communication, host readiness, or queue effects can occupy it.
It is not automatically GPU idle or removable latency.
It is not automatically recurrence-specific.

## Validity controls

Each context has three warm-up requests.
Three clean requests precede three observed requests.
Three clean requests follow the observed requests.
The report must retain both clean medians and their drift.
It must retain observer overhead against each clean median.
Teacher-forced output identity must hold in all lanes.

The native async scheduler remains enabled.
This is a stronger generic baseline control, not a candidate mechanism.
The first lane uses physical GPUs 4 and 5.
Its paired controls use the same devices and process.
The historical TP2 timing used GPUs 4 and 6, so absolute cross-run equality is not assumed.
