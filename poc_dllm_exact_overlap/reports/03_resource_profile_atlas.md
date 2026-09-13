# Resource profile atlas

## Evidence boundary

Hardware counters could not be collected because the node denies NVIDIA GPU
performance counters (`ERR_NVGPUCTRPERM`). Consequently, TensorCore, SM, HBM,
and NVLink intensities in the figure are qualitative source-derived labels, not
measured percentages. Actual pairwise contention is the decisive evidence.

| operation | expected dominant resource | measured concurrency fact |
|---|---|---|
| DeepEP dispatch | NVLink + communication SM/startup | hurts attention, shared, and local expert in the tested shapes |
| fused routed expert | TensorCore/SM/HBM | can hide part of another complete wave's combine; cannot profitably hide local-vs-remote split |
| DeepEP combine | NVLink + communication SM/startup | overlaps 0.03--0.07 ms with another complete expert wave; phase/tensor-size dependent |
| attention | TensorCore/SM/HBM | dispatch overlap is negative in all six sampled states |
| shared expert | TensorCore/SM/HBM | combine overlap sometimes positive, dispatch overlap negative |

## Main system fact

DeepEP communication is not “free NVLink work.” The current kernels consume
enough SM/memory-system resources that dispatch beside attention, shared expert,
or local routed expert causes contention. The only stable positive pattern is a
pipeline across complete independent waves, where communication issue/progress
can coexist with an already prepared expert wave.

This is consistent with DeepEP's asynchronous event interface, but also explains
why merely removing the Python wait does not create a request-level win. The
library's current documentation explicitly exposes `EventOverlap` for legal
independent work: https://github.com/deepseek-ai/DeepEP .
