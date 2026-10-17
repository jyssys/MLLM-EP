# Latency gate and measurement scope

The specification requires task-quality success before distributed latency measurements or prototype implementation.
The quality controller does not launch a distributed approximation engine.
Independent single-GPU replicas evaluate accuracy arms. They are not PP4, TP4, or an asynchronous pipeline.

Single-GPU evaluation wall times include Python execution, cache growth, and observer costs.
These times are not optimized serving latency and cannot establish architecture speedup.
Teacher-forced CUDA events validate causal input availability on one GPU.
They do not measure stage overlap across GPUs or a four-GPU latency oracle.

No measured-cost asynchronous latency oracle is available before the quality gate passes.
No zero-cost simulation can substitute for this gate.
The analytical 16/7 and 16/4 examples remain conditional bounds, not latency results.

If quality fails, the following measurements remain `NOT_RUN`:

| Comparison | Measurement status |
|---|---|
| Exact optimized 1-GPU serving | NOT_RUN |
| Exact PP4 stage communication and decode | NOT_RUN |
| Exact TP4 serving | NOT_RUN |
| DP4 aggregate serving throughput | NOT_RUN |
| Warm-up 1 asynchronous PP4 | NOT_RUN |
| Warm-up 0 asynchronous PP4 | NOT_RUN |
| Measured-cost asynchronous latency oracle | NOT_RUN |
| Concurrency 1/2/4/8 latency and throughput frontier | NOT_RUN |
| Dynamic-depth extension | NOT_RUN |

The final report records the actual gate outcome. This file does not claim that an untested configuration is impossible.
