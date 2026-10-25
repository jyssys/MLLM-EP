# Implementation scope

## Native components

The IFM adapter uses load_native_model from the pinned official xLLM source.
The Foil adapter uses AutoModelForCausalLM with the official Foil-1 remote code.
Neither adapter changes the architecture, learned parameters, attention math, or recurrent pass count.
Native FA4 is an official xLLM dispatch option.
Its numerical qualification remains explicit.

## New research code

| File | Responsibility |
| --- | --- |
| adapters.py | Official loading, tokenization, full-prefix inference, shifted NLL, and metadata |
| probes.py | Reversible router/expert observation, correct reuse keys, and pre-GEMM row omission |
| study.py | Routing statistics, primary intervention grid, and clean baseline timing |
| generation.py | Free-running modified trajectories, answer parsing, and stop accounting |
| timing.py | Separate CUDA-event spans and native-weight microbenchmarks |
| profiling.py | Actual CPU/CUDA timelines, official API parity, and contention controls |
| backend_diagnostic.py | Actual-Q/K/V FP32 attention comparisons and routed-model drift |
| dispatch.py | Locked, resumable queues of independent single-GPU workers |
| resources.py | Existing ownership-aware burner helper, scoped to this campaign |
| analyze.py | Sequence-level bootstrap intervals and machine-readable summaries |
| reports.py and plots.py | Tables and editable figures from completed evidence |
| provenance.py, archive.py, verify.py | Version locks, lossless raw preservation, and integrity checks |

## Optimization boundary

P2 interventions actually omit selected rows before expert arithmetic.
They are diagnostic quality executors, not optimized serving implementations.
The selection path can synchronize the GPU and allocate temporary tensors.
Oracle arms also require extra reference execution.
Their wall times therefore do not measure achievable serving speedup.

P3 did not run.
No new kernel, persistent model patch, EP, TP, PP, or speculative engine was implemented.
No unrelated prior PoC was resumed.

## Incomplete measurements

Actual tensor-core instruction reductions and HBM bandwidth remain NOT_MEASURED.
Selector and cache traffic costs remain unqualified for a production implementation.
Dispatch/combine critical-path percentages remain UNRESOLVED.
GSM8K/MATH natural accuracy uses eight-question pilots rather than 128-question evaluations.
The task-quality margin remains unverified.
The negative timing gate prevents an unsupported positive verdict.
