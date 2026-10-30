# Profiling, Accounting, and Limits

All GPU measurements in this file are profiler-only observations.
They are separate from the clean finite-cohort latency tables.

## Profile configuration

Each profile uses two real requests, input length512, output cap4, and chunk128.
The model executes fixed Ouro R4 or Huginn R32 in BF16.
Each run uses physical GPU0 for P and GPU1 for D.
Both workers use the same released numerical correction as the fixed-cut maps.
Warm-up occurs before capture. PyTorch records CPU and CUDA activities with shapes.
The native exporter preserves each rank in `cycle-00000.tar.gz`.

The archive contains `trace.json`, metadata, operator tables, event records, and a summary.
The analysis reads the archive without extracting or modifying it.
The archive manifest records the trace checksum.

## Actual kernel time

Kernel busy time is the union of captured CUDA kernel intervals on one GPU.
The kernel window extends from that GPU's first kernel to its last kernel.
It is not request E2E or the complete serving window.

| Model and cut | P kernel busy union ms | D kernel busy union ms | P kernel window ms | D kernel window ms |
|---|---:|---:|---:|---:|
| Ouro Cut4 | 175.06 | 59.40 | 1,311.69 | 80.93 |
| Ouro Cut2 | 88.03 | 151.65 | 867.47 | 741.10 |
| Huginn Cut32 | 629.34 | 204.36 | 2,345.94 | 363.95 |
| Huginn Cut16 | 319.38 | 523.49 | 1,216.13 | 1,518.21 |

The midpoint moves GPU work from P to D.
It does not remove the complete model work.
The summed per-device busy unions remain similar for these full and midpoint cuts.
That sum is not the multi-GPU critical path.

The traces show substantial intervals without CUDA kernels.
Those intervals can contain CPU preparation, dependency waits, GPU copies, or profiler overhead.
They must not be reported as clean-serving GPU idle percentages.

## Concurrent GPU execution

The analysis aligns both ranks with each trace's `baseTimeNanoseconds`.
It intersects the actual kernel unions, not ten-ms visualization bins.
All four two-request profiles have 0.00 ms of captured cross-device kernel overlap.
This is a measured result for those profiled two-request cohorts only.
It does not establish zero overlap for the eight-request clean maps.
It also does not establish that profiling preserves clean scheduling.

The wavefront order can finish most P work before the first receiver begins its kernels.
The final handoff and CPU submission path can further delay D.
These observations explain why moving work is not automatically parallel GPU execution.
They do not quantify an end-to-end gain or loss by themselves.

## GEMM and Attention accounting

`profiler_kernel_breakdown.csv` preserves actual kernel counts and inclusive durations.
Its names identify Attention and GEMM kernel families.
The captured External-id chain maps `nvjet_tst` launches to `aten::mm`.
The classifier therefore includes those observed kernels in the GEMM family.

`profiler_directly_correlated_GEMM_time.csv` uses CPU matrix dimensions and External IDs.
It separates directly mapped FFN, Attention-projection, and Huginn adapter GEMMs.
CUDA Graph replay can omit a matching CPU operation.
These mapped module times are incomplete, not total FFN or Attention-projection time.
Normalization-name matches are also incomplete.
Unclassified kernels remain explicit in the tables.

`profiler_observed_GEMM_shapes.csv` preserves actual observed matrix shapes.
Graph replay does not invoke Python or aten dispatch again.
Thus CPU operation counts must not be reported as all replayed GEMM invocations.
GPU kernel counts remain available separately.

The qualified numerical path pads selected Down projections to 256-row tiles.
Huginn also pads its reinjection adapter to that geometry.
This adds real work, especially for small Decode batches.
Every measured candidate and control retains that cost.
The experiment is not an optimized production-vLLM throughput claim.

## E2E component percentages

`e2e_component_shares.csv` contains per-request percentages, including background requests.
`e2e_component_shares_aggregate.csv` divides component sums by summed all-request E2E.
The components are exclusive host-observed intervals.
They cover the denominator exactly.
No request partition was excluded for missing or nonmonotonic observations.

| Stored component | Meaning | What it does not mean |
|---|---|---|
| `harness_delivery` | Scheduled arrival to coordinator admission | HTTP or network latency |
| `coordinator_P_queue` | Admission through reservation and P acceptance | Pure P-GPU queue time |
| `P_interval_observed` | P acceptance to observed compute-ready state | Exclusive GPU computation |
| `exposed_handoff_observed` | Observed compute-ready state to D receipt | Pure DMA time |
| `D_wait_and_continuation_to_first_token` | D receipt to first output delivery | Pure Decode or pure GPU compute |
| `generation_after_first_token` | First output delivery to final output delivery | GPU-only token generation |

Cut0 has a different valid partition because P performs no request computation.
Its reservation and D-Prefill intervals remain explicit.
In a loaded-D cohort, the initial background requests still use Full Cut R.
The aggregate `cut` column identifies the target cut.
The per-request table records `actual_request_cut` and `is_target` separately.
The sum of request intervals is not cohort wall time.
GPU kernels and transfer windows can overlap across requests.
No dispatch/GEMM/DMA percentage is fabricated from these host intervals.

## Transport accounting

All 145 snapshot cohort payload checks reconcile with actual NIXL byte counters.
The six native streaming control cells retain total transport counters separately.
Their KV/hidden split uses the known one-carrier final hidden size.
That split is not an independent per-segment reconciliation.

An L512 Ouro Full Cut transfers 402,653,184 KV bytes and 4,096 hidden bytes per request.
Cut2 transfers 201,326,592 KV bytes and 2,097,152 hidden bytes.
The corresponding snapshot transfer counts are seven and four per request in the recorded calibration cell.

An L512 Huginn Full Cut transfers 1,427,331,840 KV bytes and 21,120 hidden bytes per request.
Cut16 transfers 713,687,040 KV bytes and 10,813,440 hidden bytes.
The corresponding snapshot transfer counts are 22 and 11 per request in that calibration cell.

These counts come from actual transfer completions.
They are not estimates of link bandwidth.
Control bytes, exact hardware link traffic, and pure KV DMA latency are NOT_MEASURED.
The topology reports NV18 connectivity. The transport backend is NIXL/UCX.
The study does not infer a particular physical link bandwidth from that topology label.

## Main limitations

The map uses a token budget equal to the token chunk size.
It does not tune a larger pure-Prefill batch budget.
Small token chunks do not inherently require small GEMMs when multiple requests can share a batch.
The current map therefore cannot prove a Loop-specific GEMM-efficiency advantage.

The matched snapshot control reserves continuation staging capacity even at Full Cut R.
Dedicated native full PD needs less registered hidden HBM.
The fixed KV pool does not give native PD extra capacity from that saving.
No maximum-HBM capacity tuning or nine-arm capacity comparison ran.

The Huginn KV pool can constrain long-request admission before four compute slots fill.
That is a runtime resource condition, not a general architectural limit.
Internal logical-layer and token-prefix migration remain unimplemented.
Original-HF fidelity and benchmark accuracy remain unmeasured here.

## Preserved failures

Four initial GPU job namespaces failed during adapter development.
The first pair failed on a conflicting wrapper argument named `peer`.
The second pair failed on duplicate receiver-prefix initialization.
The corrected adapter then passed both models' finite qualification.
The failed namespaces and logs remain unchanged.

An early CPU parser inspected a temporary profiler file before archive finalization.
That parser failed when the exporter removed its temporary directory.
The final parser reads only completed archives. No GPU rerun was needed.

Some completed runs emitted CUDA-context warnings during UCX teardown.
Their logs remain available.
Successful worker status alone did not establish correctness.
The output, KV, hidden, and work checks supply that evidence.
