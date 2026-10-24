# Single-GPU critical path

## Clean bounded hardware inference

| Prompt tokens | Independent processes | Mean TTFT, ms | Mean delivered ITL, ms | Logical KV, GiB | Peak allocated, GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2 | 201.82 | 18.759 | 0.750 | 57.113 |
| 2048 | 2 | 380.09 | 20.359 | 1.500 | 57.113 |
| 4096 | 2 | 754.13 | 22.961 | 3.000 | 57.113 |
| 8192 | 2 | 1521.89 | 28.802 | 6.000 | 57.113 |
| 16384 | 2 | 3023.05 | 39.989 | 12.000 | 57.113 |
| 32768 | 2 | 6152.68 | 62.748 | 24.000 | 57.113 |
| 65408 | 2 | 14298.74 | 108.342 | 47.906 | 57.113 |


## Separate heavy GPU trace

| Heavy trace | Complete R4 windows | Attention exclusive GPU service, ms | Sampling interval, ms | Trace-local ratio |
| --- | --- | --- | --- | --- |
| profile_P16384_N1 | 14 | 23.021 | 42.001 | 54.81% |
| profile_P2048_N1 | 14 | 3.367 | 22.913 | 14.70% |
| profile_P32768_N1 | 14 | 45.373 | 64.406 | 70.45% |
| profile_P65408_N1 | 14 | 89.901 | 109.262 | 82.28% |
| profile_P8192_N4 | 14 | 12.614 | 33.940 | 37.17% |


Each retained window spans consecutive GPU sampling completions.
It contains 96 actual attention kernels.
Partial capture boundaries do not enter this table.
Attention-exclusive GPU service excludes overlap with other recorded kernels.
The ratio uses the same heavy-trace sampling interval.
It is not a clean host E2E fraction.

Long context creates a substantial measured attention interval.
The old Thinking-model eager result of approximately 7.3% does not describe this native Base-model configuration.
The clean ITL also increases with context.
These observations support long-context attention diagnosis.
They do not establish an integrated distributed speedup.

The actual B1/32K FA4 launch grid is 16×1×1 across 1,440 recorded kernels.
The device reports 132 SMs and 50 MiB L2.
These are launch and hardware facts, not measured occupancy.
The local sequence-parallel control addresses this generic launch-geometry limitation.
Counter permission remains unavailable, so no measured SM utilization is inferred.

`profile_kernel_families.json` retains GEMM, KV-write, metadata, copy, and pointwise service.
QKV and FFN GEMMs remain unresolved within their combined family.
Pointwise kernels also include normalization, RoPE, and residual work.
No exclusive clean FFN percentage is claimed.
The trace shows many small kernels even though CUDA Graphs reduce CPU submission count.
Kernel count alone is not removable E2E time.

`counters/` retains the absolute-path Nsight Compute attempt.
Unavailable HBM counters remain missing.
Logical KV bytes divided by latency are not achieved DRAM bandwidth.

For an interval fraction f, the conditional Amdahl speedup is `1 / (1-f+f/s)`.
Using trace-local f gives a trace-based model only.
It does not measure clean request improvement.
Cold request E2E also contains prefill, which is outside the decode-only operator.

The following percentages are conditional reductions of the heavy-trace GPU sampling interval.
Evidence: `MODEL_UPPER_BOUND`.
They are not measured clean TPOT or cold request E2E improvements.

| Prompt | B | 1.5× attention | 2× attention | 4× attention | Infinite attention |
| --- | --- | --- | --- | --- | --- |
| 16384 | 1 | 18.3% | 27.4% | 41.1% | 54.8% |
| 2048 | 1 | 4.9% | 7.3% | 11.0% | 14.7% |
| 32768 | 1 | 23.5% | 35.2% | 52.8% | 70.4% |
| 65408 | 1 | 27.4% | 41.1% | 61.7% | 82.3% |
| 8192 | 4 | 12.4% | 18.6% | 27.9% | 37.2% |
