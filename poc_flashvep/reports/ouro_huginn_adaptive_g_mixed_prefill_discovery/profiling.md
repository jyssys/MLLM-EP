# Separate heavy profiling

The profile uses producer-stream CUDA events after a complete warm-up workload.
Its raw records retain interval start, interval end, and elapsed milliseconds.
Host submission plans remain a different clock.
The figures identify their evidence lane.

| Model | Arm | Nominal G | Operation | Intervals | Median stream ms | P95 stream ms |
| --- | --- | --- | --- | --- | --- | --- |
| huginn | group | 2 | coda | 69 | 1.2144 | 2.2972 |
| huginn | group | 2 | prelude | 67 | 2.4369 | 7.4684 |
| huginn | group | 2 | recurrent | 619 | 2.0297 | 9.7028 |
| huginn | group | 4 | coda | 69 | 1.2151 | 2.2893 |
| huginn | group | 4 | prelude | 67 | 2.4038 | 8.6604 |
| huginn | group | 4 | recurrent | 583 | 2.0300 | 2.0386 |
| huginn | hybrid | 2 | coda | 44 | 1.2941 | 2.3461 |
| huginn | hybrid | 2 | prelude | 43 | 2.5136 | 12.8970 |
| huginn | hybrid | 2 | recurrent | 616 | 2.0325 | 5.6502 |
| huginn | mixed | 1 | coda | 44 | 1.2961 | 2.3212 |
| huginn | mixed | 1 | prelude | 43 | 2.5497 | 9.3663 |
| huginn | mixed | 1 | recurrent | 624 | 2.0285 | 5.5799 |
| ouro | group | 2 | coda | 39 | 0.0771 | 0.0849 |
| ouro | group | 2 | prelude | 43 | 0.0535 | 0.0989 |
| ouro | group | 2 | recurrent | 158 | 5.1807 | 42.8079 |
| ouro | group | 4 | coda | 38 | 0.0780 | 0.0910 |
| ouro | group | 4 | prelude | 42 | 0.0515 | 0.0864 |
| ouro | group | 4 | recurrent | 149 | 5.1677 | 5.2654 |
| ouro | hybrid | 2 | coda | 41 | 0.0772 | 0.1468 |
| ouro | hybrid | 2 | prelude | 45 | 0.0520 | 0.0892 |
| ouro | hybrid | 2 | recurrent | 155 | 5.1924 | 23.0330 |
| ouro | mixed | 1 | coda | 41 | 0.0780 | 0.1446 |
| ouro | mixed | 1 | prelude | 45 | 0.0537 | 0.0840 |
| ouro | mixed | 1 | recurrent | 157 | 5.1873 | 23.3322 |

A recurrent interval can contain multiple required prompt passes.
It includes GPU stream elapsed time and possible launch gaps.
It is not a hardware-counter measurement of kernel-only busy time.
CPU scheduling can overlap GPU execution.
The report does not sum CPU and GPU intervals into an E2E speedup.
It does not sum overlapping stream intervals into a 100% critical-path table.

Grouped prefill retains native per-pass attention/GEMM execution.
The patch does not fuse CUDA kernels or create a multi-loop CUDA Graph.
Mixed G1 can co-pack prompt and decode rows at different depths.
Its query width can exceed graph support and use the native eager fallback.
The clean raw rows preserve actual capture/replay/fallback increments.

`observer_parity.csv` contains three ON/OFF blocks per model.
Their elapsed-time ratios describe the light plan observer, not the heavy tensor audit.
The clean confirmation excludes that observer.
Exclusive GPU idle, kernel launch attribution, and HBM counters remain unresolved in this scope.
