# Exact single-GPU controls

The native cache stores physical layers inside each page block.
Its per-layer view has a 1.5 MiB block stride for K alone.
The layer-major control reduces that stride to 64 KiB.
It preserves the public tensor axes and depth-specific page ownership.
It does not reduce valid KV payload or model execution depth.

| Prompt | B | Lane | Start pairs | Native ITL, ms | Layer-major ITL, ms | ITL speedup, 95% CI | Cohort E2E speedup, 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1 | PARALLEL | 3 | 20.172 | 20.138 | 1.002× [0.999,1.004] | 0.967× [0.898,1.007] |
| 2048 | 1 | QUIET | 3 | 20.167 | 20.170 | 1.000× [0.998,1.001] | 1.003× [0.998,1.011] |
| 8192 | 4 | PARALLEL | 3 | 31.423 | 31.331 | 1.003× [1.001,1.004] | 0.999× [0.995,1.002] |
| 32768 | 1 | PARALLEL | 3 | 62.349 | 62.418 | 0.999× [0.998,1.000] | 0.994× [0.965,1.011] |
| 32768 | 1 | QUIET | 3 | 62.305 | 62.276 | 1.000× [1.000,1.001] | 1.007× [1.006,1.010] |
| 65408 | 1 | PARALLEL | 3 | 107.488 | 107.465 | 1.000× [0.997,1.003] | 1.037× [0.907,1.197] |
| 65408 | 1 | QUIET | 3 | 107.184 | 107.169 | 1.000× [0.999,1.001] | 0.998× [0.987,1.006] |


The confidence interval resamples independent matched process-start pairs.
Technical repeats and model layers are not independent statistical samples.
Quiet pairs use the same physical GPU within each comparison.
The parallel discovery lane remains separate.
The near-64K Quiet Lane E2E speedup is 0.998×, with 95% CI [0.987, 1.006].

The large parallel E2E outlier did not reproduce as a meaningful Quiet Lane improvement.

The dedicated initialized-page test compares three physical layers at all four depths.
Its twelve sampled Q/K/V/attention snapshots are byte-exact across layouts.
The corresponding two-token hardware outputs and depth arrays also match.
`layout_correctness.json` retains every comparison.
This limited probe is not benchmark quality evidence.

## Strong generic local attention

Pinned FA4 rejects internal SplitKV on SM90.
The study preserves that failure and tests a generic alternative.
It exposes context-page segments as extra FA4 batch rows.
The segments share the original K/V storage without a KV repack.
Query replication and the stable merge execute inside the measured CUDA Graph.
This is a Flash-Decoding-style control, not a loop-aware method.

| Visible keys | B | 128 MiB pressure | Best measured local S | Process starts | Local segmented wall, ms | Native S1 / segmented |
| --- | --- | --- | --- | --- | --- | --- |
| 2049 | 1 | False | 8 | 3 | 0.0432 | 1.086× |
| 2049 | 1 | True | 4 | 3 | 0.0502 | 1.032× |
| 2049 | 8 | False | 1 | 3 | 0.0655 | 1.000× |
| 2049 | 8 | True | 1 | 3 | 0.0756 | 1.000× |
| 8193 | 1 | False | 8 | 3 | 0.0634 | 1.997× |
| 8193 | 1 | True | 8 | 3 | 0.0724 | 1.870× |
| 8193 | 8 | False | 1 | 3 | 0.1931 | 1.000× |
| 8193 | 8 | True | 1 | 3 | 0.2033 | 1.000× |
| 32769 | 1 | False | 8 | 3 | 0.1267 | 3.496× |
| 32769 | 1 | True | 8 | 3 | 0.1370 | 3.264× |
| 32769 | 8 | False | 1 | 3 | 0.7028 | 1.000× |
| 32769 | 8 | True | 1 | 3 | 0.7096 | 1.000× |
| 65409 | 1 | False | 8 | 3 | 0.2125 | 4.045× |
| 65409 | 1 | True | 8 | 3 | 0.2224 | 3.857× |
| 65409 | 8 | False | 1 | 3 | 1.3723 | 1.000× |
| 65409 | 8 | True | 1 | 3 | 1.3830 | 1.000× |


The best S is a measured tuning envelope over S2/S4/S8/S16.
The envelope also retains native S1 when it is faster.
It is not a validated online controller.
The control uses the same source fixtures and registered numerical tolerances.
It tests whether CP gains reflect missing generic local sequence parallelism.
No full-model integration or accuracy gain is inferred from this operator.
