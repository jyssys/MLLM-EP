# Loop-aware KV residency and distributed attention discovery

## Verdict

**NO_GO_INCREMENTAL for a new loop-specific method.**
**CONDITIONAL for generic attention engineering, not a serving speedup claim.**

Long context exposes a real attention interval and substantial exact KV payload.
Real context-sharded attention runs on 2/4/8 physical H100 GPUs.
Short-context collective overhead and long-context operator scaling differ substantially.
However, the implementation is generic exact context parallelism.
No added loop-aware placement or prefetch policy demonstrates an incremental benefit.
The study does not convert that generic operator signal into a new method claim.

The generic local S8 operator reaches 3.264× at 32K and 3.857× near 64K in the cache-pressure diagnostic.
Against the best local control, B1 CP2 and CP4 are slower in every tested controlled context.
CP8 reaches 1.139× near 64K, with 95% CI [1.121, 1.151].
At 32K, CP8 is 0.982×, with 95% CI [0.975, 0.988].
These are operator measurements, not model serving speedups.

The near-64K Quiet Lane E2E speedup is 0.998×, with 95% CI [0.987, 1.006].

Thus, neither a large exact-layout gain nor a loop-specific distributed gain is established.

## Scope and completed work

Model: ByteDance/Ouro-1.4B Base, BF16, original Static R4.
Checkpoint: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Runtime: unmodified RLT `4a350344027960a48384626aa646f1765fda6d4f`.
The baseline uses native Async/Mixed, paged KV, FA4, and CUDA Graphs.
Physical GPUs 0–7 supplied independent workers and real distributed rank groups.
No Loop PP implementation or scheduling-policy research was resumed.

Completed model jobs: 140.
Retained model timing cohorts: 295.
Paired distributed operator rows: 4464.
The raw archive retains failed attempts as well as completed attempts.

## Native hardware baseline

| Prompt tokens | Independent processes | Mean TTFT, ms | Mean delivered ITL, ms | Logical KV, GiB | Peak allocated, GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2 | 201.82 | 18.759 | 0.750 | 57.113 |
| 2048 | 2 | 380.09 | 20.359 | 1.500 | 57.113 |
| 4096 | 2 | 754.13 | 22.961 | 3.000 | 57.113 |
| 8192 | 2 | 1521.89 | 28.802 | 6.000 | 57.113 |
| 16384 | 2 | 3023.05 | 39.989 | 12.000 | 57.113 |
| 32768 | 2 | 6152.68 | 62.748 | 24.000 | 57.113 |
| 65408 | 2 | 14298.74 | 108.342 | 47.906 | 57.113 |


These are real generated-token measurements with EOS ignored and cap 64.
They are not a natural accuracy or sustainable-serving benchmark.
The 65,408-token prompt remains below the 65,536 total-position limit.
An exact 65,536-token prompt plus generation was skipped.

## Physical attention interval

| Heavy trace | Complete R4 windows | Attention exclusive GPU service, ms | Sampling interval, ms | Trace-local ratio |
| --- | --- | --- | --- | --- |
| profile_P16384_N1 | 14 | 23.021 | 42.001 | 54.81% |
| profile_P2048_N1 | 14 | 3.367 | 22.913 | 14.70% |
| profile_P32768_N1 | 14 | 45.373 | 64.406 | 70.45% |
| profile_P65408_N1 | 14 | 89.901 | 109.262 | 82.28% |
| profile_P8192_N4 | 14 | 12.614 | 33.940 | 37.17% |


The table uses heavy-trace GPU intervals, not clean E2E percentages.
The clean critical-path fraction remains `UNRESOLVED` where causal attribution is incomplete.
The increasing attention interval is nevertheless a physical long-context observation.
The old eager Thinking-model 7.3% result is not reused as the present denominator.

## Exact generic controls

| Prompt | B | Lane | Start pairs | Native ITL, ms | Layer-major ITL, ms | ITL speedup, 95% CI | Cohort E2E speedup, 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1 | PARALLEL | 3 | 20.172 | 20.138 | 1.002× [0.999,1.004] | 0.967× [0.898,1.007] |
| 2048 | 1 | QUIET | 3 | 20.167 | 20.170 | 1.000× [0.998,1.001] | 1.003× [0.998,1.011] |
| 8192 | 4 | PARALLEL | 3 | 31.423 | 31.331 | 1.003× [1.001,1.004] | 0.999× [0.995,1.002] |
| 32768 | 1 | PARALLEL | 3 | 62.349 | 62.418 | 0.999× [0.998,1.000] | 0.994× [0.965,1.011] |
| 32768 | 1 | QUIET | 3 | 62.305 | 62.276 | 1.000× [1.000,1.001] | 1.007× [1.006,1.010] |
| 65408 | 1 | PARALLEL | 3 | 107.488 | 107.465 | 1.000× [0.997,1.003] | 1.037× [0.907,1.197] |
| 65408 | 1 | QUIET | 3 | 107.184 | 107.169 | 1.000× [0.999,1.001] | 0.998× [0.987,1.006] |


Layer-major storage is a generic allocator change.
Smaller physical pools save empty reservation, not valid KV bytes.
Host/peer movement tests charge original BF16 bytes and completion waits.
Already resident HBM does not supply a free prefetch opportunity.

The internal FA4 SplitKV option fails on this SM90 backend.
The study therefore tests generic local sequence segmentation with the same FA4 kernel.
It retains K/V in place and merges exact partial attention.
The control separates missing local kernel parallelism from a need for additional GPUs.

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


## Physical distributed result

The strong local sequence-parallel control is essential to interpretation.
It uses established Flash-Decoding principles, not recurrence-specific state.
The following B1 comparison includes a 128 MiB cache-pressure diagnostic.

| Visible keys | GPUs | Best local S | Local wall, ms | CP wall, ms | CP speedup over strong local, 95% CI |
| --- | --- | --- | --- | --- | --- |
| 2049 | 2 | 4 | 0.0502 | 0.0833 | 0.603× [0.520,0.659] |
| 2049 | 4 | 4 | 0.0502 | 0.0984 | 0.510× [0.370,0.664] |
| 2049 | 8 | 4 | 0.0502 | 0.0869 | 0.578× [0.561,0.598] |
| 8193 | 2 | 8 | 0.0724 | 0.1268 | 0.571× [0.510,0.608] |
| 8193 | 4 | 8 | 0.0724 | 0.1004 | 0.720× [0.684,0.743] |
| 8193 | 8 | 8 | 0.0724 | 0.1005 | 0.720× [0.670,0.751] |
| 32769 | 2 | 8 | 0.1370 | 0.2917 | 0.470× [0.436,0.491] |
| 32769 | 4 | 8 | 0.1370 | 0.1847 | 0.742× [0.735,0.750] |
| 32769 | 8 | 8 | 0.1370 | 0.1396 | 0.982× [0.975,0.988] |
| 65409 | 2 | 8 | 0.2224 | 0.4978 | 0.447× [0.429,0.458] |
| 65409 | 4 | 8 | 0.2224 | 0.2873 | 0.774× [0.768,0.780] |
| 65409 | 8 | 8 | 0.2224 | 0.1953 | 1.139× [1.121,1.151] |


The initial speedup over unsplit FA4 is not automatically a Multi-GPU requirement.
Any remaining operator advantage still needs a full-model and quality comparison.

| Visible keys | GPUs | Cache condition | Process starts | Local graph wall, ms | Distributed graph wall, ms | Operator speedup, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 2049 | 2 | repeated KV | 3 | 0.0470 | 0.0781 | 0.619× [0.479,0.693] |
| 2049 | 2 | pressure | 3 | 0.0530 | 0.0833 | 0.644× [0.547,0.702] |
| 8193 | 2 | repeated KV | 3 | 0.1276 | 0.1178 | 1.097× [0.932,1.188] |
| 8193 | 2 | pressure | 3 | 0.1358 | 0.1268 | 1.078× [0.959,1.141] |
| 16385 | 2 | repeated KV | 3 | 0.2317 | 0.1668 | 1.389× [1.372,1.408] |
| 32769 | 2 | repeated KV | 3 | 0.4413 | 0.2811 | 1.574× [1.457,1.633] |
| 32769 | 2 | pressure | 3 | 0.4475 | 0.2917 | 1.539× [1.423,1.599] |
| 65409 | 2 | repeated KV | 3 | 0.8569 | 0.4891 | 1.753× [1.685,1.789] |
| 65409 | 2 | pressure | 3 | 0.8621 | 0.4978 | 1.733× [1.661,1.774] |
| 2049 | 4 | repeated KV | 3 | 0.0472 | 0.0753 | 0.631× [0.584,0.706] |
| 2049 | 4 | pressure | 3 | 0.0518 | 0.0984 | 0.561× [0.383,0.682] |
| 8193 | 4 | repeated KV | 3 | 0.1277 | 0.0943 | 1.358× [1.305,1.462] |
| 8193 | 4 | pressure | 3 | 0.1357 | 0.1004 | 1.353× [1.283,1.398] |
| 16385 | 4 | repeated KV | 3 | 0.2319 | 0.1142 | 2.031× [2.023,2.044] |
| 32769 | 4 | repeated KV | 3 | 0.4402 | 0.1773 | 2.485× [2.402,2.527] |
| 32769 | 4 | pressure | 3 | 0.4484 | 0.1847 | 2.428× [2.405,2.454] |
| 65409 | 4 | repeated KV | 3 | 0.8531 | 0.2806 | 3.041× [2.975,3.086] |
| 65409 | 4 | pressure | 3 | 0.8619 | 0.2873 | 3.000× [2.974,3.021] |
| 2049 | 8 | repeated KV | 3 | 0.0480 | 0.0882 | 0.555× [0.457,0.621] |
| 2049 | 8 | pressure | 3 | 0.0525 | 0.0869 | 0.605× [0.585,0.627] |
| 8193 | 8 | repeated KV | 3 | 0.1283 | 0.0990 | 1.316× [1.112,1.499] |
| 8193 | 8 | pressure | 3 | 0.1359 | 0.1005 | 1.356× [1.253,1.413] |
| 16385 | 8 | repeated KV | 3 | 0.2319 | 0.1156 | 2.022× [1.917,2.113] |
| 32769 | 8 | repeated KV | 3 | 0.4418 | 0.1373 | 3.241× [2.894,3.551] |
| 32769 | 8 | pressure | 3 | 0.4475 | 0.1396 | 3.206× [3.178,3.225] |
| 65409 | 8 | repeated KV | 3 | 0.8578 | 0.1983 | 4.349× [3.940,4.682] |
| 65409 | 8 | pressure | 3 | 0.8598 | 0.1953 | 4.404× [4.340,4.439] |


These are operator speedups, not full-model E2E speedups.
The main matrix covers real captured states at four depths and three physical layers.
Batch expansion copies a captured state and does not represent independent natural requests.
The pressure control tests an important warm-cache sensitivity.
An integrated model cycles through 96 namespaces rather than one repeated fixture.

## Answers to the research questions

1. Exact KV/attention becomes physically important at long context. The report keeps trace-local shares separate from clean E2E attribution.
2. Payload grows with context and concurrency. B2 near 64K or B4 at 32K exceeds one H100 before non-KV allocations.
3. Exact generic layout and local sequence-parallel controls are measured. They do not establish a loop-specific memory method.
4. The distributed implementation is itself the generic CP control. No distinct loop-aware method beats it.
5. Per-rank operator KV shrinks through real ownership. Full-model capacity, natural throughput, and E2E improvement remain unmeasured.
6. Static R4 reads the same valid context extent at each depth. No causal state pair proves an additional placement advantage.
7. We preserve original BF16 KV and test numerical error. Full distributed natural generation and benchmark accuracy remain `NOT_RUN`.
8. FlashLoop, HLA, LLA, ResidualQuant, and MELT change representation or computation. Preserving BF16 distinguishes scope, not automatic novelty.

## Limitations and decision

The strongest same-resource integrated TP baseline is absent from the pinned RLT engine.
Native TP support in another engine is not ruled out.
The DP calibration measures independent hardware cohorts, not SLO capacity.
The complete clean FFN/attention/host causal partition remains partial.
Initial discovery groups lack an at-start experiment-source snapshot.
Later confirmations retain rank-specific source snapshots.
The measured attention/merge graph definition remains unchanged across those confirmations.

The current evidence does not justify a new loop-specific distributed runtime.
It can justify a separate generic attention-baseline integration question after the strong local control.
No new model engine, KV approximation, training, quantization, or hidden-state prediction was implemented.

Branch: `flashvep/ouro-loop-aware-kv-residency-poc`.
The final Git commit identifies the exact delivery.
`reproduce.md` contains commands and artifact locations.
