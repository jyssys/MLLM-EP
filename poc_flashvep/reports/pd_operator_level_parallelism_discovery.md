# Operator-level Prefill–Decode parallelism

## Decision

`NO_GO_INTERFERENCE` for concurrent MLP row offloading on the tested H200s. The idle-window arm has `MEASURED_PHYSICAL` speedup at the one-block boundary, with zero measured overlap between the remote GEMM and the D actor. It remains a temporal scheduling result. A serving-time advantage over whole-request, token or layer sharing is unproven.

## Physical design

The study loads real BF16 weights from block 0 of Qwen3-8B and Qwen2.5-7B-Instruct. It draws inputs from actual LongBench token IDs and executes original Q/K/V geometry, norms, RoPE, FA3 attention, cuBLAS projections and the vLLM fused SiLU×up kernel. Tests use prepared local ragged K/V and physical H200 NVLink P2P. CUDA events time isolated actors, ranks and streams. The complete MLP split timing uses synchronized wall time at the source boundary. The decode actor is a block-level attention-plus-MLP proxy in the targeted test. Full 36-layer P and D server passes are separate native controls.

The partition keeps the prefill attention and RMS normalization on P. P sends the selected normalized rows to D. P and D perform their own MLP rows, and P waits for D's result before it can complete the layer dependency. This uses both activation transfers even when D has original resident weights. At 8,192 rows and a quarter split, the round trip is 32 MiB. A half split is 64 MiB. Applying these fractions to all 36 layers of a 16K request would add 2.25 GiB or 4.5 GiB respectively, before handoff metadata. Those are analytical full-model payloads. The actual trials measure one block.

## Complementarity and concurrent release

The first Qwen3 sweep has 486 operator/mode cells covering native-style GEMMs, FA3, peer DMA and NCCL on 18 geometries. In isolated warmed GPU replays at P4096/D B4, KV16K, decode FA3 used 19.60% Tensor activity and 54.73% DRAM bandwidth activity. The P MLP reached 61.92% and 26.73%. The individual kernels expose a plausible division of resources. At that geometry ordinary stream overlap extended D FA3 p95 from roughly 0.10 ms to 1.12 ms. Priority reduced the paired p95 to 0.42 ms. Qwen2.5's analogous attention/MLP pair still raised D p95 14.78% in ordinary streams and 12.03% with priority. The overlap data therefore need an actual queueing and dependency test.

The targeted repeat uses 31 samples per arm, rotates the arm order, includes the source prefix of the block, and records the remote GEMM/decode interval intersection. B1, 8,192 query rows and 16,384 cached rows are near the observed dense native geometry.

| Remote rows | Release | P block p50 gain | D actor p95 change | Median overlap |
|---:|---|---:|---:|---:|
| 1,024 | Decode launched during P front | 5.84% | +1.38% | 0.000 ms |
| 2,048 | Decode launched during P front | 11.24% | −11.94% | 0.000 ms |
| 4,096 | Decode launched during P front | 21.32% | +0.27% | 0.000 ms |
| 1,024 | Same P-front dependency | 5.10% | +227.21% | 0.334 ms |
| 2,048 | Same P-front dependency | 10.85% | +407.93% | 0.564 ms |
| 4,096 | Same P-front dependency | 20.87% | +839.68% | 1.031 ms |

Across the 24 targeted overlapping-release cells, none met both ≥10% P critical-path reduction and ≤5% D p95 slowdown. The 24 early-D cases yielded 15 numerical passes. All 15 have zero median measured overlap. In the 4,096-row early-D transfer the P block fell from 7.20 to 5.66 ms. Concurrent release fell from 7.06 to 5.59 ms while D p95 rose from 0.124 to 1.16 ms.

The independent 256/512/1,024-row retile pass at the actual B1 geometry kept the full P2P dependency. Best D p95 harm was still +63.87% at a 256-row quantum and 25% remote share, with +10.73% P gain. At 50% remote share, 256-row tiles yielded +14.16% P gain and +70.97% D harm. All six retile arms failed the joint screen. CUDA scheduling priority did not protect that critical path.

## Physical SM partitions

The initial Torch Event test could allocate Green Contexts but failed cross-context timing. The dedicated driver test repeated the actual work with context-scoped event pairs and checked the allocated SM resources. All 40 gated timing cells completed. In the 112D/16P split, B1 D attention changed +2.87% and −1.36% for P4096 and P8192. The corresponding P MLP took 6.82× and 6.44× its isolated full-GPU time. B4 D attention slowed 43.54% and 38.61%. Every tested P/D pair at 96D/32P, 64D/64P and 32D/96P had D slowdown greater than 5%. These are same-GPU operator timings. The Green test did not run a two-way activation dependency under that partition.

## Correctness and scope

All 48 targeted cells passed the frozen relative L2 tolerance with the original attention/MLP dependency. The physical sweep's first two models had maximum row-output relative L2 of 0.0001053 and 0.0001193. Task quality, full-model online offload, a POD kernel comparison and an optimized token/layer runtime rerun are `NOT_RUN`. Native DP, chunking, NIXL and D-local complete-request deflection are the separate online controls. Existing token/layer scheduling and operators split by resource already have close precedents. The credible open issue is whether a measured deadline-safe idle window rewards the extra boundaries over an already optimized elastic PD policy. [Prior work](pd_resource_complementary_prior_work.md)

[Slowdown matrix](../pd_resource_complementary_parallelism/figures/operator_pair_slowdown.png) · [P/D break-even](../pd_resource_complementary_parallelism/figures/operator_offload_break_even.png) · [Targeted physical records](../pd_resource_complementary_parallelism/results/operator_targeted.jsonl.gz) · [Tiling records](../pd_resource_complementary_parallelism/results/operator_quanta.jsonl.gz)
