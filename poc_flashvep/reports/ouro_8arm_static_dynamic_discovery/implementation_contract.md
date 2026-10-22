# Implementation contract

## Model and physical placement

The checkpoint is base `ByteDance/Ouro-1.4B` at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
It has 24 physical layers, hidden size 2048, 16 query heads, 16 KV heads, and intermediate size 5632.
Static execution uses four recurrent loops and 96 independent depth-indexed KV slots.
Weights remain BF16. FP32 runs are diagnostics, not serving baselines.

Single uses physical GPU4.
Four-process logical ranks 0,1,2,3 map to physical GPUs4,5,6,7.
`environment.json` records UUIDs and topology. Runtime constructors check the expected UUIDs.

## Custom engine family

`request_runtime.py:path_for` defines the request-level PP visit order.
Layer PP visits ranks0–3 four times. Each rank owns six physical layers and their four depth-indexed KV sets.
Loop PP visits ranks0–3 once. Each rank owns all 24 physical weights and one loop's KV slots.
The prompt remains one complete request. There is no token-chunk pipeline.

Each rank has one worker process and one stage queue.
Workers publish completed GPU data through persistent CUDA IPC buffers.
They synchronize the producer before the receiver copies the published tensor.
Different requests can occupy different ranks. The heavy trace must verify actual kernel overlap.
CUDA IPC imports can make all four worker PIDs appear on each GPU.
Those entries do not represent sixteen independent model workers.

Custom TP4 shards Q/K/V heads and SwiGLU intermediate channels into four physical owners.
Each rank has four query heads, four KV heads, and intermediate width 1408.
Row projections use real NCCL All-Reduce on FP32 partial sums before the final BF16 cast.
The embedding and LM head remain replicated. This differs from native vocabulary-parallel execution.
The model retains the original decoder, normalization, RoPE, and residual order.

The inherited custom TP column and row projections call `torch.mm` with BF16 operands and `out_dtype=torch.float32`.
They do not convert those operands into a full FP32 GEMM.
Column projections cast their FP32 outputs to BF16 before the next operation.
Row projections reduce FP32 partials before their BF16 cast.
These explicit output buffers and casts differ from the ordinary unsharded `nn.Linear` path.
The custom implementation also keeps separate Q, K, V, gate, and up projections.
It therefore does not match native fused QKV and merged gate/up projection execution.
Those implementation differences limit a pure-architecture interpretation of custom TP latency.
The heavy trace must assess their actual kernel costs.
Do not attribute all custom TP delay to network communication or recurrent structure.

## Native production-oriented control

The native control uses vLLM 0.26.0 with FlashAttention3 and default optimized graph execution.
The actual log records FlashAttention version 3.
The native `ouro.py` SHA256 is `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff`.
Its source uses `QKVParallelLinear`, `MergedColumnParallelLinear`, and `RowParallelLinear`.
It reuses physical weights across four loops and creates separate attention/cache slots for each loop.

The inspected native forward always executes `total_ut_steps` loops.
It constructs an early-exit gate, but the Static forward does not call that gate.
The custom Static family evaluates the original unused gate after each loop.
This small execution difference remains part of the engine qualification. It is not a measured architecture gain.
Native Static code does not establish True Dynamic support.

The native configuration keeps custom All-Reduce enabled.
Its logical reductions must not all be called NCCL kernels without a trace.
Custom TP4 explicitly uses PyTorch NCCL reductions.
Ouro is dense. This study does not execute MoE Dispatch/Combine All-to-All.

The profile parser separately classifies `cross_device_reduce_1stage` and `cross_device_reduce_2stage` as native custom reduction kernels.
The inspected CUDA header defines both kernel symbols.
The installed Python wrapper calls the custom reduction operator from its `all_reduce` method.
The installed wrapper SHA256 is `3ed3ff9bbcaf361071ce6af77b879f5d269ba816366f56619cb3c21d486c52e7`.
The inspected header SHA256 is `95e7490d4f03cfac8a30f2806937aab9c788a0f06057f165643e87008081139c`.
The header belongs to local checkout `5acd95906b7c7a54dde89396d2bb06fe28ebeed0`.
Its path is `/home/esjung/external/vllm-cp-composition-audit/csrc/custom_all_reduce.cuh`.
This source audit supports kernel-name classification. It does not prove binary identity or a logical call count.
The parser preserves unclassified collective-like names for review.
A zero NCCL-kernel count must not imply absent TP communication.

## True Dynamic policy

The reference is official CDB at commit `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The method uses cumulative hazard, q=0.5, minimum R2, maximum R4, and last_exited KV.
Prefill executes R4. Decode can physically stop before later loops.
All custom Dynamic arms use no-refill token waves in canonical request order.
The prototype uses B1 stage calls. The official CDB control can pack four rows.
That packing difference remains explicit.

Forced diagnostics vary exit depth across requests and token positions.
They do not count as actual Dynamic task accuracy.
Live gate runs record their actual depth traces without forcing agreement.
The TP probe concatenates four local KV-head shards before comparison with Single.
It does not overwrite one rank's KV with another rank's tensor.

## Timing and evidence boundaries

The custom prepared-service clock excludes per-cohort graph construction.
Later cohorts also record preparation-inclusive resident-model wall time.
Earlier TP pilot records lack the preparation-inclusive field.
Report incomplete coverage as PARTIAL rather than estimating missing per-request times.
Clean replay, task generation, numerical diagnostics, and heavy profiling remain separate lanes.
The historical strict P512 numerical failure remains FAIL.
Actual task accuracy cannot retroactively pass that numerical gate.
