# Static-4 communication-only dispatch slicing — final summary

Static4 answered the functional question positively but the method-feasibility
question negatively for the current composition. It exactly separated
communication granularity from expert-compute granularity: four real DeepEP
dispatches, decode priority after S2, exact reconstruction, and one prefill
expert execution. The median maximum slice quantum fell to 0.455 ms, bounded
CIDD fell from Full's 6.484 ms/layer to zero observed inflation, and
ExpertReloadAmp stayed 1.0x.

The implementation nevertheless fails the runtime gate. CPU/D2H slice
planning plus rank synchronization and per-slice finalization increased
prefill wall time from 388.453 to 1,158.193 ms (+199.64%) and raised the victim
peak by 732.176 ms. This is beyond both the 15% Strong-GO gate and the 35%
Structural-GO fast-path band. The authoritative verdict is therefore
`STATIC4_OVERHEAD_DOMINATES`, not a speedup claim.

A narrowly scoped DeepEP/CUDA direct-buffer micro-PoC remains technically
worthwhile because the difficult semantic and one-shot expert constraints
passed, and the dominant costs are explicitly prototype mechanisms. It must
precompute/device-build slice metadata, place received rows directly into one
expert-major buffer, remove the arrival barrier, and provide a unified or
batched finalize. It should not proceed to adaptive scheduling until it first
brings overhead below 35%, then below 15%, while improving victim ITL.

```text
GPU_SET: 2,3,6,7
TOPOLOGY: TP2-DP2-EP4-PP1
DEEPEP_BACKEND: deepep_high_throughput
MODEL_REVISION: 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c

STATIC4_IMPLEMENTATION_TYPE: STOCK_DEEPEP_PYTORCH_COMPOSITION
COMBINE_MODE: PROTO_PER_SLICE_COMBINE
RESULT_SCOPE: NATIVE_SERVING_PROTOTYPE_MOE_PATH

FULL_CIDD_MS_PER_LAYER: 6.484
CP512_CIDD_MS_PER_LAYER: 1.652
LAYERED_CIDD_MS: 6.484/layer (LAYERED_EP_PHYSICAL_REPLAY)
STATIC4_CIDD_MS_PER_LAYER: -0.114 raw; 0.000 nonnegative
STATIC4_CIDD_REDUCTION_VS_FULL: 100% bounded estimate

FULL_MAX_DISPATCH_QUANTUM_MS: 33.351 median (57.763 max)
CP512_MAX_DISPATCH_QUANTUM_MS: 29.967 median
STATIC4_MAX_SLICE_DISPATCH_MS: 0.455 median; 0.632 mean [0.445,0.889]
STATIC4_TOTAL_4SLICE_DISPATCH_MS: 1.347/layer

STATIC4_SLICE_BYTES_S1: 7,530,667/layer
STATIC4_SLICE_BYTES_S2: 7,528,619/layer
STATIC4_SLICE_BYTES_S3: 7,526,400/layer
STATIC4_SLICE_BYTES_S4: 7,525,376/layer
STATIC4_SLICE_BYTE_CV: 0.000540

FULL_EXPERT_RELOAD_AMP: 1.000x
CP512_EXPERT_RELOAD_AMP: 3.821x median
STATIC4_EXPERT_RELOAD_AMP: 1.000x

FULL_EXPERT_TOTAL_MS: 34.840
CP512_EXPERT_TOTAL_MS: 90.986
STATIC4_EXPERT_TOTAL_MS: 28.048
STATIC4_EXPERT_DELTA_VS_FULL: -19.49%

STATIC4_SLICE_BUILD_MS: 226.744/request
STATIC4_REPACK_MS: 6.242/request
STATIC4_EXTRA_COMBINE_MS: +7.806/request (plus 4.972 scatter)

FULL_PREFILL_WALL_MS: 388.453 (paired clean-native lane)
CP512_PREFILL_WALL_MS: 1,110.626 (instrumented M0 lane)
STATIC4_PREFILL_WALL_MS: 1,158.193
STATIC4_PREFILL_OVERHEAD_PCT: +199.64% [95% CI +179.62,+217.89]

FULL_PEAK_VSEL_MS: 401.647 in separate instrumented M0 lane
STATIC4_PEAK_VSEL_MS: UNAVAILABLE_AS_CROSS_LANE_ABSOLUTE; paired victim peak 879.313
STATIC4_VSEL_REDUCTION: NO; paired victim peak worsened by +732.176 ms
VSEL_SCOPE: bounded live victim continuation; diagnostic and clean lanes not numerically merged

CORRECTNESS_MAX_ABS_ERR: 0.0 across 576 rank-layer comparisons
CORRECTNESS_COSINE: minimum 0.9999998808
DETERMINISTIC_NEXT_TOKEN_EQUAL: YES
CORRECTNESS_VERDICT: PASS_EXACT_BF16_OUTPUT

OVERALL_VERDICT: STATIC4_OVERHEAD_DOMINATES
NEXT_ACTION: bounded DeepEP/CUDA direct-buffer fast-path micro-PoC only; no adaptive policy
BRANCH: flashvep/mllm-ep-static4-comm-only-poc
COMMIT: PENDING_FINAL_COMMIT
```
