# Dispatch–Combine asymmetry discovery — completed evidence

**Primary verdict: DISPATCH_CRITICAL_RANK_TAIL_DOMINANT.** Large asymmetry also exists in standalone MoE LLM (GENERIC_EP_DISPATCH_COMBINE_ASYMMETRY), but native visual adds a strong first-layer producer-readiness excess over same-model text. It is NOT supported as intrinsic visual EP transport excess: equal-token route replay removes the visual difference, and visual is not slower than standalone LLM.

Attribution is **INFERRED_RESIDUAL**, supported by the clean layer trajectory, installed source and producer-ready replay. Exclusive network-tail vs producer-wait fractions remain unresolved; the generic secondary verdict refers to the comparable remaining-layer base ratios, not identical all-layer visual/text ratios.

| Arm | tokens | Dispatch ms/layer | Expert | Combine | mean request D/C ratio |
|---|---:|---:|---:|---:|---:|
| VL visual low | 175 | 0.9131 | 0.4020 | 0.1131 | 8.085 |
| VL visual medium | 721 | 0.9305 | 0.4507 | 0.0826 | 11.339 |
| VL visual high | 2029 | 1.1092 | 0.5264 | 0.1246 | 9.051 |
| VL text low | 174 | 0.4703 | 0.3973 | 0.1249 | 3.796 |
| VL text medium | 724 | 0.5498 | 0.4298 | 0.1219 | 4.747 |
| VL text high | 2033 | 0.5184 | 0.5114 | 0.1331 | 3.927 |
| Qwen3 LLM text low | 174 | 1.4125 | 0.4005 | 0.4538 | 3.114 |
| Qwen3 LLM text medium | 724 | 1.4733 | 0.4376 | 0.4471 | 3.296 |
| Qwen3 LLM text high | 2033 | 1.4405 | 0.5070 | 0.4140 | 3.559 |


High same-model Dispatch delta=0.5909ms/layer [0.5654,0.6149]95%CI; paired gain/excess=115.3% [105.7,124.6]. High replay difference=-0.0024ms; CI spans zero. The repeat CI does not generalize beyond the single frozen input in each bin.

Pure wire bandwidth and exclusive packing/movement/sync subdivisions are unresolved. Logical accounting and ApparentThroughput are in communication_summary.json. Internal API times are perturbed and never headline; profiler kernel classes are diagnostic. Clean envelope overhead and confidence intervals are reported, not hidden. No optimization/speedup/quality claim.

## Required final block

```text
GPU_SET: 0,1,4,5
PHYSICAL_TO_LOGICAL_GPU_MAPPING: 0→0,1→1,4→2,5→3
GPU_MODEL: H10080GB HBM3
GPU_TOPOLOGY: allowed GPUs NV18/NVSwitch
MODEL_MLLM: Qwen3-VL-30B-A3B-Instruct @9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
MODEL_LLM: Qwen3-30B-A3B @ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
TOPOLOGY: TRUE TP2-DP2-EP4-PP1
DEEPEP_BACKEND: high-throughput + sparse Triton
DEEPEP_VERSION: 1.2.1+73b6ea4
VLLM_VERSION: 0.20.0+cu129
MLLM_VISUAL_HIGH_TOKENS: 2029 total /1980 visual
MLLM_VISUAL_HIGH_DISPATCH_MS: 1.109241
MLLM_VISUAL_HIGH_COMBINE_MS: 0.124616
MLLM_VISUAL_HIGH_RATIO: 9.051039 (mean request ratio)
MLLM_TEXT_HIGH_TOKENS: 2033
MLLM_TEXT_HIGH_DISPATCH_MS: 0.518389
MLLM_TEXT_HIGH_COMBINE_MS: 0.133124
MLLM_TEXT_HIGH_RATIO: 3.927057
LLM_TEXT_HIGH_TOKENS: 2033
LLM_TEXT_HIGH_DISPATCH_MS: 1.440524
LLM_TEXT_HIGH_COMBINE_MS: 0.414042
LLM_TEXT_HIGH_RATIO: 3.559221
VISUAL_MINUS_MLLM_TEXT_DISPATCH_PCT: 115.327871 paired
VISUAL_MINUS_LLM_TEXT_DISPATCH_PCT: -22.997438 unpaired model-block means
MLLM_TEXT_MINUS_LLM_TEXT_DISPATCH_PCT: -64.013898 unpaired
DISPATCH_CONTROL_COUNT_MS: N/A exclusive; composite direct/API and classified kernels in artifacts
DISPATCH_PACK_LAYOUT_MS: N/A exclusive; fused kernel
DISPATCH_DATA_MOVEMENT_MS: N/A exclusive; fused movement/polling
DISPATCH_RECEIVE_TAIL_SYNC_MS: N/A exclusive; visual first-layer Dispatch 29.597ms is composite
DISPATCH_EPILOGUE_MS: N/A exclusive
DISPATCH_RESIDUAL_UNATTRIBUTED_MS: not forced into free fit constant
COMBINE_PREP_MS: N/A exclusive
COMBINE_DATA_MOVEMENT_MS: N/A exclusive
COMBINE_REDUCTION_SCATTER_MS: N/A exclusive
COMBINE_SYNC_EPILOGUE_MS: N/A exclusive
COMBINE_RESIDUAL_UNATTRIBUTED_MS: no fabricated exclusive subdivision
DISPATCH_APPARENT_THROUGHPUT_GBS: 27.146, all-rank logical useful bytes/composite critical-rank duration, NOT true bandwidth
DISPATCH_EFFECTIVE_BW_GBS: N/A
COMBINE_EFFECTIVE_BW_GBS: N/A
ROUTE_CONTROLLED_REPLAY_STATUS: PASS, layers0/24/47 only
ROUTE_CONTROLLED_VISUAL_MINUS_TEXT_DISPATCH_MS: -0.002448, CI spans0
INSTRUMENTATION_OVERHEAD_STATUS: clean measured 8.34%/7.65%; internal composite durations UNVALIDATED where perturbed; Nsight diagnostic only
CROSS_MODEL_PAIRED_TIMING: NOT_RUN; separate model-block comparison only, unlike paired same-VL visual/text
GPU_FINAL_STATE: experiment and burner stopped; 0,1,4,5 verified empty at user-requested handoff; do not restart
PRIMARY_DOMINANT_COMPONENT: count/rendezvous/producer-readiness composite, NOT proven pure data movement
GENERIC_VS_MLLM_SPECIFIC: generic asymmetry plus visual native readiness excess; NO intrinsic visual transport promotion
OVERALL_VERDICT: DISPATCH_CRITICAL_RANK_TAIL_DOMINANT
NEXT_ACTION: Do not build an MLLM-specific transport method from this result. Investigate generic count/rendezvous launch protocol and vision-producer admission separately; no speedup claim.
BRANCH: flashvep/dispatch-combine-asymmetry-discovery
COMMIT: see artifacts/release.json evidence commit and repository branch HEAD
```
