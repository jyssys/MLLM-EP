# Native Layered Prefill on TRUE EP4 — final discovery

**MODALITY_COMMON_RESIDUAL_FOUND**, scoped to the tested visual input-producer/runtime path. **No intrinsic common EP collective bottleneck is established; do not promote a new communication method.**

Actual native LP is correct; resident decode emits a token through all 48 layers on every stage. Full/LP hidden and logits are bit-exact in 30 acceptance comparisons, with eight unforced tokens identical. A fresh pooled v7 check adds 12 passing comparisons. Current rich prefill route IDs/weights match in 30 comparisons.

| Visual input arm | TTFT ms [95% CI] | Peak VSEL ms [95% CI] | P99 ITL ms [95% CI] | ReloadAmp | Active mixed expert total ms |
| --- | --- | --- | --- | --- | --- |
| FULL | 311.03 [300.76, 325.31] | 112.09 [109.06, 115.58] | 112.81 [106.73, 120.27] | 1.000 [1.000, 1.000] | 25.25 [25.21, 25.33] |
| CP512 | 619.05 [584.06, 665.78] | 113.62 [109.42, 119.22] | 119.85 [112.40, 127.42] | 3.798 [3.798, 3.798] | 83.18 [82.68, 83.55] |
| LP_AUTO | 635.80 [615.63, 668.62] | 117.97 [113.96, 122.35] | 113.29 [108.93, 119.45] | 1.000 [1.000, 1.000] | 25.54 [25.09, 26.38] |
| LP_G8 | 1069.27 [1036.02, 1128.59] | 121.33 [116.98, 127.51] | 118.24 [111.65, 125.75] | 1.000 [1.000, 1.000] | 24.95 [24.88, 25.00] |

10 paired clean repeats/arm/input across five engine restarts; components/routes use separate three-repeat lanes.

CP512 versus text adds **105.96 [100.20, 112.57] ms** peak VSEL; frozen G8 adds **101.95 [95.98, 109.78] ms**. The extra peak is concentrated in the first producer iteration. Subsequent LP stages retain low expert reload and reasonable per-token latency. G4/8/16/24/48 all execute natively; increasing groups does not remove the initial visual producer floor. CP expert fragmentation is real but **CP-specific**, not common EP evidence.

The secondary lossless producer diagnostic and lifecycle clocks are in [modality_analysis.md](modality_analysis.md). A prepare interval includes peer readiness and is **not isolated A2A service time**. No post-hoc replay, simulator, full benchmark rerun, model approximation or new method is used.

All measured Pareto points are in [pareto_analysis.md](pareto_analysis.md); the positive residual does not overturn the native LP efficiency success. The sample pool is narrow; shared-fabric foreign load and prototype overhead constrain generalization.

## Required final fields

```text
GPU_SET: 0,1,4,5; H100 80GB; NV18/NVSwitch (UUIDs in baseline_readiness.md)
MODEL: Qwen3-VL-30B-A3B-Instruct @ 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c; BF16; d2048/m768/E128/K8/L48
TOPOLOGY: TP2-DP2-EP4-PP1; linear 32 experts/rank; source SP, no counted duplicate TP paths
DEEPEP_BACKEND: deepep_high_throughput + sparse Triton; DeepEP 1.2.1+73b6ea4
UPSTREAM_LAYERED_COMMIT: 053f80e5201a7c0ab56e468e3d578907a2ca9cc3 (official TP-only code)
LP_IMPLEMENTATION_SCOPE: CONTROLLED_NATIVE_LP_EP4_HARNESS
NATIVE_LP_CORRECTNESS: PASS; exact hidden/logits, first token +8 continuation; rich route hashes PASS
DECODE_CONTINUES_EVERY_ITERATION: PASS; full48 layers / one real token per resident / per stage
PREFILL_LAYER_GROUP_EXECUTION_VERIFIED: PASS; persistent hidden/residual; 48 prefill layers exactly once
CP512_TTFT_MS: 619.05 [584.06, 665.78]
CP512_PEAK_VSEL_MS: 113.62 [109.42, 119.22]
CP512_ITL_P99_MS: 119.85 [112.40, 127.42]
CP512_EXPERT_RELOAD_AMP: 3.7978 [3.7978, 3.7978]
CP512_DISPATCH_MS_PER_LAYER: 2.1740; local prepare span, not isolated wire
LP_AUTO_STAGE_COUNT: 4
LP_AUTO_TTFT_MS: 635.80 [615.63, 668.62]
LP_AUTO_PEAK_VSEL_MS: 117.97 [113.96, 122.35]
LP_AUTO_ITL_P99_MS: 113.29 [108.93, 119.45]
LP_AUTO_EXPERT_RELOAD_AMP: 1.0000 [1.0000, 1.0000]
LP_AUTO_DISPATCH_MS_PER_ACTIVE_LAYER: 1.0146; includes peer readiness
LP_BEST_GROUP_COUNT: 8; frozen discovery minimum, NOT_SLO_ELIGIBLE; not final-selected winner
LP_BEST_TTFT_MS: 1069.27 [1036.02, 1128.59]
LP_BEST_PEAK_VSEL_MS: 121.33 [116.98, 127.51]
LP_BEST_ITL_P99_MS: 118.24 [111.65, 125.75]
LP_BEST_EXPERT_RELOAD_AMP: 1.0000 [1.0000, 1.0000]
LP_BEST_DISPATCH_MS_PER_ACTIVE_LAYER: 1.0180; includes peer readiness
CP_VISUAL_MINUS_TEXT_DISPATCH_PCT: 39.68 [36.45, 41.36]; n3 local-span diagnostic
LP_VISUAL_MINUS_TEXT_DISPATCH_PCT: 151.34 [124.76, 166.03]; n3 local-span diagnostic
CP_VISUAL_MINUS_TEXT_VSEL_PCT: 2418.47 [1312.16, 3677.03]; n10, text denominator near zero, prefer absolute delta
LP_VISUAL_MINUS_TEXT_VSEL_PCT: 571.19 [455.35, 703.08]; n10, text denominator small, prefer absolute delta
COMMON_EP_COMPONENT: NONE_ISOLATED_AS_AN_INTRINSIC_EP_LOWER_BOUND
COMMON_MODALITY_COMPONENT: INITIAL_MULTIMODAL_DELIVERY_SETUP_AND_VISION_PRODUCER_READINESS
PARETO_GAP_EXISTS: TESTED_VISUAL_INPUT_PRODUCER_GAP_ONLY; NO_COMMUNICATION_METHOD_PROMOTION
EP_COMMON_RESIDUAL: NOT_CONFIRMED
MODALITY_COMMON_RESIDUAL: SUPPORTED_TESTED_INPUT_PRODUCER_PATH
OVERALL_VERDICT: MODALITY_COMMON_RESIDUAL_FOUND
NEXT_ACTION: Retire LP expert-fragmentation claim; study lossless multimodal producer isolation before any new communication method.
BRANCH: flashvep/native-layered-prefill-ep4-discovery
COMMIT: Publication evidence SHA recorded in artifacts/release.json after evidence freeze
```
