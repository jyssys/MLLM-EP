# Foundation completion summary

The model/image/true-EP paths are real measurements. The current routed-stage
cost model is functional but fails untouched EP2 and EP4 accuracy gates, so it
cannot screen small Part-2 gains. EP8 is structural plus local-kernel evidence
only.

## Readiness

```json
{
  "ACTUAL_EP2_READY": true,
  "ACTUAL_EP4_READY": true,
  "EP8_PHYSICAL_VALIDATION": "NOT_AVAILABLE",
  "EP8_PROJECTION_READY": "UNVALIDATED_SENSITIVITY_ONLY",
  "LOW_RESOURCE_SEMANTIC_READY": true,
  "LOW_RESOURCE_TARGET_EP2_EP4_READY": true,
  "QUALITY_BASELINE_READY": true,
  "READY_FOR_PART2_PHYSICAL_EP4": true,
  "READY_FOR_PART2_VIRTUAL_EP2_EP4": false,
  "SIM_EP2_READY": false,
  "SIM_EP4_READY": false,
  "VIRTUAL_EP8_FUNCTIONAL": true
}
```

## Required final summary

```text
MODEL_REVISION / SHAPES / DTYPE: 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c / d2048,m768,E128,k8,48 layers,patch16,merge2 / BF16
PHYSICAL_GPU_UUIDS / TARGET_TOPOLOGIES: 4=GPU-6076e2f2-5b63-3761-5586-56ceb7df8139; 5=GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730; 6=GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28; 7=GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b / EP2=TP1-DP2-EP2; EP4=TP2-DP2-EP4; EP8=TP2-DP4-EP8 projection
MMSTAR_REFERENCE_N_SCORE / EP4_N_SCORE / REPORTED_GAP: 1500,71.985% / 1500,72.220% / reference -0.115pp, EP4 0.120pp
CHARTQA_REFERENCE_N_SCORE / EP4_N_SCORE / REPORTED_GAP: 2500,85.600% / 2500,85.400% / reference -1.200pp, EP4 -1.400pp
PUBLIC_PROTOCOL_STATUS: PUBLIC_PROTOCOL_PARTIAL
REFERENCE_VS_EP_QUALITY_DELTA_AND_CI: MMStar +0.400pp [-0.600,+1.467]; ChartQA -0.127pp [-0.519,+0.282]; both pass -1pp NI operational gate
DETERMINISTIC_DIVERGENCE_CAUSE_AND_STATUS: accumulated BF16 TP/EP reduction-order and kernel drift; full layer localization incomplete; NUMERICAL_PARITY_PARTIAL
ONE_GPU / TWO_GPU_SEMANTIC_QUALITY_STATUS: COMPLETE 192/192 each; exact-token 90.104%, parsed-answer 96.875%, paired accuracy +0.521pp CI[-1.042,+2.604]
ACTUAL_EP2 / ACTUAL_EP4_PATH_PROOF: PASS TP1-DP2-EP2 64 experts/rank DeepEP / PASS TP2-DP2-EP4 32 experts/rank DeepEP; nonzero sparse dispatch-compute-combine evidence
CALIBRATION / DEV / TEST_REQUEST_COUNTS: 32 / 64 / 16 requests (16 / 32 / 8 two-request groups)
PROFILING_OVERHEAD_BEFORE_AFTER: 13.550% -> 5.616% median; EP-only 4.056%; WARN_5_TO_10_PERCENT
BASELINE_TTFT / FIXED32 / FIXED128 / NATURAL_E2E: 2973.293 / 7150.996 / 16585.785 / 3220.822 ms median
VALIDATED_ROUTED_STAGE_E2E_SHARE: DIAGNOSTIC 2.745% of first-token wall; profiler median overhead above 5% target
SIM_EP2_MEDIAN_APE / P90_APE: 12.962% / 22.728% — SIM_UNVALIDATED
SIM_EP4_MEDIAN_APE / P90_APE: 14.510% / 41.084% — SIM_UNVALIDATED
SIM_TTFT / FIXED_E2E / NATURAL_E2E_VALIDATION: UNVALIDATED / UNVALIDATED / UNVALIDATED
GAIN_ERROR_PP / SIGN_ACCURACY / SPEARMAN: EP2 10.647/68.202pp, 0.789, 0.381; EP4 21.010/54.215pp, 0.864, 0.762
PREDICTION_INTERVAL_COVERAGE_AND_WIDTH: EP2 0.875 / 1.114; EP4 1.000 / 1.133; intervals too wide for small-gain screening
VALIDATED_DOMAIN / OOD_LIMITS: structural replay validated for frozen Qwen BF16 k8 real routes on observed image shapes; latency predictor fails gate; new k/resolution/backend/graph mode/topology is OOD
EP2_TO_EP4_BLIND_TRANSFER_ERROR: median/P90 11.418% / 17.688%
EP8_FUNCTIONAL / LOCAL_COMPUTE / COMM / E2E_EVIDENCE: STRUCTURAL_VERIFIED / LOCAL_KERNEL_CALIBRATED / EXTRAPOLATED / UNVALIDATED_EXTRAPOLATION
HOST1_TARGET2_4_8_TEST: semantic COMPLETE; target predictions HOST_INVARIANT; EP8 all rows OOD and sensitivity-only
HOST2_TARGET2_4_8_TEST: semantic COMPLETE; target predictions HOST_INVARIANT; EP8 all rows OOD and sensitivity-only
READY_FOR_PART2_PHYSICAL_EP4: TRUE
READY_FOR_PART2_VIRTUAL_EP2_EP4: FALSE
EP8_PROJECTION_READY: UNVALIDATED_SENSITIVITY_ONLY
FROZEN_BUNDLE_PATH: /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/frozen_baseline_bundle.json
EXACT_1GPU_AND_2GPU_RUN_COMMANDS: see reports/actual_cli_usage.md
REMAINING_BLOCKERS_AND_RESUME_COMMANDS: latency calibration remains SIM_UNVALIDATED; collect additional restart-stratified physical EP2/EP4 anchors, then rerun fit_cost_model and a new untouched image-disjoint test
NEW_RESEARCH_OPTIMIZATION_IMPLEMENTED: NO
```

No new research optimization was implemented.
