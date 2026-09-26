
# Part 2 final summary

FOUNDATION_BUNDLE_AND_BASELINE_STATUS: Part1 frozen bundle READY; public protocol PARTIAL; physical EP4 READY; Part1 simulator UNVALIDATED
MODEL_REVISION / GPU_SET / ACTUAL_TOPOLOGY: 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c / 4,5,6,7 / TP2-DP2-SP1-EP4-PP1
PUBLIC_REPRODUCTION_STATUS: PUBLIC_PROTOCOL_PARTIAL; Part1 full MMStar and ChartQA reused only for the unchanged R0/K8 control
EVALUATION_LANE_AND_METRIC: OFFICIAL_REPRODUCTION_PUBLIC_PROTOCOL_PARTIAL / MMStar exact-option accuracy + ChartQA relaxed accuracy
SAMPLING_TOP_K / EXPERT_TOP_K_TEXT / EXPERT_TOP_K_VISION: 20 / 8 / {8,6,4}
DISCOVERY_COUNT / HELDOUT_COUNT: 96 image-question requests x 9 / 192 image-hash-disjoint requests x baseline+2 frozen candidate(s)
RESOLUTION_GRID_AND_ACTUAL_VISUAL_TOKEN_COUNTS: {"R0": {"duplicate_grids": 0, "mean_actual_ratio": 1.0, "mean_visual_tokens": 398.5833333333333}, "R025": {"duplicate_grids": 11, "mean_actual_ratio": 0.3662379771853333, "mean_visual_tokens": 105.5625}, "R05": {"duplicate_grids": 11, "mean_actual_ratio": 0.5316629274000867, "mean_visual_tokens": 194.84375}}
PRIMARY_K_GRID: visual {8,6,4}; text=8; generated decode=8
NO_OP_PARITY: {"image_grid_equal": 96, "input_ids_equal": 96, "output_tokens_exact": 92, "parsed_answer_equal": 96, "processed_pixels_equal": 96, "score_outcome_equal": 96} over n=96; runtime={"custom_mixed_k_adapter_p0_ms": 3215.6212990000004, "custom_overhead_fraction": -0.13717311582572256, "custom_overhead_pct": -13.717311582572256, "evidence": "MEASURED_TRUE_EP4", "native_fused_p0_ms": 3726.8441190000003, "schema_version": "mllm_resolution_expert_tradeoff.noop_runtime.v1", "scope": "TRUE_EP4_DETERMINISTIC_ONE_TOKEN_PAIRED_REPRESENTATIVE_COHORT"}
BEST_RESOLUTION_ONLY: R05_K8
BEST_EXPERT_ONLY: R0_K6
BEST_INDEPENDENT_TWO_AXIS: resolution-then-k=R05_K6; k-then-resolution=R05_K6
BEST_JOINT_PAIR_ONLY: R05_K6
BEST_JOINT_EP_AWARE: R0_K8
TASK_SENSITIVITY_AND_INTERACTION: TRADEOFF_SUPPORTED; {"R025_K4": {"bootstrap_95ci_pp": [-10.416666666666668, 2.083333333333333], "interaction_pp": -4.166666666666666}, "R025_K6": {"bootstrap_95ci_pp": [-6.25, 4.166666666666666], "interaction_pp": -1.0416666666666665}, "R05_K4": {"bootstrap_95ci_pp": [-6.25, 2.083333333333333], "interaction_pp": -2.083333333333333}, "R05_K6": {"bootstrap_95ci_pp": [-8.333333333333332, 4.166666666666666], "interaction_pp": -2.083333333333333}}
MATCHED_PAIR_COMPARISON: [{"left": "R05_K8", "mean_pair_difference_fraction": 0.022318628475851976, "quality_bootstrap_95ci_pp": [-8.333333333333332, 3.125], "right": "R0_K4", "right_minus_left_quality_pp": -2.083333333333333, "right_minus_left_ttft_ms": 333.1415715000003, "ttft_evidence": "PAIRED_CLEAN_TRUE_EP4_REPRESENTATIVE_COHORT"}]
ACTUAL_EXECUTED_VISION_PAIRS_REDUCTION: 0.000%
PHYSICAL_PADDED_WORK_REDUCTION: N/A_BACKEND_DID_NOT_EXPOSE_PADDING; logical valid/invalid route slots reported separately
EP_CRITICAL_PATH_CHANGE: 0.000% reduction for R0_K8 vs R0_K8 on the paired representative TRUE-EP4 cohort
HELDOUT_QUALITY_DELTA_AND_CI: {"R05_K6": {"ci_pp": [-5.208333333333334, 4.166666666666666], "correct_to_wrong": 10, "delta_pp": -0.5208333333333333, "finish_reason_counts": {"stop": 192}, "mean_output_token_delta_vs_baseline": 155.84375, "mean_output_tokens": 275.53125, "p90_output_tokens": 253.70000000000007, "wrong_to_correct": 9}, "R05_K8": {"ci_pp": [-8.854166666666668, 0.5208333333333333], "correct_to_wrong": 14, "delta_pp": -4.166666666666666, "finish_reason_counts": {"stop": 192}, "mean_output_token_delta_vs_baseline": 101.89583333333333, "mean_output_tokens": 221.58333333333334, "p90_output_tokens": 187.40000000000003, "wrong_to_correct": 6}}
TRUE_EP4_TTFT_GAIN / NATURAL_E2E_GAIN / FIXED_DECODE_GAIN: 0.000% / 0.000% / fixed32=0.000%, fixed128=0.000%
SIM_EP4_GAIN_AND_UNCERTAINTY / SIM_EP8_LABEL: EP4_STRUCTURAL_REPLAY; EP8_UNVALIDATED_EXTRAPOLATION
EP_SPECIFIC_INCREMENT_VS_STRONGEST_CONTROL: 0.000 pp; GENERIC_COMPRESSION_ONLY
ONLINE_SELECTOR_CAUSALITY_AND_COST: NOT_IMPLEMENTED; uniform request-level factorial only, no task label/answer/future routing used online
PRIOR_BASELINES_COMPLETED_OR_MISSING: FastMMoE visual-k component=C2 REIMPLEMENTED_COMPONENT_CONTROL; full FastMMoE/MoDES ports=NOT_RUN; MACS/ReaLB boundary audited
TRADEOFF_STATUS / EP_SPECIFIC_STATUS / QUALITY_STATUS: TRADEOFF_SUPPORTED / GENERIC_COMPRESSION_ONLY / QUALITY_UNRESOLVED
VERDICT: GENERIC_ONLY
NOT_TESTED_AND_LIMITATIONS: public wrapper remains partial; physical padding not exposed; distinct-image global-batch4 K6/K4 unavailable after reproduced external-DP synchronization hang; Part1 EP2/EP4 simulator unvalidated; EP8 has no physical collective/E2E truth; no saliency/video/basis/training
NEXT_ACTION: Stop this EP-aware grid family as a main method.
DO_NOT_IMPLEMENT_UNBOUNDED_PRODUCTION_RUNTIME: true

## Machine-readable summary

```json
{
  "schema_version": "mllm_resolution_expert_tradeoff.summary.v1",
  "baseline": "PART1_FOUNDATION_READY_PUBLIC_PROTOCOL_PARTIAL",
  "controls": {
    "C0_reference": "R0_K8",
    "C1_resolution_only": "R05_K8",
    "C2_expert_only": "R0_K6",
    "C3_independent_k_then_resolution": "R05_K6",
    "C3_independent_resolution_then_k": "R05_K6",
    "C4_joint_pair_only": "R05_K6",
    "C5_joint_ep_critical_path": "R0_K8"
  },
  "frozen_candidates": [
    {
      "config": "R05_K8",
      "reason": "DISCOVERY_BEST_NONBASELINE_EP_SCREEN_C0_WON"
    },
    {
      "config": "R05_K6",
      "reason": "DISCOVERY_JOINT_TOTAL_PAIR"
    }
  ],
  "no_op_parity": {
    "image_grid_equal": 96,
    "input_ids_equal": 96,
    "output_tokens_exact": 92,
    "parsed_answer_equal": 96,
    "processed_pixels_equal": 96,
    "score_outcome_equal": 96
  },
  "no_op_runtime": {
    "custom_mixed_k_adapter_p0_ms": 3215.6212990000004,
    "custom_overhead_fraction": -0.13717311582572256,
    "custom_overhead_pct": -13.717311582572256,
    "evidence": "MEASURED_TRUE_EP4",
    "native_fused_p0_ms": 3726.8441190000003,
    "schema_version": "mllm_resolution_expert_tradeoff.noop_runtime.v1",
    "scope": "TRUE_EP4_DETERMINISTIC_ONE_TOKEN_PAIRED_REPRESENTATIVE_COHORT"
  },
  "ep_specific_increment_pp": 0.0,
  "ep_specific_status": "GENERIC_COMPRESSION_ONLY",
  "quality_status": "QUALITY_UNRESOLVED",
  "best_joint_ep": "R0_K8",
  "best_natural_e2e_gain_pct": 0.0,
  "verdict": "GENERIC_ONLY",
  "public_protocol_status": "PUBLIC_PROTOCOL_PARTIAL",
  "ep8_status": "UNVALIDATED_EXTRAPOLATION",
  "batch4_secondary_status": {
    "schema_version": "mllm_resolution_expert_tradeoff.runtime_limit.v1",
    "scope": "DISTINCT_IMAGE_GLOBAL_BATCH4_SECONDARY",
    "runtime": "vLLM 0.20.0 external DP2 / TP2 / EP4",
    "R0_R05_R025_K8": "MEASURED_TRUE_EP4_COMPLETE",
    "K6": "RUNTIME_UNAVAILABLE_EXTERNAL_DP_SHM_BROADCAST_HANG",
    "K4": "NOT_RUN_AFTER_REPRODUCED_K6_RUNTIME_HANG",
    "primary_batch1_status": "MEASURED_TRUE_EP4_COMPLETE",
    "interpretation": "The secondary heterogeneous global-batch4 limitation is a runtime failure, not a method speedup or hypothesis NO-GO. No incomplete K6 timing is used."
  },
  "do_not_implement_unbounded_production_runtime": true,
  "tradeoff_status": "TRADEOFF_SUPPORTED",
  "best_config": "R0_K8",
  "actual_executed_visual_pair_reduction_pct": 0.0,
  "actual_ep_critical_path_gain_pct": 0.0,
  "actual_ttft_gain_pct": 0.0,
  "actual_fixed32_gain_pct": 0.0,
  "actual_fixed128_gain_pct": 0.0,
  "heldout_quality": {
    "R05_K6": {
      "delta_pp": -0.5208333333333333,
      "ci_pp": [
        -5.208333333333334,
        4.166666666666666
      ],
      "correct_to_wrong": 10,
      "wrong_to_correct": 9,
      "mean_output_tokens": 275.53125,
      "p90_output_tokens": 253.70000000000007,
      "mean_output_token_delta_vs_baseline": 155.84375,
      "finish_reason_counts": {
        "stop": 192
      }
    },
    "R05_K8": {
      "delta_pp": -4.166666666666666,
      "ci_pp": [
        -8.854166666666668,
        0.5208333333333333
      ],
      "correct_to_wrong": 14,
      "wrong_to_correct": 6,
      "mean_output_tokens": 221.58333333333334,
      "p90_output_tokens": 187.40000000000003,
      "mean_output_token_delta_vs_baseline": 101.89583333333333,
      "finish_reason_counts": {
        "stop": 192
      }
    }
  },
  "resolution_grid": {
    "R0": {
      "mean_visual_tokens": 398.5833333333333,
      "mean_actual_ratio": 1.0,
      "duplicate_grids": 0
    },
    "R05": {
      "mean_visual_tokens": 194.84375,
      "mean_actual_ratio": 0.5316629274000867,
      "duplicate_grids": 11
    },
    "R025": {
      "mean_visual_tokens": 105.5625,
      "mean_actual_ratio": 0.3662379771853333,
      "duplicate_grids": 11
    }
  },
  "physical_padded_work": "N/A_BACKEND_DID_NOT_EXPOSE_PADDING",
  "online_selector": "NOT_IMPLEMENTED_UNIFORM_REQUEST_LEVEL_GRID_ONLY"
}
```

Part 1's reusable foundation remains valid regardless of this tested-family
verdict. No saliency model, training, video extension, or unbounded production
runtime was implemented.
