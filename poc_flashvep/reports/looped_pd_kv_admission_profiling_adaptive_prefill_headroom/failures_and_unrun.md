# Failures and unrun configurations

No failed namespace is overwritten.
A completed response cohort can still fail profiler export or native shutdown.
Those failures remain failures.

| Terminal status | Configurations |
| --- | --- |
| COMPLETE | 223 |
| FAILED | 9 |
| NOT_RUN | 99 |

## Failed jobs

| Run | Model | Return code | Reason |
| --- | --- | --- | --- |
| calibration_huginn_l2560_c2560_serial16 | huginn | 1 | Source-hash startup guard rejected a concurrent harness edit. The namespace remains failed. A boundary edit lock protects later starts. |
| long_profile_ouro_PD_start1_both_n128 | ouro | 1 | All measured responses completed. Native artifact processing exceeded the fixed 300-second wait. See the dedicated export root-cause report. |
| recovery_huginn_3500_32_control_gate8 | huginn | 1 | Instrumented serving watchdog expired. Native shutdown also exceeded its deadline. The first blocked operation is not established. |
| recovery_huginn_3500_32_native_cuda_only_profile4 | huginn | 1 | Native PD startup/shutdown deadline expired. A completed response cohort does not override a failed terminal status. |
| recovery_huginn_3500_32_prewarmed_profile24 | huginn | 1 | Native PD startup/shutdown deadline expired. Logs and partial artifacts remain preserved. |
| recovery_huginn_3500_32_prewarmed_profile24_descriptor | huginn | 1 | Native PD startup/shutdown deadline expired. The later explicit native shutdown configuration uses a new namespace. |
| recovery_ouro_8192_64_original_gate8 | ouro | 1 | An instrumentation hook read the nonexistent Request.num_output_tokens field. The corrected hook uses generated_token_ids. |
| warm_epoch_ouro_native_gate2 | ouro | 1 | Startup warm-up used the wrong PD statistics API. The corrected worker-counter API passes the separate regression and retry. |
| warm_secondary_aggregated_huginn_l3072_c2048_gate8 | huginn | 1 | CUDA OOM at a 104-MiB allocation. One smaller supported KV-pool retry passes its gate and pilot. The original OOM remains failed. |

## Unrun or unfinished configurations

Older superseded plans remain visible.
Their absence is not a correctness failure or a performance measurement.

| Run | Model | Status | Interpretation |
| --- | --- | --- | --- |
| aggregated_huginn_l2560_c2048_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c2048_q0p4_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c2048_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c2048_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c512_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c512_q0p4_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c512_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_huginn_l2560_c512_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| aggregated_ouro_l6144_c2048_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c2048_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c2048_q0p6_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c2048_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c2048_q1p0_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c2048_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c512_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c512_q0p6_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c512_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c512_q1p0_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_huginn_l2560_c512_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c2048_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c2048_q1p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c2048_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c2048_q2p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c2048_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c512_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c512_q1p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c512_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c512_q2p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_aggregated_ouro_l6144_c512_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2048_q0p6_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2048_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2048_q1p0_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2048_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2560_q0p6_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2560_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2560_q1p0_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_c2560_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_huginn_l2560_native_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c2048_q1p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c2048_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c2048_q2p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c2048_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c6144_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c6144_q2p5_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| async_epoch_ouro_l6144_c6144_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_DP_start1_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_DP_start2_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_DP_start3_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_PD_start1_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_PD_start2_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_huginn_primary_PD_start3_n128 | huginn | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_DP_start1_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_DP_start2_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_DP_start3_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_PD_start1_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_PD_start2_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| confirm_ouro_primary_PD_start3_n128 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_packed_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_packed_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_packed_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_Pslots2_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_legacy_observed_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_legacy_observed_q0p8_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_huginn_legacy_observed_q1p2_pilot24 | huginn | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_packed_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_packed_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_packed_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_credit8g_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_legacy_observed_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_legacy_observed_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_legacy_observed_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_packed_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_packed_q2p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| generic_control_ouro_packed_q3p0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| logging_control_huginn_async_q1p0_n24 | huginn | NOT_RUN | No measurement or success is inferred |
| logging_control_huginn_sync_q1p0_n24 | huginn | NOT_RUN | No measurement or success is inferred |
| primary_huginn_l2560_o32_c2048_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| primary_huginn_l2560_o32_c2560_gate8 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_huginn_l2560_both_native_cuda_primary24 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_huginn_l2560_both_native_cuda_primary24_bounded3_v3 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_huginn_l2560_both_native_cuda_primary24_bounded3_v3_full_P_fallback_v4 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_huginn_l2560_decode_primary24 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_huginn_l2560_prefill_primary24 | huginn | NOT_RUN | No measurement or success is inferred |
| profile_ouro_l6144_both_native_cuda_primary24 | ouro | NOT_RUN | No measurement or success is inferred |
| profile_ouro_l6144_both_native_cuda_primary24_bounded3_v3 | ouro | NOT_RUN | No measurement or success is inferred |
| profile_ouro_l6144_both_native_cuda_primary24_bounded3_v3_full_P_fallback_v4 | ouro | NOT_RUN | No measurement or success is inferred |
| profile_ouro_l6144_decode_primary24 | ouro | NOT_RUN | No measurement or success is inferred |
| profile_ouro_l6144_prefill_primary24 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_original_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_pool25088_drain0_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_pool25088_drain0_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_pool25088_gate8 | ouro | NOT_RUN | No measurement or success is inferred |
| recovery_ouro_8192_64_workspace128_pool25088_pilot24 | ouro | NOT_RUN | No measurement or success is inferred |
