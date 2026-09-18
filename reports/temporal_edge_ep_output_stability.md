# Temporal-Edge EP: actual expert-branch output stability

This targeted trace re-executed current-block branches at routed layers (1, 5, 10, 14, 19) on 16 fixed GSM8K requests (0--7 discovery, 8--15 held out). The official MoE result was returned unchanged; these timings are not performance measurements.

There are 782,876 exact adjacent live-MASK STAY branches. Raw output relative-L2 P50/P90/P99 is 0.3500/0.8811/1.4794; cosine P50 is 0.9431. Only 3.349% are within 5% relative-L2, so identity persistence does not imply numerical reuse safety. R1 (current-weight reweighting) removes router-weight drift but cannot remove raw branch drift; R0 is worse (P50 L2 0.3672).

The observational collector matched the frozen reference on NFE for 16/16 requests and correctness for 16/16, with 0 remaining MASK tokens.

The discovery-selected current-observable E3 thresholds are `{'tau_h': 0.02, 'tau_w': 0.05, 'min_age': 2, 'precision': 1.0, 'recall': 0.023122207248741044, 'coverage': 0.0008233861883281219, 'selected': 326}`. Held-out E3 coverage/precision/recall is `{'records': 386950, 'selected': 196, 'coverage_of_stay': 0.0005065253908773744, 'precision_for_safe': 1.0, 'recall_of_safe': 0.01617028297995215}`; E4 is `{'records': 386950, 'selected': 81, 'coverage_of_stay': 0.00020932937071973124, 'precision_for_safe': 1.0, 'recall_of_safe': 0.006682616945796551}`. Full distributions, layer/age breakdowns and all four oracle thresholds are in `artifacts/dllm_native_ep_discovery/branch_stability_summary.json`.
