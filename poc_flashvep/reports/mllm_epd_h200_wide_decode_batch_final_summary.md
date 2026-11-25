# H200 wide decode batch token split final summary

Serving status **`HOLD`** · Motivation **`P_HEAVY_D_IDLE_NOT_MEASURED`** · Candidate evidence **`CONDITIONAL_ONLINE_PILOT`**

**Q1. Did the local runtime pass the absolute performance sanity check?** Both paths completed the same 32 ordered Calibration requests. Stock and EPD TTFT p95 were 2226.21 and 2175.52 ms. The results use different GPU counts.

**Q2. What isolated D capacity was reached at each cap?** 1 3.31 req/s, 174.4 output tok/s · 2 5.98 req/s, 315.2 output tok/s · 4 9.70 req/s, 511.3 output tok/s · 8 13.67 req/s, 720.2 output tok/s · 16 13.65 req/s, 719.4 output tok/s · 32 13.91 req/s, 732.8 output tok/s · 64 13.60 req/s, 716.5 output tok/s · free 13.36 req/s, 703.8 output tok/s

**Q3. Were prepared B32 and B64 formed?** B32 1/1 arms reached the target, maximum prepared decode requests 32 · B64 0/1 arms reached the target, maximum prepared decode requests 63

**Q4. Which cap had the best observed trade-off?** The fastest observed isolated cap within ITL p99 ≤34 ms and TPOT p95 ≤33 ms was B8 13.668 req/s, 720.2 output tok/s, ITL p99 27.04 ms, TPOT p95 8.33 ms. The raw throughput peak was B32 13.908 req/s, 732.8 output tok/s, ITL p99 62.90 ms, TPOT p95 19.11 ms. Isolated latency quantiles do not measure per-request absolute Decode SLO or prove an online winner. Clean online NativeCommon screening completed 44/44 cells.

**Q5. Which H200 cap/QPS conditions satisfied the combined motivation gate?** No cap/QPS condition passed the combined gate. B4 QPS5 diagnostic recorded P busy 93.90%, request/token P proximity 103.12%/100.69%, recent P wait 99.66%, ≥160 ms D quiet overlapping busy and backlog 25.13% of P-busy time, and clean/profile/clean Decode SLO 97.00%/94.00%/96.00%. Free QPS5 diagnostic recorded P busy 95.84%, request/token P proximity 114.57%/105.09%, recent P wait 94.41%, ≥160 ms D quiet overlapping busy and backlog 24.60% of P-busy time, and clean/profile/clean Decode SLO 99.00%/97.00%/95.00%. The separate clean Free screen at 2–8 QPS changed TTFT p95 5164→19209 ms and ITL p99 11.93→13.64 ms.

**Q6. Did Free QPS5 pass qualification?** 0/6 distinct Free QPS5 captures passed Clean/Profile/Clean (0/7 overall). The discovery report records every capture, every failed criterion, and clean Decode SLO. Repeated trace-only failures trigger the limited H5 pilot only with independent supporting evidence.

**Q7. Was an E-busy, P-heavy and D-idle regime qualified?** The simultaneous E-busy, P-busy and D-quiet condition is supported at diagnostic level. 7 captures had E kernel busy 87.60%–90.95% with P pressure and operational D quiet overlap.

**Q8. Did Static Split reduce TTFT?** N1 static10-B4 vs native-common-B4 TTFT p95 18841.9/13232.9 ms (+42.4%) · N1 static10-free vs native-common-free TTFT p95 19344.6/13445.2 ms (+43.9%).

**Q9. Did token splitting collapse the actual D batch?** idle10-B4 vs native-common-B4 prepared max 4/4, single-request forward shares 72.28%/67.82% (DIAGNOSTIC/DIAGNOSTIC) · static10-B4 vs native-common-B4 prepared max 1/4, single-request forward shares 100.00%/67.82% (DIAGNOSTIC/DIAGNOSTIC) · idle10-free vs native-common-free prepared max 4/5, single-request forward shares 81.85%/69.80% (DIAGNOSTIC/DIAGNOSTIC) · static10-free vs native-common-free prepared max 1/5, single-request forward shares 100.00%/69.80% (DIAGNOSTIC/DIAGNOSTIC)

**Q10. Did Idle-aware Split improve on Static?** B4 QPS5 Idle minus Static TTFT p95 -5494.6 ms · Free QPS5 Idle minus Static TTFT p95 -6197.4 ms. idle10-B4 selected 23/100, fell back on 77, moved 25150 tokens (2.28%), prefix-ready D quiet 34.78%, selected admission wait p95 3909.12 ms. idle10-free selected 33/100, fell back on 67, moved 32994 tokens (2.99%), prefix-ready D quiet 27.27%, selected admission wait p95 3275.34 ms.

**Q11. Was ITL and TPOT protected with paired intervals?** Independent-start paired 95% CIs are `NOT_MEASURED`. N1 idle10-B4 versus native-common-B4 ITL p99 14.55/13.06 ms, TPOT p95 6.06/5.09 ms, absolute Decode SLO 92.00%/96.00%. N1 static10-B4 versus native-common-B4 ITL p99 6.70/13.06 ms, TPOT p95 4.81/5.09 ms, absolute Decode SLO 97.00%/96.00%. N1 idle10-free versus native-common-free ITL p99 13.14/13.11 ms, TPOT p95 5.40/5.73 ms, absolute Decode SLO 94.00%/95.00%. N1 static10-free versus native-common-free ITL p99 6.03/13.11 ms, TPOT p95 4.84/5.73 ms, absolute Decode SLO 98.00%/95.00%.

**Q12. Was there practical benefit against Native Free?** Practical benefit remains unestablished. It requires lower TTFT against Strong Native Free with the absolute Decode SLO, paired ITL/TPOT bounds, Joint SLO/goodput, and quality guardrails met. N1 idle10-B4 versus native-strong-free TTFT p95 -4.44%, ITL p99 ratio 1.148, TPOT p95 ratio 1.197, candidate Decode SLO 92.00%. N1 static10-B4 versus native-strong-free TTFT p95 +34.90%, ITL p99 ratio 0.529, TPOT p95 ratio 0.951, candidate Decode SLO 97.00%. N1 idle10-free versus native-strong-free TTFT p95 -5.87%, ITL p99 ratio 1.037, TPOT p95 ratio 1.067, candidate Decode SLO 94.00%. N1 static10-free versus native-strong-free TTFT p95 +38.50%, ITL p99 ratio 0.476, TPOT p95 ratio 0.956, candidate Decode SLO 98.00%.

**Q13. Is there enough evidence for modality-aware follow-up?** `HOLD`. The current evidence does not establish a quality-preserving wide-batch serving benefit.

## Campaign closeout

The two QPS5 pilots were limited to Free and B4. All eight clean NativeStrong, NativeCommon, Static10 and Idle10 cohorts completed 100 requests each. Static10 increased TTFT p95 under both caps. Idle10 improved on Static10 and missed the 95% absolute Decode SLO under both caps. The campaign ended after the pilots. Additional ratios, online equal-output replay, independent N3 confirmation and Heldout50 were `NOT_RUN`. The independent D-only arms did use the frozen 5,269-token replay.

A suffix-specific kernel cost, time-weighted actual decode geometry and active KV peak were `NOT_MEASURED`. Output counts varied under Natural EOS. The scoped Static10 correctness checks and an N1 task-score diagnostic cannot establish quality noninferiority.

## Pilot output and task quality

The unchanged Natural EOS outputs were scored offline. Exact output token sequences matched same-cap NativeCommon on static10 FREE 79/100 · idle10 FREE 82/100 · static10 B4 76/100 · idle10 B4 81/100. Official MMLongBench-Doc answer extraction and independent quality intervals remain unavailable. The per-task CSV retains scorer coverage and point estimates.

## Evidence reports

- [Calibration32 runtime sanity](mllm_epd_h200_runtime_performance_sanity_check.md)
- [Isolated capacity and actual decode geometry](mllm_epd_h200_wide_decode_batch_capacity.md)
- [Native online grid and Burner comparison](mllm_epd_h200_decode_batch_qps_motivation.md)
- [P/D combined gates](mllm_epd_h200_qualified_prefill_heavy_discovery.md)
- [Static candidate results](mllm_epd_h200_batch_preserving_static_token_split.md)
- [Idle-aware candidate results](mllm_epd_h200_batch_preserving_idle_token_split.md)
- [Common timestamp analysis](mllm_epd_h200_wide_batch_token_split_critical_path.md)
- [Frozen campaign specification](mllm_epd_h200_wide_decode_batch_token_split_poc_spec.md)

## Evidence tables and inventory

- [Analysis receipt and table row counts](../mllm_epd_h200_wide_decode_batch_token_split/results/receipt.json)
- [Runtime fingerprint, manifest hashes, topology, launch arguments, and arrival seeds](../mllm_epd_h200_wide_decode_batch_token_split/results/execution_metadata.json)
- [Motivation audit and all eight quiet-window thresholds](../mllm_epd_h200_wide_decode_batch_token_split/results/discovery_gates_wide.json)
- [Post-campaign GPU burner verification](../mllm_epd_h200_wide_decode_batch_token_split/results/burner_restore.json)
- [Supplementary figures and timeline selection rule](../mllm_epd_h200_wide_decode_batch_token_split/results/supplementary_figures_receipt.json)
- [Calibration32 matched request inputs, outputs, and latencies](../mllm_epd_h200_wide_decode_batch_token_split/results/wide_sanity_requests.csv)
- [Pilot output equality, task-score scope, and pinned scorer provenance](../mllm_epd_h200_wide_decode_batch_token_split/results/wide_pilot_quality.json)
- [Pilot task-score point estimates by dataset and method](../mllm_epd_h200_wide_decode_batch_token_split/results/wide_pilot_quality_by_task.csv)
- [Per-cap isolated decode capacity](../mllm_epd_h200_wide_decode_batch_token_split/results/wide_decode_capacity.csv)
- [Native QPS grid, scope and case status](../mllm_epd_h200_wide_decode_batch_token_split/results/native_grid.csv)
- [Actual wide decode geometry and D admission](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_cap.csv)
- [Clean / profile / clean observer criteria](../mllm_epd_h200_wide_decode_batch_token_split/results/observer_qualification.csv)
- [D quiet overlap and denominators](../mllm_epd_h200_wide_decode_batch_token_split/results/quiet_overlap.csv)
- [Same-cap candidate comparisons](../mllm_epd_h200_wide_decode_batch_token_split/results/same_cap_contrasts.csv)
- [Practical Free Native comparisons](../mllm_epd_h200_wide_decode_batch_token_split/results/free_native_contrasts.csv)
- [Independent-start paired intervals](../mllm_epd_h200_wide_decode_batch_token_split/results/contrast_ci.csv)
- [Additive host critical path](../mllm_epd_h200_wide_decode_batch_token_split/results/critical_path.csv)
- [Joint SLO breakdown](../mllm_epd_h200_wide_decode_batch_token_split/results/slo_breakdown.csv)
- [Measurements unavailable](../mllm_epd_h200_wide_decode_batch_token_split/results/missing.csv)
- [Failed executions](../mllm_epd_h200_wide_decode_batch_token_split/results/execution_failures.csv)

## Measured figures

- [D observed quiet window CDF natural 1x off deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/D_observed_quiet_window_CDF_natural_1x_off_deferred_all_q5.pdf)
- [D observed quiet window CDF natural 1x on 4 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/D_observed_quiet_window_CDF_natural_1x_on_4_deferred_all_q5.pdf)
- [D observed quiet window CDF natural 1x on 5 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/D_observed_quiet_window_CDF_natural_1x_on_5_deferred_all_q5.pdf)
- [D observed quiet window CDF natural 3x on 5 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/D_observed_quiet_window_CDF_natural_3x_on_5_deferred_all_q5.pdf)
- [D only KV bound and allocation](../mllm_epd_h200_wide_decode_batch_token_split/results/D_only_KV_bound_and_allocation.pdf)
- [D only output ITL TPOT by cap](../mllm_epd_h200_wide_decode_batch_token_split/results/D_only_output_ITL_TPOT_by_cap.pdf)
- [D quiet coverage 1x off deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/D_quiet_coverage_1x_off_deferred_all.pdf)
- [D quiet coverage 1x on 4 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/D_quiet_coverage_1x_on_4_deferred_all.pdf)
- [D quiet coverage 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/D_quiet_coverage_1x_on_5_deferred_all.pdf)
- [D quiet coverage 3x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/D_quiet_coverage_3x_on_5_deferred_all.pdf)
- [E P D sampled queue Free QPS5 first burners on](../mllm_epd_h200_wide_decode_batch_token_split/results/E_P_D_sampled_queue_Free_QPS5_first_burners_on.pdf)
- [PD overlap Free QPS5 first burners on diagnostic](../mllm_epd_h200_wide_decode_batch_token_split/results/PD_overlap_Free_QPS5_first_burners_on_diagnostic.pdf)
- [TTFT ITL tradeoff natural 1x off deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/TTFT_ITL_tradeoff_natural_1x_off_deferred_all.pdf)
- [TTFT ITL tradeoff natural 1x on 4 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/TTFT_ITL_tradeoff_natural_1x_on_4_deferred_all.pdf)
- [TTFT ITL tradeoff natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/TTFT_ITL_tradeoff_natural_1x_on_5_deferred_all.pdf)
- [TTFT ITL tradeoff natural 3x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/TTFT_ITL_tradeoff_natural_3x_on_5_deferred_all.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q2](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q2.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q3](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q3.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q4](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q4.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q5.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q6](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q6.pdf)
- [decode prepared distribution natural 1x on 5 deferred all q8](../mllm_epd_h200_wide_decode_batch_token_split/results/decode_prepared_distribution_natural_1x_on_5_deferred_all_q8.pdf)
- [independent decode capacity](../mllm_epd_h200_wide_decode_batch_token_split/results/independent_decode_capacity.pdf)
- [kernel busy cap qps natural 1x off deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/kernel_busy_cap_qps_natural_1x_off_deferred_all.pdf)
- [kernel busy cap qps natural 1x on 4 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/kernel_busy_cap_qps_natural_1x_on_4_deferred_all.pdf)
- [kernel busy cap qps natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/kernel_busy_cap_qps_natural_1x_on_5_deferred_all.pdf)
- [kernel busy cap qps natural 3x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/kernel_busy_cap_qps_natural_3x_on_5_deferred_all.pdf)
- [native cap qps heatmap natural 1x off deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/native_cap_qps_heatmap_natural_1x_off_deferred_all.pdf)
- [native cap qps heatmap natural 1x on 4 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/native_cap_qps_heatmap_natural_1x_on_4_deferred_all.pdf)
- [native cap qps heatmap natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/native_cap_qps_heatmap_natural_1x_on_5_deferred_all.pdf)
- [native cap qps heatmap natural 3x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/native_cap_qps_heatmap_natural_3x_on_5_deferred_all.pdf)
- [nvml utilization cap qps natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/nvml_utilization_cap_qps_natural_1x_on_5_deferred_all.pdf)
- [pilot host critical path natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/pilot_host_critical_path_natural_1x_on_5_deferred_all.pdf)
- [pilot joint SLO failures natural 1x on 5 deferred all](../mllm_epd_h200_wide_decode_batch_token_split/results/pilot_joint_SLO_failures_natural_1x_on_5_deferred_all.pdf)
- [selected prefix ready D state natural 1x on 5 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/selected_prefix_ready_D_state_natural_1x_on_5_deferred_all_q5.pdf)
- [ttft CDF natural 1x on 5 deferred all q2](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q2.pdf)
- [ttft CDF natural 1x on 5 deferred all q3](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q3.pdf)
- [ttft CDF natural 1x on 5 deferred all q4](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q4.pdf)
- [ttft CDF natural 1x on 5 deferred all q5](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q5.pdf)
- [ttft CDF natural 1x on 5 deferred all q6](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q6.pdf)
- [ttft CDF natural 1x on 5 deferred all q8](../mllm_epd_h200_wide_decode_batch_token_split/results/ttft_CDF_natural_1x_on_5_deferred_all_q8.pdf)
- [wide sanity absolute latency](../mllm_epd_h200_wide_decode_batch_token_split/results/wide_sanity_absolute_latency.pdf)
