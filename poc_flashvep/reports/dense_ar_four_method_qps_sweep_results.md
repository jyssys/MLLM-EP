# Four-method QPS sweep

## Protocol

The frozen main grid completed 108 independent N128 starts and one separate Native reference start.
Each of nine workload cells compares Native, qualified Continuous DP2, Best Static, and Revised Dynamic.
Each arm uses three paired arrival seeds.
No method selects settings from fresh confirmation results.
Each output cap uses its own viewed QPS4.8 selection.
Those settings remain fixed across its QPS sweep.
The selected Static and Colocated arms are not exhaustive per-QPS optima.
Colocated selection maximizes Balanced goodput, not a separately tuned Decode-priority frontier.

| Output cap | Offered QPS | Static choice | Revised admission-wait limit |
|---|---|---|---:|
| 16 |2.5,4.8,6.4 | Last8,100% |250 ms |
| 64 |2.5,4.8,6.4 | Native Last0 |0 ms |
| 200 |1.5,2.5,4.8 | Native Last0 |0 ms |

Input length remains8000 tokens.
Native and PD variants use P slots4 and D slots16.
P has a32000-token budget.
D has a128000-token budget.
Layer-policy reservation count is at most 32.
Native retains its scheduler's normal KV admission and Source lifetime.
Every GPU retains42 GiB KV.
The [Continuous report](dense_ar_continuous_colocated_correctness_and_baseline.md) records its independent chunk and sequence tuning.

The model revision remains `b968826d9c46dd6066d109eabc6255188de91218`.
Runtime remains vLLM0.26.0+cu129, PyTorch2.11.0+cu129, and NIXL1.4.1.
GPU0/1 UUIDs, source hashes, prompt IDs, and exact commands accompany every result.
Original BF16, exact Attention, Full36 Decode, greedy sampling, and disabled thinking remain unchanged.

The fresh128 LongBench prompts exclude previously viewed IDs and long token prefixes.
No repeated filler creates additional Prefill work.
Ignore EOS fixes the requested output cap.
Natural EOS and bursty arrivals remain separate `NOT_RUN` generalization tests.

## Independent uncertainty

Each plotted latency is the mean of independent start-level quantiles.
It is not a quantile of pooled tokens.
Each95% interval uses the empirical start bootstrap.
Three paired starts produce27 paired bootstrap draws.
The limited sample count does not establish a production guarantee.

Native-relative protection requires both ITL p99 and TPOT p95 confidence upper bounds at most 1.05.
Absolute SLO attainment remains a separate result.
Every frozen regime uses request TTFT, request-average TPOT, and request-maximum ITL for joint attainment.
First ITL is included.
The raw metrics also contain first and steady ITL separately.
No completed confirmation start contained a coalesced token interval in the checked records.

## Output16 confirmation

| QPS | Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output tokens/s | Decode 5% CI gate |
|---|---|---:|---:|---:|---:|---|
|2.5 | Native |1714.59 |26.39 |25.62 |39.37 | Reference |
|2.5 | Continuous |874.22 |50.37 |29.90 |39.64 | NOT ESTABLISHED |
|2.5 | Static Last8 |1526.64 |27.72 |26.66 |39.34 | NOT ESTABLISHED |
|2.5 | Revised |1789.67 |27.99 |26.97 |39.21 | NOT ESTABLISHED |
|4.8 | Native |10937.12 |26.98 |25.42 |54.11 | Reference |
|4.8 | Continuous |3026.39 |35.84 |29.32 |70.13 | NOT ESTABLISHED |
|4.8 | Static Last8 |4550.11 |28.98 |27.39 |66.29 | NOT ESTABLISHED |
|4.8 | Revised |5317.00 |28.26 |27.03 |64.59 | NOT ESTABLISHED |
|6.4 | Native |16696.71 |28.09 |26.16 |54.62 | Reference |
|6.4 | Continuous |7066.23 |35.04 |27.66 |74.04 | NOT ESTABLISHED |
|6.4 | Static Last8 |10022.04 |27.35 |26.01 |66.74 | PASS |
|6.4 | Revised |10036.89 |27.18 |26.02 |66.95 | PASS |

Static and Revised improve the Native-relative trade-off at QPS6.4.
Revised does not establish additional TTFT improvement over Static.
Colocated has the lowest TTFT and highest raw generation rate in those overloaded cells.
Its Decode tails exceed the Native-relative criterion.
Thus, the measurements show a service trade-off, not universal superiority of one arm.

## Output64 confirmation

| QPS | Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output tokens/s |
|---|---|---:|---:|---:|---:|
|2.5 | Native |3346.49 |28.81 |27.32 |151.32 |
|2.5 | Continuous |953.18 |73.01 |30.22 |154.57 |
|2.5 | Static Last0 |3144.24 |26.31 |25.77 |152.26 |
|2.5 | Revised |3292.01 |28.40 |27.35 |151.68 |
|4.8 | Native |12683.36 |30.02 |27.18 |199.84 |
|4.8 | Continuous |3286.91 |35.73 |29.33 |267.13 |
|4.8 | Static Last0 |12420.54 |30.00 |27.35 |204.41 |
|4.8 | Revised |12068.34 |29.52 |27.28 |205.55 |
|6.4 | Native |18026.74 |28.51 |26.34 |207.39 |
|6.4 | Continuous |8620.14 |35.93 |29.57 |268.34 |
|6.4 | Static Last0 |18246.26 |31.36 |28.18 |205.35 |
|6.4 | Revised |17984.05 |27.98 |26.37 |205.10 |

Revised migrates0/384 requests at every tested Output64 QPS.
Static Last0 uses the identical Native implementation in independent starts.
Differences between these zero-migration arms are not Layer Split gains.
They demonstrate runtime variation and controller effects.

## Output200 confirmation

All lower-load Output200 cells completed with the same frozen settings.

| QPS | Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | E2E p95 (ms) | Output tokens/s |
|---|---|---:|---:|---:|---:|---:|
|1.5 | Native |6569.584 |29.966 |29.431 |12079.977 |267.492 |
|1.5 | Continuous |1342.899 |69.827 |34.086 |7798.452 |281.832 |
|1.5 | Static Last0 |6327.666 |29.072 |28.455 |11706.640 |270.497 |
|1.5 | Revised |6123.228 |28.685 |27.892 |11418.117 |273.145 |
|2.5 | Native |11386.296 |32.066 |30.871 |17242.935 |389.650 |
|2.5 | Continuous |1558.682 |54.021 |29.370 |7294.235 |446.666 |
|2.5 | Static Last0 |10407.077 |31.967 |30.983 |16194.467 |396.663 |
|2.5 | Revised |10257.877 |31.110 |30.019 |16076.621 |397.480 |
|4.8 | Native |28009.778 |30.735 |29.522 |33880.716 |426.423 |
|4.8 | Continuous |11565.490 |35.711 |29.914 |17036.011 |581.780 |
|4.8 | Static Last0 |29750.239 |31.634 |30.765 |35676.996 |413.103 |
|4.8 | Revised |27177.089 |30.604 |29.087 |32753.567 |434.138 |

Revised completes no Layer Offloading at any tested Output200 QPS.
Its small latency difference from Native is not an Offloading speedup.
Its maximum ITL reaches 198.89 ms in one start despite a near-Native p99.
Native maxima are49.34/47.09/46.37 ms.
The report does not hide that additional tail risk behind a passing p99 ratio.

For that worst interval, q107 advanced from output count 17 to 18.

- Coordinator ITL: 198.889 ms.
- Worker-timestamp interval: 28.077 ms.
- Change in delivery lag: 170.812 ms.

These values satisfy the interval timestamp identity.
They do not identify an exclusive GPU slowdown.
The exact delivery-stall component remains `NOT_IDENTIFIED`.

## Maximum ITL and threshold counts

At Output16/QPS4.8, Balanced ITL exceedances were 0/0/0 for Native, Static, and Revised.
Colocated recorded 18/76/23 exceedances.
At Output16/QPS6.4, Revised and Native again recorded 0/0/0.
Static recorded 0/0/4 and Colocated22/21/28.

At Output64/QPS4.8, Native, Static Last0, and Revised recorded 0/0/0 Balanced exceedances.
Colocated recorded 86/222/93.
These are interval counts, not request counts or p99 values.
The complete per-start metric table preserves every maximum and threshold count.

## Capacity and queue interpretation

Independent P-only batch4 capacity was 3.58564 requests/s.
The QPS grid brackets that reference rather than using arbitrary integer loads only.
Native P-ready queues grow at the overloaded Output16 anchor without KV refusal.
The main report gives the per-start queue observations.

Output200/QPS4.8 is not a qualified pure Prefill-compute bottleneck.
Native P KV allocation attempts failed 30510/22253/20156 times across the three starts.
Those are repeated allocator attempts, not rejected request counts.
Minimum P free KV blocks reached 113.
All requests still completed and returned normal credits.
The evidence indicates Source KV retention and longer Decode pressure in that cell.
It does not indicate a KV leak.

Prepared Decode capacity at batch16 was approximately 35.02/8.43/2.67 request-equivalents per second for Output16/64/200.
Those resident-KV capacity diagnostics are not sustainable online capacities.
Output200/QPS4.8 exceeds the corresponding independent Decode reference.
The report does not combine that point with Output16's cleaner queue evidence to claim same-window P saturation.

Clean P kernel busy and D physical idle remain `NOT_MEASURED`.
Therefore, the strict physical saturation gate is not established.
The observed TTFT/Decode trade-off is real within the marker-free protocol.
It is not a qualified physical-slack capacity measurement.
Decode kernel slowdown versus Host scheduling remains `NOT_IDENTIFIED` without a qualified kernel trace.

## Complete evidence and figures

- [Independent start metrics CSV](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/independent_start_metrics.csv).
- [All cell estimates and paired confidence intervals](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/four_method_qps_results.json).
- [Frozen selection](dense_ar_decode_safe_elastic_layer_pd_20261113/configs/heldout_selection.json).
- [QPS versus TTFT](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_A_qps_vs_ttft.png).
- [QPS versus TPOT](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_B_qps_vs_tpot.png).
- [QPS versus generation throughput](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_C_qps_vs_generation_throughput.png).
- [QPS versus ITL](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_D_qps_vs_itl.png).
- [Balanced joint attainment](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_E_balanced_joint_attainment.png).
- [Balanced goodput](dense_ar_decode_safe_elastic_layer_pd_20261113/figures/figure_F_balanced_slo_goodput.png).

All rates include arrival-to-drain observation time.
Offered QPS, realized arrival QPS, actual submission QPS, and completed QPS remain distinct fields.
No finite N128 result claims sustainable capacity.
