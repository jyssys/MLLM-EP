# Grouped Encode open-loop N24 pilot

Very-long Fixed16, QPS 1.465, seed 2026112809. Each measured arm uses the same H200 allocation if the matched checks pass. These pilot cohorts cover one server start. Quantiles describe completed requests and attainment uses the planned denominator.

| Arm | Status | Complete | TTFT p95 (s) | TPOT p95 (ms) | ITL p99 (ms) | Decode | Joint req/s | D-local |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | MEASURED_ONLINE | 24/24 | 10.626 | 4.331 | 11.093 | 24/24 | 0.333 | 0 |
| Whole-request Isolation | MEASURED_ONLINE | 24/24 | 9.370 | 4.568 | 10.637 | 24/24 | 0.295 | 3 |
| Grouped Late P | MEASURED_ONLINE | 24/24 | 10.838 | 4.408 | 13.935 | 24/24 | 0.236 | 0 |
| Grouped Early P | MEASURED_ONLINE | 24/24 | 10.956 | 3.564 | 10.895 | 24/24 | 0.338 | 0 |
| Grouped Guarded D | MEASURED_ONLINE | 24/24 | 10.307 | 3.920 | 8.153 | 24/24 | 0.398 | 2 |

| Matched reference → candidate | Reply prompts / outputs identical | Δ TTFT p95 | Δ TPOT p95 | Δ ITL p99 | Δ Joint goodput | Decode relative to reference | 15% TTFT gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Native → Whole-request Isolation | 24/24 / 24/24 | -11.82% | +5.46% | -4.11% | -11.37% | FAIL | FAIL |
| Native → Grouped Late P | 24/24 / 24/24 | +2.00% | +1.77% | +25.61% | -29.22% | FAIL | FAIL |
| Native → Grouped Early P | 24/24 / 24/24 | +3.11% | -17.70% | -1.79% | +1.49% | PASS | FAIL |
| Native → Grouped Guarded D | 24/24 / 24/24 | -3.00% | -9.50% | -26.51% | +19.75% | PASS | FAIL |
| Whole-request Isolation → Grouped Early P | 24/24 / 24/24 | +16.94% | -21.96% | +2.42% | +14.50% | PASS | FAIL |
| Whole-request Isolation → Grouped Guarded D | 24/24 / 24/24 | +10.00% | -14.18% | -23.36% | +35.11% | PASS | FAIL |
| Grouped Late P → Grouped Early P | 24/24 / 24/24 | +1.09% | -19.13% | -21.81% | +43.39% | PASS | FAIL |
| Grouped Late P → Grouped Guarded D | 24/24 / 24/24 | -4.90% | -11.07% | -41.49% | +69.18% | PASS | FAIL |
| Grouped Early P → Grouped Guarded D | 24/24 / 24/24 | -5.93% | +9.97% | -25.17% | +17.99% | FAIL | FAIL |

Native-relative Decode limits are +5% on TPOT p95 and ITL p99, at least 95% absolute Decode attainment, and a pass count at least equal to Native. Grouped comparisons against other references isolate the time and recipient policy.

| Arm | Actual route counts | P submitted | P logical rows | E submit→ACK p50 (ms) | P submit→ready p50 (ms) | P submit before E ACK | Proven early Visual retire, P / D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | P_to_D 24 | 24 | 626486 | 1962.2 | 3869.5 | 0 | 0 of 0/0 of 0 |
| Whole-request Isolation | D_local 3, P_to_D 21 | 21 | 546737 | 1868.9 | 3918.7 | 0 | 0 of 0/0 of 0 |
| Grouped Late P | late_p 24 | 24 | 626486 | 2673.4 | 3729.9 | 0 | 0 of 18/0 of 0 |
| Grouped Early P | early_p 18, late_p 6 | 24 | 626486 | 2452.8 | 4008.6 | 18 | 1 of 18/0 of 0 |
| Grouped Guarded D | early_d 2, early_p 16, late_p 6 | 22 | 579342 | 2537.7 | 3514.1 | 16 | 2 of 16/2 of 2 |

A proven early retirement precedes even the producer's lower completion bound. Observed upper-bound counts are retained separately in the receipt.

Observed identity compares prompt and generated token sequences. Scored task quality is outside this pilot.

[Raw-derived paired receipts, events and pressure samples](group_online_pilot.json)
