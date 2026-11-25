# H200 qualified prefill pressure and D quiet windows

Qualified observer captures: 0/7. Duplicate receipts for the same start and cohort count once.

| Start | Cap | QPS | Main100 cycles | Evidence | E busy | P busy | D busy | P scheduler wait | P req / token capacity proximity | D quiet ≥160 ms | P busy and backlog overlap | Decode SLO before / profile / after | Motivation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b4_q5_short_burners_on | 4 | 5.00 | 1 | DIAGNOSTIC | 90.40% | 93.90% | 63.76% | 99.66% | 103.12% / 100.69% | 7 | 25.13% | 97.00% / 94.00% / 96.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_burners_off | free | 5.00 | 1 | DIAGNOSTIC | 87.68% | 95.00% | 63.49% | 87.31% | 111.71% / 104.08% | 8 | 21.49% | 92.00% / 95.00% / 95.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_burners_on | free | 5.00 | 1 | DIAGNOSTIC | 87.60% | 95.84% | 64.11% | 94.41% | 114.57% / 105.09% | 7 | 24.60% | 99.00% / 97.00% / 95.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_full_lowfreq_burners_on | free | 5.00 | 1 | DIAGNOSTIC | 88.64% | 93.97% | 59.77% | 100.00% | 108.85% / 99.17% | 8 | 31.30% | 99.00% / 98.00% / 96.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_short_burners_off | free | 5.00 | 1 | DIAGNOSTIC | 90.94% | 96.20% | 60.56% | 99.57% | 105.98% / 103.70% | 7 | 23.61% | 95.00% / 94.00% / 94.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_short_burners_on | free | 5.00 | 1 | DIAGNOSTIC | 90.28% | 95.20% | 64.66% | 99.36% | 100.25% / 99.84% | 6 | 22.38% | 97.00% / 97.00% / 99.00% | P_HEAVY_D_IDLE_NOT_MEASURED |
| free_q5_sustained_r3 | free | 5.00 | 3 | DIAGNOSTIC | 90.95% | 94.52% | 55.82% | 89.92% | 105.05% / 102.21% | 23 | 29.12% | 94.33% / 91.33% / 91.67% | P_HEAVY_D_IDLE_NOT_MEASURED |

## Observer qualification

| Start | Collector | Clean / profile / clean | Failed metric and absolute deviations | Failed combined prerequisites |
| --- | --- | --- | --- | --- |
| b4_q5_short_burners_on | yes | no | duration_seconds (clean 2.42%, profile 7.41%), TTFT_p50_ms (clean 17.71%, profile 52.96%), TTFT_p95_ms (clean 3.99%, profile 12.58%), ITL_p99_ms (clean 5.71%, profile 0.28%), TPOT_p95_ms (clean 0.60%, profile 11.05%) | trace_qualified, absolute_decode_slo |
| free_q5_burners_off | yes | no | TTFT_p50_ms (clean 10.68%, profile 13.14%) | trace_qualified |
| free_q5_burners_on | yes | no | TPOT_p95_ms (clean 0.32%, profile 5.25%) | trace_qualified |
| free_q5_full_lowfreq_burners_on | yes | no | TTFT_p50_ms (clean 2.36%, profile 19.06%), TTFT_p95_ms (clean 1.02%, profile 5.50%), ITL_p99_ms (clean 5.52%, profile 1.06%) | trace_qualified |
| free_q5_short_burners_off | yes | no | TTFT_p50_ms (clean 4.32%, profile 35.32%), TTFT_p95_ms (clean 0.79%, profile 12.35%), ITL_p99_ms (clean 4.14%, profile 5.02%), TPOT_p95_ms (clean 1.00%, profile 16.48%) | trace_qualified, absolute_decode_slo |
| free_q5_short_burners_on | yes | no | TTFT_p50_ms (clean 14.92%, profile 44.37%), TTFT_p95_ms (clean 6.95%, profile 13.96%), TPOT_p95_ms (clean 3.48%, profile 18.04%) | trace_qualified |
| free_q5_sustained_r3 | yes | no | TPOT_p95_ms (clean 2.22%, profile 12.95%) | trace_qualified, absolute_decode_slo |

P-capacity receipts used here · `../h200_wide_runs/capacity_prefill/analysis/capacity_prefill_free.json`

Kernel-busy percentages use unioned CUPTI activity in the central p20–p80 arrival interval. P scheduler backlog is a recent sampled scheduler waiting state. P service proximity compares central completions with a finite isolated drain. Batch mix, KV drain, and cohort boundaries can yield a ratio above 100%. Quiet requires no D kernel activity and no coordinator-visible D work from P-ready through D completion. Overlap uses P kernel busy time as denominator and requires coincident sampled P backlog. It is possible to observe high E kernel busy and still meet all P and D criteria.

The combined gate requires P busy ≥80%, independent P capacity proximity ≥85% for requests and tokens, real scheduler backlog, low P starvation, at least two eligible quiet windows with ≥10% of P busy time overlapping sampled P backlog, at least 32,768 tokens of D KV headroom, and ≥95% absolute decode SLO. The clean/profile/clean tolerance is 5%. The combined gate uses the profiled cohort's Decode SLO.
Pilot eligibility also considers clean-before and clean-after attainment. Physical and operational windows measure observed availability. A future suffix fit or `DECODE_SAFE_HEADROOM_CONFIRMED` requires completed candidate evidence.
