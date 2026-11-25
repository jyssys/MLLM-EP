# H200 batch-preserving idle-aware token split

The following candidate cohorts completed. Any exploratory pilot remains bounded by its authorization and does not qualify the underlying workload.

`CONDITIONAL_ONLINE_PILOT` selected Free QPS5, five spare-GPU burners running, B4 QPS5, same burner state. Static10 correctness first for each cap. Run exactly two pilot conditions. No ratio optimization or Final GO without qualified motivation and independent performance confirmation.

Free rationale · Three diagnostic Main100 traces satisfy every other motivation check. Clean Decode SLO was 95–99%. The sustained trace supports P pressure and overlap but missed the 95% Decode SLO.

B4 rationale · Diagnostic P Busy 93.9%, B4's 160ms quiet coverage 25.1%, isolated D 9.703 req/s, with 95% clean screen and 97%/96% clean brackets. The profiled run's Decode SLO was 94%. The H5 clean SLO criterion uses separate unprofiled cohorts.

| Start | Method | QPS | Output | Scope | Requests | TTFT p50 / p95 ms | ITL p99 ms | TPOT p95 ms | Total tokens/s | Output tokens/s | Requests/s | Decode / Joint SLO | SLO-goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_idle10 | idle10-B4 | 5.00 | Natural EOS | clean | 100 | 6525.81 / 13347.33 | 14.55 | 6.06 | 32315.30 | 159.79 | 2.92 | 92.00% / 28.00% | 54.20 |
| pilot_free_idle10 | idle10-free | 5.00 | Natural EOS | clean | 100 | 5885.51 / 13147.12 | 13.14 | 5.40 | 32248.16 | 155.32 | 2.91 | 94.00% / 43.00% | 69.95 |

## Correctness

`NOT_MEASURED` for this campaign.

These receipts cover the launched policy's exact reconstruction checks. Task-level quality requires separate evidence.

## Realized split and D state

| Start | Method | QPS | Split selected | Completed | Moved tokens | Moved fraction | Prefix-ready D inactive and no pending | Selected wait p50 ms | Selected wait p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_idle10 | idle10-B4 | 5.00 | 23 | 100 | 25150 | 2.28% | 34.78% | 400.28 | 3909.12 |
| pilot_free_idle10 | idle10-free | 5.00 | 33 | 100 | 32994 | 2.99% | 27.27% | 303.15 | 3275.34 |

## Decode geometry and practical comparison

| Start | Cohort | Batch evidence | Prepared max | Prepared mean | Single-request fraction | Suffix selected | Suffix zero active D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_idle10 | idle10_b4_5_observed | DIAGNOSTIC | 4 | 1.4214833759590793 | 72.28% | 26 | 26 |
| pilot_free_idle10 | idle10_free_5_observed | DIAGNOSTIC | 4 | 1.216090888012984 | 81.85% | 40 | 40 |

Candidate versus same-cap Native comparisons require shared
arrival trace, output mode, runtime, topology, and admission discipline. Practical benefit additionally requires the
Free Native reference. A change in ITL alone cannot establish a TTFT benefit. An actually smaller prepared-batch
distribution under split can confound the effect even under the same configured cap.

### Conditional pilot N1 contrasts

| N1 contrast | Baseline | Candidate | QPS | Baseline TTFT p95 ms | Candidate TTFT p95 ms | TTFT p95 change % | ITL p99 ratio | TPOT p95 ratio | Baseline Decode SLO | Candidate Decode SLO | Baseline Joint SLO | Candidate Joint SLO | Baseline SLO-goodput | Candidate SLO-goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Same condition | native-common-B4 | idle10-B4 | 5.00 | 13232.86 | 13347.33 | 0.87 | 1.11 | 1.19 | 96.00% | 92.00% | 52.00% | 28.00% | 71.57 | 54.20 |
| Same condition | native-strong-B4 | idle10-B4 | 5.00 | 13104.29 | 13347.33 | 1.85 | 1.12 | 1.17 | 99.00% | 92.00% | 55.00% | 28.00% | 93.97 | 54.20 |
| Same condition | static10-B4 | idle10-B4 | 5.00 | 18841.93 | 13347.33 | -29.16 | 2.17 | 1.26 | 97.00% | 92.00% | 17.00% | 28.00% | 33.87 | 54.20 |
| Strong Native Free | native-strong-free | idle10-B4 | 5.00 | 13967.34 | 13347.33 | -4.44 | 1.15 | 1.20 | 97.00% | 92.00% | 48.00% | 28.00% | 63.71 | 54.20 |
| Same condition | native-common-free | idle10-free | 5.00 | 13445.19 | 13147.12 | -2.22 | 1.00 | 0.94 | 95.00% | 94.00% | 51.00% | 43.00% | 88.60 | 69.95 |
| Same condition | native-strong-free | idle10-free | 5.00 | 13967.34 | 13147.12 | -5.87 | 1.04 | 1.07 | 97.00% | 94.00% | 48.00% | 43.00% | 63.71 | 69.95 |
| Strong Native Free | native-strong-free | idle10-free | 5.00 | 13967.34 | 13147.12 | -5.87 | 1.04 | 1.07 | 97.00% | 94.00% | 48.00% | 43.00% | 63.71 | 69.95 |
| Same condition | static10-free | idle10-free | 5.00 | 19344.56 | 13147.12 | -32.04 | 2.18 | 1.12 | 98.00% | 94.00% | 17.00% | 43.00% | 32.22 | 69.95 |

### Same-cap paired cohorts

| Start | Compared native arm | Candidate | QPS | Output | Metric | Native | Candidate | Candidate / Native | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4661.815 | 6525.805 | 1.400 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 13232.858 | 13347.326 | 1.009 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 13.061 | 14.554 | 1.114 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.086 | 6.060 | 1.192 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | SLO_goodput | 71.570 | 54.197 | 0.757 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4400.890 | 6525.805 | 1.483 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 13104.286 | 13347.326 | 1.019 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 12.954 | 14.554 | 1.124 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.182 | 6.060 | 1.170 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | SLO_goodput | 93.970 | 54.197 | 0.577 | MEASURED_ONLINE |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4009.084 | 4598.902 | 1.147 | DIAGNOSTIC |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 12381.103 | 11675.133 | 0.943 | DIAGNOSTIC |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 14.108 | 13.852 | 0.982 | DIAGNOSTIC |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.146 | 5.513 | 1.071 | DIAGNOSTIC |
| pilot_b4_idle10 | native-common-B4 | idle10-B4 | 5.0 | Natural EOS | SLO_goodput | 93.667 | 83.960 | 0.896 | DIAGNOSTIC |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 3773.507 | 4598.902 | 1.219 | DIAGNOSTIC |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 12024.368 | 11675.133 | 0.971 | DIAGNOSTIC |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 13.255 | 13.852 | 1.045 | DIAGNOSTIC |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.399 | 5.513 | 1.021 | DIAGNOSTIC |
| pilot_b4_idle10 | native-strong-B4 | idle10-B4 | 5.0 | Natural EOS | SLO_goodput | 95.458 | 83.960 | 0.880 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4796.438 | 5885.509 | 1.227 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13445.188 | 13147.123 | 0.978 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 13.113 | 13.144 | 1.002 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.725 | 5.403 | 0.944 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 88.603 | 69.949 | 0.789 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4948.516 | 5885.509 | 1.189 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13967.337 | 13147.123 | 0.941 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.677 | 13.144 | 1.037 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.062 | 5.403 | 1.067 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 63.714 | 69.949 | 1.098 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 3963.153 | 5241.734 | 1.323 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12121.379 | 12710.582 | 1.049 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.247 | 12.727 | 1.039 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.383 | 5.170 | 0.960 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 68.869 | 75.882 | 1.102 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4331.642 | 5241.734 | 1.210 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12547.589 | 12710.582 | 1.013 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.917 | 12.727 | 0.985 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.280 | 5.170 | 0.979 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 84.969 | 75.882 | 0.893 | DIAGNOSTIC |

### Free Native paired cohorts

| Start | Compared native arm | Candidate | QPS | Output | Metric | Native | Candidate | Candidate / Native | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4796.438 | 5885.509 | 1.227 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13445.188 | 13147.123 | 0.978 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 13.113 | 13.144 | 1.002 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.725 | 5.403 | 0.944 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 88.603 | 69.949 | 0.789 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4948.516 | 5885.509 | 1.189 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13967.337 | 13147.123 | 0.941 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.677 | 13.144 | 1.037 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.062 | 5.403 | 1.067 | MEASURED_ONLINE |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 63.714 | 69.949 | 1.098 | MEASURED_ONLINE |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 3963.153 | 5241.734 | 1.323 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12121.379 | 12710.582 | 1.049 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.247 | 12.727 | 1.039 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.383 | 5.170 | 0.960 | DIAGNOSTIC |
| pilot_free_idle10 | native-common-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 68.869 | 75.882 | 1.102 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4331.642 | 5241.734 | 1.210 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12547.589 | 12710.582 | 1.013 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.917 | 12.727 | 0.985 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.280 | 5.170 | 0.979 | DIAGNOSTIC |
| pilot_free_idle10 | native-strong-free | idle10-free | 5.0 | Natural EOS | SLO_goodput | 84.969 | 75.882 | 0.893 | DIAGNOSTIC |
