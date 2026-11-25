# H200 batch-preserving static token split

The following candidate cohorts completed. Any exploratory pilot remains bounded by its authorization and does not qualify the underlying workload.

`CONDITIONAL_ONLINE_PILOT` selected Free QPS5, five spare-GPU burners running, B4 QPS5, same burner state. Static10 correctness first for each cap. Run exactly two pilot conditions. No ratio optimization or Final GO without qualified motivation and independent performance confirmation.

Free rationale · Three diagnostic Main100 traces satisfy every other motivation check. Clean Decode SLO was 95–99%. The sustained trace supports P pressure and overlap but missed the 95% Decode SLO.

B4 rationale · Diagnostic P Busy 93.9%, B4's 160ms quiet coverage 25.1%, isolated D 9.703 req/s, with 95% clean screen and 97%/96% clean brackets. The profiled run's Decode SLO was 94%. The H5 clean SLO criterion uses separate unprofiled cohorts.

| Start | Method | QPS | Output | Scope | Requests | TTFT p50 / p95 ms | ITL p99 ms | TPOT p95 ms | Total tokens/s | Output tokens/s | Requests/s | Decode / Joint SLO | SLO-goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_static10 | static10-B4 | 5.00 | Natural EOS | clean | 100 | 8703.33 / 18841.93 | 6.70 | 4.81 | 27768.75 | 135.07 | 2.51 | 97.00% / 17.00% | 33.87 |
| pilot_free_static10 | static10-free | 5.00 | Natural EOS | clean | 100 | 9645.49 / 19344.56 | 6.03 | 4.84 | 27494.87 | 134.38 | 2.48 | 98.00% / 17.00% | 32.22 |

## Correctness

| Start | Method | Evidence | Receipt |
| --- | --- | --- | --- |
| pilot_b4_correctness_static10 | static10 | CORRECTNESS_PASS | pilot_b4_correctness_static10/analysis/regression.json |
| pilot_free_correctness_static10 | static10 | CORRECTNESS_PASS | pilot_free_correctness_static10/analysis/regression.json |

These receipts cover the launched policy's exact reconstruction checks. Task-level quality requires separate evidence.

## Realized split and D state

| Start | Method | QPS | Split selected | Completed | Moved tokens | Moved fraction | Prefix-ready D inactive and no pending | Selected wait p50 ms | Selected wait p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_static10 | static10-B4 | 5.00 | 100 | 100 | 110296 | 10.00% | 7.00% | 5858.66 | 7873.67 |
| pilot_free_static10 | static10-free | 5.00 | 100 | 100 | 110296 | 10.00% | 9.00% | 6328.40 | 8024.54 |

## Decode geometry and practical comparison

| Start | Cohort | Batch evidence | Prepared max | Prepared mean | Single-request fraction | Suffix selected | Suffix zero active D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_static10 | static10_b4_5_observed | DIAGNOSTIC | 1 | 1.0 | 100.00% | 100 | 100 |
| pilot_free_static10 | static10_free_5_observed | DIAGNOSTIC | 1 | 1.0 | 100.00% | 100 | 100 |

Candidate versus same-cap Native comparisons require shared
arrival trace, output mode, runtime, topology, and admission discipline. Practical benefit additionally requires the
Free Native reference. A change in ITL alone cannot establish a TTFT benefit. An actually smaller prepared-batch
distribution under split can confound the effect even under the same configured cap.

### Conditional pilot N1 contrasts

| N1 contrast | Baseline | Candidate | QPS | Baseline TTFT p95 ms | Candidate TTFT p95 ms | TTFT p95 change % | ITL p99 ratio | TPOT p95 ratio | Baseline Decode SLO | Candidate Decode SLO | Baseline Joint SLO | Candidate Joint SLO | Baseline SLO-goodput | Candidate SLO-goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Same condition | native-common-B4 | static10-B4 | 5.00 | 13232.86 | 18841.93 | 42.39 | 0.51 | 0.95 | 96.00% | 97.00% | 52.00% | 17.00% | 71.57 | 33.87 |
| Same condition | native-strong-B4 | static10-B4 | 5.00 | 13104.29 | 18841.93 | 43.78 | 0.52 | 0.93 | 99.00% | 97.00% | 55.00% | 17.00% | 93.97 | 33.87 |
| Strong Native Free | native-strong-free | static10-B4 | 5.00 | 13967.34 | 18841.93 | 34.90 | 0.53 | 0.95 | 97.00% | 97.00% | 48.00% | 17.00% | 63.71 | 33.87 |
| Same condition | native-common-free | static10-free | 5.00 | 13445.19 | 19344.56 | 43.88 | 0.46 | 0.85 | 95.00% | 98.00% | 51.00% | 17.00% | 88.60 | 32.22 |
| Same condition | native-strong-free | static10-free | 5.00 | 13967.34 | 19344.56 | 38.50 | 0.48 | 0.96 | 97.00% | 98.00% | 48.00% | 17.00% | 63.71 | 32.22 |
| Strong Native Free | native-strong-free | static10-free | 5.00 | 13967.34 | 19344.56 | 38.50 | 0.48 | 0.96 | 97.00% | 98.00% | 48.00% | 17.00% | 63.71 | 32.22 |

### Same-cap paired cohorts

| Start | Compared native arm | Candidate | QPS | Output | Metric | Native | Candidate | Candidate / Native | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4661.815 | 8703.328 | 1.867 | MEASURED_ONLINE |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 13232.858 | 18841.929 | 1.424 | MEASURED_ONLINE |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 13.061 | 6.701 | 0.513 | MEASURED_ONLINE |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.086 | 4.814 | 0.947 | MEASURED_ONLINE |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | SLO_goodput | 71.570 | 33.873 | 0.473 | MEASURED_ONLINE |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4400.890 | 8703.328 | 1.978 | MEASURED_ONLINE |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 13104.286 | 18841.929 | 1.438 | MEASURED_ONLINE |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 12.954 | 6.701 | 0.517 | MEASURED_ONLINE |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.182 | 4.814 | 0.929 | MEASURED_ONLINE |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | SLO_goodput | 93.970 | 33.873 | 0.360 | MEASURED_ONLINE |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 4009.084 | 8740.113 | 2.180 | DIAGNOSTIC |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 12381.103 | 17920.077 | 1.447 | DIAGNOSTIC |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 14.108 | 6.033 | 0.428 | DIAGNOSTIC |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.146 | 4.805 | 0.934 | DIAGNOSTIC |
| pilot_b4_static10 | native-common-B4 | static10-B4 | 5.0 | Natural EOS | SLO_goodput | 93.667 | 31.524 | 0.337 | DIAGNOSTIC |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p50_ms | 3773.507 | 8740.113 | 2.316 | DIAGNOSTIC |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TTFT_p95_ms | 12024.368 | 17920.077 | 1.490 | DIAGNOSTIC |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | ITL_p99_ms | 13.255 | 6.033 | 0.455 | DIAGNOSTIC |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | TPOT_p95_ms | 5.399 | 4.805 | 0.890 | DIAGNOSTIC |
| pilot_b4_static10 | native-strong-B4 | static10-B4 | 5.0 | Natural EOS | SLO_goodput | 95.458 | 31.524 | 0.330 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4796.438 | 9645.492 | 2.011 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13445.188 | 19344.562 | 1.439 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 13.113 | 6.034 | 0.460 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.725 | 4.838 | 0.845 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 88.603 | 32.224 | 0.364 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4948.516 | 9645.492 | 1.949 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13967.337 | 19344.562 | 1.385 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.677 | 6.034 | 0.476 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.062 | 4.838 | 0.956 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 63.714 | 32.224 | 0.506 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 3963.153 | 8678.747 | 2.190 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12121.379 | 18125.308 | 1.495 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.247 | 6.490 | 0.530 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.383 | 4.831 | 0.898 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 68.869 | 35.560 | 0.516 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4331.642 | 8678.747 | 2.004 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12547.589 | 18125.308 | 1.445 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.917 | 6.490 | 0.502 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.280 | 4.831 | 0.915 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 84.969 | 35.560 | 0.419 | DIAGNOSTIC |

### Free Native paired cohorts

| Start | Compared native arm | Candidate | QPS | Output | Metric | Native | Candidate | Candidate / Native | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4796.438 | 9645.492 | 2.011 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13445.188 | 19344.562 | 1.439 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 13.113 | 6.034 | 0.460 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.725 | 4.838 | 0.845 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 88.603 | 32.224 | 0.364 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4948.516 | 9645.492 | 1.949 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 13967.337 | 19344.562 | 1.385 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.677 | 6.034 | 0.476 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.062 | 4.838 | 0.956 | MEASURED_ONLINE |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 63.714 | 32.224 | 0.506 | MEASURED_ONLINE |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 3963.153 | 8678.747 | 2.190 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12121.379 | 18125.308 | 1.495 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.247 | 6.490 | 0.530 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.383 | 4.831 | 0.898 | DIAGNOSTIC |
| pilot_free_static10 | native-common-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 68.869 | 35.560 | 0.516 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p50_ms | 4331.642 | 8678.747 | 2.004 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TTFT_p95_ms | 12547.589 | 18125.308 | 1.445 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | ITL_p99_ms | 12.917 | 6.490 | 0.502 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | TPOT_p95_ms | 5.280 | 4.831 | 0.915 | DIAGNOSTIC |
| pilot_free_static10 | native-strong-free | static10-free | 5.0 | Natural EOS | SLO_goodput | 84.969 | 35.560 | 0.419 | DIAGNOSTIC |
