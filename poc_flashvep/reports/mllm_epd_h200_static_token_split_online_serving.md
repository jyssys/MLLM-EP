# H200 Static Token Split Online Serving

Phase F의 P-heavy 확인 이후 후보 실험은 `NOT_RUN`이다. Phase B의 QPS2 Batch1 대조군은 Admission/Geometry 진단 근거다. E는 원래 Vision을 유지하고 P contiguous Prefix와 D contiguous Suffix를 사용한다. D Active Decode가 없어야 Suffix 제출을 허용한다.

| 후보 | Phase F 진행 |
| --- | --- |
| Static5-B1 | NOT_RUN |
| Static10-B1 | NOT_RUN |
| Static15-B1 | NOT_RUN |
| Static20-B1 | NOT_RUN |

### 요청 지연과 작업량

완료된 Cohort가 없어 `NOT_RUN`으로 표시한다.

### 처리량과 SLO

완료된 Cohort가 없어 `NOT_RUN`으로 표시한다.

## Phase B QPS2 Equal-output 통제

### 요청 지연과 작업량

| Method | QPS | 출력 계약 | Run/Cohort | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | 출력 토큰 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common/native_common_b1_2_batch_clean_before` | 2,479.37 | 2,105.74 | 5,116.49 | 4.63 | 6.07 | 4.78 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_after` | 2,715.92 | 2,226.30 | 5,350.26 | 4.65 | 5.78 | 4.81 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_before` | 2,563.81 | 2,111.62 | 5,147.30 | 4.68 | 6.15 | 4.83 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_observed` | 2,548.57 | 2,147.75 | 5,163.10 | 4.66 | 6.56 | 4.83 | 5269 | DIAGNOSTIC |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_after` | 3,022.30 | 2,236.54 | 6,174.54 | 4.63 | 6.01 | 4.80 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_before` | 2,695.20 | 2,158.22 | 5,498.68 | 4.67 | 5.95 | 4.81 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_observed` | 2,739.88 | 2,053.03 | 6,125.62 | 4.64 | 5.83 | 4.82 | 5269 | DIAGNOSTIC |

NativeCommon Unrestricted에서 NativeCommon-B1으로 바꾸자 ITL p99가 11.321→6.562ms였다. Static10-B1의 5.827ms와 비교한 절대 격차는 5.494→0.735ms였다. 부호 있는 Native−Static 차이는 +5.494→+0.735ms였다. 이는 Counter를 켠 독립 N1의 방향성 근거다. 인과 기여율이나 적격 C3 우위를 확정하지 않는다.

Static 성능의 기여는 Same Batch1 Native와 Paired Independent Starts에서 평가한다. Unrestricted Native 대비 결과는 실용 기준으로 남긴다. H100/H200, Natural/Replay, Batch Counter/ Clean의 측정군을 따로 판단한다.
