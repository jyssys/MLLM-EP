# H200 MLLM EPD Prefill-heavy와 Batch-matched Token Split 종합

연구 판정은 `HOLD`다. Motivation 판정은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 현재 독립 Stage Capacity 상태는 측정 완료 4/4이다.

**Q1. H200 환경과 Dataset을 재구축했는가?** 예. 고정 모델·Media·NPZ·Manifest 해시와 H200 Runtime Fingerprint를 검증했다. Native N8 상태 8/8, Greedy 6/8, Token10 두 기준 8/8이다. Wheel SHA와 Driver/Hardware는 H100과 다르다.

**Q2. Actual Decode Batch를 1로 통일했는가?** Prepared Decode 최대값 native-common-free 4, idle10-B1 1, native-common-B1 1, static10-B1 1을 관측했다. 같은 QPS2 Equal-output Counter Cohort의 근거다.

**Q3. Batch1 이후 Static10과 ITL 격차는?** NativeCommon Unrestricted에서 NativeCommon-B1으로 바꾸자 ITL p99가 11.321→6.562ms였다. Static10-B1의 5.827ms와 비교한 절대 격차는 5.494→0.735ms였다. 부호 있는 Native−Static 차이는 +5.494→+0.735ms였다. 이는 Counter를 켠 독립 N1의 방향성 근거다. 인과 기여율이나 적격 C3 우위를 확정하지 않는다.

**Q4. P-heavy와 D Idle이 동시 발생하는 QPS를 찾았는가?** `P_HEAVY_D_IDLE_NOT_MEASURED`. 핵심 Gate는 적격 Trace, 실제 P Backlog 및 독립 Capacity다.

**Q5. P-heavy의 근거가 부족한 이유는?** - `discovery_gate_free_q5.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified이다. - `discovery_gate_free_q5_deferred.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified이다. - `discovery_gate_q3.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, p_kernel_busy, p_service_proximity, p_not_starved, d_quiet_window_coverage이다. - `discovery_gate_q4.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, p_kernel_busy, d_quiet_window_coverage이다. - `discovery_gate_q5.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다. - `discovery_gate_q5_deferred.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다. - `discovery_gate_q5_sustained_r3.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다. - `discovery_gate_q5_sustained_r3_buffer2.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다. - `discovery_gate_q6.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.

**Q6. 실제 Prefix-ready에서 D가 얼마나 자주 비어 있었는가?** Phase B Static10-B1의 100건은 D HTTP 비활성 32.00%, 비활성·대기 없음 32.00%, Coordinator Wait p95 2,471.53ms였다. 적격 P-heavy 결과는 별도다.

**Q7. 확인된 P-heavy에서 Static Split이 TTFT를 개선했는가?** Phase F의 완료 Cohort는 0개다. Batch1과 Unrestricted의 Paired 기준을 별도로 평가한다.

**Q8. ITL/TPOT가 유지됐는가?** Phase B 진단의 수치는 표에 있다. 적격 P-heavy에서의 Decode 비열등성과 품질은 Paired CI 및 Absolute 95%로 별도 검사한다.

**Q9. Static10 대비 Idle10의 추가 이득은?** Phase G의 완료 Cohort는 0개다. QPS2 진단의 Split 실행률, 이전 Token, Native 경로 및 D 대기는 별도 보고서에 남겼다.

**Q10. 기존 H100 실패의 원인은?** Static10에서 P HTTP 절감보다 D Admission 대기가 더 컸다는 경로 변화가 직접 관측됐다. Actual Decode Batch 차이는 관측 간섭 실패 N1에 해당한다. Prefill-heavy와 독립 Stage Capacity는 이전에 미입증이었다.

**Q11. Modality-aware로 확장할 근거가 있는가?** `HOLD`. 실제로 확인된 P-heavy, Batch1 일치, TTFT 개선과 Decode·품질 보호가 필요하다.


## 결과표

### 판정의 근거

QPS 5·6의 Batch1 진단에서 P Kernel Busy는 각각 96.83%, 93.81%였다. P Busy와 최근 P Backlog에 겹친 160ms 이상 D 무작업·무대기 구간은 각각 4.72%, 7.99%였다. 각각 한 번으로 사전 기준인 두 구간과 P Busy 대비 10%에 미달했다. 300-arrival 반복에서 Decode Trace Buffer를 2GiB로 늘리자 기록 손실은 사라졌고 구간 겹침은 1.64%였다. 이 반복도 Clean 간 ITL p99 편차 5.53%로 적격 기준을 통과하지 못했다. 전체 Trace는 진단 근거로 보존했다.

독립 D의 완료율은 Batch1에서 3.32 req/s, Unrestricted에서 7.19 req/s였다. P의 E 임베딩 준비 후 로컬 KV 반환 기준은 3.18 req/s였다. Batch1에서 두 Stage의 처리율이 가까워지고 고부하 D 여유가 줄었다는 해석과 일치한다. Unrestricted QPS5 진단에서는 긴 D 무작업·무대기 겹침이 두 번의 계측에서 26–27%였으나 두 계측 모두 적격 기준에 미달했다. 온라인과 독립 Burst의 Batch·입력 길이·관측 구간이 다르므로 100%를 넘는 Proximity는 절대 처리 한계로 해석하지 않는다.

QPS2 Clean 전후 평균에서 NativeCommon Unrestricted의 ITL p99 12.19ms는 Batch1에서 5.97ms였고, Static10-B1은 5.98ms였다. Static10-B1은 NativeCommon-B1 대비 평균 P HTTP가 311ms 짧았고 평균 D Coordinator Wait가 347ms 길었으며 평균 TTFT가 219ms 길었다. 이 수치는 독립 Start 한 번의 진단 비교다. Idle10-B1의 TTFT p95는 4.96초였으며, Unrestricted NativeCommon의 4.47초보다 높았다. Phase F와 G의 적격 P-heavy 비교 및 신뢰구간은 미실행이다.

실험은 GPU 4·5·6을 사용했고 나머지 다섯 대에서는 버너가 가동됐다. 실험 GPU에는 동시 버너가 없었으며, 공유 CPU·메모리·전력·열의 간접 간섭을 분리하는 전부 종료 대조 실험은 수행하지 않았다. 종료 후 여덟 GPU의 `train_lm.py` 프로세스를 확인했다.

[핵심 결과와 종료 GPU 상태](../mllm_epd_h200_prefill_heavy_batch_matched/evidence/h200_results.json)

### 요청 지연과 작업량

| Method | QPS | 출력 계약 | Run/Cohort | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | 출력 토큰 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Natural EOS | `a_common/native_common_free_2` | 2,592.29 | 2,175.35 | 5,230.94 | 4.80 | 11.95 | 5.26 | 5468 | MEASURED_ONLINE |
| native-strong-free | 2.0 | Natural EOS | `a_strong_v2/native_strong_free_2` | 2,268.76 | 2,174.25 | 4,587.53 | 4.79 | 11.79 | 5.11 | 5523 | MEASURED_ONLINE |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_batch_clean_after` | 2,280.33 | 2,227.89 | 4,390.59 | 4.79 | 12.37 | 5.13 | 5269 | MEASURED_ONLINE |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_batch_clean_before` | 2,266.99 | 2,171.21 | 4,552.83 | 4.80 | 12.00 | 5.28 | 5269 | MEASURED_ONLINE |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_observed` | 2,371.15 | 2,118.23 | 4,683.82 | 4.81 | 11.32 | 5.12 | 5269 | DIAGNOSTIC |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_after` | 2,475.71 | 2,084.52 | 5,052.37 | 4.66 | 6.33 | 4.90 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_before` | 2,513.48 | 2,179.80 | 4,871.38 | 4.64 | 6.09 | 4.81 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_observed` | 2,658.85 | 2,144.68 | 5,377.93 | 4.64 | 6.18 | 4.79 | 5269 | DIAGNOSTIC |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common/native_common_b1_2_batch_clean_before` | 2,479.37 | 2,105.74 | 5,116.49 | 4.63 | 6.07 | 4.78 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_after` | 2,715.92 | 2,226.30 | 5,350.26 | 4.65 | 5.78 | 4.81 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_before` | 2,563.81 | 2,111.62 | 5,147.30 | 4.68 | 6.15 | 4.83 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_observed` | 2,548.57 | 2,147.75 | 5,163.10 | 4.66 | 6.56 | 4.83 | 5269 | DIAGNOSTIC |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_after` | 3,022.30 | 2,236.54 | 6,174.54 | 4.63 | 6.01 | 4.80 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_before` | 2,695.20 | 2,158.22 | 5,498.68 | 4.67 | 5.95 | 4.81 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_observed` | 2,739.88 | 2,053.03 | 6,125.62 | 4.64 | 5.83 | 4.82 | 5269 | DIAGNOSTIC |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_batch_clean_after` | 2,449.55 | 2,016.10 | 4,878.60 | 4.60 | 6.26 | 4.77 | 5269 | MEASURED_ONLINE |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_batch_clean_before` | 2,594.95 | 2,148.00 | 5,178.64 | 4.62 | 6.30 | 4.83 | 5269 | MEASURED_ONLINE |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_observed` | 2,507.90 | 2,121.68 | 5,114.21 | 4.60 | 5.80 | 4.74 | 5269 | DIAGNOSTIC |

### 처리량과 SLO

| Method | QPS | 출력 계약 | 입력+출력 tok/s | 출력 tok/s | 완료 req/s | SLO-goodput tok/s | TTFT Pass | Decode Pass | Joint SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Natural EOS | 20,584.58 | 101.55 | 1.86 | 75.71 | 93.00% | 95.00% | 88.00% |
| native-strong-free | 2.0 | Natural EOS | 21,161.27 | 105.44 | 1.91 | 96.25 | 99.00% | 96.00% | 95.00% |
| native-common-free | 2.0 | Per-request equal-output replay | 21,170.22 | 100.65 | 1.91 | 99.32 | 99.00% | 99.00% | 98.00% |
| native-common-free | 2.0 | Per-request equal-output replay | 21,158.74 | 100.60 | 1.91 | 93.27 | 98.00% | 98.00% | 96.00% |
| native-common-free | 2.0 | Per-request equal-output replay | 21,054.24 | 100.10 | 1.90 | 92.16 | 98.00% | 98.00% | 96.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,837.47 | 99.07 | 1.88 | 92.58 | 94.00% | 98.00% | 92.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,921.51 | 99.47 | 1.89 | 96.09 | 96.00% | 99.00% | 95.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,678.48 | 98.31 | 1.87 | 87.40 | 92.00% | 100.00% | 92.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,899.94 | 99.37 | 1.89 | 96.33 | 94.00% | 99.00% | 93.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,688.76 | 98.36 | 1.87 | 80.57 | 87.00% | 97.00% | 85.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,811.56 | 98.95 | 1.88 | 89.69 | 92.00% | 99.00% | 91.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,757.05 | 98.69 | 1.87 | 88.63 | 93.00% | 100.00% | 93.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,326.69 | 96.64 | 1.83 | 70.05 | 72.00% | 98.00% | 70.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,589.75 | 97.89 | 1.86 | 85.48 | 87.00% | 99.00% | 86.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,351.52 | 96.76 | 1.84 | 80.21 | 86.00% | 100.00% | 86.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,874.24 | 99.25 | 1.88 | 93.97 | 96.00% | 99.00% | 95.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,799.24 | 98.89 | 1.88 | 87.67 | 90.00% | 100.00% | 90.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,912.17 | 99.43 | 1.89 | 97.39 | 94.00% | 100.00% | 94.00% |

## 재현과 남은 Gate

원본 Setup과 이번 H200 Run은 같은 서버에 보존한다. 핵심 자료는 `h200_runs/<run>/state`, `configs`, `raw`, `analysis`에 있고 각 Report에 Evidence Level을 표시했다. 집계는 `python -m poc_flashvep.mllm_epd_h200_prefill_heavy_batch_matched.analysis --runs <h200_runs> --output <new-output>`으로 재생성한다.

재생성된 Figure는 `actual_decode_batch_histogram`, `critical_path_natural`, `critical_path_replay`, `d_idle_cdf_a_common_native_common_free_2`, `d_idle_cdf_a_strong_v2_native_strong_free_2`, `d_idle_cdf_b_common_free_native_common_free_2_batch_clean_after`, `d_idle_cdf_b_common_free_native_common_free_2_batch_clean_before`, `d_idle_cdf_b_common_free_native_common_free_2_observed`, `d_idle_cdf_b_idle10_idle10_b1_2_batch_clean_after`, `d_idle_cdf_b_idle10_idle10_b1_2_batch_clean_before`, `d_idle_cdf_b_idle10_idle10_b1_2_observed`, `d_idle_cdf_b_native_common_native_common_b1_2_batch_clean_before`, `d_idle_cdf_b_native_common_v2_native_common_b1_2_batch_clean_after`, `d_idle_cdf_b_native_common_v2_native_common_b1_2_batch_clean_before`, `d_idle_cdf_b_native_common_v2_native_common_b1_2_observed`, `d_idle_cdf_b_static10_static10_b1_2_batch_clean_after`, `d_idle_cdf_b_static10_static10_b1_2_batch_clean_before`, `d_idle_cdf_b_static10_static10_b1_2_observed`, `d_idle_cdf_b_strong_b1_native_strong_b1_2_batch_clean_after`, `d_idle_cdf_b_strong_b1_native_strong_b1_2_batch_clean_before`, `d_idle_cdf_b_strong_b1_native_strong_b1_2_observed`, `d_idle_cdf_c_native_b1_screen_native_common_b1_0p5`, `d_idle_cdf_c_native_b1_screen_native_common_b1_1`, `d_idle_cdf_c_native_b1_screen_native_common_b1_2`, `d_idle_cdf_c_native_b1_screen_native_common_b1_3`, `d_idle_cdf_c_native_b1_screen_native_common_b1_4`, `d_idle_cdf_c_native_b1_screen_native_common_b1_5`, `d_idle_cdf_c_native_b1_screen_native_common_b1_6`, `d_idle_cdf_c_native_b1_screen_native_common_b1_8`, `d_idle_cdf_c_native_free_screen_native_common_free_1`, `d_idle_cdf_c_native_free_screen_native_common_free_3`, `d_idle_cdf_c_native_free_screen_native_common_free_5`, `d_idle_cdf_c_native_free_screen_native_common_free_8`, `d_idle_cdf_d_profile_b1_q3_native_common_b1_3_clean_after`, `d_idle_cdf_d_profile_b1_q3_native_common_b1_3_clean_before`, `d_idle_cdf_d_profile_b1_q3_native_common_b1_3_profile`, `d_idle_cdf_d_profile_b1_q4_sampled_native_common_b1_4_clean_after`, `d_idle_cdf_d_profile_b1_q4_sampled_native_common_b1_4_clean_before`, `d_idle_cdf_d_profile_b1_q4_sampled_native_common_b1_4_profile`, `d_idle_cdf_d_profile_b1_q5_deferred_native_common_b1_5_clean_after`, `d_idle_cdf_d_profile_b1_q5_deferred_native_common_b1_5_clean_before`, `d_idle_cdf_d_profile_b1_q5_deferred_native_common_b1_5_profile`, `d_idle_cdf_d_profile_b1_q5_sampled_native_common_b1_5_clean_after`, `d_idle_cdf_d_profile_b1_q5_sampled_native_common_b1_5_clean_before`, `d_idle_cdf_d_profile_b1_q5_sampled_native_common_b1_5_profile`, `d_idle_cdf_d_profile_b1_q6_sampled_native_common_b1_6_clean_after`, `d_idle_cdf_d_profile_b1_q6_sampled_native_common_b1_6_clean_before`, `d_idle_cdf_d_profile_b1_q6_sampled_native_common_b1_6_profile`, `d_idle_cdf_d_profile_free_q5_deferred_native_common_free_5_clean_after`, `d_idle_cdf_d_profile_free_q5_deferred_native_common_free_5_clean_before`, `d_idle_cdf_d_profile_free_q5_deferred_native_common_free_5_profile`, `d_idle_cdf_d_profile_free_q5_sampled_native_common_free_5_clean_after`, `d_idle_cdf_d_profile_free_q5_sampled_native_common_free_5_clean_before`, `d_idle_cdf_d_profile_free_q5_sampled_native_common_free_5_profile`, `host_timeline_a_common_native_common_free_2`, `host_timeline_a_strong_v2_native_strong_free_2`, `host_timeline_b_common_free_native_common_free_2_batch_clean_after`, `host_timeline_b_common_free_native_common_free_2_batch_clean_before`, `host_timeline_b_common_free_native_common_free_2_observed`, `host_timeline_b_idle10_idle10_b1_2_batch_clean_after`, `host_timeline_b_idle10_idle10_b1_2_batch_clean_before`, `host_timeline_b_idle10_idle10_b1_2_observed`, `host_timeline_b_native_common_native_common_b1_2_batch_clean_before`, `host_timeline_b_native_common_v2_native_common_b1_2_batch_clean_after`, `host_timeline_b_native_common_v2_native_common_b1_2_batch_clean_before`, `host_timeline_b_native_common_v2_native_common_b1_2_observed`, `host_timeline_b_static10_static10_b1_2_batch_clean_after`, `host_timeline_b_static10_static10_b1_2_batch_clean_before`, `host_timeline_b_static10_static10_b1_2_observed`, `host_timeline_b_strong_b1_native_strong_b1_2_batch_clean_after`, `host_timeline_b_strong_b1_native_strong_b1_2_batch_clean_before`, `host_timeline_b_strong_b1_native_strong_b1_2_observed`, `host_timeline_c_native_b1_screen_native_common_b1_0p5`, `host_timeline_c_native_b1_screen_native_common_b1_1`, `host_timeline_c_native_b1_screen_native_common_b1_2`, `host_timeline_c_native_b1_screen_native_common_b1_3`, `host_timeline_c_native_b1_screen_native_common_b1_4`, `host_timeline_c_native_b1_screen_native_common_b1_5`, `host_timeline_c_native_b1_screen_native_common_b1_6`, `host_timeline_c_native_b1_screen_native_common_b1_8`, `host_timeline_c_native_free_screen_native_common_free_1`, `host_timeline_c_native_free_screen_native_common_free_3`, `host_timeline_c_native_free_screen_native_common_free_5`, `host_timeline_c_native_free_screen_native_common_free_8`, `host_timeline_d_profile_b1_q3_native_common_b1_3_clean_after`, `host_timeline_d_profile_b1_q3_native_common_b1_3_clean_before`, `host_timeline_d_profile_b1_q3_native_common_b1_3_profile`, `host_timeline_d_profile_b1_q4_sampled_native_common_b1_4_clean_after`, `host_timeline_d_profile_b1_q4_sampled_native_common_b1_4_clean_before`, `host_timeline_d_profile_b1_q4_sampled_native_common_b1_4_profile`, `host_timeline_d_profile_b1_q5_deferred_native_common_b1_5_clean_after`, `host_timeline_d_profile_b1_q5_deferred_native_common_b1_5_clean_before`, `host_timeline_d_profile_b1_q5_deferred_native_common_b1_5_profile`, `host_timeline_d_profile_b1_q5_sampled_native_common_b1_5_clean_after`, `host_timeline_d_profile_b1_q5_sampled_native_common_b1_5_clean_before`, `host_timeline_d_profile_b1_q5_sampled_native_common_b1_5_profile`, `host_timeline_d_profile_b1_q6_sampled_native_common_b1_6_clean_after`, `host_timeline_d_profile_b1_q6_sampled_native_common_b1_6_clean_before`, `host_timeline_d_profile_b1_q6_sampled_native_common_b1_6_profile`, `host_timeline_d_profile_free_q5_deferred_native_common_free_5_clean_after`, `host_timeline_d_profile_free_q5_deferred_native_common_free_5_clean_before`, `host_timeline_d_profile_free_q5_deferred_native_common_free_5_profile`, `host_timeline_d_profile_free_q5_sampled_native_common_free_5_clean_after`, `host_timeline_d_profile_free_q5_sampled_native_common_free_5_clean_before`, `host_timeline_d_profile_free_q5_sampled_native_common_free_5_profile`, `isolated_capacity_vs_online`, `kernel_busy_by_qps`, `native_qps_latency_natural`, `native_qps_latency_replay`, `offload_latency_2_replay`, `p_ready_d_state_natural`, `p_ready_d_state_replay`, `slo_breakdown_natural`, `slo_breakdown_replay`, `ttft_cdf_0.5_natural`, `ttft_cdf_1_natural`, `ttft_cdf_2_natural`, `ttft_cdf_2_replay`, `ttft_cdf_3_natural`, `ttft_cdf_4_natural`, `ttft_cdf_5_natural`, `ttft_cdf_6_natural`, `ttft_cdf_8_natural`, `ttft_itl_tradeoff_natural`, `ttft_itl_tradeoff_replay`이다.

- [사양서](mllm_epd_h200_prefill_heavy_batch_matched_poc_spec.md)
- [환경 및 초기 정합성](mllm_epd_h200_environment_and_correctness.md)
- [Decode Batch1 통제](mllm_epd_h200_decode_batch_matched_control.md)
- [독립 Stage Capacity](mllm_epd_h200_independent_stage_capacity.md)
- [Prefill-heavy 탐색](mllm_epd_h200_prefill_heavy_discovery.md)
- [Decode Idle과 Prefix-ready](mllm_epd_h200_decode_idle_prefix_ready_analysis.md)
- [Static Token Split](mllm_epd_h200_static_token_split_online_serving.md)
- [Idle-aware Token Split](mllm_epd_h200_idle_aware_token_split_online_serving.md)
- [TTFT 및 ITL 임계경로](mllm_epd_h200_token_split_critical_path.md)
