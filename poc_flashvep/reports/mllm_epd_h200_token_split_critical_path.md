# H200 Token Split의 TTFT 및 ITL 임계경로

요청별 Scheduled Arrival부터 First Client Token까지 겹치지 않는 Host 경계를 합산한다. 평균의 가법 분해와 TTFT p95의 분포 변화는 분리한다. 아래 모든 경계의 단위는 ms다. Host HTTP 구간에는 스케줄링, 전송, GPU 계산과 응답이 포함된다.

| Run/Cohort | Method | Arrival to proxy | Proxy to E submit | E HTTP | E complete to P submit | P HTTP | P ready to D submit | D submit to first client token | 합계 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a_common/native_common_free_2` | native-common-free | 3.34 | 0.04 | 1,182.12 | 0.18 | 1,341.40 | 0.20 | 65.01 | 2,592.29 |
| `a_strong_v2/native_strong_free_2` | native-strong-free | 2.69 | 0.03 | 906.37 | 0.17 | 1,293.04 | 0.19 | 66.26 | 2,268.76 |
| `b_common_free/native_common_free_2_batch_clean_after` | native-common-free | 2.88 | 0.04 | 707.31 | 0.19 | 1,509.28 | 0.19 | 60.46 | 2,280.33 |
| `b_common_free/native_common_free_2_batch_clean_before` | native-common-free | 2.67 | 0.03 | 910.02 | 0.17 | 1,290.42 | 0.17 | 63.50 | 2,266.99 |
| `b_common_free/native_common_free_2_observed` | native-common-free | 2.71 | 0.03 | 776.58 | 0.17 | 1,531.59 | 0.18 | 59.88 | 2,371.15 |
| `b_idle10/idle10_b1_2_batch_clean_after` | idle10-B1 | 3.03 | 0.04 | 757.88 | 0.19 | 1,313.96 | 320.06 | 80.54 | 2,475.71 |
| `b_idle10/idle10_b1_2_batch_clean_before` | idle10-B1 | 2.70 | 0.04 | 766.55 | 0.17 | 1,264.07 | 399.89 | 80.08 | 2,513.48 |
| `b_idle10/idle10_b1_2_observed` | idle10-B1 | 2.55 | 0.04 | 794.99 | 0.18 | 1,454.90 | 322.54 | 83.65 | 2,658.85 |
| `b_native_common/native_common_b1_2_batch_clean_before` | native-common-B1 | 3.14 | 0.04 | 987.89 | 0.17 | 1,220.32 | 211.10 | 56.72 | 2,479.37 |
| `b_native_common_v2/native_common_b1_2_batch_clean_after` | native-common-B1 | 2.72 | 0.08 | 1,043.98 | 0.16 | 1,353.04 | 260.20 | 55.73 | 2,715.92 |
| `b_native_common_v2/native_common_b1_2_batch_clean_before` | native-common-B1 | 2.26 | 0.04 | 822.34 | 0.18 | 1,417.91 | 269.30 | 51.78 | 2,563.81 |
| `b_native_common_v2/native_common_b1_2_observed` | native-common-B1 | 2.33 | 0.04 | 767.93 | 0.18 | 1,433.12 | 287.84 | 57.13 | 2,548.57 |
| `b_static10/static10_b1_2_batch_clean_after` | static10-B1 | 2.88 | 0.04 | 1,018.34 | 0.20 | 1,202.12 | 671.30 | 127.42 | 3,022.30 |
| `b_static10/static10_b1_2_batch_clean_before` | static10-B1 | 2.79 | 0.04 | 1,071.04 | 0.19 | 945.93 | 552.41 | 122.81 | 2,695.20 |
| `b_static10/static10_b1_2_observed` | static10-B1 | 2.69 | 0.04 | 1,081.34 | 0.19 | 1,028.89 | 511.43 | 115.30 | 2,739.88 |
| `b_strong_b1/native_strong_b1_2_batch_clean_after` | native-strong-B1 | 3.38 | 0.04 | 731.92 | 0.18 | 1,408.92 | 246.71 | 58.39 | 2,449.55 |
| `b_strong_b1/native_strong_b1_2_batch_clean_before` | native-strong-B1 | 2.44 | 0.04 | 900.28 | 0.16 | 1,360.69 | 273.86 | 57.46 | 2,594.95 |
| `b_strong_b1/native_strong_b1_2_observed` | native-strong-B1 | 2.67 | 0.04 | 851.27 | 0.19 | 1,351.29 | 243.28 | 59.16 | 2,507.90 |

QPS2 Phase B의 Static10-B1−NativeCommon-B1에서 P HTTP -18.195ms, P-ready→D-submit +460.204ms, 전체 평균 TTFT +542.928ms였다. 같은 출력 계약의 Host 경계 진단이며 순수 GPU 연산 비용은 별개다.

D Suffix의 GPU-exclusive 시간, KV Import 내부와 D First Worker Output은 실제 해당 Trace의 구분이 가능한 범위까지만 기록한다. D Submit→First Client의 Host 경계에서 Suffix 전체 비용을 역산하지 않는다.

NativeCommon Unrestricted에서 NativeCommon-B1으로 바꾸자 ITL p99가 11.321→6.562ms였다. Static10-B1의 5.827ms와 비교한 절대 격차는 5.494→0.735ms였다. 부호 있는 Native−Static 차이는 +5.494→+0.735ms였다. 이는 Counter를 켠 독립 N1의 방향성 근거다. 인과 기여율이나 적격 C3 우위를 확정하지 않는다.

[H100 Critical Path](mllm_epd_protected_token_critical_path_v1.md)의 Static10 Replay 수치는 P HTTP −305.329ms, D Coordinator +971.956ms, 평균 TTFT +933.063ms였다. Hardware/Runtime이 바뀐 H200의 차이에 이 값을 직접 차감하지 않는다.
