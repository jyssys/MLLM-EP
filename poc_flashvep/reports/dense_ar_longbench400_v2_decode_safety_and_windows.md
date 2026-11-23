# Decode Tail 보호와 Window 판정

상태: FINAL.
생성 시각: 2026-11-23T02:04:00.568194+00:00.
경과 연구 시간: 18.953시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 동결 계약

TTFT 제한은 5000ms다. 모든 Client-visible ITL 제한은 34ms다. Request TPOT 제한은 33ms/token이다.
상대 Gate는 ITL p99와 TPOT p95의 Paired CI 상한이 모두 1.05 이하여야 한다.
NativeStrong과 같은 D 정책을 사용하는 Common-policy Native를 모두 대조군으로 사용한다.
기존 절대 Decode Attainment 95% 조건도 별도로 검사한다.
TTFT 개선으로 Decode 실패를 상쇄하지 않는다. N1 또는 N2는 CI Gate를 통과할 수 없다.

## 두 Native 대비 Tail Gate

| Trace | QPS | Method | Starts | ITL/Strong [CI] | TPOT/Strong [CI] | ITL/Common [CI] | TPOT/Common [CI] | 상대 PASS | 절대 PASS | 통합 PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c3_main_serving_v2 | 1.4 | Deflection | 1 | 0.5285 [CI 없음] | 0.8881 [CI 없음] | 1.0014 [CI 없음] | 1.0001 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 1.4 | Hybrid | 1 | 0.5323 [CI 없음] | 0.8995 [CI 없음] | 0.9653 [CI 없음] | 0.9990 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 1.4 | Layer | 1 | 0.5308 [CI 없음] | 0.8881 [CI 없음] | 0.9627 [CI 없음] | 0.9863 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 1.4 | StockToken | 1 | 0.5546 [CI 없음] | 0.9094 [CI 없음] | 0.6903 [CI 없음] | 0.9954 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 2.0 | Deflection | 3 | 0.4495 [0.4351, 0.4652] | 0.8827 [0.8769, 0.8885] | 0.9939 [0.9918, 0.9955] | 0.9991 [0.9967, 1.0018] | True | True | True |
| c3_main_serving_v2 | 2.0 | Hybrid | 3 | 0.4578 [0.4428, 0.4739] | 0.8977 [0.8946, 0.9012] | 0.9558 [0.9543, 0.9575] | 0.9824 [0.9774, 0.9881] | True | True | True |
| c3_main_serving_v2 | 2.0 | Layer | 3 | 0.4560 [0.4415, 0.4720] | 0.8870 [0.8823, 0.8919] | 0.9521 [0.9499, 0.9543] | 0.9707 [0.9657, 0.9751] | True | True | True |
| c3_main_serving_v2 | 2.0 | StockToken | 3 | 0.5062 [0.4911, 0.5207] | 0.9410 [0.9403, 0.9416] | 0.6349 [0.5364, 0.7657] | 0.7713 [0.6330, 0.9421] | True | True | True |
| c3_main_serving_v2 | 2.6 | StockToken | 1 | 0.5473 [CI 없음] | 0.9621 [CI 없음] | 0.5353 [CI 없음] | 0.6545 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 3.2 | Deflection | 1 | 0.3791 [CI 없음] | 0.8627 [CI 없음] | 0.9980 [CI 없음] | 1.0000 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 3.2 | Layer | 1 | 0.3805 [CI 없음] | 0.8651 [CI 없음] | 0.9422 [CI 없음] | 0.9729 [CI 없음] | False | True | False |
| c3_main_serving_v2 | 3.2 | StockToken | 1 | 0.4308 [CI 없음] | 0.9393 [CI 없음] | 0.5159 [CI 없음] | 0.6525 [CI 없음] | False | True | False |
| c3_heldout_serving_v2 | 2.0 | Deflection | 3 | 0.4587 [0.4430, 0.4781] | 0.9199 [0.9122, 0.9293] | 0.9990 [0.9963, 1.0023] | 0.9981 [0.9951, 1.0014] | True | True | True |
| c3_heldout_serving_v2 | 2.0 | Hybrid | 3 | 0.4641 [0.4489, 0.4843] | 0.9268 [0.9206, 0.9346] | 0.9630 [0.9583, 0.9677] | 0.9820 [0.9783, 0.9859] | True | True | True |
| c3_heldout_serving_v2 | 2.0 | Layer | 3 | 0.4622 [0.4467, 0.4823] | 0.9248 [0.9189, 0.9319] | 0.9592 [0.9545, 0.9630] | 0.9799 [0.9754, 0.9840] | True | True | True |
| c3_heldout_serving_v2 | 2.0 | StockToken | 3 | 0.5031 [0.4810, 0.5268] | 0.9653 [0.9491, 0.9816] | 0.6633 [0.4918, 0.9339] | 0.7129 [0.5456, 0.9393] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Hybrid | 3 | 0.4806 [0.4005, 0.6187] | 0.9093 [0.8926, 0.9262] | 1.0018 [0.9958, 1.0073] | 0.9983 [0.9869, 1.0117] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Layer | 3 | 0.4819 [0.4017, 0.6193] | 0.9077 [0.8919, 0.9208] | 1.0045 [1.0017, 1.0072] | 0.9965 [0.9909, 1.0039] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | StockToken | 3 | 0.4820 [0.3990, 0.6246] | 0.9127 [0.8929, 0.9304] | 1.0048 [0.9974, 1.0147] | 1.0021 [0.9921, 1.0144] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Hybrid | 3 | 0.4210 [0.3697, 0.4806] | 0.8141 [0.7894, 0.8380] | 0.9879 [0.9725, 1.0044] | 0.9829 [0.9817, 0.9838] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Layer | 3 | 0.4210 [0.3696, 0.4821] | 0.8207 [0.7927, 0.8503] | 0.9879 [0.9778, 0.9983] | 0.9908 [0.9842, 0.9988] | True | False | False |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | StockToken | 3 | 0.4277 [0.3787, 0.4845] | 0.8269 [0.8034, 0.8514] | 1.0036 [0.9835, 1.0209] | 0.9982 [0.9940, 1.0014] | True | False | False |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Hybrid | 3 | 1.0323 [1.0027, 1.0639] | 1.0444 [1.0123, 1.0871] | 1.0333 [1.0161, 1.0539] | 1.0444 [1.0118, 1.0867] | False | False | False |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Layer | 3 | 0.9815 [0.9677, 0.9950] | 1.0057 [1.0001, 1.0127] | 0.9824 [0.9586, 1.0084] | 1.0057 [0.9989, 1.0143] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | StockToken | 3 | 1.0066 [0.9952, 1.0162] | 1.0124 [1.0077, 1.0162] | 1.0075 [1.0066, 1.0085] | 1.0124 [1.0092, 1.0150] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Hybrid | 3 | 0.3101 [0.2803, 0.3578] | 0.7836 [0.7683, 0.8024] | 0.8417 [0.8333, 0.8494] | 0.8471 [0.8445, 0.8496] | True | True | True |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | Layer | 3 | 0.3105 [0.2829, 0.3558] | 0.7850 [0.7700, 0.8038] | 0.8428 [0.8299, 0.8536] | 0.8486 [0.8461, 0.8515] | True | False | False |
| c3_continuous_iso8_output_bridge_v2 | 4.0 | StockToken | 3 | 0.3454 [0.3139, 0.3965] | 0.8860 [0.8696, 0.9061] | 0.9377 [0.9247, 0.9477] | 0.9578 [0.9538, 0.9615] | True | False | False |


## C3 Client Tail의 Host 경계

각 실행의 p99 이상 ITL 구간을 조사한다. 표는 같은 요청의 Worker Emission과 Client Delivery를 사용한다.
여러 Token의 동시 전달은 원래 0ms Client Interval을 유지한다.
Prefill, NIXL API 및 새 등록의 중첩 수는 서로 더할 수 있는 비용이 아니다.
Stream Wait의 GPU-exclusive 비중은 `NOT_MEASURED`다. 중첩만으로 개별 원인을 확정하지 않는다.

| QPS | Start | Method | 전체 ITL 구간 | p99 이상 구간 | Prefill Host 중첩 | NIXL Host 중첩 | 새 D 등록 중첩 | Tail Mean Worker Gap ms | Tail Mean Delivery 변화 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 0 | NativeCommon1 | 10478 | 105 | 0 | 0 | 0 | 8.200 | 1.332 |
| 2.0 | 0 | Hybrid | 10209 | 103 | 0 | 0 | 0 | 8.936 | 3.636 |
| 2.0 | 0 | StockToken | 11716 | 118 | 0 | 0 | 0 | 9.284 | 0.709 |
| 2.0 | 0 | Layer | 11018 | 111 | 0 | 0 | 0 | 8.404 | 0.726 |
| 2.0 | 0 | NativeStrong | 10906 | 110 | 0 | 82 | 82 | 22.709 | 0.282 |
| 2.0 | 0 | NativeCommon2 | 11806 | 119 | 5 | 0 | 0 | 12.218 | 1.154 |
| 2.0 | 0 | NativeCommon0 | 10677 | 107 | 0 | 0 | 0 | 12.955 | 2.611 |
| 2.0 | 0 | Deflection | 10445 | 105 | 0 | 0 | 0 | 8.292 | 3.996 |
| 2.0 | 1 | NativeCommon2 | 10607 | 107 | 2 | 0 | 0 | 10.808 | 5.633 |
| 2.0 | 1 | NativeCommon0 | 10465 | 105 | 0 | 0 | 0 | 12.017 | 0.455 |
| 2.0 | 1 | NativeCommon1 | 10482 | 105 | 0 | 0 | 0 | 8.239 | 1.454 |
| 2.0 | 1 | Hybrid | 10376 | 104 | 0 | 0 | 0 | 8.860 | 7.948 |
| 2.0 | 1 | Deflection | 10918 | 110 | 0 | 0 | 0 | 8.287 | 2.089 |
| 2.0 | 1 | StockToken | 10892 | 109 | 0 | 0 | 0 | 9.929 | 0.439 |
| 2.0 | 1 | NativeStrong | 11383 | 114 | 0 | 87 | 87 | 25.211 | 1.606 |
| 2.0 | 1 | Layer | 10498 | 105 | 0 | 0 | 0 | 8.561 | 0.407 |
| 2.0 | 2 | NativeStrong | 10598 | 106 | 2 | 76 | 76 | 25.287 | 0.584 |
| 2.0 | 2 | NativeCommon1 | 10478 | 105 | 0 | 0 | 0 | 8.307 | 4.044 |
| 2.0 | 2 | NativeCommon2 | 10593 | 106 | 3 | 0 | 0 | 11.460 | 0.641 |
| 2.0 | 2 | NativeCommon0 | 11789 | 118 | 0 | 0 | 0 | 26.224 | 0.868 |
| 2.0 | 2 | Deflection | 10157 | 102 | 0 | 0 | 0 | 8.220 | 0.796 |
| 2.0 | 2 | Layer | 10494 | 105 | 0 | 0 | 0 | 8.393 | 0.869 |
| 2.0 | 2 | StockToken | 9898 | 99 | 0 | 0 | 0 | 8.748 | 0.733 |
| 2.0 | 2 | Hybrid | 10860 | 109 | 0 | 0 | 0 | 8.969 | 0.683 |

[c3_heldout_serving_v2_decode_tail_host_summary_v2.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_serving_v2_decode_tail_host_summary_v2.csv)

## C0의 모든-ITL 실패 위치

C0 Held-out의 고유 실행 전체에서 Client ITL>34ms 구간을 조사했다.
Native 실행은 여러 Cut에서 재사용되므로 한 번만 계산한다.
Client ITL = Worker Emission Gap + Delivery Delay 변화다.
Worker Gap≤34ms인 Client 위반은 Delivery Delay 증가에서 나타난다.
이 등식은 GC, Socket, CPU Scheduling 또는 File I/O 중 하나를 원인으로 확정하지 않는다.
모든 Client 위반은 원래 SLO 판정에 그대로 남긴다. Timestamp를 교정하지 않는다.

| Arm | Output | 고유 실행 | Requests | ITL 구간 | 모든-ITL 실패 요청 | Worker Gap≤34ms인 위반 구간 | Worker Gap>34ms인 위반 구간 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Layer | 8 | 48 | 768 | 5376 | 0 | 0 | 0 |
| Layer | 16 | 48 | 768 | 11520 | 0 | 0 | 0 |
| Layer | 64 | 48 | 768 | 48384 | 4 | 4 | 0 |
| Layer | 128 | 48 | 768 | 97536 | 14 | 14 | 0 |
| Layer | 192 | 48 | 768 | 146688 | 9 | 9 | 0 |
| Native | 8 | 12 | 192 | 1344 | 0 | 0 | 0 |
| Native | 16 | 12 | 192 | 2880 | 0 | 0 | 0 |
| Native | 64 | 12 | 192 | 12096 | 1 | 1 | 0 |
| Native | 128 | 12 | 192 | 24384 | 2 | 2 | 0 |
| Native | 192 | 12 | 192 | 36672 | 5 | 4 | 1 |
| Token | 8 | 48 | 768 | 5376 | 0 | 0 | 0 |
| Token | 16 | 48 | 768 | 11520 | 0 | 0 | 0 |
| Token | 64 | 48 | 768 | 48384 | 3 | 3 | 0 |
| Token | 128 | 48 | 768 | 97536 | 21 | 21 | 0 |
| Token | 192 | 48 | 768 | 146688 | 7 | 7 | 0 |

표의 실행 수는 여러 Input/Cut 조건의 기술 통계다. 이를 한 Cell의 독립 Start 수로 사용하지 않는다.

[c0_heldout_client_tail_violations_v2.csv](dense_ar_longbench400_v2_20261122/figures/c0_heldout_client_tail_violations_v2.csv)


## Kernel 및 Window

| Method | QPS | Role | 증거 | Kernel Busy % | ROI 초 | Observer Gate |
| --- | --- | --- | --- | --- | --- | --- |
| NativeStrong | 2.0 | prefill | DIAGNOSTIC | 79.093 | 20.448 | DIAGNOSTIC |
| NativeStrong | 2.0 | decode | DIAGNOSTIC | 56.798 | 20.009 | DIAGNOSTIC |
| NativeStrong | 4.8 | prefill | DIAGNOSTIC | 99.547 | 20.266 | DIAGNOSTIC |
| NativeStrong | 4.8 | decode | DIAGNOSTIC | 49.275 | 20.038 | DIAGNOSTIC |
| Layer | 4.8 | prefill | MEASURED_KERNEL_ACTIVITY | 99.444 | 20.357 | PASS |
| Layer | 4.8 | decode | MEASURED_KERNEL_ACTIVITY | 71.198 | 20.017 | PASS |
| StockToken | 4.8 | prefill | DIAGNOSTIC | 99.477 | 20.406 | DIAGNOSTIC |
| StockToken | 4.8 | decode | DIAGNOSTIC | 91.776 | 20.194 | DIAGNOSTIC |
| Hybrid | 4.8 | prefill | MEASURED_KERNEL_ACTIVITY | 76.793 | 20.002 | PASS |
| Hybrid | 4.8 | decode | MEASURED_KERNEL_ACTIVITY | 93.160 | 20.018 | PASS |
| NativeCommon0 | 4.8 | prefill | DIAGNOSTIC | 99.524 | 20.428 | DIAGNOSTIC |
| NativeCommon0 | 4.8 | decode | DIAGNOSTIC | 55.817 | 20.014 | DIAGNOSTIC |
| NativeCommon1 | 4.8 | prefill | MEASURED_KERNEL_ACTIVITY | 99.555 | 20.267 | PASS |
| NativeCommon1 | 4.8 | decode | MEASURED_KERNEL_ACTIVITY | 67.346 | 20.013 | PASS |
| NativeCommon2 | 4.8 | prefill | DIAGNOSTIC | 99.518 | 20.424 | DIAGNOSTIC |
| NativeCommon2 | 4.8 | decode | DIAGNOSTIC | 58.889 | 20.015 | DIAGNOSTIC |
| NativeStrong | 1.4 | prefill | MEASURED_KERNEL_ACTIVITY | 48.588 | 20.537 | PASS |
| NativeStrong | 1.4 | decode | MEASURED_KERNEL_ACTIVITY | 38.100 | 20.030 | PASS |
| NativeStrong | 2.6 | prefill | DIAGNOSTIC | 98.723 | 20.338 | DIAGNOSTIC |
| NativeStrong | 2.6 | decode | DIAGNOSTIC | 49.328 | 20.018 | DIAGNOSTIC |
| NativeStrong | 3.2 | prefill | MEASURED_KERNEL_ACTIVITY | 99.546 | 20.257 | PASS |
| NativeStrong | 3.2 | decode | MEASURED_KERNEL_ACTIVITY | 48.424 | 20.015 | PASS |
| NativeStrong | 4.0 | prefill | MEASURED_KERNEL_ACTIVITY | 99.535 | 20.282 | PASS |
| NativeStrong | 4.0 | decode | MEASURED_KERNEL_ACTIVITY | 49.917 | 20.033 | PASS |
| NativeStrong | 6.0 | prefill | DIAGNOSTIC | 99.517 | 20.354 | DIAGNOSTIC |
| NativeStrong | 6.0 | decode | DIAGNOSTIC | 50.497 | 20.024 | DIAGNOSTIC |
| NativeStrong | 0.8 | prefill | MEASURED_KERNEL_ACTIVITY | 22.112 | 20.005 | PASS |
| NativeStrong | 0.8 | decode | MEASURED_KERNEL_ACTIVITY | 24.569 | 20.008 | PASS |

Kernel Busy는 Clean/Profile/Clean Gate가 통과한 Capture만 적격 수치로 사용한다.
Observer Gate는 전체 Trace와 ROI 노출 Request를 각각 검사한다. 허용 변화는 5%다.
Kernel Gap은 모든 Engine의 Physical Idle 또는 실행 가능한 Decode-safe Window와 같지 않다.
Worker Deadline Residual은 마지막 Token 시각+34ms의 관측 Proxy다.
이 Proxy는 후보 준비, KV Credit, Stream 가용성 및 미래 Arrival 안전성을 보장하지 않는다.
Online Dynamic Insertion은 검증된 Window와 Tail Predictor가 없으면 `NOT_RUN`이다.
GPU Operator-exclusive GEMM/Attention, SM 및 HBM 비중은 적격 계측이 없으면 `NOT_MEASURED`다.
NIXL Host API와 Kernel Gap의 시간 중첩은 인과 관계를 입증하지 않는다.

[c3_kernel_copy_cpc_results_v2.json — lossless archive manifest](dense_ar_longbench400_v2_20261122/publication/derived_tables/analysis_c3_kernel_copy_cpc_results_v2.json.manifest.json)

[c3_kernel_busy_and_queue_v2.csv](dense_ar_longbench400_v2_20261122/figures/c3_kernel_busy_and_queue_v2.csv)
