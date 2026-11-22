# LongBench400: Input × Output × Cut Winner Map

증거 수준은 `MEASURED_ONLINE`이다. 이 문서의 Primary Map은 C0 Batch1/1/1 Mechanism Control이다.
C3 Continuous Serving의 성능으로 해석하지 않는다. Natural EOS Main400과도 구분한다.

## 실행 계약

- Qwen3-8B Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- BI0, Original BF16, Exact Causal Attention, Full36 Decode다.
- B8192/S1024, KV42GiB/GPU, KV Block128, Source Credit4다.
- 실제 P Prefill/D Suffix/D Decode Batch는 1/1/1이다. D는 Whole-request Admission을 사용한다.
- 각 Start는 원문 Prompt 16개를 사용한다. Output은 정확히 8/16/64/128 Tokens로 통제한다.
- Primary Cut은 Last2/4/8/12다. Token Suffix는 인접 16-Token 길이로 Aggregate Rows를 맞춘다.
- Input별 공통 QPS는 0.910229/0.871211/0.822364/0.763308이다.
- Source Capsule SHA256은 `08918a2e25f957ede7387297119b491ee21fd6ea26cd380ceb6d7e328da0d772`다.

각 Cell은 독립 Start 3회다. 사전에 지정한 Core Cell 3개는 5회다.
실제 P/D Partition 기록 198개 Paired Cell/Start Block에서 양축의 Aggregate P/D Rows가 정확히 같았다.
GEMM Rows가 같아도 Attention Pair와 Payload Bytes는 다르다. Native의 한 Query 재계산은 별도 비용이다.
CI는 Start별 Paired Bootstrap을 완전 열거했다. n3은 27개, n5는 3125개 재표본을 사용한다.
CI는 Pointwise이며 다중 비교 보정을 하지 않았다. 64개 Cell 전체의 Familywise 오류를 제어하지 않는다.
Request나 Token을 독립 Server Start로 취급하지 않았다.

## 전체 판정

| 판정 | Cell 수 |
|---|---:|
| LAYER_CONFIRMED | 22 |
| TOKEN_CONFIRMED | 16 |
| TIE_OR_INCONCLUSIVE | 21 |
| INVALID | 5 |
| NOT_RUN | 0 |

`LAYER_CONFIRMED`와 `TOKEN_CONFIRMED`는 TTFT 방향과 Winner의 Decode Gate를 함께 확인한 판정이다.
Native 대비 ITL p99와 TPOT p95 CI 상한은 모두 1.05 이하여야 한다. Absolute Decode Gate도 별도로 검사한다.
어느 축이 다른 축을 이겼다는 사실은 Native보다 좋은 Serving 성과를 뜻하지 않는다.
`INVALID` Cell 5개도 아래 표에 남긴다. 실패한 Decode Gate를 제거하거나 Threshold를 바꾸지 않았다.

## 64개 Cell

각 비율은 Layer/Token이다. 대괄호는 95% CI다. Goodput은 Frozen Joint SLO-qualified Output Tokens/s다.

### Controlled Output8

| Input Bucket | Cut | Starts | Actual GPU Pair | TTFT p95 L/T [CI] | SLO-goodput L/T [CI] | 판정 |
|---|---:|---:|---|---|---|---|
| 3K–6K | 2 | 3 | 2/3 | 0.9437 [0.9241, 0.9653] | 1.0003 [1.0001, 1.0005] | LAYER_CONFIRMED |
| 3K–6K | 4 | 5 | 2/3 | 0.9239 [0.8346, 1.0337] | 1.0017 [1.0009, 1.0026] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 8 | 5 | 2/3 | 1.1386 [1.0828, 1.1828] | 0.9995 [0.9990, 1.0000] | TOKEN_CONFIRMED |
| 3K–6K | 12 | 3 | 2/3 | 1.0013 [0.9391, 1.0545] | 0.9989 [0.9981, 0.9998] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 2 | 3 | 6/7 | 1.0231 [1.0116, 1.0327] | 0.9987 [0.9982, 0.9995] | TOKEN_CONFIRMED |
| 6K–10K | 4 | 3 | 6/7 | 0.9596 [0.9295, 0.9847] | 0.9997 [0.9990, 1.0003] | LAYER_CONFIRMED |
| 6K–10K | 8 | 3 | 6/7 | 1.1222 [1.0148, 1.2583] | 0.9984 [0.9975, 0.9992] | TOKEN_CONFIRMED |
| 6K–10K | 12 | 3 | 6/7 | 0.9703 [0.9625, 0.9785] | 0.9996 [0.9988, 1.0003] | LAYER_CONFIRMED |
| 10K–20K | 2 | 3 | 4/5 | 0.9991 [0.9495, 1.0471] | 0.9989 [0.9979, 0.9995] | TIE_OR_INCONCLUSIVE |
| 10K–20K | 4 | 3 | 4/5 | 1.0458 [1.0235, 1.0718] | 0.9987 [0.9972, 0.9999] | TOKEN_CONFIRMED |
| 10K–20K | 8 | 3 | 4/5 | 1.0770 [1.0365, 1.1184] | 0.9980 [0.9970, 0.9991] | INVALID |
| 10K–20K | 12 | 3 | 4/5 | 1.0488 [1.0157, 1.0730] | 0.9992 [0.9986, 0.9999] | INVALID |
| 20K–36K | 2 | 3 | 0/1 | 1.0679 [1.0527, 1.0863] | 0.9945 [0.9926, 0.9969] | TOKEN_CONFIRMED |
| 20K–36K | 4 | 3 | 0/1 | 1.1036 [1.0774, 1.1191] | 0.9920 [0.9902, 0.9942] | TOKEN_CONFIRMED |
| 20K–36K | 8 | 3 | 0/1 | 1.1479 [1.1053, 1.1784] | 0.9956 [0.9924, 0.9980] | TOKEN_CONFIRMED |
| 20K–36K | 12 | 3 | 0/1 | 1.1526 [1.0837, 1.2099] | 0.9944 [0.9895, 0.9982] | TOKEN_CONFIRMED |

### Controlled Output16

| Input Bucket | Cut | Starts | Actual GPU Pair | TTFT p95 L/T [CI] | SLO-goodput L/T [CI] | 판정 |
|---|---:|---:|---|---|---|---|
| 3K–6K | 2 | 3 | 2/3 | 0.8966 [0.8652, 0.9185] | 1.0009 [1.0006, 1.0012] | LAYER_CONFIRMED |
| 3K–6K | 4 | 3 | 2/3 | 0.9482 [0.8910, 1.0230] | 1.0012 [1.0008, 1.0015] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 8 | 3 | 2/3 | 0.9300 [0.8766, 1.0136] | 1.0257 [0.9998, 1.0565] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 12 | 3 | 2/3 | 0.9992 [0.9146, 1.0678] | 0.9993 [0.9988, 0.9999] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 2 | 3 | 6/7 | 1.0675 [0.9600, 1.2069] | 1.0007 [0.9990, 1.0025] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 4 | 3 | 6/7 | 0.9436 [0.8993, 0.9964] | 0.9995 [0.9992, 0.9998] | LAYER_CONFIRMED |
| 6K–10K | 8 | 3 | 6/7 | 0.9755 [0.9364, 1.0048] | 0.9993 [0.9985, 1.0002] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 12 | 3 | 6/7 | 1.0403 [0.9843, 1.0936] | 0.9999 [0.9997, 1.0000] | TIE_OR_INCONCLUSIVE |
| 10K–20K | 2 | 3 | 4/5 | 1.0455 [1.0242, 1.0653] | 0.9999 [0.9991, 1.0008] | TOKEN_CONFIRMED |
| 10K–20K | 4 | 3 | 4/5 | 1.0600 [1.0541, 1.0648] | 0.9982 [0.9966, 0.9992] | TOKEN_CONFIRMED |
| 10K–20K | 8 | 3 | 4/5 | 1.0821 [1.0712, 1.0972] | 0.9977 [0.9962, 0.9992] | TOKEN_CONFIRMED |
| 10K–20K | 12 | 3 | 4/5 | 1.0298 [0.9864, 1.0825] | 0.9991 [0.9989, 0.9994] | TIE_OR_INCONCLUSIVE |
| 20K–36K | 2 | 3 | 0/1 | 1.0559 [1.0285, 1.0909] | 0.9956 [0.9933, 0.9981] | TOKEN_CONFIRMED |
| 20K–36K | 4 | 3 | 0/1 | 1.0794 [1.0521, 1.1106] | 0.9929 [0.9914, 0.9949] | INVALID |
| 20K–36K | 8 | 5 | 0/1 | 1.1405 [1.1074, 1.1774] | 0.9963 [0.9922, 0.9993] | TOKEN_CONFIRMED |
| 20K–36K | 12 | 3 | 0/1 | 1.1014 [1.0585, 1.1383] | 0.9973 [0.9968, 0.9977] | TOKEN_CONFIRMED |

### Controlled Output64

| Input Bucket | Cut | Starts | Actual GPU Pair | TTFT p95 L/T [CI] | SLO-goodput L/T [CI] | 판정 |
|---|---:|---:|---|---|---|---|
| 3K–6K | 2 | 3 | 2/3 | 0.9793 [0.9171, 1.0435] | 0.9773 [0.9484, 1.0040] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 4 | 3 | 2/3 | 0.9466 [0.8883, 1.0294] | 1.0016 [0.9999, 1.0040] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 8 | 3 | 2/3 | 0.9730 [0.9410, 1.0018] | 1.0006 [0.9996, 1.0024] | TIE_OR_INCONCLUSIVE |
| 3K–6K | 12 | 3 | 2/3 | 0.9904 [0.9599, 1.0133] | 1.0000 [0.9992, 1.0015] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 2 | 3 | 6/7 | 0.9669 [0.9555, 0.9806] | 1.0007 [0.9993, 1.0025] | LAYER_CONFIRMED |
| 6K–10K | 4 | 3 | 6/7 | 0.9911 [0.9812, 1.0021] | 0.9992 [0.9978, 1.0006] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 8 | 3 | 6/7 | 0.9797 [0.9641, 0.9942] | 0.9986 [0.9977, 0.9998] | LAYER_CONFIRMED |
| 6K–10K | 12 | 3 | 6/7 | 0.9866 [0.9595, 1.0147] | 0.9597 [0.8885, 1.0485] | TIE_OR_INCONCLUSIVE |
| 10K–20K | 2 | 3 | 4/5 | 0.9700 [0.9420, 0.9895] | 1.0444 [1.0117, 1.0699] | LAYER_CONFIRMED |
| 10K–20K | 4 | 3 | 4/5 | 1.0061 [0.9912, 1.0168] | 0.9989 [0.9985, 0.9993] | TIE_OR_INCONCLUSIVE |
| 10K–20K | 8 | 3 | 4/5 | 0.9780 [0.9629, 0.9913] | 1.0169 [0.9995, 1.0508] | LAYER_CONFIRMED |
| 10K–20K | 12 | 3 | 4/5 | 0.9660 [0.9568, 0.9725] | 1.0263 [1.0001, 1.0566] | LAYER_CONFIRMED |
| 20K–36K | 2 | 3 | 0/1 | 1.0582 [1.0360, 1.0767] | 0.9947 [0.9921, 0.9974] | TOKEN_CONFIRMED |
| 20K–36K | 4 | 3 | 0/1 | 1.0919 [1.0684, 1.1121] | 0.9983 [0.9962, 1.0002] | TOKEN_CONFIRMED |
| 20K–36K | 8 | 3 | 0/1 | 1.0058 [0.9631, 1.0414] | 0.9989 [0.9968, 1.0001] | TIE_OR_INCONCLUSIVE |
| 20K–36K | 12 | 3 | 0/1 | 0.9424 [0.9264, 0.9683] | 1.0011 [0.9983, 1.0035] | LAYER_CONFIRMED |

### Controlled Output128

| Input Bucket | Cut | Starts | Actual GPU Pair | TTFT p95 L/T [CI] | SLO-goodput L/T [CI] | 판정 |
|---|---:|---:|---|---|---|---|
| 3K–6K | 2 | 3 | 2/3 | 0.9489 [0.9210, 0.9711] | 0.9930 [0.9571, 1.0247] | INVALID |
| 3K–6K | 4 | 3 | 2/3 | 0.9644 [0.9477, 0.9815] | 0.9852 [0.9580, 1.0100] | INVALID |
| 3K–6K | 8 | 3 | 2/3 | 0.9766 [0.9688, 0.9832] | 1.0309 [1.0053, 1.0610] | LAYER_CONFIRMED |
| 3K–6K | 12 | 3 | 2/3 | 0.9845 [0.9731, 0.9940] | 1.0282 [1.0016, 1.0610] | LAYER_CONFIRMED |
| 6K–10K | 2 | 3 | 6/7 | 0.9801 [0.9659, 0.9905] | 1.0100 [1.0010, 1.0204] | LAYER_CONFIRMED |
| 6K–10K | 4 | 3 | 6/7 | 0.9800 [0.9763, 0.9845] | 1.0203 [1.0048, 1.0507] | LAYER_CONFIRMED |
| 6K–10K | 8 | 3 | 6/7 | 1.0019 [0.9916, 1.0146] | 0.9833 [0.9523, 1.0005] | TIE_OR_INCONCLUSIVE |
| 6K–10K | 12 | 3 | 6/7 | 0.9907 [0.9826, 1.0003] | 1.0160 [0.9976, 1.0502] | TIE_OR_INCONCLUSIVE |
| 10K–20K | 2 | 3 | 4/5 | 0.9748 [0.9566, 0.9935] | 1.0297 [1.0074, 1.0539] | LAYER_CONFIRMED |
| 10K–20K | 4 | 3 | 4/5 | 0.9848 [0.9711, 0.9994] | 1.0049 [1.0005, 1.0093] | LAYER_CONFIRMED |
| 10K–20K | 8 | 3 | 4/5 | 0.9685 [0.9621, 0.9785] | 1.0129 [1.0060, 1.0173] | LAYER_CONFIRMED |
| 10K–20K | 12 | 3 | 4/5 | 0.9882 [0.9788, 0.9975] | 1.0061 [1.0016, 1.0101] | LAYER_CONFIRMED |
| 20K–36K | 2 | 3 | 0/1 | 0.9896 [0.9766, 1.0000] | 1.0060 [1.0005, 1.0116] | TIE_OR_INCONCLUSIVE |
| 20K–36K | 4 | 3 | 0/1 | 0.9551 [0.9304, 0.9835] | 1.0663 [1.0052, 1.1463] | LAYER_CONFIRMED |
| 20K–36K | 8 | 3 | 0/1 | 0.9547 [0.9467, 0.9694] | 1.0168 [1.0056, 1.0263] | LAYER_CONFIRMED |
| 20K–36K | 12 | 3 | 0/1 | 0.9102 [0.8816, 0.9556] | 1.1477 [1.0313, 1.3010] | LAYER_CONFIRMED |

## 해석 범위

작은 Cut이면 항상 Layer가 이긴다는 가설은 성립하지 않았다.
Input, Controlled Output, Cut과 Queue Regime에 따라 Winner가 달랐다.
같은 Input에서 Output만 바뀌어도 D Residency가 달라진다. 다만 모든 구간의 인과성을 이 표 하나로 확정하지 않는다.

원래 Discovery의 상대 부하는 Serial P+D+Decode Reference로 정했다.
이 값은 실제 Pipeline Bottleneck Capacity가 아니다. 같은 절대 QPS라도 Output128과 Output8의 부하 상태는 다르다.
따라서 전체 64개 Cell을 Low-load라고 부르지 않는다.
별도 Proper-pipeline-load Supplement는 Bottleneck Rate 추정치 주변을 검사한다. 단일 Start는 Winner CI가 아니다.

## Native-relative Serving 비교

I0/Output8/Last2의 Layer/Token TTFT 비율은 0.9437 [0.9241,0.9653]이었다.
하지만 같은 Cell의 Layer/Native 비율은 1.0029 [0.9696,1.0324]였다.
이는 Axis Win이지만 Native 대비 TTFT Win은 아니다.

I3/Output16/Last8의 Layer/Token 비율은 1.1405 [1.1074,1.1774]였다.
이 Cell에서는 Token이 우세했다. 두 축 모두 Native 대비 TTFT와 SLO-goodput 이득을 보였다.
이 결과도 16개 요청의 Finite-cohort 측정이다. Sustainable Capacity로 표현하지 않는다.

## 예측 한계

기존 Serial-calibrated State-updating Replay는 명확한 Winner 38개 중 32개를 예측했다.
Accuracy는 32/38, 84.21%였다. 유효한 59개 Cell의 평균 TTFT Regret은 0.841%였다.
p95 Regret은 5.966%였고 최대는 11.530%였다. 모든 명확한 오분류는 실제 Layer Winner를 Token으로 예측한 경우였다.
이 Model은 전체 Controlled Arrival Trace와 Output 길이를 안다. 검증된 1–2개 Online Knob으로 해석하지 않는다.

## 재현 자료

- [전체 Cell과 Native-relative Metric](dense_ar_longbench400_2d_20261121/analysis/axis_matrix_heldout_confirmation_r2.json).
- [Original Discovery와 Pipeline Load 구분](dense_ar_longbench400_pipeline_load_reference_audit.md).
- [Held-out Winner Prediction과 Regret](dense_ar_longbench400_2d_20261121/analysis/c0_axis_winner_prediction_audit_v1.json).
- [TTFT Ratio Figure](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_ttft_ratio.png).
- [SLO-goodput Ratio Figure](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_goodput_ratio.png).

C3 Best-static Confirmation과 Genuine Hybrid의 비교는 별도 보고서에 기록한다.
미실행 C3나 Dynamic 결과로 이 C0 Map을 대체하지 않는다.

## 양축 모두 Decode Gate를 통과한 더 엄격한 부분집합

원래 Winner Label은 이긴 Arm의 Decode Gate를 요구한다.
상대 Arm의 실패도 원래 표와 NI Heatmap에 남긴다.
아래 추가 감사는 Layer와 Token이 모두 같은 Native Decode Gate를 통과할 것을 요구한다.
원래 Label과 SLO는 변경하지 않는다. 더 엄격한 해석 범위를 따로 표시한다.

| 두 Arm 모두 안전한 원래 판정 | Cell 수 |
|---|---:|
| LAYER_CONFIRMED | 15 |
| TIE_OR_INCONCLUSIVE | 16 |
| TOKEN_CONFIRMED | 16 |

양축 모두 안전한 Cell은 47/64개다.
서로 다른 Cell의 Layer Winner와 Token Winner를 실제 Continuous Serving의 보편 법칙으로 확대하지 않는다.
축 사이 TTFT Win과 Native 대비 TTFT Win은 여전히 다른 판정이다.

| 양축 모두 안전하지만 Native도 이긴 TTFT Winner | Cell 수 |
|---|---:|
| Layer | 4 |
| Token | 15 |

Native TTFT Win도 해당 Cell의 Paired CI 상한<1을 요구한다.
Goodput 우위는 별도 CI다. TTFT만으로 Goodput Win을 선언하지 않는다.
64개 Cell 전체의 Familywise CI는 아니다. 전체 실패 Arm도 보존했다.

- [모든 Cell의 양축 Safety와 원래 Label](dense_ar_longbench400_2d_20261121/analysis/axis64_both_axes_decode_safety_audit_v1.json)
