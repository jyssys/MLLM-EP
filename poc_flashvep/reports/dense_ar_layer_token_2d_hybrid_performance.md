# Genuine 2D Hybrid: 성능과 강한 대조군

이 문서는 C0, C3 Controlled Output과 C3 Natural400을 구분한다.
실측, Structural Accounting과 Offline Oracle을 같은 결과로 합산하지 않는다.

## C0 Exact Iso-work Confirmation

Layer8, Token8과 Hybrid4/Suffix 약 L/8의 실제 P/D Partition Rows를 검사했다.
Prompt KV와 Query/Layer Coverage Gate도 통과해야 유효한 Cell로 처리한다.
Actual Batch는 1/1/1이다. Source Credit은 4다.
모든 Worker는 BI0/BF16/B8192/S1024/KV42GiB/Block128이다.
Cell당 원본 Prompt는 16개다. 각 Cell은 독립 Server Start 3회를 완료했다.
비율은 Start별 Metric의 평균을 비교한다. CI는 Paired Start Bootstrap 27개 재표본으로 계산했다.
Pointwise Empirical 95% CI다. 다중 비교 보정을 하지 않았다.
Native는 D에서 마지막 Prompt Query를 다시 계산한다. Native를 같은 Offloaded Rows의 Arm이라고 부르지 않는다.

| Input Bucket | Output | QPS | 완료 Paired Start | Hybrid/Layer TTFT p95 [95% CI] | Hybrid/Token TTFT p95 [95% CI] | Hybrid/Native Goodput [95% CI] | 상태 |
|---|---:|---:|---:|---|---|---|---|
| I0 | 16 | 7.960658 | 3 | 1.100965 [1.026596,1.212455] | 1.021182 [0.938549,1.102130] | 0.885354 [0.874271,0.900239] | TIE_OR_NO_HYBRID_WIN |
| I0 | 64 | 2.352185 | 3 | 1.096742 [1.066608,1.121987] | 1.049831 [1.041724,1.059495] | 0.928918 [0.884113,0.973696] | TIE_OR_NO_HYBRID_WIN |
| I1 | 16 | 5.190199 | 3 | 1.052635 [1.010992,1.079705] | 1.059327 [1.044482,1.077379] | 1.092537 [1.054603,1.131081] | TIE_OR_NO_HYBRID_WIN |
| I1 | 64 | 2.290158 | 3 | 0.998300 [0.982873,1.014628] | 1.030955 [1.026288,1.035681] | 0.937000 [0.912322,0.964879] | TIE_OR_NO_HYBRID_WIN |
| I2 | 16 | 2.493804 | 3 | 0.962730 [0.944861,0.982918] | 1.044157 [1.027332,1.056120] | 1.111479 [1.052917,1.176752] | TIE_OR_NO_HYBRID_WIN |
| I2 | 64 | 2.19754 | 3 | 1.023307 [0.999882,1.039994] | 0.955279 [0.928091,0.979394] | 0.913110 [0.884765,0.946812] | TIE_OR_NO_HYBRID_WIN |
| I3 | 16 | 1.255875 | 3 | 0.925459 [0.894922,0.959027] | 1.046442 [1.020605,1.066546] | 1.621152 [1.204449,2.203379] | TIE_OR_NO_HYBRID_WIN |
| I3 | 64 | 1.253873 | 3 | 0.984181 [0.972197,0.996592] | 0.980561 [0.971238,0.990277] | 1.552476 [1.165218,2.185855] | HYBRID_ISOWORK_AXIS_WIN |

Hybrid의 TTFT CI가 두 Pure Axis보다 낮아도 Best-tuned Axis 전체를 이긴 것은 아니다.
해당 비교는 같은 Iso-work Frontier다. 다른 Cut의 최적 Static은 별도 대조군이다.
같은 Rows도 Attention, LM Head, Kernel Launch와 Transfer 비용을 같게 만들지 않는다.

### 모든 완료 Cell의 Decode Gate

| Input / Output | Axis | Native-relative ITL/TPOT 5% Gate | Absolute Decode≥95% | Native-relative TTFT CI Win |
|---|---|---|---|---|
| I0/O16 | Layer | False | True | False |
| I0/O16 | Token | True | True | False |
| I0/O16 | Hybrid | True | True | False |
| I0/O64 | Layer | True | True | False |
| I0/O64 | Token | True | True | False |
| I0/O64 | Hybrid | True | False | False |
| I1/O16 | Layer | True | True | True |
| I1/O16 | Token | True | True | True |
| I1/O16 | Hybrid | True | True | True |
| I1/O64 | Layer | True | False | False |
| I1/O64 | Token | True | True | False |
| I1/O64 | Hybrid | True | True | False |
| I2/O16 | Layer | True | True | True |
| I2/O16 | Token | True | True | True |
| I2/O16 | Hybrid | True | True | True |
| I2/O64 | Layer | True | True | False |
| I2/O64 | Token | True | True | False |
| I2/O64 | Hybrid | True | False | False |
| I3/O16 | Layer | True | True | True |
| I3/O16 | Token | True | True | True |
| I3/O16 | Hybrid | True | True | True |
| I3/O64 | Layer | True | True | True |
| I3/O64 | Token | True | True | True |
| I3/O64 | Hybrid | True | True | True |

## C0의 더 강한 Static Discovery

다음 표는 8개 Workload Cell에서 실행한 200-start, 25-action Discovery를 요약한다.
각 Cell은 Native와 24개 Non-native 2D Action을 포함한다.
Layer-only, Token-only Boundary와 Interior Hybrid를 모두 유지했다.
각 Family의 최소 TTFT p95를 Decode Point Gate 통과 후보 안에서 골랐다.
이 표는 독립 Start 1회다. Confidence-interval Winner나 Held-out Best-static Serving Gain이 아니다.
서로 다른 Action은 작업량이 다르다. 이 표를 Exact Iso-work로 표시하지 않는다.

| Input / Output | Family | 적격/전체 후보 | Discovery 최소 TTFT p95(ms) | Goodput(output tokens/s) | m / Suffix Fraction | Raw Name |
|---|---|---:|---:|---:|---|---|
| I0/O16 | Native | 1/1 | 712.971 | 110.513 | 0 / 0.000000 | HybridGrid_I0_O16_Native_v1 |
| I0/O16 | Layer | 4/4 | 899.280 | 100.903 | 2 / 0.000000 | HybridGrid_I0_O16_m2_f0.0_v1 |
| I0/O16 | Token | 4/4 | 1041.625 | 94.439 | 0 / 0.062500 | HybridGrid_I0_O16_m0_f0.0625_v1 |
| I0/O16 | Hybrid | 15/16 | 1116.357 | 91.979 | 4 / 0.062500 | HybridGrid_I0_O16_m4_f0.0625_v1 |
| I0/O64 | Native | 1/1 | 2293.441 | 130.347 | 0 / 0.000000 | HybridGrid_I0_O64_Native_v1 |
| I0/O64 | Layer | 1/4 | 2541.727 | 125.030 | 4 / 0.000000 | HybridGrid_I0_O64_m4_f0.0_v1 |
| I0/O64 | Token | 4/4 | 2621.450 | 123.328 | 0 / 0.125000 | HybridGrid_I0_O64_m0_f0.125_v1 |
| I0/O64 | Hybrid | 16/16 | 2663.373 | 122.665 | 2 / 0.062500 | HybridGrid_I0_O64_m2_f0.0625_v1 |
| I1/O16 | Native | 1/1 | 1316.781 | 66.681 | 0 / 0.000000 | HybridGrid_I1_O16_Native_v1 |
| I1/O16 | Layer | 4/4 | 1039.734 | 74.720 | 8 / 0.000000 | HybridGrid_I1_O16_m8_f0.0_v1 |
| I1/O16 | Token | 4/4 | 920.004 | 76.169 | 0 / 0.125000 | HybridGrid_I1_O16_m0_f0.125_v1 |
| I1/O16 | Hybrid | 16/16 | 863.930 | 79.691 | 2 / 0.125000 | HybridGrid_I1_O16_m2_f0.125_v1 |
| I1/O64 | Native | 1/1 | 2593.186 | 123.121 | 0 / 0.000000 | HybridGrid_I1_O64_Native_v1 |
| I1/O64 | Layer | 4/4 | 2779.227 | 119.122 | 2 / 0.000000 | HybridGrid_I1_O64_m2_f0.0_v1 |
| I1/O64 | Token | 4/4 | 2710.133 | 119.540 | 0 / 0.062500 | HybridGrid_I1_O64_m0_f0.0625_v1 |
| I1/O64 | Hybrid | 15/16 | 3033.260 | 114.770 | 2 / 0.062500 | HybridGrid_I1_O64_m2_f0.0625_v1 |
| I2/O16 | Native | 1/1 | 2857.134 | 32.220 | 0 / 0.000000 | HybridGrid_I2_O16_Native_v1 |
| I2/O16 | Layer | 4/4 | 1351.528 | 43.463 | 12 / 0.000000 | HybridGrid_I2_O16_m12_f0.0_v1 |
| I2/O16 | Token | 4/4 | 1280.859 | 43.839 | 0 / 0.333333 | HybridGrid_I2_O16_m0_f0.3333333333333333_v1 |
| I2/O16 | Hybrid | 16/16 | 1167.442 | 44.484 | 4 / 0.250000 | HybridGrid_I2_O16_m4_f0.25_v1 |
| I2/O64 | Native | 1/1 | 3237.066 | 109.756 | 0 / 0.000000 | HybridGrid_I2_O64_Native_v1 |
| I2/O64 | Layer | 4/4 | 3226.158 | 109.661 | 2 / 0.000000 | HybridGrid_I2_O64_m2_f0.0_v1 |
| I2/O64 | Token | 3/4 | 3538.023 | 105.827 | 0 / 0.062500 | HybridGrid_I2_O64_m0_f0.0625_v1 |
| I2/O64 | Hybrid | 15/16 | 3852.215 | 102.365 | 2 / 0.062500 | HybridGrid_I2_O64_m2_f0.0625_v1 |
| I3/O16 | Native | 1/1 | 5605.850 | 12.393 | 0 / 0.000000 | HybridGrid_I3_O16_Native_v1 |
| I3/O16 | Layer | 4/4 | 2858.091 | 22.302 | 12 / 0.000000 | HybridGrid_I3_O16_m12_f0.0_v1 |
| I3/O16 | Token | 4/4 | 2212.192 | 22.426 | 0 / 0.333333 | HybridGrid_I3_O16_m0_f0.3333333333333333_v1 |
| I3/O16 | Hybrid | 16/16 | 2179.600 | 22.376 | 2 / 0.333333 | HybridGrid_I3_O16_m2_f0.3333333333333333_v1 |
| I3/O64 | Native | 1/1 | 5620.614 | 48.023 | 0 / 0.000000 | HybridGrid_I3_O64_Native_v1 |
| I3/O64 | Layer | 4/4 | 3864.107 | 75.140 | 8 / 0.000000 | HybridGrid_I3_O64_m8_f0.0_v1 |
| I3/O64 | Token | 4/4 | 4038.896 | 74.546 | 0 / 0.125000 | HybridGrid_I3_O64_m0_f0.125_v1 |
| I3/O64 | Hybrid | 16/16 | 3966.519 | 74.253 | 2 / 0.125000 | HybridGrid_I3_O64_m2_f0.125_v1 |

Discovery의 Token-only Boundary는 Suffix Fraction 1/16, 1/8, 1/4, 1/3을 포함한다.
이 Boundary를 Last2/4/8/12의 Exact Rows Control과 같은 Partition 크기로 간주하지 않는다.
이 Static Grid의 QPS는 기존 Serial Service Reference에 기반한다.
따라서 QPS를 측정된 Sustainable Capacity의 고정 비율로 표현하지 않는다.

## Output만 바꾼 반증

I3의 별도 검사는 QPS1.253873, Source, Pair, Prompt와 Arrival을 고정했다.
Output16의 Hybrid/Token TTFT p95는 1.070249 [1.054657,1.081369]였다.
Output64에서는 0.980561 [0.971238,0.990277]였다.
각각 Paired Independent Start 3회다. 이 Controlled Output 역전은 C0 범위다.
이를 Natural400 또는 C3 Serving의 역전으로 옮기지 않는다.

## 실제 Continuous Bridge와 Natural400

C3 Controlled Bridge는 Output16/64와 D Cap1/2를 사전에 동결했다.
강한 Compiled Stock Token과 진단용 Mode0 Token을 모두 포함한다.
C0의 Hybrid 후보가 느린 Token Control만 이긴다면 Strong Serving GO로 처리하지 않는다.
Actual Continuous Batch, Padding, Compile Family와 Credit도 별도로 기록한다.
해당 결과는 [Controlled C3 Bridge](dense_ar_longbench400_controlled_continuous_bridge.md)에 기록한다.
Main Natural400은 별도의 Discovery-frozen Static 선택과 독립 Arrival Seed를 사용한다.
최종 Serving Gate는 NativeStrong과 같은 D 정책의 Native 대비 두 Decode CI를 모두 요구한다.
C0 성능만으로 C3 Gate를 통과 처리하지 않는다.

원래 최종 Main400/Heldout200의140 Starts는 모두 완료했다.
원래 Hybrid의 일부 Decode Gate는 실패했다. 그 결과를 그대로 보존했다.
별도 등록 수정 후 Natural400/QPS1.8/2.8은 각각 독립 Start3회에서 두 Native Decode Gate를 통과했다.
하지만 Hybrid/강한 Stock Token TTFT 비율은 각각1.096645 [1.036991,1.147475]와1.344281 [1.312721,1.378300]이었다.
같은 순서의 Goodput 비율은0.979117 [0.967172,0.988945]와0.845429 [0.698429,0.953074]였다.
즉, 수정한 C3에서도 Hybrid가 강한 Token보다 우수하다는 근거는 확보하지 못했다.
이는 시험한 Dataset, 후보, Runtime과 부하의 결과다. 모든 2D 설계의 불가능성은 아니다.

Natural400의 Safety PASS도 고정 Output16/64의 모든 계약에 자동 적용되지 않는다.
해당 동일 조건의 추가 결과는 별도 Controlled Bridge에 기록한다.
실패한 Compile 경로를 조용히 성능 Baseline으로 바꾸지 않았다.

마지막 별도 Source의 Eager-prefill/Native-compiled-decode 검사는 Output16/Cap1의 독립 Start3회를 완료했다.
수정 Hybrid는 두 Native의 Decode5%와 모든 Start의 Absolute Gate를 통과했다.
그러나 Hybrid/강한 Token TTFT는1.121429 [1.050122,1.179215]였다.
Goodput은0.965898 [0.949395,0.988077]이었다. Decode 보호의 회복은 강한 Pure-axis 대비 승리가 아니었다.
같은 Source의 기존 Hybrid 대비 TTFT CI도1을 포함했다. 새 Kernel 정책만으로 TTFT 이득을 확정하지 않는다.
전체21회와 변경하지 않은 대조군은 [최종 Kernel 정책 검사](dense_ar_longbench400_eager_prefill_native_decode_results.md)에 있다.

## 근거

- [Exact Iso-work Confirmation](dense_ar_longbench400_2d_20261121/analysis/iso8_hybrid_heldout_confirmation_v1.json)
- [전체 Static Discovery 설계](dense_ar_longbench400_2d_20261121/configs/hybrid_static_search_frozen_v1.json)
- [Static Discovery 전체 결과](dense_ar_longbench400_2d_20261121/analysis/c0_hybrid_static_discovery_full_ledger_v1.json)
- [Exact Output Counterfactual](dense_ar_longbench400_2d_20261121/analysis/iso8_I3_exact_output_counterfactual_v1.json)
- [Final C3 설계](dense_ar_longbench400_2d_20261121/configs/final_continuous_execution_plan_v1.json)
