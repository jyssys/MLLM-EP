# Final C3: 실측 Table

이 문서는 Raw Client Timestamp에서 다시 계산한 값만 표시한다.
같은 Cell의 Arm은 Dataset, Arrival Trace와 GPU Pair가 같다.
기본 Metric은 Start별 통계량의 평균이다. 관측 전체 Max 열만 모든 유효 Start의 단일 최댓값이다.
Request를 독립 Start로 재표본하지 않는다.
N3의 비율 CI는 Paired Start Bootstrap 27개 재표본을 완전 열거했다.
이 CI는 Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.
N1에는 Start-level CI가 없다. 유한 Cohort 처리량은 Sustainable Capacity가 아니다.

표의 Metric은 소수 3자리, 비율은 소수 6자리로 표시한다. Gate는 반올림 전 값으로 판정한다.

Source Capsule SHA256: `a2d5ebc4cb39eff66a8fbba3b4c0b83a8957d79e50c11ff94a529b2cdb2ca6e3`.
Execution Plan SHA256: `b249fc9c962c300cb4ad5002f063f483f7b0cf555b6a13f19beed9e8dca696d0`.
Main Dataset SHA256: `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`.
모든 Worker는 BI0, Original BF16, B8192/S1024, KV42GiB, Block128을 사용한다.
Natural EOS를 허용한다. Output Cap은 2048이다.
Task-quality Gate는 다른 BI0 Batch의 Greedy Token 완전 일치를 뜻하지 않는다.

## Heldout200 / QPS1.8

Cell 상태: `MEASURED_ONLINE_CONFIRMED`. 예정 독립 Start: 3.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 3 | 1,955.388 | 27.781 | 348.150 | 428.916 | 460.345 | 247.464 | 44.852 | 24.475 | 0.568 | 1.812 |
| BestHybrid | 3 | 4,327.216 | 7.645 | 15.520 | 62.120 | 82.096 | 9.727 | 44.909 | 39.949 | 0.925 | 1.806 |
| BestLayer | 3 | 4,889.332 | 7.454 | 9.263 | 59.446 | 113.119 | 8.369 | 44.923 | 40.659 | 0.890 | 1.807 |
| BestToken | 3 | 3,671.162 | 7.281 | 9.309 | 38.353 | 49.027 | 8.457 | 45.366 | 44.760 | 0.993 | 1.808 |
| BestWholeDeflection | 3 | 3,899.296 | 7.142 | 8.764 | 30.283 | 45.410 | 7.916 | 45.425 | 44.827 | 0.980 | 1.809 |
| NativeCommon0 | 3 | 12,272.380 | 7.409 | 13.768 | 50.682 | 55.959 | 8.107 | 44.079 | 18.185 | 0.423 | 1.770 |
| NativeCommon1 | 3 | 12,388.480 | 7.120 | 8.629 | 54.699 | 83.217 | 7.878 | 43.889 | 18.354 | 0.425 | 1.766 |
| NativeCommon2 | 3 | 12,291.788 | 7.228 | 8.770 | 52.397 | 60.013 | 7.967 | 43.884 | 18.488 | 0.425 | 1.769 |
| NativeCommon3 | 3 | 12,318.560 | 7.228 | 8.749 | 79.405 | 169.394 | 7.897 | 43.781 | 18.491 | 0.430 | 1.768 |
| NativeStrong | 3 | 12,293.687 | 7.434 | 14.315 | 47.718 | 48.693 | 8.165 | 44.114 | 17.952 | 0.423 | 1.769 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [4901, 4989, 4960] | 8/9 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [4983, 4957, 4977] | 3/3 | 8192/8192 | 8192/8192 | [4] | [true,false,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [4984, 4948, 4983] | 3/3 | 8192/8192 | 8192/8192 | [4] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestToken | [5101, 5009, 4949] | 4/4 | 8192/8192 | 8192/8192 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [5035, 4989, 5035] | 3/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [5017, 4930, 4989] | 4/4 | 8192/4 | 8192/4 | [0] | [true,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [5019, 4932, 4956] | 4/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [4937, 4968, 4977] | 4/4 | 8192/4 | 8192/4 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon3 | [4992, 4932, 4934] | 4/4 | 8192/4 | 8192/4 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [4987, 4988, 4984] | 4/4 | 8192/4 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.351987 [0.294400,0.402228] | 1.084193 [0.984518,1.226784] | 1.191371 [1.168692,1.211739] | 2.225270 [1.523568,3.702404] | FAIL | True |
| BestHybrid / NativeCommon0 | 0.352598 [0.293872,0.402888] | 1.127287 [1.009791,1.292281] | 1.199883 [1.176017,1.219098] | 2.196753 [1.539817,3.581745] | FAIL | True |
| BestHybrid / BestToken | 1.178705 [1.142755,1.207547] | 1.667167 [1.557531,1.763912] | 1.150177 [1.138472,1.165422] | 0.892496 [0.880725,0.902199] | FAIL | True |
| BestHybrid / BestLayer | 0.885032 [0.876541,0.897196] | 1.675421 [1.497526,1.839457] | 1.162363 [1.126672,1.194884] | 0.982534 [0.940839,1.017631] | FAIL | True |
| BestLayer / NativeStrong | 0.397711 [0.329962,0.454595] | 0.647116 [0.596967,0.690567] | 1.024956 [1.005671,1.040682] | 2.264826 [1.538782,3.708623] | PASS | True |
| BestLayer / NativeCommon2 | 0.397772 [0.330454,0.454583] | 1.056199 [1.046995,1.064503] | 1.050423 [1.044123,1.058790] | 2.199178 [1.491307,3.656803] | FAIL | True |
| BestLayer / BestToken | 1.331821 [1.273739,1.377662] | 0.995073 [0.958974,1.040450] | 0.989517 [0.970875,1.013719] | 0.908361 [0.865468,0.959102] | PASS | True |
| BestToken / NativeStrong | 0.298622 [0.253976,0.341786] | 0.650320 [0.600291,0.708127] | 1.035815 [1.018331,1.058204] | 2.493311 [1.690847,4.202031] | FAIL | True |
| BestToken / NativeCommon3 | 0.298019 [0.254384,0.341020] | 1.064019 [1.021113,1.105578] | 1.070990 [1.044973,1.096957] | 2.420726 [1.647766,4.197142] | FAIL | True |
| BestWholeDeflection / NativeStrong | 0.317179 [0.278895,0.361600] | 0.612258 [0.579444,0.641615] | 0.969491 [0.965554,0.972765] | 2.497017 [1.679960,4.263597] | PASS | True |
| BestWholeDeflection / NativeCommon1 | 0.314752 [0.277948,0.359013] | 1.015695 [1.011842,1.018924] | 1.004818 [1.002015,1.008349] | 2.442368 [1.644527,4.274646] | PASS | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## Heldout200 / QPS2.8

Cell 상태: `MEASURED_ONLINE_CONFIRMED`. 예정 독립 Start: 3.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 3 | 5,170.774 | 73.829 | 390.509 | 576.401 | 593.538 | 292.741 | 70.256 | 15.749 | 0.268 | 2.798 |
| BestHybrid | 3 | 16,267.209 | 7.473 | 9.352 | 279.856 | 450.253 | 8.386 | 59.222 | 14.893 | 0.257 | 2.381 |
| BestLayer | 3 | 18,081.958 | 7.472 | 9.384 | 78.753 | 142.419 | 8.354 | 57.790 | 13.730 | 0.248 | 2.320 |
| BestToken | 3 | 12,704.612 | 7.345 | 10.091 | 29.718 | 40.227 | 8.482 | 63.924 | 23.273 | 0.368 | 2.549 |
| BestWholeDeflection | 3 | 11,649.151 | 7.141 | 8.828 | 121.613 | 189.349 | 7.980 | 63.202 | 25.651 | 0.388 | 2.553 |
| NativeCommon0 | 3 | 35,038.804 | 7.129 | 8.751 | 107.343 | 173.698 | 7.882 | 46.663 | 6.224 | 0.130 | 1.885 |
| NativeCommon1 | 3 | 34,860.884 | 7.213 | 9.046 | 88.033 | 187.380 | 7.938 | 46.762 | 6.117 | 0.128 | 1.887 |
| NativeCommon2 | 3 | 34,913.217 | 7.204 | 8.937 | 94.035 | 184.696 | 7.900 | 46.863 | 6.142 | 0.128 | 1.885 |
| NativeStrong | 3 | 34,727.624 | 7.440 | 14.334 | 40.037 | 44.564 | 8.101 | 46.964 | 6.034 | 0.130 | 1.891 |
| TokenCap2Control | 3 | 12,681.703 | 7.292 | 9.376 | 91.509 | 176.757 | 8.370 | 64.264 | 23.796 | 0.365 | 2.538 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [5010, 5020, 5036] | 11/19 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [4977, 4972, 4975] | 4/3 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [4983, 4981, 4985] | 4/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | MATCHED_NO_COMPILE |
| BestToken | [5007, 4975, 5064] | 4/4 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [4948, 4944, 4960] | 3/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [4932, 4932, 4986] | 3/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [4960, 4975, 4933] | 3/2 | 8192/2 | 8192/2 | [0] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [4987, 4976, 4950] | 3/3 | 8192/3 | 8192/4 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [4953, 4992, 4953] | 3/3 | 8192/3 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| TokenCap2Control | [5090, 5036, 5067] | 4/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.468423 [0.448714,0.483440] | 0.652431 [0.609859,0.716985] | 1.035124 [1.025320,1.044248] | 2.468250 [1.908160,3.275263] | PASS | True |
| BestHybrid / NativeCommon2 | 0.465933 [0.444660,0.482159] | 1.046443 [1.031095,1.065651] | 1.061483 [1.051506,1.070857] | 2.424590 [1.911297,3.220754] | FAIL | True |
| BestHybrid / BestToken | 1.280418 [1.241426,1.308032] | 0.926764 [0.876507,0.992655] | 0.988622 [0.969307,1.013199] | 0.639911 [0.473817,0.910780] | PASS | True |
| BestHybrid / TokenCap2Control | 1.282731 [1.250897,1.307221] | 0.997422 [0.906955,1.092567] | 1.001814 [0.971494,1.033056] | 0.625842 [0.473984,0.848081] | FAIL | True |
| BestHybrid / BestLayer | 0.899638 [0.893896,0.905885] | 0.996579 [0.979445,1.010376] | 1.003804 [1.000340,1.007726] | 1.084689 [1.032224,1.128277] | PASS | True |
| BestLayer / NativeStrong | 0.520679 [0.501976,0.534354] | 0.654670 [0.604332,0.732068] | 1.031200 [1.017462,1.043894] | 2.275536 [1.847590,2.901076] | PASS | True |
| BestLayer / NativeCommon1 | 0.518689 [0.497651,0.534142] | 1.037395 [1.013947,1.060096] | 1.052359 [1.046456,1.057330] | 2.244556 [1.852476,2.863629] | FAIL | True |
| BestLayer / BestToken | 1.423259 [1.388589,1.446708] | 0.929945 [0.892850,0.983665] | 0.984876 [0.967638,1.009057] | 0.589949 [0.424456,0.849062] | PASS | True |
| BestLayer / TokenCap2Control | 1.425830 [1.399183,1.443365] | 1.000846 [0.925718,1.082669] | 0.998017 [0.969854,1.028831] | 0.576978 [0.424611,0.790700] | FAIL | True |
| BestToken / NativeStrong | 0.365836 [0.348027,0.384462] | 0.703988 [0.614641,0.818171] | 1.047036 [1.020450,1.073944] | 3.857177 [2.302409,6.682430] | FAIL | True |
| BestToken / NativeCommon2 | 0.363891 [0.345013,0.383424] | 1.129135 [1.073835,1.176397] | 1.073699 [1.055870,1.088122] | 3.788949 [2.307344,6.728246] | FAIL | True |
| BestWholeDeflection / NativeStrong | 0.335443 [0.302428,0.359575] | 0.615881 [0.573385,0.671902] | 0.985110 [0.979936,0.990932] | 4.251390 [2.314599,7.408804] | PASS | True |
| BestWholeDeflection / NativeCommon0 | 0.332464 [0.298123,0.357582] | 1.008777 [1.001265,1.016601] | 1.012444 [1.009027,1.016561] | 4.121163 [2.240824,7.210305] | PASS | True |
| TokenCap2Control / NativeStrong | 0.365176 [0.349546,0.381107] | 0.654117 [0.558268,0.791631] | 1.033250 [1.000814,1.071591] | 3.943884 [2.340109,6.740148] | FAIL | True |
| TokenCap2Control / NativeCommon1 | 0.363780 [0.347229,0.379812] | 1.036518 [0.953056,1.138545] | 1.054450 [1.025500,1.085288] | 3.890190 [2.346313,6.726636] | FAIL | True |
| TokenCap2Control / BestToken | 0.998197 [0.991293,1.004383] | 0.929160 [0.880679,0.977736] | 0.986833 [0.974210,1.000327] | 1.022479 [0.994185,1.073806] | PASS | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## Main400 / QPS1.0

Cell 상태: `MEASURED_ONLINE_SINGLE_START`. 예정 독립 Start: 1.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 1 | 1,306.015 | 10.796 | 193.507 | 394.517 | 394.517 | 53.027 | 27.892 | 21.763 | 0.828 | 1.008 |
| BestHybrid | 1 | 2,051.739 | 7.576 | 9.234 | 55.290 | 55.290 | 8.604 | 27.604 | 26.565 | 0.983 | 1.008 |
| BestLayer | 1 | 2,093.794 | 7.530 | 8.890 | 455.478 | 455.478 | 8.277 | 27.750 | 27.145 | 0.990 | 1.008 |
| BestToken | 1 | 1,865.766 | 7.232 | 8.689 | 67.218 | 67.218 | 8.152 | 28.229 | 27.881 | 0.995 | 1.008 |
| BestWholeDeflection | 1 | 1,918.853 | 7.193 | 8.558 | 62.449 | 62.449 | 7.931 | 27.789 | 27.688 | 0.998 | 1.008 |
| NativeCommon0 | 1 | 2,324.680 | 7.359 | 11.911 | 83.908 | 83.908 | 8.219 | 27.804 | 27.206 | 0.990 | 1.008 |
| NativeCommon1 | 1 | 2,483.007 | 7.179 | 8.531 | 58.859 | 58.859 | 7.926 | 27.837 | 27.714 | 0.998 | 1.008 |
| NativeCommon2 | 1 | 2,343.437 | 7.259 | 8.719 | 187.919 | 187.919 | 7.966 | 27.800 | 27.505 | 0.995 | 1.008 |
| NativeCommon3 | 1 | 2,354.320 | 7.234 | 8.752 | 67.075 | 67.075 | 7.946 | 27.965 | 27.839 | 0.998 | 1.008 |
| NativeStrong | 1 | 2,325.976 | 7.420 | 13.422 | 67.742 | 67.742 | 8.357 | 27.781 | 27.390 | 0.993 | 1.008 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [11063] | 4/4 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [10951] | 3/2 | 8192/8192 | 8192/8192 | [4] | [true,false,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [11008] | 3/2 | 8192/8192 | 8192/8192 | [4] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestToken | [11197] | 3/2 | 8192/7936 | 8192/7936 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [11023] | 2/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [11029] | 3/4 | 8192/4 | 8192/4 | [0] | [true,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [11042] | 3/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [11027] | 3/3 | 8192/3 | 8192/4 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon3 | [11093] | 3/3 | 8192/3 | 8192/4 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [11020] | 3/4 | 8192/4 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.882098 [N1, CI 없음] | 0.687965 [N1, CI 없음] | 1.029613 [N1, CI 없음] | 0.969883 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / NativeCommon0 | 0.882590 [N1, CI 없음] | 0.775228 [N1, CI 없음] | 1.046890 [N1, CI 없음] | 0.976432 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / BestToken | 1.099676 [N1, CI 없음] | 1.062764 [N1, CI 없음] | 1.055502 [N1, CI 없음] | 0.952817 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / BestLayer | 0.979914 [N1, CI 없음] | 1.038661 [N1, CI 없음] | 1.039596 [N1, CI 없음] | 0.978654 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / NativeStrong | 0.900179 [N1, CI 없음] | 0.662358 [N1, CI 없음] | 0.990397 [N1, CI 없음] | 0.991038 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / NativeCommon2 | 0.893472 [N1, CI 없음] | 1.019641 [N1, CI 없음] | 1.038921 [N1, CI 없음] | 0.986892 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / BestToken | 1.122217 [N1, CI 없음] | 1.023206 [N1, CI 없음] | 1.015300 [N1, CI 없음] | 0.973600 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestToken / NativeStrong | 0.802143 [N1, CI 없음] | 0.647336 [N1, CI 없음] | 0.975473 [N1, CI 없음] | 1.017911 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestToken / NativeCommon3 | 0.792486 [N1, CI 없음] | 0.992743 [N1, CI 없음] | 1.025925 [N1, CI 없음] | 1.001490 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestWholeDeflection / NativeStrong | 0.824967 [N1, CI 없음] | 0.637594 [N1, CI 없음] | 0.949096 [N1, CI 없음] | 1.010892 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestWholeDeflection / NativeCommon1 | 0.772794 [N1, CI 없음] | 1.003176 [N1, CI 없음] | 1.000617 [N1, CI 없음] | 0.999092 [N1, CI 없음] | N1, NOT_CONFIRMED | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## Main400 / QPS1.8

Cell 상태: `MEASURED_ONLINE_CONFIRMED`. 예정 독립 Start: 3.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 3 | 1,495.860 | 17.939 | 283.659 | 414.948 | 431.132 | 178.006 | 49.960 | 27.184 | 0.635 | 1.807 |
| BestHybrid | 3 | 3,018.808 | 7.628 | 11.707 | 65.764 | 68.652 | 9.419 | 49.624 | 46.381 | 0.964 | 1.802 |
| BestLayer | 3 | 3,254.888 | 7.507 | 9.077 | 276.338 | 701.754 | 8.288 | 49.596 | 48.742 | 0.993 | 1.802 |
| BestToken | 3 | 2,839.826 | 7.276 | 8.950 | 87.062 | 125.648 | 8.239 | 50.455 | 50.024 | 0.994 | 1.803 |
| BestWholeDeflection | 3 | 3,068.499 | 7.194 | 8.593 | 237.170 | 457.769 | 7.945 | 49.993 | 49.237 | 0.990 | 1.805 |
| NativeCommon0 | 3 | 4,302.649 | 7.552 | 17.003 | 135.896 | 317.916 | 8.489 | 49.576 | 47.269 | 0.969 | 1.799 |
| NativeCommon1 | 3 | 4,678.077 | 7.234 | 8.548 | 882.867 | 2,524.485 | 7.913 | 49.659 | 47.371 | 0.958 | 1.798 |
| NativeCommon2 | 3 | 4,400.438 | 7.294 | 8.936 | 91.912 | 129.550 | 8.014 | 49.829 | 48.139 | 0.971 | 1.798 |
| NativeCommon3 | 3 | 4,367.251 | 7.275 | 8.886 | 91.915 | 149.758 | 8.012 | 49.654 | 48.334 | 0.973 | 1.798 |
| NativeStrong | 3 | 4,307.192 | 7.483 | 14.620 | 93.208 | 147.320 | 8.292 | 49.581 | 47.911 | 0.973 | 1.799 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [11101, 11132, 10953] | 8/8 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [10987, 11030, 11032] | 3/3 | 8192/8192 | 8192/8192 | [4] | [true,false,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [11030, 10997, 10997] | 3/3 | 8192/8192 | 8192/8192 | [4] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestToken | [11177, 11199, 11200] | 3/5 | 8192/8192 | 8192/8192 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [11046, 11089, 11103] | 3/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [11045, 11062, 10964] | 3/4 | 8192/4 | 8192/4 | [0] | [true,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [11033, 11082, 11033] | 3/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [11072, 11115, 11069] | 3/5 | 8192/5 | 8192/8 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon3 | [11095, 11000, 11042] | 3/5 | 8192/5 | 8192/8 | [0] | [true,true,false,null,true] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [11029, 11030, 11010] | 3/4 | 8192/4 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.700876 [0.683088,0.716358] | 0.800780 [0.650349,1.008575] | 1.135822 [1.115406,1.158138] | 0.968065 [0.957878,0.981397] | FAIL | True |
| BestHybrid / NativeCommon0 | 0.701616 [0.687279,0.714975] | 0.688548 [0.614511,0.775966] | 1.109471 [1.096561,1.123605] | 0.981223 [0.958468,1.004606] | FAIL | True |
| BestHybrid / BestToken | 1.063026 [1.029261,1.097238] | 1.308057 [1.103043,1.540976] | 1.143143 [1.126128,1.158259] | 0.927170 [0.920955,0.931854] | FAIL | True |
| BestHybrid / BestLayer | 0.927469 [0.902141,0.949805] | 1.289800 [1.073209,1.538877] | 1.136384 [1.116053,1.159082] | 0.951573 [0.946041,0.955530] | FAIL | True |
| BestLayer / NativeStrong | 0.755687 [0.719199,0.794162] | 0.620856 [0.586322,0.663160] | 0.999505 [0.998845,1.000309] | 1.017332 [1.002458,1.037385] | PASS | True |
| BestLayer / NativeCommon2 | 0.739674 [0.700855,0.785896] | 1.015754 [0.992340,1.044274] | 1.034290 [1.028116,1.039499] | 1.012526 [0.994645,1.037139] | PASS | True |
| BestLayer / BestToken | 1.146158 [1.139623,1.155887] | 1.014155 [1.001439,1.027895] | 1.005948 [0.997368,1.014186] | 0.974355 [0.973367,0.975404] | PASS | True |
| BestToken / NativeStrong | 0.659322 [0.622592,0.696074] | 0.612190 [0.577458,0.659043] | 0.993596 [0.986319,1.001485] | 1.044108 [1.028740,1.065367] | PASS | True |
| BestToken / NativeCommon3 | 0.650255 [0.612557,0.690617] | 1.007202 [0.987857,1.024800] | 1.028361 [1.025433,1.031249] | 1.034983 [1.017478,1.059480] | PASS | True |
| BestWholeDeflection / NativeStrong | 0.712413 [0.667070,0.761667] | 0.587747 [0.557581,0.632373] | 0.958159 [0.950780,0.966147] | 1.027664 [1.010016,1.050396] | PASS | True |
| BestWholeDeflection / NativeCommon1 | 0.655932 [0.606640,0.705676] | 1.005237 [1.000457,1.009898] | 1.004094 [1.003095,1.005206] | 1.039386 [1.012217,1.078283] | PASS | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## Main400 / QPS2.8

Cell 상태: `MEASURED_ONLINE_CONFIRMED`. 예정 독립 Start: 3.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 3 | 2,204.664 | 36.098 | 343.887 | 528.607 | 640.428 | 254.004 | 77.107 | 22.528 | 0.387 | 2.791 |
| BestHybrid | 3 | 9,215.782 | 7.524 | 9.126 | 192.322 | 356.628 | 8.335 | 73.852 | 35.503 | 0.498 | 2.689 |
| BestLayer | 3 | 11,017.683 | 7.549 | 9.779 | 58.886 | 63.117 | 8.450 | 73.101 | 25.421 | 0.367 | 2.658 |
| BestToken | 3 | 5,579.426 | 7.356 | 8.919 | 94.213 | 165.480 | 8.194 | 76.457 | 66.923 | 0.885 | 2.773 |
| BestWholeDeflection | 3 | 7,119.072 | 7.189 | 8.702 | 64.422 | 67.822 | 7.941 | 75.752 | 58.280 | 0.768 | 2.738 |
| NativeCommon0 | 3 | 35,345.656 | 7.160 | 8.589 | 105.809 | 235.207 | 7.901 | 62.264 | 7.407 | 0.129 | 2.256 |
| NativeCommon1 | 3 | 35,012.130 | 7.268 | 8.997 | 136.742 | 219.345 | 7.972 | 62.416 | 7.454 | 0.130 | 2.259 |
| NativeCommon2 | 3 | 35,027.635 | 7.271 | 9.066 | 219.539 | 295.135 | 7.963 | 62.288 | 7.475 | 0.130 | 2.258 |
| NativeStrong | 3 | 34,932.249 | 7.511 | 14.845 | 64.854 | 68.871 | 8.328 | 62.507 | 7.021 | 0.128 | 2.261 |
| TokenCap2Control | 3 | 5,795.941 | 7.360 | 9.304 | 167.153 | 307.706 | 8.308 | 76.161 | 66.473 | 0.879 | 2.770 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [11078, 11057, 11014] | 12/12 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [10990, 10979, 10989] | 3/4 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [11008, 11024, 10971] | 3/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | MATCHED_NO_COMPILE |
| BestToken | [10990, 11019, 11073] | 4/5 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [11065, 11078, 11064] | 3/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [11033, 11081, 11000] | 4/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [11079, 11034, 11045] | 4/2 | 8192/2 | 8192/2 | [0] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [11082, 11007, 11008] | 4/5 | 8192/5 | 8192/8 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [11066, 11098, 11012] | 4/4 | 8192/4 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| TokenCap2Control | [10947, 11025, 11025] | 4/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.263819 [0.249473,0.286244] | 0.614753 [0.541357,0.699818] | 1.000908 [0.983555,1.018373] | 5.056473 [3.348846,6.715805] | PASS | True |
| BestHybrid / NativeCommon2 | 0.263100 [0.248092,0.287095] | 1.006578 [0.975884,1.043421] | 1.046756 [1.043745,1.050312] | 4.749484 [3.287554,6.027864] | FAIL | True |
| BestHybrid / BestToken | 1.651744 [1.482651,1.868662] | 1.023149 [0.992882,1.051341] | 1.017272 [1.005299,1.031329] | 0.530510 [0.326095,0.750590] | FAIL | True |
| BestHybrid / TokenCap2Control | 1.590041 [1.381405,1.824533] | 0.980897 [0.949684,1.011096] | 1.003322 [0.995442,1.009751] | 0.534099 [0.329257,0.755219] | PASS | True |
| BestHybrid / BestLayer | 0.836454 [0.749025,0.989539] | 0.933197 [0.917516,0.948088] | 0.986483 [0.983222,0.989542] | 1.396592 [1.233961,1.693871] | PASS | True |
| BestLayer / NativeStrong | 0.315401 [0.289808,0.333080] | 0.658760 [0.581679,0.742160] | 1.014622 [1.000337,1.029133] | 3.620581 [2.357089,5.161493] | PASS | True |
| BestLayer / NativeCommon1 | 0.314682 [0.289768,0.332118] | 1.087001 [1.082651,1.092160] | 1.059946 [1.057794,1.062689] | 3.410599 [2.317960,4.715049] | FAIL | True |
| BestLayer / BestToken | 1.974698 [1.505425,2.495660] | 1.096392 [1.062797,1.125659] | 1.031211 [1.022454,1.042227] | 0.379861 [0.229070,0.577006] | FAIL | True |
| BestLayer / TokenCap2Control | 1.900931 [1.404133,2.436478] | 1.051115 [1.010905,1.098025] | 1.017070 [1.011726,1.022273] | 0.382430 [0.231327,0.580603] | FAIL | True |
| BestToken / NativeStrong | 0.159721 [0.133656,0.193242] | 0.600844 [0.545145,0.665599] | 0.983914 [0.977609,0.989352] | 9.531337 [8.950303,10.310970] | PASS | True |
| BestToken / NativeCommon2 | 0.159286 [0.132914,0.193827] | 0.983803 [0.968119,0.998232] | 1.028984 [1.018417,1.038245] | 8.952670 [8.039871,10.135779] | PASS | True |
| BestWholeDeflection / NativeStrong | 0.203797 [0.184983,0.227253] | 0.586194 [0.525321,0.657408] | 0.953556 [0.941632,0.966542] | 8.300351 [7.955747,8.618268] | PASS | True |
| BestWholeDeflection / NativeCommon0 | 0.201413 [0.182972,0.224976] | 1.013137 [1.008887,1.017308] | 1.005097 [1.003570,1.006618] | 7.868094 [7.598579,8.285736] | PASS | True |
| TokenCap2Control / NativeStrong | 0.165920 [0.136799,0.207741] | 0.626725 [0.566455,0.703989] | 0.997594 [0.987491,1.010051] | 9.467294 [8.896384,10.208168] | PASS | True |
| TokenCap2Control / NativeCommon1 | 0.165541 [0.136402,0.207741] | 1.034140 [0.989771,1.073105] | 1.042157 [1.035103,1.050239] | 8.918222 [8.077719,10.048108] | FAIL | True |
| TokenCap2Control / BestToken | 1.038806 [0.986972,1.090187] | 1.043075 [1.020287,1.064646] | 1.013904 [1.003891,1.025013] | 0.993281 [0.988128,0.998517] | FAIL | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## Main400 / QPS4.5

Cell 상태: `MEASURED_ONLINE_SINGLE_START`. 예정 독립 Start: 1.

| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agg | 1 | 7,505.652 | 220.884 | 404.131 | 716.486 | 716.486 | 289.484 | 117.402 | 2.188 | 0.025 | 4.249 |
| BestHybrid | 1 | 54,371.745 | 7.536 | 9.392 | 68.325 | 68.325 | 8.417 | 76.676 | 7.814 | 0.110 | 2.796 |
| BestLayer | 1 | 57,364.681 | 7.571 | 9.899 | 67.373 | 67.373 | 8.420 | 75.325 | 7.297 | 0.107 | 2.741 |
| BestToken | 1 | 43,787.549 | 7.491 | 10.055 | 372.407 | 372.407 | 8.568 | 83.263 | 9.735 | 0.122 | 3.023 |
| BestWholeDeflection | 1 | 49,920.766 | 7.211 | 8.816 | 67.406 | 67.406 | 7.945 | 80.322 | 8.417 | 0.113 | 2.880 |
| NativeCommon0 | 1 | 86,290.599 | 7.188 | 8.556 | 470.315 | 470.315 | 7.931 | 62.720 | 3.304 | 0.060 | 2.271 |
| NativeCommon1 | 1 | 86,381.159 | 7.284 | 8.896 | 309.408 | 309.408 | 7.934 | 62.583 | 3.284 | 0.060 | 2.269 |
| NativeCommon2 | 1 | 86,266.056 | 7.311 | 9.055 | 56.186 | 56.186 | 7.981 | 62.898 | 3.285 | 0.060 | 2.270 |
| NativeStrong | 1 | 86,234.771 | 7.510 | 13.785 | 69.013 | 69.013 | 8.310 | 62.244 | 2.953 | 0.058 | 2.271 |
| TokenCap2Control | 1 | 44,839.240 | 7.403 | 10.096 | 70.672 | 70.672 | 8.467 | 84.112 | 9.755 | 0.115 | 2.997 |

### Actual Work와 Runtime

| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |
|---|---|---|---|---|---|---|---|
| Agg | [11053] | 23/25 | 8192/8192 | 8192/8192 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestHybrid | [10971] | 3/4 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | MATCHED_NO_COMPILE |
| BestLayer | [10994] | 3/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | MATCHED_NO_COMPILE |
| BestToken | [11016] | 4/5 | 8192/8192 | 8192/8192 | [8] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| BestWholeDeflection | [11155] | 3/1 | 8192/8192 | 8192/8192 | [4] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon0 | [11049] | 4/1 | 8192/1 | 8192/1 | [0] | [true,true,false,1,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon1 | [11034] | 4/2 | 8192/2 | 8192/2 | [0] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |
| NativeCommon2 | [11085] | 4/5 | 8192/5 | 8192/8 | [0] | [true,true,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| NativeStrong | [10961] | 4/4 | 8192/4 | 8192/4 | [0] | [false,false,false,null,false] | INSTALLED_DEFAULT_COMPILE |
| TokenCap2Control | [11226] | 4/2 | 8192/8192 | 8192/8192 | [8] | [true,true,false,2,false] | INSTALLED_DEFAULT_COMPILE |

D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.
Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.
Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.

### Paired 비교

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| BestHybrid / NativeStrong | 0.630508 [N1, CI 없음] | 0.681319 [N1, CI 없음] | 1.012897 [N1, CI 없음] | 2.646090 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / NativeCommon2 | 0.630280 [N1, CI 없음] | 1.037214 [N1, CI 없음] | 1.054753 [N1, CI 없음] | 2.378366 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / BestToken | 1.241717 [N1, CI 없음] | 0.934064 [N1, CI 없음] | 0.982375 [N1, CI 없음] | 0.802621 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / TokenCap2Control | 1.212593 [N1, CI 없음] | 0.930307 [N1, CI 없음] | 0.994122 [N1, CI 없음] | 0.800962 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestHybrid / BestLayer | 0.947826 [N1, CI 없음] | 0.948793 [N1, CI 없음] | 0.999737 [N1, CI 없음] | 1.070831 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / NativeStrong | 0.665215 [N1, CI 없음] | 0.718090 [N1, CI 없음] | 1.013163 [N1, CI 없음] | 2.471062 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / NativeCommon1 | 0.664088 [N1, CI 없음] | 1.112806 [N1, CI 없음] | 1.061259 [N1, CI 없음] | 2.221959 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / BestToken | 1.310068 [N1, CI 없음] | 0.984476 [N1, CI 없음] | 0.982633 [N1, CI 없음] | 0.749531 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestLayer / TokenCap2Control | 1.279341 [N1, CI 없음] | 0.980516 [N1, CI 없음] | 0.994384 [N1, CI 없음] | 0.747982 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestToken / NativeStrong | 0.507771 [N1, CI 없음] | 0.729414 [N1, CI 없음] | 1.031069 [N1, CI 없음] | 3.296811 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestToken / NativeCommon2 | 0.507587 [N1, CI 없음] | 1.110432 [N1, CI 없음] | 1.073677 [N1, CI 없음] | 2.963249 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestWholeDeflection / NativeStrong | 0.578894 [N1, CI 없음] | 0.639518 [N1, CI 없음] | 0.956018 [N1, CI 없음] | 2.850558 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| BestWholeDeflection / NativeCommon0 | 0.578519 [N1, CI 없음] | 1.030343 [N1, CI 없음] | 1.001691 [N1, CI 없음] | 2.547860 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| TokenCap2Control / NativeStrong | 0.519967 [N1, CI 없음] | 0.732360 [N1, CI 없음] | 1.018886 [N1, CI 없음] | 3.303640 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| TokenCap2Control / NativeCommon1 | 0.519086 [N1, CI 없음] | 1.134919 [N1, CI 없음] | 1.067253 [N1, CI 없음] | 2.970607 [N1, CI 없음] | N1, NOT_CONFIRMED | True |
| TokenCap2Control / BestToken | 1.024018 [N1, CI 없음] | 1.004038 [N1, CI 없음] | 0.988183 [N1, CI 없음] | 1.002071 [N1, CI 없음] | N1, NOT_CONFIRMED | True |

TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.
NativeStrong과 같은 D 정책의 Native를 모두 검사한다.
안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.

## 모든 개별 Start

| Workload | QPS | Start | Seed | GPU Pair | Arm | Raw Name | 상태 |
|---|---:|---:|---:|---|---|---|---|
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | NativeStrong | FinalC3_Heldout200_q1.8_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | BestLayer | FinalC3_Heldout200_q1.8_s0_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | BestToken | FinalC3_Heldout200_q1.8_s0_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | BestHybrid | FinalC3_Heldout200_q1.8_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q1.8_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q1.8_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q1.8_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q1.8_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon3 | FinalC3_Heldout200_q1.8_s0_NativeCommon3_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 0 | 2026112411 | [0, 1] | Agg | FinalC3_Heldout200_q1.8_s0_Agg_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | NativeStrong | FinalC3_Heldout200_q1.8_s1_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | BestLayer | FinalC3_Heldout200_q1.8_s1_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | BestToken | FinalC3_Heldout200_q1.8_s1_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | BestHybrid | FinalC3_Heldout200_q1.8_s1_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q1.8_s1_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q1.8_s1_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q1.8_s1_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q1.8_s1_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | NativeCommon3 | FinalC3_Heldout200_q1.8_s1_NativeCommon3_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 1 | 2026112412 | [0, 1] | Agg | FinalC3_Heldout200_q1.8_s1_Agg_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | NativeStrong | FinalC3_Heldout200_q1.8_s2_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | BestLayer | FinalC3_Heldout200_q1.8_s2_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | BestToken | FinalC3_Heldout200_q1.8_s2_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | BestHybrid | FinalC3_Heldout200_q1.8_s2_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q1.8_s2_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q1.8_s2_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q1.8_s2_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q1.8_s2_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | NativeCommon3 | FinalC3_Heldout200_q1.8_s2_NativeCommon3_v1 | MEASURED_ONLINE |
| Heldout200 | 1.8 | 2 | 2026112413 | [0, 1] | Agg | FinalC3_Heldout200_q1.8_s2_Agg_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | NativeStrong | FinalC3_Heldout200_q2.8_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | BestLayer | FinalC3_Heldout200_q2.8_s0_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | BestToken | FinalC3_Heldout200_q2.8_s0_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | BestHybrid | FinalC3_Heldout200_q2.8_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q2.8_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | TokenCap2Control | FinalC3_Heldout200_q2.8_s0_TokenCap2Control_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q2.8_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q2.8_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q2.8_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 0 | 2026112411 | [0, 1] | Agg | FinalC3_Heldout200_q2.8_s0_Agg_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | NativeStrong | FinalC3_Heldout200_q2.8_s1_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | BestLayer | FinalC3_Heldout200_q2.8_s1_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | BestToken | FinalC3_Heldout200_q2.8_s1_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | BestHybrid | FinalC3_Heldout200_q2.8_s1_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q2.8_s1_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | TokenCap2Control | FinalC3_Heldout200_q2.8_s1_TokenCap2Control_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q2.8_s1_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q2.8_s1_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q2.8_s1_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 1 | 2026112412 | [0, 1] | Agg | FinalC3_Heldout200_q2.8_s1_Agg_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | NativeStrong | FinalC3_Heldout200_q2.8_s2_NativeStrong_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | BestLayer | FinalC3_Heldout200_q2.8_s2_BestLayer_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | BestToken | FinalC3_Heldout200_q2.8_s2_BestToken_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | BestHybrid | FinalC3_Heldout200_q2.8_s2_BestHybrid_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | BestWholeDeflection | FinalC3_Heldout200_q2.8_s2_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | TokenCap2Control | FinalC3_Heldout200_q2.8_s2_TokenCap2Control_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | NativeCommon0 | FinalC3_Heldout200_q2.8_s2_NativeCommon0_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | NativeCommon1 | FinalC3_Heldout200_q2.8_s2_NativeCommon1_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | NativeCommon2 | FinalC3_Heldout200_q2.8_s2_NativeCommon2_v1 | MEASURED_ONLINE |
| Heldout200 | 2.8 | 2 | 2026112413 | [0, 1] | Agg | FinalC3_Heldout200_q2.8_s2_Agg_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | NativeStrong | FinalC3_Main400_q1.0_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | BestLayer | FinalC3_Main400_q1.0_s0_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | BestToken | FinalC3_Main400_q1.0_s0_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | BestHybrid | FinalC3_Main400_q1.0_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | BestWholeDeflection | FinalC3_Main400_q1.0_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | NativeCommon0 | FinalC3_Main400_q1.0_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | NativeCommon1 | FinalC3_Main400_q1.0_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | NativeCommon2 | FinalC3_Main400_q1.0_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | NativeCommon3 | FinalC3_Main400_q1.0_s0_NativeCommon3_v1 | MEASURED_ONLINE |
| Main400 | 1.0 | 0 | 2026112411 | [6, 7] | Agg | FinalC3_Main400_q1.0_s0_Agg_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | NativeStrong | FinalC3_Main400_q1.8_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | BestLayer | FinalC3_Main400_q1.8_s0_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | BestToken | FinalC3_Main400_q1.8_s0_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | BestHybrid | FinalC3_Main400_q1.8_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | BestWholeDeflection | FinalC3_Main400_q1.8_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon0 | FinalC3_Main400_q1.8_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon1 | FinalC3_Main400_q1.8_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon2 | FinalC3_Main400_q1.8_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | NativeCommon3 | FinalC3_Main400_q1.8_s0_NativeCommon3_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 0 | 2026112411 | [0, 1] | Agg | FinalC3_Main400_q1.8_s0_Agg_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | NativeStrong | FinalC3_Main400_q1.8_s1_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | BestLayer | FinalC3_Main400_q1.8_s1_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | BestToken | FinalC3_Main400_q1.8_s1_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | BestHybrid | FinalC3_Main400_q1.8_s1_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | BestWholeDeflection | FinalC3_Main400_q1.8_s1_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | NativeCommon0 | FinalC3_Main400_q1.8_s1_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | NativeCommon1 | FinalC3_Main400_q1.8_s1_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | NativeCommon2 | FinalC3_Main400_q1.8_s1_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | NativeCommon3 | FinalC3_Main400_q1.8_s1_NativeCommon3_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 1 | 2026112412 | [2, 3] | Agg | FinalC3_Main400_q1.8_s1_Agg_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | NativeStrong | FinalC3_Main400_q1.8_s2_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | BestLayer | FinalC3_Main400_q1.8_s2_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | BestToken | FinalC3_Main400_q1.8_s2_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | BestHybrid | FinalC3_Main400_q1.8_s2_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | BestWholeDeflection | FinalC3_Main400_q1.8_s2_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | NativeCommon0 | FinalC3_Main400_q1.8_s2_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | NativeCommon1 | FinalC3_Main400_q1.8_s2_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | NativeCommon2 | FinalC3_Main400_q1.8_s2_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | NativeCommon3 | FinalC3_Main400_q1.8_s2_NativeCommon3_v1 | MEASURED_ONLINE |
| Main400 | 1.8 | 2 | 2026112413 | [4, 5] | Agg | FinalC3_Main400_q1.8_s2_Agg_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | NativeStrong | FinalC3_Main400_q2.8_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | BestLayer | FinalC3_Main400_q2.8_s0_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | BestToken | FinalC3_Main400_q2.8_s0_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | BestHybrid | FinalC3_Main400_q2.8_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | BestWholeDeflection | FinalC3_Main400_q2.8_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | TokenCap2Control | FinalC3_Main400_q2.8_s0_TokenCap2Control_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon0 | FinalC3_Main400_q2.8_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon1 | FinalC3_Main400_q2.8_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | NativeCommon2 | FinalC3_Main400_q2.8_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 0 | 2026112411 | [0, 1] | Agg | FinalC3_Main400_q2.8_s0_Agg_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | NativeStrong | FinalC3_Main400_q2.8_s1_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | BestLayer | FinalC3_Main400_q2.8_s1_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | BestToken | FinalC3_Main400_q2.8_s1_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | BestHybrid | FinalC3_Main400_q2.8_s1_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | BestWholeDeflection | FinalC3_Main400_q2.8_s1_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | TokenCap2Control | FinalC3_Main400_q2.8_s1_TokenCap2Control_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | NativeCommon0 | FinalC3_Main400_q2.8_s1_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | NativeCommon1 | FinalC3_Main400_q2.8_s1_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | NativeCommon2 | FinalC3_Main400_q2.8_s1_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 1 | 2026112412 | [2, 3] | Agg | FinalC3_Main400_q2.8_s1_Agg_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | NativeStrong | FinalC3_Main400_q2.8_s2_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | BestLayer | FinalC3_Main400_q2.8_s2_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | BestToken | FinalC3_Main400_q2.8_s2_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | BestHybrid | FinalC3_Main400_q2.8_s2_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | BestWholeDeflection | FinalC3_Main400_q2.8_s2_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | TokenCap2Control | FinalC3_Main400_q2.8_s2_TokenCap2Control_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | NativeCommon0 | FinalC3_Main400_q2.8_s2_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | NativeCommon1 | FinalC3_Main400_q2.8_s2_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | NativeCommon2 | FinalC3_Main400_q2.8_s2_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 2.8 | 2 | 2026112413 | [4, 5] | Agg | FinalC3_Main400_q2.8_s2_Agg_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | NativeStrong | FinalC3_Main400_q4.5_s0_NativeStrong_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | BestLayer | FinalC3_Main400_q4.5_s0_BestLayer_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | BestToken | FinalC3_Main400_q4.5_s0_BestToken_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | BestHybrid | FinalC3_Main400_q4.5_s0_BestHybrid_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | BestWholeDeflection | FinalC3_Main400_q4.5_s0_BestWholeDeflection_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | TokenCap2Control | FinalC3_Main400_q4.5_s0_TokenCap2Control_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | NativeCommon0 | FinalC3_Main400_q4.5_s0_NativeCommon0_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | NativeCommon1 | FinalC3_Main400_q4.5_s0_NativeCommon1_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | NativeCommon2 | FinalC3_Main400_q4.5_s0_NativeCommon2_v1 | MEASURED_ONLINE |
| Main400 | 4.5 | 0 | 2026112411 | [6, 7] | Agg | FinalC3_Main400_q4.5_s0_Agg_v1 | MEASURED_ONLINE |

Raw Name의 `raw/<name>/configuration.json`, `requests.jsonl`과 `actual_queries_*.json`을 함께 사용한다.
누락되거나 실패한 Start를 보간하지 않는다. Actual Natural Output 수가 다르면 그 차이도 보존한다.
