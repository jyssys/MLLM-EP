# LongBench400: C0 후보의 Controlled Continuous Bridge

상태: `MEASURED_ONLINE_CONTROLLED_OUTPUT`.
54개 Start를 완료했다. 안전성 및 성능 판정은 아래 표에 분리했다.

## 목적과 범위

기존 C0의 I3/Output64에서 작은 Iso-work Hybrid 이득을 확인했다.
그 결과가 실제 Continuous Scheduler에서도 유지되는지 검사한다.
기존 Natural400 결과와 고정 Output 결과는 별도로 보고한다.
유리한 C0 Cell을 선택했다는 사실도 명시한다.

반례였던 Output16과 공통 D Admission 변경도 함께 검사한다.
모든 조건을 사전에 동결했다. 불리한 조건을 결과 뒤에 제외하지 않는다.
Main Natural400의 Static 후보를 이 Bridge 결과로 다시 고르지 않는다.

## 동결한 실행 계약

모델은 Qwen3-8B다. Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16, BI0, Exact Causal Attention을 사용한다. 모든 Decode는 Full36 Layers다.
모든 Worker는 B8192/S1024와 KV42GiB/Block128을 유지한다.

데이터는 원본 LongBench Prompt 16개다. Input은 I3의 20K–36K 구간이다.
Dataset은 `iso8_input3_heldout_exact16.json`이다.
SHA256은 `6429dde7a7dd44156636e7f9c8f041d7b7d0b6f91c80bc2cce1697dc51e8c780`이다.
Context를 늘리거나 Filler를 넣지 않았다.

| 조건 | Controlled Output | D Running Cap | Offered QPS | 독립 Start |
|---|---:|---:|---:|---:|
| 짧은 Output 반례 | 16 | 1 | 1.253873 | 3 |
| 기존 C0 후보 | 64 | 1 | 1.253873 | 3 |
| 같은 Output의 Admission 변경 | 64 | 2 | 1.253873 | 3 |

각 조건은 같은 여섯 Arm을 사용한다. 총 예정 성능 Start는 54회다.
Source Capsule SHA256은 `a2d5ebc4cb39eff66a8fbba3b4c0b83a8957d79e50c11ff94a529b2cdb2ca6e3`이다.
GPU Pair는 모든 새 Start에서 4/5다.
Arrival Seed는 2026112551/2026112552/2026112553이다.
같은 Seed에서는 Prompt, Arrival Offset과 Output Count가 같다.
Output은 `ignore_eos=True`로 정확히 제어한다. 이를 Natural EOS 결과로 표시하지 않는다.

## 대조군

| Arm | Runtime 역할 | Source Credit | D 정책 |
|---|---|---:|---|
| NativeStrong | 변경하지 않은 Full-model Compile 기준선 | 0 | 원래 Scheduler |
| NativeCommon | 같은 D 정책을 적용한 강한 Native | 0 | Compute/READ Deferral 및 Cap1/2 |
| Layer Last8 | Row-aware Mode0 Partial Path | 8 | 공통 D 정책 |
| Stock Token Last8-equivalent | 원래 Full-model Compile 유지 | 8 | 공통 D 정책 |
| Token Mode0 Control | 진단용 Generic Row-aware 경로 | 8 | 공통 D 정책 |
| Hybrid Last4+Suffix 약 L/8 | Genuine 2D Mode0 Path | 8 | 공통 D 정책 |

Stock Token을 느린 Mode0 Control로 교체하지 않는다.
두 Token 경로를 모두 남긴다.
이 두 Runtime Family는 Adapter와 Numerical Path도 다를 수 있다.
따라서 그 차이를 Compile Flag 하나의 순수 효과로 해석하지 않는다.

Source Credit은 Actual GPU Batch가 아니다.
D Running Cap도 Actual Batch 수를 강제하지 않는다.
P는 Continuous Scheduler를 사용한다.
D는 공통 Running Cap 안에서 정상 Scheduling과 Batching을 수행한다.
C0의 Whole-request Barrier를 사용하지 않는다.

## 작업량과 정합성

Layer8, 두 Token8 경로와 Hybrid4는 같은 Logical P/D Rows를 사용한다.
16개 요청의 P Rows는 각각 12,676,608이다. D Rows는 각각 3,621,888이다.
Attention Work와 Transfer Payload는 같지 않다.
Actual Batch와 Model-input Padding도 별도로 감사한다.
같은 Logical Rows를 같은 Physical Kernel Work라고 부르지 않는다.

새 Execution/Admission 조합은 Natural32/400 정합성을 먼저 통과해야 한다.
같은 Source와 Signature의 Full400 Proof가 있으면 정확히 확인한 뒤 재사용한다.
Mode0 Token Control도 별도 새 조합 Proof를 요구한다.
KV Import, Query/Layer Coverage, Hidden Lifetime과 Task Quality를 검사한다.
정합성 실패 조합의 성능은 실행하지 않는다.

## 판정 기준과 한계

새 C3 Cell 안에서는 Paired Independent Start 3회의 비율 CI를 계산한다.
Bootstrap 27개 재표본을 완전 열거한다.
Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않는다.
NativeStrong과 NativeCommon 대비 ITL p99/TPOT p95의 CI 상한≤1.05를 각각 검사한다.
Absolute Decode SLO도 별도로 유지한다.

Hybrid가 같은 Cut의 Pure Axis를 이겨도 전역 Best-tuned Axis를 이겼다는 뜻은 아니다.
Natural400의 강한 Static 비교는 별도로 수행한다.
이 16-request 유한 Cohort의 처리량을 Sustainable Capacity라고 부르지 않는다.

기존 C0는 다른 Source, GPU6/7과 다른 Arrival Seed를 사용했다.
따라서 C0와 새 C3의 차이에 Paired CI를 계산하지 않는다.
새 C3 안의 Cap1→2 Intervention은 같은 Source, Pair, Prompt와 Arrival을 유지한다.
이 Intervention도 GPU-exclusive Operator 비용을 직접 분해하지는 않는다.

## 재현 자료

- [동결한 Bridge 설계](dense_ar_longbench400_2d_20261121/configs/controlled_continuous_I3_bridge_frozen_v1.json)
- [새 C3 분석](dense_ar_longbench400_2d_20261121/analysis/controlled_continuous_I3_bridge_results_v1.json)
- [기존 C0 Exact Output 반증](dense_ar_longbench400_2d_20261121/analysis/iso8_I3_exact_output_counterfactual_v1.json)

실행기는 `controlled_continuous_bridge.py`다.
분석기는 `controlled_bridge_analysis.py`다.
정합성, 성능, Decode Gate와 최종 Serving 판정은 서로 구분한다.

## 완료한 C3 측정

각 값은 독립 Start 3개의 Metric 평균이다. 모든 조건은 N16이다.
O16은 Start당 256 Tokens다. O64는 Start당 1024 Tokens다.
CI는 같은 Seed의 Server Start를 재표본했다. 27개 조합을 완전 열거했다.
Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.

| Output / D-cap | Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(output tokens/s) |
|---|---|---:|---:|---:|---:|
| 16/1 | NativeStrong | 5990.934762 | 8.834114 | 7.960411 | 10.101162 |
| 16/1 | NativeCommon | 5998.110018 | 8.743584 | 7.910327 | 9.766540 |
| 16/1 | Layer | 3827.557570 | 9.269328 | 8.284512 | 18.834064 |
| 16/1 | StockToken | 3090.121936 | 9.086028 | 8.103135 | 20.007729 |
| 16/1 | TokenMode0Control | 3245.818788 | 9.026437 | 8.236663 | 19.731816 |
| 16/1 | Hybrid | 3462.432325 | 9.183098 | 8.294218 | 19.285070 |
| 64/1 | NativeStrong | 6016.105228 | 8.775282 | 8.294307 | 39.339836 |
| 64/1 | NativeCommon | 5963.214548 | 8.772384 | 8.949966 | 38.243973 |
| 64/1 | Layer | 4153.714367 | 9.180973 | 8.536141 | 69.581230 |
| 64/1 | StockToken | 4117.757585 | 11.416515 | 8.349869 | 65.091528 |
| 64/1 | TokenMode0Control | 4405.101895 | 12.224034 | 8.625944 | 66.143871 |
| 64/1 | Hybrid | 4335.582235 | 11.137192 | 8.585070 | 65.309085 |
| 64/2 | NativeStrong | 6028.638995 | 8.798262 | 8.276803 | 39.355838 |
| 64/2 | NativeCommon | 6016.425827 | 8.779430 | 8.267584 | 38.079843 |
| 64/2 | Layer | 4267.599970 | 9.591968 | 8.549697 | 69.012640 |
| 64/2 | StockToken | 4139.416271 | 9.975171 | 8.367431 | 66.620802 |
| 64/2 | TokenMode0Control | 4457.308656 | 11.415907 | 8.612205 | 66.201094 |
| 64/2 | Hybrid | 4290.757817 | 11.368101 | 8.567740 | 67.123393 |

### 같은 조건의 비교와 Decode Gate

| Output / D-cap | Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |
|---|---|---|---|---|---|---|---|
| 16/1 | Layer/NativeStrong | 0.638892 [0.614012,0.673988] | 1.049265 [1.046805,1.051544] | 1.040714 [1.036832,1.045948] | 1.864544 [1.577016,2.164616] | False | True |
| 16/1 | Layer/NativeCommon | 0.638127 [0.614301,0.672262] | 1.060129 [1.050559,1.067696] | 1.047303 [1.046179,1.048781] | 1.928428 [1.579772,2.371584] | False | True |
| 16/1 | Layer/StockToken | 1.238643 [1.150068,1.319969] | 1.020174 [1.011246,1.031168] | 1.022383 [1.014183,1.029650] | 0.941339 [0.916775,0.976904] | True | True |
| 16/1 | Layer/TokenMode0Control | 1.179227 [1.113201,1.239878] | 1.026909 [1.024734,1.029751] | 1.005809 [1.001988,1.010798] | 0.954502 [0.933484,0.983056] | True | True |
| 16/1 | StockToken/NativeStrong | 0.515800 [0.465193,0.586711] | 1.028516 [1.015177,1.039850] | 1.017929 [1.012419,1.023745] | 1.980735 [1.615660,2.361121] | True | True |
| 16/1 | StockToken/NativeCommon | 0.515183 [0.465410,0.585207] | 1.039165 [1.032178,1.047396] | 1.024374 [1.016718,1.032351] | 2.048600 [1.618479,2.586891] | True | True |
| 16/1 | TokenMode0Control/NativeStrong | 0.541788 [0.495243,0.605896] | 1.021770 [1.018829,1.025103] | 1.034703 [1.034288,1.035051] | 1.953420 [1.605145,2.318866] | True | True |
| 16/1 | TokenMode0Control/NativeCommon | 0.541140 [0.495474,0.604344] | 1.032350 [1.023597,1.040608] | 1.041254 [1.035173,1.046252] | 2.020349 [1.607947,2.540635] | True | True |
| 16/1 | TokenMode0Control/StockToken | 1.050385 [1.031832,1.064605] | 0.993441 [0.984359,1.005409] | 1.016479 [1.011046,1.021602] | 0.986210 [0.979267,0.994791] | True | True |
| 16/1 | Hybrid/NativeStrong | 0.577945 [0.543026,0.621225] | 1.039504 [1.029399,1.048201] | 1.041933 [1.039307,1.045020] | 1.909193 [1.587365,2.247345] | True | True |
| 16/1 | Hybrid/NativeCommon | 0.577254 [0.543278,0.619634] | 1.050267 [1.038571,1.065165] | 1.048530 [1.040467,1.056698] | 1.974606 [1.590137,2.462272] | False | True |
| 16/1 | Hybrid/Layer | 0.904606 [0.884370,0.921670] | 0.990697 [0.980050,1.000915] | 1.001172 [0.994541,1.007548] | 1.023946 [1.006284,1.038216] | True | True |
| 16/1 | Hybrid/StockToken | 1.120484 [1.053338,1.174115] | 1.010683 [0.993421,1.031227] | 1.023581 [1.015203,1.032203] | 0.963881 [0.949123,0.983897] | True | True |
| 16/1 | Hybrid/TokenMode0Control | 1.066736 [1.020771,1.102865] | 1.017356 [1.009206,1.025671] | 1.006988 [1.004111,1.010376] | 0.977359 [0.969157,0.989183] | True | True |
| 64/1 | Layer/NativeStrong | 0.690432 [0.655981,0.724735] | 1.046231 [1.019585,1.077897] | 1.029157 [1.023422,1.034188] | 1.768722 [1.519240,2.051539] | False | True |
| 64/1 | Layer/NativeCommon | 0.696556 [0.659884,0.734697] | 1.046577 [1.015697,1.087578] | 0.953762 [0.864505,1.034169] | 1.819404 [1.611271,2.047263] | False | True |
| 64/1 | Layer/StockToken | 1.008732 [0.991199,1.030300] | 0.804183 [0.713833,0.947341] | 1.022308 [1.017254,1.028119] | 1.068975 [1.011488,1.137771] | True | True |
| 64/1 | Layer/TokenMode0Control | 0.942933 [0.924793,0.963465] | 0.751059 [0.706006,0.812355] | 0.989589 [0.986045,0.992742] | 1.051968 [1.030675,1.082217] | True | True |
| 64/1 | StockToken/NativeStrong | 0.684456 [0.641355,0.721205] | 1.300986 [1.093341,1.463693] | 1.006699 [1.001703,1.012478] | 1.654596 [1.395565,2.000999] | False | False |
| 64/1 | StockToken/NativeCommon | 0.690526 [0.645171,0.731112] | 1.301415 [1.099195,1.454338] | 0.932950 [0.848386,1.010247] | 1.702008 [1.480258,1.996828] | False | False |
| 64/1 | TokenMode0Control/NativeStrong | 0.732218 [0.709299,0.752202] | 1.393008 [1.278118,1.485911] | 1.039984 [1.031696,1.046717] | 1.681346 [1.456889,1.958175] | False | True |
| 64/1 | TokenMode0Control/NativeCommon | 0.738713 [0.713520,0.762531] | 1.393468 [1.262251,1.504214] | 0.963796 [0.876227,1.043219] | 1.729524 [1.545274,1.954093] | False | True |
| 64/1 | TokenMode0Control/StockToken | 1.069782 [1.042990,1.106575] | 1.070733 [0.879521,1.342393] | 1.033063 [1.028288,1.038189] | 1.016167 [0.977798,1.052829] | False | True |
| 64/1 | Hybrid/NativeStrong | 0.720663 [0.691912,0.743658] | 1.269155 [1.122443,1.391210] | 1.035056 [1.029862,1.041117] | 1.660126 [1.478527,1.872436] | False | False |
| 64/1 | Hybrid/NativeCommon | 0.727055 [0.697643,0.749089] | 1.269574 [1.136177,1.373781] | 0.959229 [0.866890,1.042507] | 1.707696 [1.568265,1.868534] | False | False |
| 64/1 | Hybrid/Layer | 1.043784 [1.006882,1.069321] | 1.213073 [1.045364,1.352555] | 1.005732 [1.002727,1.008062] | 0.938602 [0.911392,0.973677] | False | False |
| 64/1 | Hybrid/StockToken | 1.052899 [1.016506,1.088930] | 0.975533 [0.786695,1.236895] | 1.028168 [1.020029,1.036408] | 1.003342 [0.936411,1.059468] | False | False |
| 64/1 | Hybrid/TokenMode0Control | 0.984218 [0.967055,1.000025] | 0.911090 [0.755541,1.089585] | 0.995262 [0.988847,1.000455] | 0.987379 [0.956369,1.014941] | False | False |
| 64/2 | Layer/NativeStrong | 0.707888 [0.674701,0.736584] | 1.090212 [1.073143,1.109474] | 1.032971 [1.029773,1.035930] | 1.753555 [1.508250,2.034367] | False | True |
| 64/2 | Layer/NativeCommon | 0.709325 [0.674743,0.737608] | 1.092550 [1.068612,1.114347] | 1.034123 [1.033151,1.035303] | 1.812314 [1.509738,2.229236] | False | True |
| 64/2 | Layer/StockToken | 1.030967 [1.008864,1.056008] | 0.961584 [0.879746,1.053545] | 1.021783 [1.013179,1.030638] | 1.035902 [1.010207,1.059691] | False | True |
| 64/2 | Layer/TokenMode0Control | 0.957439 [0.933119,0.978347] | 0.840228 [0.742339,1.010572] | 0.992742 [0.987153,0.998996] | 1.042470 [1.029259,1.062308] | True | True |
| 64/2 | StockToken/NativeStrong | 0.686625 [0.639027,0.730140] | 1.133766 [1.018798,1.261876] | 1.010950 [1.005138,1.016380] | 1.692781 [1.489812,1.933526] | False | False |
| 64/2 | StockToken/NativeCommon | 0.688019 [0.639611,0.731151] | 1.136198 [1.014543,1.267364] | 1.012077 [1.004529,1.019715] | 1.749503 [1.491289,2.118816] | False | False |
| 64/2 | TokenMode0Control/NativeStrong | 0.739356 [0.717121,0.764818] | 1.297519 [1.099351,1.455682] | 1.040523 [1.031426,1.047783] | 1.682116 [1.452975,1.960359] | False | True |
| 64/2 | TokenMode0Control/NativeCommon | 0.740857 [0.717761,0.765877] | 1.300302 [1.098552,1.468839] | 1.041683 [1.034683,1.047480] | 1.738481 [1.454413,2.148401] | False | True |
| 64/2 | TokenMode0Control/StockToken | 1.076796 [1.045210,1.127403] | 1.144432 [0.882064,1.401056] | 1.029253 [1.015313,1.041090] | 0.993700 [0.975189,1.013817] | False | True |
| 64/2 | Hybrid/NativeStrong | 0.711729 [0.684255,0.747377] | 1.292085 [1.091670,1.464590] | 1.035151 [1.028813,1.040002] | 1.705551 [1.421852,2.016349] | False | False |
| 64/2 | Hybrid/NativeCommon | 0.713174 [0.684865,0.748405] | 1.294856 [1.090887,1.477865] | 1.036305 [1.032187,1.039373] | 1.762701 [1.423246,2.209339] | False | False |
| 64/2 | Hybrid/Layer | 1.005426 [0.977359,1.031402] | 1.185169 [0.986834,1.355336] | 1.002110 [0.998605,1.005146] | 0.972625 [0.938310,1.004184] | False | False |
| 64/2 | Hybrid/StockToken | 1.036561 [0.999455,1.084776] | 1.139640 [0.882317,1.374887] | 1.023939 [1.012236,1.034689] | 1.007544 [0.947962,1.064133] | False | False |
| 64/2 | Hybrid/TokenMode0Control | 0.962634 [0.950580,0.979656] | 0.995812 [0.976792,1.012920] | 0.994837 [0.992232,0.997600] | 1.013932 [0.965764,1.066781] | True | False |

성능 우위와 Decode 보호를 별도로 검사한다.
Layer와 Hybrid의 Serving 주장은 두 Native 대조군의 Gate를 모두 요구한다.
Mode0 Token 대비 우위만으로 강한 Stock Token 대비 우위를 주장하지 않는다.

### Actual Geometry와 Padding

| Output / D-cap | Arm | P/D 최대 Request Batch | P/D 추가 Prefill Padding Query 합 |
|---|---|---|---|
| 16/1 | NativeStrong | 2/1 | 0/0 |
| 16/1 | NativeCommon | 2/1 | 0/0 |
| 16/1 | Layer | 2/1 | 0/0 |
| 16/1 | StockToken | 2/1 | 10/0 |
| 16/1 | TokenMode0Control | 2/1 | 0/0 |
| 16/1 | Hybrid | 2/1 | 0/0 |
| 64/1 | NativeStrong | 2/1 | 0/0 |
| 64/1 | NativeCommon | 2/1 | 0/0 |
| 64/1 | Layer | 2/1 | 0/0 |
| 64/1 | StockToken | 2/1 | 10/0 |
| 64/1 | TokenMode0Control | 2/1 | 0/0 |
| 64/1 | Hybrid | 2/1 | 0/0 |
| 64/2 | NativeStrong | 2/1 | 0/0 |
| 64/2 | NativeCommon | 2/1 | 0/0 |
| 64/2 | Layer | 2/2 | 0/0 |
| 64/2 | StockToken | 2/2 | 10/0 |
| 64/2 | TokenMode0Control | 2/2 | 0/0 |
| 64/2 | Hybrid | 2/2 | 0/0 |

Padding 합은 3개 Start의 Model-input Query Slot 합이다.
이를 Partial Adapter의 실제 GEMM Rows 또는 GPU-exclusive 시간으로 환산하지 않는다.

### 같은 Request의 Host 경계

단위는 ms다. 값은 Arm에서 Stock Token을 뺀 평균이다.
Prepare는 Host 입력 준비 경계다. GPU Kernel 시작이 아니다.
여섯 구간은 Mean TTFT 차이에 합산된다. TTFT p95는 합산하지 않는다.

| Output / D-cap | Arm − StockToken | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 | 이후 Residency 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16/1 | Layer | 491.077983 | 91.336817 | -0.030677 | -4.167957 | -52.593155 | 0.007706 | 525.630717 | 3.012390 |
| 16/1 | Hybrid | 262.382070 | 75.456237 | -0.032921 | -7.523866 | -19.375260 | -0.001480 | 310.904781 | 2.806982 |
| 16/1 | TokenMode0Control | 92.389828 | 35.585338 | -0.141320 | -12.786027 | -3.963271 | -0.019179 | 111.065368 | 2.476693 |
| 64/1 | Layer | 484.740717 | 94.431893 | -0.812022 | -452.191103 | -53.482872 | 0.129315 | 72.815929 | 13.583861 |
| 64/1 | Hybrid | 266.513558 | 72.316344 | -0.980789 | -147.441734 | -12.457925 | 0.013739 | 177.963192 | 15.962377 |
| 64/1 | TokenMode0Control | 82.279369 | 35.255341 | -0.982424 | 88.679713 | 5.202655 | 0.013732 | 210.448385 | 17.097228 |
| 64/2 | Layer | 505.172084 | 93.038500 | 0.429944 | -404.011656 | -73.251606 | -0.916843 | 120.460424 | 17.098814 |
| 64/2 | Hybrid | 268.028734 | 71.109292 | -0.330791 | -166.646289 | -4.220155 | -0.969913 | 166.970877 | 17.032322 |
| 64/2 | TokenMode0Control | 85.518394 | 35.810787 | 0.527011 | 51.718928 | 47.568533 | -0.946065 | 220.197588 | 18.202008 |

이후 Residency는 첫 Output 이후의 Client 관측 구간이다.
이를 TTFT 또는 순수 Decode GPU 시간에 더하지 않는다.

### Output 및 Admission Intervention

| Intervention | Arm | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] |
|---|---|---|---|---|---|
| O64/O16, D-cap1 | NativeStrong | 1.004201 [1.000201,1.007113] | 0.993340 [0.980732,1.005167] | 1.041945 [1.036769,1.046267] | 3.894585 [3.887154,3.904350] |
| O64/O16, D-cap1 | NativeCommon | 0.994182 [0.989337,0.998220] | 1.003294 [0.992016,1.012972] | 1.131428 [1.044604,1.247970] | 3.915816 [3.670539,4.286400] |
| O64/O16, D-cap1 | Layer | 1.085213 [1.053520,1.117732] | 0.990468 [0.963638,1.026967] | 1.030374 [1.028996,1.032000] | 3.694435 [3.575083,3.794765] |
| O64/O16, D-cap1 | StockToken | 1.332555 [1.206869,1.452189] | 1.256491 [1.053755,1.415677] | 1.030449 [1.017832,1.042029] | 3.253319 [2.937832,3.514689] |
| O64/O16, D-cap1 | TokenMode0Control | 1.357162 [1.237718,1.451781] | 1.354248 [1.228742,1.457790] | 1.047262 [1.039676,1.056008] | 3.352143 [3.096034,3.613284] |
| O64/O16, D-cap1 | Hybrid | 1.252178 [1.172408,1.316411] | 1.212792 [1.085358,1.306469] | 1.035067 [1.029802,1.039055] | 3.386510 [3.150120,3.664746] |
| Cap2/Cap1, O64 | NativeCommon | 1.008923 [1.004432,1.013322] | 1.000803 [0.998841,1.003154] | 0.923756 [0.836822,1.002044] | 0.995708 [0.909543,1.062588] |
| Cap2/Cap1, O64 | Layer | 1.027418 [1.010898,1.040333] | 1.044766 [1.003304,1.088663] | 1.001588 [1.000463,1.002723] | 0.991828 [0.988160,0.995809] |
| Cap2/Cap1, O64 | StockToken | 1.005260 [1.001154,1.011219] | 0.873749 [0.792200,0.967081] | 1.002103 [0.995943,1.009449] | 1.023494 [0.965073,1.074770] |
| Cap2/Cap1, O64 | TokenMode0Control | 1.011851 [1.000024,1.022734] | 0.933890 [0.869455,0.977112] | 0.998407 [0.995904,1.000765] | 1.000865 [0.996825,1.006811] |
| Cap2/Cap1, O64 | Hybrid | 0.989661 [0.950860,1.035597] | 1.020733 [0.794122,1.304298] | 0.997981 [0.993603,1.002764] | 1.027780 [0.960351,1.088767] |

Output Intervention은 Cap에 따른 KV Reservation 변화도 포함한다.
Output Length와 Decode Residency만의 순수 효과로 단정하지 않는다.
Admission Intervention은 Source, Prompt, Arrival, Output과 Axis를 고정한다.
D-cap2는 실제 Decode Batch2를 강제하지 않는다.

### 한계와 재현 자료

이 결과는 기존 C0에서 선택한 작은 Prompt 집합의 Bridge다.
전체 Natural400 또는 보지 않은 새로운 Request Population의 결과가 아니다.
C0와 C3는 Source, Pair와 Seed가 다르다. 둘 사이의 차이는 단일 정책의 인과 효과가 아니다.
강한 Natural400 Static 비교와 Dynamic Window 검증을 이 결과로 대체하지 않는다.
실제 GPU Operator별 C3 비용 비중과 미래 Decode-safe Window는 이 실험으로 식별하지 못했다.

- [전체 Metric와 Gate](dense_ar_longbench400_2d_20261121/analysis/controlled_continuous_I3_bridge_results_v1.json)
- [같은 Request의 경계와 Intervention](dense_ar_longbench400_2d_20261121/analysis/controlled_C3_bridge_causal_boundaries_v1.json)
