# Dense AR PD Joint SLO 실패 분해

## 판정과 범위

기존 Main400 Natural N3의 판정은 `SLO_FAILURE_CONFIRMED`다.
TTFT 분포와 긴 출력의 드문 ITL 실패가 Joint SLO 차이를 설명한다.
아래 계산은 Metric 분해다. Runtime 원인의 인과 기여율은 아니다.

근거 Commit은 `0e2e062260103feb53bd0afbb0ac964ebb027cd9`다.
Dataset은 동결한 LongBench400-v2 Main400이다.
Dataset SHA256은 `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`다.
모델은 Qwen3-8B, Original BF16, BI0다.
모델 Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
QPS는 2.0이다. Natural EOS와 Output Cap2048을 유지했다.
각 Method는 독립 Start3회다. 각 Start는 같은 400개 요청을 사용했다.
Arrival Seed는 `2026112271/2026112272/2026112273`이다.
아래 합계의 분모는 **1200개 요청 관측값**이다. 독립 요청 1200개나 독립 Start1200회가 아니다.

앞부분은 기존 실측을 재계산한다. 뒷부분은 새 Natural·Replay를 별도로 보고한다.
새 캠페인의 정책 실측과 기존 수치를 합쳐 평균을 계산하지 않는다.
후속 정책 비교는 [Controlled Intervention 보고서](dense_ar_pd_critical_path_controlled_interventions.md)에서 별도 Family로 보고한다.

## Metric 계약

요청 i의 출력 Token 수를 nᵢ, 측정 구간을 W라고 둔다.

- `TTFT_PASS`: 예정 Arrival부터 Client 첫 Token까지 ≤5000ms.
- `ITL_PASS`: 해당 요청의 모든 Client-visible ITL이 ≤34ms.
- `TPOT_PASS`: 요청 TPOT가 ≤33ms/token.
- `JOINT_PASS = TTFT_PASS AND ITL_PASS AND TPOT_PASS`.
- `Raw tok/s = Σnᵢ / W`.
- `SLO-goodput = Σ[nᵢ × JOINT_PASSᵢ] / W`.
- `Joint Attainment = ΣJOINT_PASSᵢ / 요청 수`.

부등호는 모두 경계를 포함한다.
TPOT의 종료 경계는 마지막 Token이 아니라 완료 통지 시각이다.
동시에 전달된 여러 Token의 추가 ITL은 0ms다.
출력 1개인 요청의 ITL 조건은 참이다. 해당 요청의 TPOT는 `None`이며 통과한다.
출력이 0개인 완료는 유효하지 않다.
Client는 이 실험의 Coordinator 수신 경계다. HTTP 전달 시간을 추가로 측정한 값이 아니다.

각 Start에서 Metric을 계산한 뒤 산술평균했다.
Start별 비율의 평균과 전체 Token 합계의 비율을 혼동하지 않는다.
Raw Throughput은 Drain을 포함한 유한 Cohort 지표다. 지속 가능한 Capacity가 아니다.

## 원래 표의 재현

`HISTORICAL_RAW_RECALCULATION`이다.
원래 15개 실행의 요청 Raw에서 각 Start의 Metric을 재계산했다.
주어진 일곱 Metric과 반올림 허용오차 내에서 일치했다.

| Method | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Raw tok/s | SLO tok/s | Joint % |
|---|---:|---:|---:|---:|---:|---:|
| Agg DP2 | 1477.006 | 277.839 | 141.579 | 118.693 | 37.482 | 52.00 |
| NativeStrong | 8161.605 | 18.723 | 8.968 | 109.402 | 70.139 | 73.50 |
| Strong Stock Token | 11483.980 | 9.478 | 8.438 | 114.035 | 69.358 | 60.75 |
| Layer | 11679.085 | 8.538 | 7.955 | 112.677 | 49.029 | 46.42 |
| Hybrid | 11767.609 | 8.572 | 8.050 | 105.978 | 49.802 | 46.50 |

전체 요청을 합친 p95는 위 Start-mean p95와 다르다.
예를 들어 Layer의 pooled p95는 14874.581ms다. Start-mean은 11679.085ms다.
두 집계 모두 [집계 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/metrics_start_mean_and_pooled.csv)에 보존했다.

Layer/Stock의 TTFT p95 비율은 1.01699다. Start-level 95% CI는 [0.78620, 1.21991]이다.
Hybrid/Stock은 1.02470 [0.73875, 1.39168]이다.
따라서 이 Natural N3에서 두 Method의 p95 차이를 통계적으로 확정하지 않는다.
반면 Goodput 비율은 Layer 0.70691 [0.57097, 0.82827], Hybrid 0.71805 [0.38935, 0.91413]이다.
CI는 paired Start3회에서 구한 27개 경험적 Bootstrap 재표본이다.
Pointwise CI다. 여러 비교에 대한 동시 보정은 하지 않았다.

## 여덟 실패 조합

각 셀은 `요청 관측 수 / 출력 Token 수`다.
요청 비율은 각 Method의 Count를 1200으로 나눈다.
Token 비율은 각 Method의 전체 Generated Token 합계로 나눈다.
각 범주의 Rate는 해당 Start의 Token 수를 해당 Start의 W로 나눈다.
Rate를 구할 때 아래 합계를 평균 W로 나누지 않는다.

| 범주 | Agg | NativeStrong | Stock Token | Layer | Hybrid |
|---|---:|---:|---:|---:|---:|
| Joint PASS | 624 / 21855 | 882 / 41348 | 729 / 40891 | 557 / 29404 | 558 / 29995 |
| TTFT Only Fail | 0 / 0 | 266 / 12030 | 467 / 24994 | 634 / 33079 | 625 / 30754 |
| ITL Only Fail | 358 / 41977 | 33 / 7386 | 4 / 462 | 5 / 3498 | 8 / 774 |
| TPOT Only Fail | 1 / 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| TTFT + ITL Fail | 0 / 0 | 19 / 2780 | 0 / 0 | 4 / 380 | 9 / 936 |
| TTFT + TPOT Fail | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ITL + TPOT Fail | 217 / 5118 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| All Fail | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 전체 | 1200 / 68952 | 1200 / 63544 | 1200 / 66347 | 1200 / 66361 | 1200 / 62459 |

Start별 Count, Fraction, Token Fraction, Rate는 [8-category CSV](dense_ar_native_ttft_joint_slo_20261124/figures/slo_eight_categories.csv)에 있다.
요청별 Boolean은 [Main400 Request CSV](dense_ar_native_ttft_joint_slo_20261124/figures/main400_request_slo_failures.csv)에 있다.
실패 요청은 삭제하지 않았다.

Layer의 TTFT 실패는 638/1200이다. Stock Token은 467/1200이다.
Layer의 ITL 실패는 9/1200이다. Stock Token은 4/1200이다.
Hybrid의 TTFT 실패는 634/1200이다. ITL 실패는 17/1200이다.
네 PD Method의 TPOT 실패는 모두 0/1200이다.
따라서 이 Family에서 TPOT가 Layer/Hybrid의 Joint 하락을 설명한다는 가설은 반증된다.

## p95가 비슷해도 5초 통과율은 다르다

| Method | TTFT Mean ms | TTFT p50 ms | TTFT p95 ms | TTFT-only Attainment % |
|---|---:|---:|---:|---:|
| Stock Token | 5006.159 | 4064.664 | 11483.980 | 61.08 |
| Layer | 6103.797 | 6265.086 | 11679.085 | 46.83 |
| Hybrid | 6622.302 | 6296.245 | 11767.609 | 47.17 |

Layer와 Stock Token의 p95 차이는 195.105ms다.
그러나 p50 차이는 2200.422ms다.
Stock의 p50는 5초 미만이다. Layer와 Hybrid의 p50는 5초를 초과한다.
p95 한 점은 5초 경계 근처의 요청 수를 나타내지 않는다.
이 차이는 [TTFT CDF Figure](dense_ar_native_ttft_joint_slo_20261124/figures/method_ttft_CDF.png)와 [Threshold CSV](dense_ar_native_ttft_joint_slo_20261124/figures/ttft_threshold_CDF.csv)로 재현한다.

## 더 낮은 ITL p99와 모든-ITL 통과는 다르다

ITL p99는 Token 간격의 분포다.
모든-ITL 통과율은 요청마다 최대 간격을 검사한다.
긴 요청은 간격이 많다. 드문 큰 간격 하나만 있어도 그 요청의 모든 출력 Token이 Goodput에서 제외된다.
이는 가능성에 관한 일반 설명이다. 아래 Count와 Token 수는 실제 관측이다.

Layer의 ITL-only 실패 5개가 3498개 Token을 제외했다.
Stock의 ITL-only 실패 4개는 462개 Token을 제외했다.
모든 ITL 실패 요청을 합치면 Layer는 9개, 3878개 Token이다.
Stock은 4개, 462개 Token이다.
해당 요청의 평균 출력은 Layer 430.889개, Stock 115.500개다.
따라서 Layer의 더 낮은 ITL p99만으로 요청별 Decode 통과나 Token 가중 손실을 판단할 수 없다.

Worker Emission과 Client Delivery도 구분했다.
각 Token 간격에서 `Client ITL = Worker Gap + Delivery Lag 변화량`을 검증했다.
아래 표의 분모도 Method별1200개 요청 관측값이다.
Worker 판정은 모든 Worker Emission Gap≤34ms다. GPU Kernel 전용 판정이 아니다.

| Method | Client Fail·Worker Pass 요청 / Token | Client·Worker 모두 Fail 요청 / Token |
|---|---:|---:|
| NativeStrong | 9 / 803 | 43 / 9363 |
| Stock Token | 4 / 462 | 0 / 0 |
| Layer | 6 / 3615 | 3 / 263 |
| Hybrid | 14 / 1364 | 3 / 346 |

Stock의 ITL 실패 4개는 모두 Worker 간격이 통과한 요청이다.
Layer의 ITL 실패 Token3878개 중3615개도 이 범주에 있다.
따라서 이 손실을 전부 D Prefill Kernel 간섭으로 설명할 수 없다.
특히 Layer의 ITL-only 실패 5개, 3498개 Token은 모두 Worker Gap이 통과한 요청이다.
Stock의 ITL-only 실패 4개, 462개 Token도 모두 이 조건을 만족한다.
Hybrid의 ITL-only 실패 8개 중7개, 690개 Token이 이 조건을 만족한다.
나머지 Hybrid 요청 1개, 84개 Token은 Worker Gap도 실패했다.
다만 Delivery Lag가 증가한 구체적 Host 원인을 이 항등식만으로 확정하지 않는다.
Worker Gap에도 Engine, Scheduler와 Host 시간이 포함된다.
Worker Gap을 GPU-exclusive Decode 시간으로 부르지 않는다.
[Worker·Client 요청 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/worker_client_decode_requests.csv)와 [위반 간격 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/worker_client_decode_violations.csv)에 개별 시각과 Lag를 보존했다.

## Raw Throughput과 Goodput 차이의 정확한 항등식

각 Start에서 다음 항등식이 성립한다.

`Goodput = Raw Token Rate − Σ(실패 범주별 제외 Token Rate)`

Layer−Stock의 Start-mean Goodput 차이는 −20.328347 tok/s다.
그중 Raw Rate 차이는 −1.357679 tok/s다.
제외 Rate 차이는 다음과 같다.

- TTFT-only 실패: +13.127853 tok/s.
- ITL-only 실패: +5.185026 tok/s.
- TTFT+ITL 실패: +0.657789 tok/s.
- 나머지 실패 조합: 0 tok/s.

위 제외 Rate 증가를 Raw 차이에서 빼면 Goodput 차이를 정확하게 재현한다.
이 항등식은 Runtime 원인의 기여율이 아니다.
GPU 시간을 이 비율로 배분하지 않는다.
Hybrid의 같은 계산도 [Goodput 항등식 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/goodput_gap_accounting.csv)에 있다.

Layer와 Stock의 전체 출력 합계는 66361개와 66347개다.
Aggregate Work 차이는 14개다. 요청별 출력 길이까지 같다는 뜻은 아니다.
Hybrid는 62459개다. Stock보다 약5.86% 적다.
Hybrid의 Raw Rate 하락에는 출력 작업량 차이도 함께 관측된다.
그 효과를 분리할 때는 별도 Equal-output Replay를 사용한다.

## Gate 제거 Counterfactual

`OFFLINE_METRIC_GATE_COUNTERFACTUAL`이다.
아래 계산은 원래 TTFT, ITL, 출력, 측정 구간을 변경하지 않는다.
Metric Gate만 제거한다. 실제 Serving 개선이 아니다.
각 셀은 `Joint % / SLO tok/s`의 Start-mean이다.

| 제거 Gate | NativeStrong | Stock Token | Layer | Hybrid |
|---|---:|---:|---:|---:|
| 없음 | 73.50 / 70.139 | 60.75 / 69.358 | 46.42 / 49.029 | 46.50 / 49.802 |
| TTFT | 95.67 / 91.769 | 99.67 / 113.260 | 99.25 / 106.059 | 98.58 / 103.137 |
| ITL | 76.25 / 82.773 | 61.08 / 70.132 | 46.83 / 54.989 | 47.17 / 51.063 |
| TPOT | 73.50 / 70.139 | 60.75 / 69.358 | 46.42 / 49.029 | 46.50 / 49.802 |
| TTFT + ITL | 100 / 109.402 | 100 / 114.035 | 100 / 112.677 | 100 / 105.978 |
| TTFT + TPOT | 95.67 / 91.769 | 99.67 / 113.260 | 99.25 / 106.059 | 98.58 / 103.137 |
| ITL + TPOT | 76.25 / 82.773 | 61.08 / 70.132 | 46.83 / 54.989 | 47.17 / 51.063 |

실패가 중첩되므로 단일 Gate 제거 이득을 무조건 더하지 않는다.
예를 들어 TTFT+ITL 실패는 한 Gate만 제거해도 계속 실패한다.
정확한 Shapley 보조 계산은 실패 요청의 손실을 실패한 Gate 수로 균등 배분한다.
이는 3-Gate Metric 게임의 Attribution이다. Runtime 인과 Attribution이 아니다.
[Gate CSV](dense_ar_native_ttft_joint_slo_20261124/figures/slo_gate_counterfactuals.csv)와 [Shapley CSV](dense_ar_native_ttft_joint_slo_20261124/figures/metric_failure_shapley.csv)에 전부 보존했다.

## Input·Output·Task 분포

아래는 Input Bucket별 TTFT 실패 관측 수다.
각 셀의 분모는 300이다. 동일 Bucket의 100개 요청을 Start3회 측정했다.

| Input | Stock Token | Layer | Hybrid |
|---|---:|---:|---:|
| 3K–6K | 107 / 300 | 153 / 300 | 162 / 300 |
| 6K–10K | 115 / 300 | 167 / 300 | 159 / 300 |
| 10K–20K | 113 / 300 | 156 / 300 | 156 / 300 |
| 20K–36K | 132 / 300 | 162 / 300 | 157 / 300 |

Layer의 실패가 가장 긴 Input에만 집중되지는 않는다.
현재 요청의 Input만으로 앞선 요청에서 누적된 Queue를 설명할 수 없다.
Task·Output Bucket별 분모와 Token 가중 손실은 [Strata CSV](dense_ar_native_ttft_joint_slo_20261124/figures/task_input_output_strata.csv)에 있다.
Task와 Output 분포는 기술 통계다. Input이나 Output의 독립적인 인과 효과를 뜻하지 않는다.
현재 Runtime의 Natural 출력은 Dataset 선정 시점의 Output Bucket을 벗어날 수도 있다.
해당 출력도 실제 길이대로 보존했다.

## 새 Family의 실패 조합과 반례

새 Main400과 Heldout Replay도 원시 요청에서 같은 Gate를 계산했다.
아래 값은 기존 v2와 별도다.
Main의 분모는1200개 요청 관측값이다. Replay의 분모는576개다.
각 셀은 `요청 수 / 실제 출력 Token 수`다.

| Family·결과 | NativeStrong | Stock | StockCap2 | Layer | Hybrid |
|---|---:|---:|---:|---:|---:|
| Main Joint PASS | 865 / 39366 | 886 / 47309 | 633 / 34536 | 585 / 33049 | 536 / 29498 |
| Main TTFT-only Fail | 252 / 10702 | 314 / 16264 | 567 / 29192 | 613 / 32311 | 661 / 33255 |
| Main ITL-only Fail | 52 / 11688 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 51 |
| Main TTFT+ITL Fail | 31 / 3811 | 0 / 0 | 0 / 0 | 2 / 208 | 2 / 172 |
| Replay Joint PASS | 467 / 22066 | 418 / 24572 | 368 / 22158 | 347 / 21036 | 334 / 18055 |
| Replay TTFT-only Fail | 79 / 3684 | 156 / 8814 | 208 / 11466 | 229 / 12588 | 239 / 12786 |
| Replay ITL-only Fail | 26 / 7403 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 2648 |
| Replay TTFT+ITL Fail | 4 / 471 | 2 / 238 | 0 / 0 | 0 / 0 | 1 / 135 |

이 두 새 Family의 표에 없는 네 TPOT 관련 범주는 모두0이다.
모든 범주의 원래8개 행은 Start별 CSV에 그대로 보존한다.
각 열의 Main 총 출력은65567,63573,63728,65568,62976개다.
Replay는 Method별11208×3=33624개다. 요청별 길이도 정확히 일치한다.

새 Main Layer는 ITL-only 실패가0건이었다.
그러나 Joint는48.75%로 Stock73.83%보다 낮았다.
이 Family의 차이는 주로 TTFT 실패 요청의 증가다.
기존 v2의 긴 ITL-only5건을 새 결과에도 그대로 적용하지 않는다.
반대로 Replay Hybrid는 ITL-only2건,2648개 Token을 잃었다.
드문 긴 출력의 가중 손실은 이 별도 Family에서도 관측됐다.

새 Main NativeStrong의 모든-ITL Attainment는93.08%였다.
95% Absolute Decode 목표에 미달했다.
Common0와 Common2는 Main에서100%였다.
따라서 NativeStrong-relative 개선만으로 새로운 방법의 이득을 선언하지 않는다.
정확한 Common-D Native와의 비교도 유지해야 한다.

새 Stock의 Main SLO-goodput은80.536 tok/s였다.
NativeStrong은66.704, Common0은77.969 tok/s였다.
Stock/NativeStrong 비율은1.20736 [1.10969, 1.40370]이다.
Stock/Common0은1.03293 [0.93101, 1.19368]이다.
같은 Reference를 고르지 않으면 개선 판정이 달라진다.
이 Stock은 기존 Strong Baseline이다. 새로운 Layer·Hybrid의 성과로 보고하지 않는다.

## 재현 명령

저장소 Root에서 다음 명령을 실행한다.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.reproduction
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.failure_analysis
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.request_trace
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.causal_analysis
```

전체 분석에는 기존 Raw가 필요하다.
Source SHA, GPU UUID, Actual Batch, Config, Request ID와 Arrival Seed는 실행별 Metadata에 있다.
새 캠페인의 Raw와 기존 Raw는 별도 Namespace다.
Timestamp와 Host/GPU 경계의 한계는 [Request Trace 보고서](dense_ar_native_token_layer_request_trace.md)에 기록한다.
