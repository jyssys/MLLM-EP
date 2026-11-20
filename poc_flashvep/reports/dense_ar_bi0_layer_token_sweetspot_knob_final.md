# BI0 Layer/Token Sweet Spot과 최소 Knob

상태: `CLOSED / HOLD`. 검증된 최소 Knob은 확보하지 못했다.

## 관측 가능한 후보

가장 직접적인 후보는 P와 D의 유효 Service Demand 균형이다.
P Demand는 Dispatch부터 Parent Ready까지의 Host Stage다.
D Demand는 D Submit부터 First Client Token까지와 이후 Decode Residency를 포함한다.
이 값들은 GPU-exclusive 비용이 아니다.
실제 Online Predictor는 완료한 이전 요청의 비용만 사용할 수 있다.
현재 요청의 미래 EOS나 완료 시각을 읽는 Predictor는 허용하지 않는다.

단일 후보 Scalar는 `rho_PD = mean(P Host Service) / mean(D Whole-request Host Demand)`다.
Online 구현에서는 이미 완료한 요청의 관측으로 Mean을 갱신할 수 있다.
그러나 현재 요청의 실제 미래 Service를 넣으면 Offline 설명값이 된다.
rho_PD>1을 실제 Winner의 검증된 Threshold로 사용하지 않는다.
이 Scalar는 GPU Busy, Safe Window 또는 다음 Decode Deadline과 다르다.

Pipeline TTFT는 P+D 합계만으로 결정되지 않는다.
같은 합계에서도 느린 Stage가 다르면 Queue 누적이 달라질 수 있다.
Source Credit은 이 Stage 사이의 Backpressure를 바꾼다.
Credit이 반환되는 시점도 같은 숫자의 Window와 다를 수 있다.
따라서 Cut 크기 하나로 Axis Winner를 설명하지 않는다.

## 실측 반증

Last4의 Fixed-output Full200 Serial P+D 비율은1.005966이다.
Start-level95% CI는 [0.989338, 1.020603]이다.
같은 Fixed-output Online의 TTFT p95 비율은1.212395다.
CI는 [1.201986, 1.221838]이다.
각 Group은 독립 Process Start3회지만 Arrival Seed는1개다.
이 비교는 Serial 합계가 확실히 Layer 우위라는 전제를 재현하지 못했다.
그러나 비슷한 Service 합계가 같은 Online TTFT를 보장하지 않는다는 점은 확인했다.

Credit1/4/8/16/32의 새 Control에서 Token의 Last8 TTFT 우위는 유지됐다.
이는 Source Credit4만이 Token 승리를 만들었다는 단순 설명과 맞지 않는다.
Historical Input8000/Output16은 별도 Family로 Bridge한다.
그 결과를 Current LongBench200의 Cut별 CI에 합치지 않는다.

그 Family의 Current BI0/KV128 독립3회에서는 Token이 빨랐다.
TTFT p95 Layer/Token 비율은1.160249다. CI는 [1.088786, 1.249619]다.
KV16의 첫 Start에서는 Layer가 빨랐지만, 별도 Held-out3회 비율은1.025844였다.
CI는 [0.962237, 1.071844]다. 이 결과는 Layer Winner로 확정하지 않는다.
따라서 KV Block 하나로 재현 가능한 BI0 Winner를 결정하는 단순 규칙도 아직 검증하지 못했다.

## Replay 검증

FIFO P/D Queue와 Source Credit을 갱신하는 Replay를 사용했다.
같은 Loaded Service로 원래 Trace를 정확하게 복원하는 검사는 Accounting이다.
그 일치만으로 새 Workload의 예측 성능을 주장하지 않는다.
P와 D의 Service Vector를 서로 바꾸는 검사는 조건부 민감도 분석이다.
그 조합은 실제로 실행 가능한 Transformer Partition이 아닐 수 있다.

새 Arrival Seed2085에서 Last4의 TTFT p95/MAE Gate는 양측이 통과했다.
같은 Replay의 Layer SLO-goodput 오차는 +15.388%였다.
사전10% Gate를 실패했다. Token의 오차는 +4.537%였다.
Last8의 Serial Service로 Loaded Layer를 예측한 p95 오차도 약+32.49%였다.
따라서 고정 Serial Cost가 모든 Queue 조건으로 전달된다는 가설은 지지되지 않는다.

## 현재 판정

Full200 Confirmation의 측정 Host Demand는 다음처럼 변했다.
같은 QPS2.35와 독립3회를 사용한다. 이 표의 rho_PD는 사후 설명값이다.

| Cut | rho_PD Layer | rho_PD Token | Mean Host Demand 관계 |
|---|---:|---:|---|
| Last2 | 2.052994 | 1.890654 | P가 D보다 큼 |
| Last4 | 1.713463 | 1.564435 | P가 D보다 큼 |
| Last8 | 1.283568 | 1.118976 | P가 D보다 큼 |
| Last12 | 0.934209 | 0.782839 | D가 P보다 큼 |

Last12에서 P/D Demand의 우세한 단계가 바뀌었다.
Token의 Credit Extension 누적도 Layer보다2424.919ms 더 컸다.
반면 Last4의 차이는103.152ms였다. Source Credit의 영향은 Cut에 따라 달랐다.
그러나 Last12의 TTFT CI는1을 포함했고 SLO-goodput 우위도 미확정이었다.
따라서 이 전환을 검증된 Axis 선택 Threshold라고 표시하지 않는다.

P/D Service Balance는 유용한 설명 후보다.
새 QPS·Output·Batch에서 검증된 Single-knob 또는 Two-knob Winner Rule은 아직 없다.
Best Fixed-axis 대비5% 추가 SLO-goodput Oracle Headroom도 검증하지 못했다.
이 상태에서 복잡한 Online Axis Selector를 구현하지 않는다.
Active Decode 사이의 Safe Window도 이번 증거로 확정하지 않는다.
다른 QPS·Output·Batch의 적격 Label과 Held-out Regret를 모두 갖춘 분류 검증은 `NOT_RUN`이다.
따라서 Single-knob이나 Two-knob의 일반화 Accuracy를 보고하지 않는다.
이 판정은 해당 Candidate의 가능성을 영구적으로 배제한다는 뜻이 아니다.

[Clean Control 통계](dense_ar_bi0_axis_rootcause_20261120/analysis/Last4_clean_control_start_statistics.json),
[Held-out Goodput 검증](dense_ar_bi0_axis_rootcause_20261120/analysis/heldout_replay_goodput_validation_v1.json)을 참조한다.
