# H200 Dynamic Budget 및 Suffix 비용 예측

기존 정책은 경쟁 P 요청의 아직 발행되지 않은 최소 Prompt Row 수를 80,000 tok/s로 나눴으며 상한은 160ms였다. 이미 발행됐지만 GPU에서 진행 중인 작업을 0 Tokens로 처리할 수 있다. 실제 Cut에서 D HTTP가 비어 있던 중앙 ROI의 Generic Natural 7건은 예측 예산 중앙값이 0ms였고 다음 다른 요청의 P-ready까지 시간 중앙값은 195.67ms였다. 추정된 Suffix 포함 D 요청 비용의 중앙값은 281.49ms였다. 160ms 상한을 높이는 경우에도 추정 예산이 0ms였다면 결과는 동일하다.

이 7건에서 최소 비용 추정치도 192.38ms였다. 10ms의 여유를 함께 쓰면 P 완료 예측을 바꾸더라도 기존 160ms 상한에서는 전체가 거절된다. 반대로 상한만 바꿔도 이미 발행된 P 작업을 0ms로 산정한 후보는 여전히 거절된다.

## 비용과 예산 경계

공통 Inline 격리 측정의 동일 요청 및 Repeat에서 0% Native와 Split을 짝지었다. D 절대 비용은 POST부터 1개 Greedy 응답까지이며 KV Import, Embedding, D 실행, 응답 비용을 포함한다. D 증분 비용은 같은 요청의 0% 대비 D 시간 차이고 P 절감은 P의 0% 대비 Prefix HTTP 차이다. Cut 결정부터 D Submit까지의 별도 30ms는 이전 Inline Staged 경계의 24.53ms 및 29.19ms를 참고한 초기 예산이다.

| BI0 Whole-inline 격리 예측량 | 요청 분리 검증 샘플 | MAE | 과소예측 p95 |
|---|---:|---:|---:|
| D 절대 비용 | 98 | 26.07ms | 90.61ms |
| D 증분 비용 | 98 | 26.45ms | 86.09ms |
| P HTTP 절감 | 98 | 27.87ms | 33.09ms |

D 절대 모델의 학습 잔차 p95 상한을 넘은 검증 샘플은 11/98이다. 초기 학습 범위는 Input 2,044–30,195 Tokens, Suffix 102–9,059 Tokens다. 같은 BI0의 새 Slice 직접 관측 중 모델 범위에 있는 28건은 모두 이전 Whole 모델의 상한 이내였고 중앙 예측 오차는 +29.25ms였다. 모델 범위 밖의 32건과 BI1에서의 비용은 별도 Calibration이 필요하다.

## 새 정책의 결정

P의 실제 Issued/Retired 진행과 Causal Attention Pair 규모, Running 및 Waiting 상태, 배치 Row 수를 사용한다. 같은 P 결과에서 이미 다른 요청이 완료되면 Split을 거절한다. 실제 미래 P-ready는 오프라인 보정의 정답으로만 쓰고, Online에서는 비용 보정 영수증과 당시 Scheduler 상태만 참조한다.

- Safe-fit은 새로운 D 작업의 절대 비용, Cut부터 D Submit까지의 고정 여유, 10ms 안전 마진을 합산해 남은 D 예산과 비교한다.
- 순이익은 잔여 P 완료 시간의 보수적 예측에서 Native 대비 증분 D 비용과 Cut 통제 비용을 차감한다.
- 같은 정책에서 D Cap 및 D Reservation은 공통 경로로 유지한다. 동시에 D Engine이 비어 있고 필요한 KV Block을 확보한 경우만 실제 예약한다.
- 지원 범위 밖의 Geometry에서는 P가 원래 요청과 KV 상태로 계속 실행한다.

160/225/300/500ms 및 Adaptive 상한은 동일한 P-ready 예측에 적용하는 정책 선택지다. 별도 Point-gain 민감도는 엄격한 D 절대 Fit 및 Reservation을 유지하고 순이익의 항만 P와 증분 D의 Point Estimate로 계산한다. 안전한 D 점유 조건과 순이익 예상의 불확실성을 독립적으로 확인하기 위한 비교다.

## Free BI0 Stock의 보정 결과

현재 Free BI0 Main100의 고정 Discovery 요청에서 54개 P Chunk만 학습에 사용했다. 같은 실행의 Heldout 41개 Chunk에서 다음 다른 P-ready 예측의 MAE는 기존 80K/160ms의 258.10ms에서 102.44ms로 줄었다. 기존 정책은 28건에서 0ms를 예측했다. 같은 80K/160ms에서 Retired Work만 반영해도 0ms는 22건으로 줄었다. 새 예측기의 p95 과대예측은 117.70ms였고 학습 잔차 p10을 적용한 하한은 Heldout 2개에서 실제 시각을 넘었다. 같은 Start의 Counter 반복 Heldout 48개에서는 MAE 120.57ms 및 하한 초과 6건이었다. 두 검증의 요청 ID는 학습에서 분리했으며 Server Start는 공유한다.

| 목표 5–30%에서 두 조건을 모두 통과한 Cut | 160ms | 225ms | 300ms | 500ms | Adaptive |
|---|---:|---:|---:|---:|---:|
| 보수적 이득, Discovery | 0 | 0 | 0 | 0 | 0 |
| 보수적 이득, Heldout | 0 | 0 | 0 | 0 | 0 |
| Point-gain, Discovery | 0 | 0 | 0 | 1 | 1 |
| Point-gain, Heldout | 0 | 0 | 0 | 1 | 1 |
| Point-gain, Counter 반복 Heldout | 0 | 0 | 0 | 1 | 1 |

Discovery 4,820-token Cut의 추정 이득은 77.06ms, D 절대 비용 추정 상한은 436.05ms, 관측된 다음 다른 P-ready는 479.86ms였다. Heldout 4,496-token Cut의 해당 값은 19.01ms, 338.57ms, 1,240.08ms였다. 두 Cut 모두 후속 P Rows가 이미 발행돼 있었다. Stock에서 사후 확인한 구간에 적용된 비교이므로 OFFLINE_ORACLE이다. Hold 적용 시 실제 시각과 해당 요청의 Cross-GPU 비용이 달라질 수 있다. 이 BI0 모델의 계수와 실측 비용은 BI1 또는 다른 D Backend를 검증하는 데 사용하지 않는다.

## B8 Eager의 별도 P-ready 보정

이번 수치 조건의 Discovery 56 Cut으로 다시 보정했다. 첫 Cohort의 Heldout 31 Cut에서 기존 MAE 306.33ms가 123.03ms로 감소했다. 새 예측 p95 과대예측은 266.98ms였고, 보수 하한도 3/31건에서 실제 시간보다 컸다. Counter 반복의 Heldout 34 Cut에서 MAE는 258.21ms에서 131.71ms로 감소했으며 하한 초과는 8/34건이었다. 더 나은 평균 오차와 실에러 꼬리는 별개로 확인해야 한다. 이 학습은 Eager P-ready만 보정하며 Eager D 절대 비용은 같은 수치 조건의 격리 실험이 필요하다.

## Active에서 관측한 비용 범위의 한계

Eager Active A의 실제 Suffix 6건은 Cut부터 Coordinator가 확인한 D 첫 Token까지 201.88–2,735.47ms를 사용했다. 모두 P가 이미 가져온 Crossing Image의 Descriptor를 다시 확보하기 위한 E export RPC를 한 번씩 실행했다. RPC는 4.55–2,470.68ms였다. 격리 상태의 Cost Table은 이와 같은 E 수신 Queue 부담을 포함한 Online 상한으로 사용할 수 없다. 특히 실행 시 D가 비어 있던 세 요청에서도 다음 P-ready가 Suffix 완료 전에 도착했다.

새 Transport readiness 선택지는 Cut에서 필요한 각 이미지의 사용할 수 있는 Inline Descriptor 여부를 조회한다. Use-once Descriptor가 빠져 있으면 비용 평가와 D 예약 전에 해당 요청을 P에서 즉시 계속 처리한다. 이 확인은 모든 Partition 비교군에 동일하게 적용할 수 있다. Producer가 준비한 Spare를 이용하는 경우도 D의 실제 Claim과 비용을 Raw 로그로 별도 검사한다.

## Matching Eager 격리 보정

Eager에서 실제 335회, 173가지 Geometry를 수행했다. 같은 요청의 0% 결과를 Whole과 Slice 각각의 Suffix 결과와 짝지었다. 두 경로 모두 87개 학습 Pair 및 요청 분리 검증 75 Pair를 사용했다.

최초 격리 실행 시도는 런타임 Overlay 전에 모듈을 읽어 이전 서버 경로에서 모델 영수증을 찾는 시작 오류가 있었다. Overlay 이후 Worker 경로를 불러오도록 수정한 다음 새 Epoch에서 전체 실험을 완료했다.

| Eager Transport | D 절대 비용 MAE | D 절대 과소예측 p95 | D 증분 비용 MAE | D 증분 과소예측 p95 |
|---|---:|---:|---:|---:|
| Whole Inline | 32.32ms | 70.20ms | 31.45ms | 69.04ms |
| Slice Inline | 27.93ms | 65.91ms | 32.22ms | 74.11ms |

격리 상태의 D export RPC는 Slice와 Whole에서 각각 145건이었다. Slice의 중앙값과 최대는 22.82ms 및 47.80ms, Whole는 19.49ms 및 47.10ms였다. A Online의 같은 Host 경계를 요구한 6건 중 5건이 격리 회귀의 Geometry 범위에 있었고 그중 4건의 실제 Cut→D 첫 Token 비용은 30ms Handoff 여유까지 포함한 Slice 모형 상한보다 높았다. 나머지 한 건의 Input 30,326 Tokens는 학습 상한 30,195 Tokens 밖이다.

B8 Eager의 Stock 보정에서 500ms 및 Adaptive에 동일 Discovery Cut 한 건이 절대 Fit을 통과했다. 160/225/300ms에서는 Fit이 없었다. 보수 이득과 Point-gain을 썼을 때 두 조건을 모두 통과한 Cut은 전 Cap에서 0건이다. 같은 시작의 두 Heldout Cohort에서는 D가 비고 정책 범위에 드는 대상 검증 Cut을 얻지 못했다.

Stock의 P Own-ready 잔여 시간은 이미 발행된 후속 Chunk의 시간을 사용하며, Hold로 실제 절감되는 P Service의 직접 관측이 아니다. 이 경계를 바꾸는 민감도는 같은 Shape의 격리 P 서비스 절감과 D 증분 비용을 사용했다.

| Cut Suffix | P 대기 | Slice P HTTP 절감 예측 | Slice 증분 D 예측 | P 절감 − D 증분 − Handoff |
|---:|---:|---:|---:|---:|
| 2,842 | 5 | 117.93ms | 151.01ms | −63.08ms |
| 3,067 | 6 | 117.58ms | 143.08ms | −55.50ms |

Whole의 두 추정 순이익은 −67.98ms 및 −68.49ms였다. P의 실제 대기열은 관측됐으나 Work 제거가 다른 요청을 얼마나 앞당겼는지는 추정에서 차감하지 않았다. Slice에서 손익분기까지 필요한 추가 Queue 절감은 각각 63.08ms와 55.50ms다. 두 번째 Cut은 절대 D Fit도 통과하지 못했다.

동일 Sample과 동일 Ridge Family의 Crossing Binary Feature가 두 비용 모형의 검증 오류와 상한을 함께 개선하지 못했다. 이 Eager 보정에서 C-Soft 비용 옵션은 비활성 상태다.

## BI1 Spare 경로의 QPS4 학습

Native-Matched BI1 B8 QPS4에서 사전 지정된 Discovery의 46 Cut만 학습했다. 아래 Heldout 평가들은 요청 ID를 학습에서 제외했고 서버 시작을 공유한다.

| 평가 | Cut 수 | 기존 80K/160ms MAE | 새 MAE | 보수 하한이 실제를 초과 |
|---|---:|---:|---:|---:|
| QPS4 최초 | 41 | 308.50ms | 109.13ms | 6 |
| QPS4 Counter | 34 | 344.99ms | 106.80ms | 3 |
| QPS5 최초 | 32 | 296.82ms | 130.18ms | 4 |
| QPS3 최초 | 37 | 395.16ms | 213.39ms | 6 |

QPS4 중앙에서는 대상 비율에서 D가 빈 Cut이 없었다. QPS5의 2,842-token Discovery Cut은 실제 다른 P-ready까지 315.74ms였고 새 예산 하한은 380.6ms였다. QPS3의 7,077-token Discovery Cut은 실제 616.00ms에 하한 842.6ms였다. 개별 하한에서도 과대예측이 발생하므로 실행 가능성을 확인하려면 실제 D 절대 비용과 Native 지연을 함께 측정한다.

## Matching BI1 Spare의 실제 격리 비용

동일 수치 조건과 Preissue 경로에서 99 Geometry를 187회 실행했다. 11개 요청의 Native 0%를 두 경로에 공통으로 짝지었으며 Whole 및 Slice에서 각각 나머지 88개 Suffix의 실제 경로가 NO_RPC_VERIFIED였다. 각 모형의 학습은 48개, 요청 분리 검증은 40개다. 모델이 지원하는 학습 Suffix는 19–6,039 Tokens다.

| BI1 Spare Transport | D 절대 MAE | D 절대 과소예측 p95 | D 증분 MAE | D 증분 과소예측 p95 |
|---|---:|---:|---:|---:|
| Whole Inline | 30.86ms | 74.70ms | 35.21ms | 92.17ms |
| Slice Inline | 20.86ms | 68.79ms | 23.18ms | 80.05ms |

아래 Cap에서는 실제 Whole/Slice 격리 비용과 QPS4에서 동결한 P-ready 모델을 사용했다. 검증된 10/20% Cut 범위에서 QPS4의 두 중앙 Cohort는 D가 빈 경우가 0건이었다. QPS3의 실제 7,077-token Cut은 짧은 격리 보정의 지원 범위 밖이다.

| QPS5 Discovery의 중앙 Cut | 160ms | 225ms | 300ms | 500ms | Adaptive |
|---|---:|---:|---:|---:|---:|
| Slice 절대 Fit | 0 | 0 | 0 | 1 | 1 |
| Slice 절대 Fit 및 보수적 이득 | 0 | 0 | 0 | 0 | 0 |
| Slice 절대 Fit 및 Point-gain | 0 | 0 | 0 | 1 | 1 |
| Slice 절대 Fit 및 격리 P-service 이득 | 0 | 0 | 0 | 0 | 0 |

1건의 2,842-token Cut은 Slice Point-gain이 +157.74ms, paired P-service 기준 순이익이 −29.33ms였다. 예측 예산은 380.6ms였다. 307.12ms로 추정한 D 절대 점유에 10ms 마진을 합하면 당시의 315.74ms Native 다른 P-ready를 1.38ms 초과한다. 이 과거 Cut에 대한 OFFLINE_ORACLE은 실행 가능한 상한의 근사치이며, 실제 Suffix를 실행했을 때의 대기는 별도 관측이 필요하다.

실제 Spare A를 사후 점검한 세 사례에서는 각 Cut→D 첫 Token이 모두 해당 Slice 격리 비용의 상한 내였다. 이 중 QPS3 두 사례는 여전히 다음 Native-ready가 해당 Suffix를 앞질렀고 24–79ms의 D 대기가 생겼다. Slice 및 Whole에서 C-Soft 조정은 지원된 비용 검증 기준을 통과하지 못했다.

## 동결한 B-new Online의 예측 검증

동일 BI1 Spare Slice의 Point-gain, 500ms Cap 및 Transport readiness Guard를 실제 Next Native-ready 결정에 사용했다. Main100 Natural QPS5/4/3 최초 및 같은 Start Counter에서 12건이 D에서 실제 Suffix를 계산했다. 실제 Cut에서 선택한 Suffix는 1,482/1,752/2,755/2,842 Tokens였으며 Ratio는 9.6–17.6%였다.

실제 비용이 예측 상한과 예산을 모두 초과한 경우는 Counter 1/12였다. 별도로 여섯 Suffix 동안 Native-ready 일곱 건이 도착했다. 가장 큰 대기는 QPS5 최초의 70.10ms였는데 실제 점유 174.55ms는 예측 비용 217.08ms와 예산 243.88ms보다 작았다. 오차의 직접적인 출처는 해당 사례에서 기존 P 완료와의 정렬이었다. 비용 상한을 지켰어도 새 작업이 미리 도착할 수 있다. 보고한 12건 중 11건은 중앙 ROI 이전에 실행됐다.

QPS5 Equal-output Replay의 세 실제 Split에서는 모두 예측 절대 비용과 예산을 지켰다. 최초 Cohort의 두 Split에서는 Native-ready 세 건이 도착했고 각 요청은 37.76–54.57ms를 기다렸다. 중앙의 2,755-token Suffix가 사용한 시간은 284.68ms, 예상 비용 304.22ms, 예산 407.88ms였다. 다음 실제 Native-ready는 246.48ms였다. 어느 Host 경계에서 남은 여유가 과대평가됐는지 직접 기록한 사례다.

[B-new Replay의 실제 예산, 비용 및 Native 대기](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bnew_bi1_preissue_replay) · [B-new Natural 예측 결과](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bnew_bi1_preissue_natural) · [BI1 Spare의 실제 격리 비용과 Cap 민감도](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bi1_preissue_matched_cost) · [BI1 Spare의 QPS4 P-ready 영수증](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_b8_bi1_preissue/ready_model) · [Matching Eager 격리 비용과 Cap 민감도](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/eager_matched_cost) · [B8 Eager P-ready 영수증](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_b8_eager/ready_model) · [Free BI0 예측 및 Cap 민감도](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_free_bi0/provisional_predictor) · [BI0 격리 비용](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bi0_historical_cost) · [보정 코드](../mllm_epd_h200_dynamic_correctness_slice_aware/calibrate_slack.py) · [Online 정책](../mllm_epd_h200_dynamic_correctness_slice_aware/slack_policy.py)
