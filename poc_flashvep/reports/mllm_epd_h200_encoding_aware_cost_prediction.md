# Encoding Features의 Heldout 예측 가치

Generic과 Modality predictor를 동일 model family, 샘플, context-disjoint train/heldout, regularization과 평가 예산에서 비교한다. Generic은 입력·prefix·suffix 길이, causal work, KV 및 batch, queue, request readiness를 이용할 수 있다. 새 feature는 실제 online scheduler에서 조회 가능한 image count/span/grid, encoding 완료 상태나 transfer/layout 측정으로 제한한다.

| Metric | 계산 |
| --- | --- |
| MAE | 같은 heldout execution에서 예측 비용과 관측 비용의 절대 오차 평균 |
| Relative error | 측정 비용이 양수인 요청의 signed relative error |
| p95 underprediction | max(측정 비용−예측 비용, 0)의 p95 |
| False-safe | 추정상 safe-fit이며 관측 비용은 해당 budget을 넘은 선택의 비율. 분모와 budget 출처 기재 |
| Excess rejection | 예측에서 거절했지만 비교한 budget 안에 관측 비용이 들어간 경우 |
| 결정을 내리는 비용 | feature 추출·선택의 실제 online host event |

미래 decode EOS 또는 trace의 미래 arrival은 online feature에서 제외한다. 기존 runtime에서 image별 ready 시점을 제공하지 않으면 request-level completion을 쓰며 세분화한 ready feature을 NOT_AVAILABLE로 남긴다. Pure predictor 개선과 Serving 개선은 별도 판정이다.

## On-demand calibration heldout

동일 λ=0.1 Ridge를 사용했고 공통 feature는 입력·prefix·suffix와 causal work다. D batch1 isolated sample에서는 queue 및 batch 변화를 학습하지 않았다. Encoding-only는 image count와 E request HTTP를 추가하며 E kernel time을 뜻하지 않는다. Boundary-only는 cross-image count와 logical BF16 payload를 추가한다. 8개 fit 요청과 정확히 같은 문서 및 image pixel hash가 없는 7개 검증 요청의 98개 cost trial을 검사했다.

| Predictor | MAE (ms) | p95 underprediction (ms) | 고정 160 ms false-safe / predicted-safe | 실제 safe 54개 중 과거절 | CPU p50 (μs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 22.67 | 57.27 | 2/34 | 22 | 3.67 |
| Encoding-only | 23.18 | 53.73 | 2/30 | 26 | 4.23 |
| Boundary-only | 22.96 | 51.82 | 2/32 | 24 | 4.15 |
| Full modality | 23.14 | 52.10 | 2/31 | 25 | 4.64 |

Full modality는 underprediction tail을 줄였지만 MAE가 증가했고 같은 160 ms reference에서 추가로 3개의 실제 safe case를 거절했다. 이는 고정 budget의 offline 검사이며 실제 C3 Idle window를 나타내지 않는다. MAE 개선과 underprediction tail 비악화를 함께 요구한 사전 규칙을 충족하지 않아 Soft는 full predictor를 활성화하지 않는다. 동일 비용을 두 번 계산하는 별도 duplication penalty도 추가하지 않는다. H-M2의 이 predictor에 대한 판정은 NO_INCREMENTAL_MODALITY_EFFECT다. [보정 모델과 budget별 수치](../mllm_epd_h200_modality_aware_dynamic_split/results/suffix_cost_model.json) · [Figure 5](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig05_generic_modality_cost_prediction.pdf)

## Inline-redirect calibration heldout

같은 요청 split, predictor family와 보정 방법을 유지했다. Transport를 신선한 epoch에서 실측해 fit은 별도로 수행했다.

| Predictor | MAE (ms) | p95 underprediction (ms) | 160 ms false-safe / predicted-safe | 실제 safe 46개 중 거절 | CPU p50 (μs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 26.0690 | 90.5902 | 1/16 | 31 | 3.95 |
| Encoding-only | 26.8670 | 87.5844 | 1/17 | 30 | 4.56 |
| Boundary-only | 26.0605 | 89.9336 | 1/16 | 31 | 4.46 |
| Full modality | 26.8655 | 87.4991 | 1/17 | 30 | 4.99 |

Full modality는 두 transport에서 p95 과소예측을 낮췄으나 평균 절대오차가 증가했다. Inline boundary-only의 MAE 차이는 0.009 ms였고 160 ms 분류 수는 Generic과 같았다. 이 점추정으로 의미 있는 추가 개선을 확정할 수 없다. Soft의 full-feature 활성화 기준은 inline에서도 미충족이었고 Soft shadow는 실행하지 않았다. [Inline fit과 heldout 원자료](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/suffix_cost_model.json) · [Inline Figure 5](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig05_generic_modality_cost_prediction.pdf)
