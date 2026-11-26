# Method C Hard/Soft Modality-aware Split

Method C는 B와 같은 scheduler에서 modality feature switch로 실행한다. 공통 candidate geometry set을 먼저 구성한다. Hard는 image 내부 cut의 eligibility만 제한한다. Soft는 비용 증가가 실측되면 해당 범위에서만 penalty를 적용한다. 관측으로 보정할 추가 비용이 없으면 Soft penalty는 0이며 그 arm은 Generic과 사실상 동등할 수 있다.

| Arm | Encoding | Boundary | Transfer cost |
| --- | --- | --- | --- |
| Generic | Off | Off | Generic |
| Encoding-only | On, 실제 제공 시 | Off | Generic |
| Hard Boundary-only | Off | Hard | Generic |
| Soft Boundary-only | Off | measured penalty | Generic |
| Transfer-aware-only | Off 또는 최소 | Off | Measured |
| Full | 실제 제공 feature | 측정에 근거한 정책 | Measured |
| Fake-boundary | Off | 동일 수의 비이미지 위치 | Generic |

Hard의 feasible ratio 감소, boundary에서 큰 suffix의 비용 증가, single-image의 제약 및 fallback을 보고한다. 모델 입력의 image metadata는 모든 arm에 그대로 유지한다. 모든 boundary와 ratio는 동일한 Full28 Prefix/Final KV, MRoPE 및 Greedy 검사를 거친다. 이전 compiled N8의 25% 경로 실패는 새 20% 또는 30%의 PASS로 간주할 수 없다. [이전 correctness 결과](mllm_epd_prefill_heavy_token_split_final_summary.md)

초기 on-demand 보정의 H-M1에서 whole image 복제와 추가 E TX가 실측됐다. 해당 encoder/boundary feature에는 MAE 개선과 p95 과소예측 비악화를 모두 요구했다. 이 규칙을 충족하는 pipeline의 cost model에서만 Soft 추가 feature를 활성화한다. 같은 cost model을 사용하는 Hard는 공통 후보의 eligibility만 바꾼다. [Embedding 복제 대조](mllm_epd_h200_image_span_transfer_cost_ablation.md) · [Predictor 대조](mllm_epd_h200_encoding_aware_cost_prediction.md)

Natural-hold shadow에서 Generic과 Hard에는 같은 stock P boundary가 주어진다. Hard는 image span 내부를 거절하고 P가 계산을 이어간다. 이 경로에는 추가 edge 삽입이나 fake boundary를 실행하지 않는다. Feature-on/off의 active 대조, block-alignment와 fake-boundary의 online 대조는 NOT_RUN이다.

실제 두 staged N8에서 image-edge와 intra-image 각각의 D suffix를 실행했다. Image-edge lead의 58개 Greedy IDs는 Native와 모두 일치했다. 두 arm 모두 Native follower를 포함한 전체 gate에서는 7/8이었다. Active Hard/Soft의 timed C3는 NOT_RUN이다. [Natural cut 정확성 및 actual D batch](mllm_epd_h200_late_bound_batch_preserving_split.md#실제-d-suffix와-native-follower가-겹치는-검증)
