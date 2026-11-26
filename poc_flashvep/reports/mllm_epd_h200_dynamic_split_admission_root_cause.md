# Static10 Admission과 Batch1 직렬화

직전 H200의 Static10은 첫 D token이 나오면 suffix 전용 예약을 해제했지만 D HTTP 요청을 EOS까지 active에 보관했다. 다음 Static10 요청도 항상 suffix를 필요로 해서 기존 active 목록이 비어야 입장할 수 있었다. 이 두 조건이 결합되어 D Cap이 Free 또는 B4여도 실제 decode가 겹치지 않았다.

| 소스 경계 | 실제 조건 |
| --- | --- |
| Early eligibility | proxy.py에서 encode 전 P HTTP outstanding, D HTTP active 및 suffix owner를 확인. Static은 모든 요청을 선택하고 Idle10은 해당 순간 상태에서 선택 |
| P response | prefill_request가 P 요청을 보내고 응답에 포함된 remote block IDs와 잘라낸 위치를 검사. 이 다음에 D admission에 진입 |
| Guard | IdleAdmission.can_enter에서 suffix 요청이면 not self.active가 필요. Prefill owner도 비어 있고 B4에서 cap/FIFO 조건이 충족되어야 함 |
| 첫 decode token | first_token이 그 요청의 prefill_owner를 비우지만 active를 보존 |
| Stream 종료 | close_admission → leave가 active에서 요청을 제거. 다음 Static10이 이 시점 이후 admission 가능 |

[Early decision, P response 및 admission 호출](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/proxy.py#L147-L191) · [First-token 및 SSE 종료 경계](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/proxy.py#L206-L232) · [Guard와 active 생명주기](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/idle_admission.py#L62-L121)

## 기존 실행 증거

| Free QPS5 Natural | P HTTP 평균 ms | D Coordinator 평균 ms | D 첫 token까지 ms | 별도 Counter 최대 Prepared |
| --- | ---: | ---: | ---: | ---: |
| NativeCommon | 2,947.76 | 0.18 | 70.23 | 5 |
| Static10 | 1,759.14 | 5,320.52 | 129.75 | 1 |
| Idle10 | 2,732.29 | 320.20 | 88.96 | 4 |

Static10의 단일 요청 decode forward는 Counter arms에서 100%였다. 위 Counter arms는 별도 observer 실행의 DIAGNOSTIC이다. Critical path 값은 clean host boundary 시간이다. 전체 요청 대기를 forward나 kernel에 임의 배분하지 않았다. Idle10의 선택된 33건 중 9건에서만 prefix-ready D가 quiet였고 D Coordinator Wait p95는 3,275.34 ms였다. [Critical path](mllm_epd_h200_wide_batch_token_split_critical_path.md) · [Static10](mllm_epd_h200_batch_preserving_static_token_split.md) · [Idle10](mllm_epd_h200_batch_preserving_idle_token_split.md) · [Counter 검증](mllm_epd_actual_batch_counter_validation_v1.md)

## 새 인과 대조

NativeCommon-B8에서 P 전체 prefill을 그대로 하면서 D에 기존 직렬 guard를 적용하는 대조는 허용된 원인 조사다. 최종 비교는 원래 Native admission을 사용한다. Dense AR의 Registration-open과 Credit-return 조작은 admission 제어의 위치와 source KV 수명이 성능과 함께 움직일 수 있음을 보였다. 이 효과 크기를 MLLM으로 옮기지 않으며, 새 예약과 KV retirement의 시간 및 실패 지점을 각각 기록한다. [Dense AR truth and limits](dense_ar_ttft_slo_regression_final_summary.md) · [Controlled interventions](dense_ar_pd_critical_path_controlled_interventions.md)

새 guard-only 실행 상태는 NOT_RUN이다. 실제 실험 결과와 source 확인을 별도 갱신한다.
