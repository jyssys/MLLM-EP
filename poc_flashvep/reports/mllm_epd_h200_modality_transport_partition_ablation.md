# Transport와 Modality Partition의 독립 효과

Generic은 Cut 선호도에 이미지 구조를 사용하지 않는다. 선택된 Cut에서 원본 Token Order, MRoPE, 이미지 경로와 필요한 Embedding Row를 보존한다. C-Hard는 Image 내부 Cut을 거절하고, C-Soft는 같은 Transport에서 측정한 추가 비용이 있을 때만 해당 비용을 선택에 반영한다. 공통 inline-redirect, B8192/S1024, KV42GiB, D Cap8과 Native Admission을 유지한다.

## 동일 Cut의 H200 전송 결과

BI0에서 동일 요청·Cut을 바꾸지 않고 Whole과 Slice를 번갈아 실행한 60쌍 중, 실제 D Clone Row 수가 줄어든 38쌍의 E NVLink TX 절감 중앙값은 32.81 MiB였다. 자르기 위치의 작업량과 두 Transport의 NIXL KV Bytes는 각 쌍에서 동일했다. D HTTP 첫 출력까지의 감소 중앙값은 38쌍에서 5.20 ms였고, 20쌍에서 감소했다. Host Latency의 감소 폭은 물리 전송량과 함께 달라진 정도를 보장하지 않는다.

L7,823 / p7,804 / n19에서는 D에 필요한 이미지 Row가 하나였다. D Clone Payload는 55,838,720 bytes에서 7,168 bytes로, E NVLink TX는 두 반복에서 약 55.84 MB에서 3,072–5,120 bytes로 줄었다. 같은 두 쌍의 D HTTP는 각각 약 2.78 ms와 109.12 ms 증가했다. P에서 이미 한 Whole-image Clone 및 원래 E Producer Allocation은 그대로 남았다. [동일 Cut 전송 및 정합성 자료](mllm_epd_h200_same_cut_embedding_transfer_ablation.md)

## Slice 이후에도 남은 추가 작업

동일 요청에서 Suffix가 한 토큰 차이 나는 Interior와 Boundary의 18개 대응쌍은 NIXL Bytes가 각각 같았다. Whole에서는 Interior가 E TX를 중앙값 32.81 MiB 더 사용했다. Slice에서는 그 차이가 중앙값 7 KiB로 줄었고, 추가 Clone과 Descriptor가 각 한 개 남았다. Slice의 Interior D Clone Host Enclosure 증가 중앙값은 10.04 ms였으며, D HTTP는 중앙값 20.23 ms 증가했다. D HTTP 차이의 범위는 −23.64~+127.97 ms였다. 이 대조는 작은 Slice도 별도 Descriptor 준비와 Clone·동기화를 필요로 할 수 있음을 보여준다. Boundary 정책이 같은 예산에서 더 좋은 Cut을 고르는지는 별도 시험의 대상이다. [경계 근처 전송 대조](mllm_epd_h200_same_cut_embedding_transfer_ablation.md)

## 2×2 비교의 판정 범위

| 대비 | 확인하는 효과 |
|---|---|
| G-Whole → M-Whole | 기존 Whole-image Clone 경로에서의 Modality Partition 효과 |
| G-Whole → G-Slice | 같은 Generic 선택에서의 전송 최적화 |
| G-Slice → M-Slice | 필요한 Row만 복제한 이후에도 남는 Modality Scheduling 효과 |
| M-Whole → M-Slice | 같은 Modality 정책에 대한 전송 최적화 효과 |

같은 Cut의 고립 비교에서 두 번째 대비의 데이터 이동 메커니즘을 검증했다. 사전 발급 BI1에서는 G-Slice와 M-Slice를 같은 Cost Model, Adaptive Budget, Point Gain, B8, Arrival Seed와 10/20% 범위로 실행했다. M-Slice의 Hard는 실제 Image 내부 Cut만 막았다. Soft의 Crossing Feature는 동일 데이터에서 heldout 오차와 tail 기준을 모두 개선하지 못해 비활성화했다.

| QPS5 Natural, 첫 Main100 Cohort | 실제 Offload | TTFT p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Decode SLO | Goodput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| G-Whole | NOT_RUN |  |  |  |  |  |
| M-Whole | NOT_RUN |  |  |  |  |  |
| G-Slice | 1/100 | 20.317 | 31.941 | 22.655 | 75% | 8.060 |
| M-Slice Hard | 0/100 | 18.602 | 31.170 | 22.776 | 78% | 17.943 |

QPS5 Hard 첫 Cohort의 Callback 23건은 모두 D Admission Busy였다. QPS5의 성능 차이는 Hard의 실제 Suffix 또는 Boundary 전송 효과로 귀속할 수 없다. QPS4에서는 B-new가 두 Main100 첫 Cohort 요청의 실제 1,752/1,482-token Suffix를 이전했다. Hard는 D가 HTTP idle인 시점에 같은 Dataset Request ID와 같은 Cut 두 개를 Image 내부라는 이유로 거절했다. QPS4 Hard의 실제 이전은 0/100, TTFT p95는 14.800 s, Decode SLO는 84%였다. B-new는 14.135 s 및 80%였다. 한 번의 시작에서 두 정책 모두 Decode 목표에 미달했다. [Actual Cut과 Reason](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/chard_bi1_preissue_natural/) · [Active Online 결과](mllm_epd_h200_dynamic_split_active_online_serving.md)

Serving의 2×2에서 Whole 두 Cell은 미실행이다. 고립 비용 보정에는 같은 입력, Cut과 새 사전 발급 경로를 사용하는 Whole/Slice를 각각 실행했다. Slice-only 이후 실제 비용 차이와 Hard가 사용할 수 있는 Cut을 줄이는 현상을 확인했다. 가장 강한 G-Slice 대비 Modality-aware의 독립적인 Online 우위는 입증하지 못했다.
