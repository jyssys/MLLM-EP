# Image Span과 Embedding Transfer 비교

같은 실제 요청의 intra-image, image-edge, generic-position cut을 비교한다. 같은 요청에서 두 실제 cut이 다르면 suffix 길이와 attention pair work도 달라진다. 이 차이를 시료별로 남기고 가까운 pair, matched stratum 또는 cost-adjusted residual로 검사한다. 같은 실제 cut을 유지하고 transfer/layout 최적화만 On/Off할 수 있는 경우에는 이를 데이터 이동의 직접 대조로 사용한다.

| 관측량 | 정의 |
| --- | --- |
| Logical embedding bytes | 토큰 범위와 embedding width, dtype에서 계산한 단위 |
| Alias/storage bytes | 동일한 backing storage를 공유하는 extent 및 producer 수명 |
| Actual copy/transfer | 경로에서 계측한 copy event 또는 byte counter. 링크 수준 물리량은 그 계측이 있을 때마다 명시 |
| Descriptor | export/import에서 실제 취급한 handle과 item 수 |
| Memory | 동시에 live인 allocation의 peak와 수명. KV와 embedding 분리 |
| 시간 | E/P/D host event 및 forward enclosure. 실제 kernel event와 다른 수치 |

Block128 위치 정렬과 candidate count는 generic 및 fake-boundary에서도 맞춘다. Image span 내부 분할 시 엄밀한 정합성 증거는 기존에 일부 관측됐다. 그 사실과 새 timing 및 storage 효과는 별개다. [기존 image 내부/경계 검사](mllm_epd_generic_token_split_correctness.md)

기존 E consumer plan은 half-open span의 시작이 cut보다 작으면 P를, 끝이 cut보다 크면 D를 포함한다. E는 원본 텐서를 보존하고 CUDA event를 descriptor와 함께 내보낸다. Consumer는 source storage를 import한 뒤 전체 이미지 텐서를 clone하고, clone 완료 후 ACK를 보낸다. 양쪽 소비자로 배정될 때 같은 backing payload를 별도 worker에서 clone하는 경로가 있다는 소스 근거다. 실제 clone 횟수와 비용은 다음 실험의 counter에서 확인한다. [Consumer plan](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/embedding_consumer_plan.py#L19-L31) · [Producer event와 descriptor](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/cuda_ipc_ec.py#L39-L86) · [Consumer clone과 ACK](https://github.com/jyssys/MLLM-EP/blob/55870ccafb210d32dc956320d02226d26f2efee6/poc_flashvep/mllm_epd_prefill_heavy_token_split/cuda_ipc_ec.py#L193-L220)

Calibration 계획은 같은 request 및 Block128 prefix block 수에서 실제 suffix가 한 토큰만 다른 9개 image-boundary/intra-image 쌍을 사용한다. Suffix는 single/few-image 16–62 tokens부터 long-document의 2,311/2,870 tokens까지 포함한다. 10% Generic 및 128-aligned Generic은 독립 geometry 대조다.

## On-demand 18개 matched pair

| Intra-image − boundary | 관측 |
| --- | --- |
| D clone와 duplicate image dispatch | 모든 pair에서 원본 BF16 이미지 텐서 한 장이 추가됨 |
| E NVLink TX, D 구간 | +3.876–53.255 MiB, 중앙값 +32.813 MiB. 모든 pair에서 추가 cloned tensor payload의 4 KiB 이내 |
| E NVLink TX, P 구간 | pair 간 ±5 KiB 이내 |
| NIXL transfer bytes | 18/18 동일. descriptor 수에는 변화가 있었음 |
| D clone host enclosure | 중앙값 +10.61 ms, 16/18 증가 |
| D one-token HTTP | 중앙값 +21.30 ms, 범위 −32.58~+68.99 ms, 15/18 증가 |
| E storage snapshot | 변화 관측 없음 |
| Producer host-marker lifetime | 중앙값 +74.20 ms. 감소한 pair와 최대 +351.54 ms 증가한 pair 공존 |

한 토큰의 차이와 동일한 KV block 수로 조합을 고정했다. 입력과 반복은 동일한 장치에서 짝지었고, figure의 선은 동일 요청 반복의 중앙값이다. NVLink 계수는 독립 실행에서 읽은 device-wide 물리 tx 변화다. 원본 텐서 크기·clone counter 및 P 구간 불변성이 함께 관측된 덕분에 추가 D 방향 비용의 출처를 좁힐 수 있다. 링크 단위 완전 귀속 또는 온라인 peak resident memory까지는 측정하지 않았다. Host latency 증감은 시료에 따라 달랐다. 이 소규모 비용 시험으로 C3 TTFT의 순이익을 추정하지 않는다.

현재 E consumer가 whole image를 복제하는 경로의 H-M1은 MODALITY_EFFECT_MEASURED다. 이 현상은 이미 token slice만 전송하는 다른 구현으로 일반화할 수 없다. [Matched trial과 control](../mllm_epd_h200_modality_aware_dynamic_split/results/modality_matched_transfer.json) · [Figure 4](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig04_image_cut_work_matched.pdf)

## Inline-redirect로 다시 실행한 18개 matched pair

Inline으로 사용할 수 있는 descriptor는 D가 받아서 쓰고, P가 이미 사용한 crossing-image descriptor는 별도 export 경로를 사용한다. 이 같은 transport에서도 18/18 intra-image arm이 원본 이미지 전체에 대한 D clone 하나를 더 수행했다. D 구간 E NVLink TX는 +3.874–53.253 MiB, 중앙값 +32.812 MiB였고 매 pair에서 복제 payload와 4 KiB 이내였다. NIXL payload와 관측한 E storage는 각 pair에서 같았다.

Intra-image의 D one-token HTTP 증가는 중앙값 +24.970 ms이며 범위는 −0.947~+60.200 ms, 17/18 pair에서 증가했다. D clone host enclosure 중앙값은 +18.444 ms, producer host-marker lifetime은 +51.933 ms였다. 복제 메커니즘은 두 transport에서 재현됐고 지연 추정과 storage snapshot은 실행별로 보존했다. [Inline matched pair 원자료](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/modality_matched_transfer.json) · [Inline Figure 4](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig04_image_cut_work_matched.pdf)

| Inline 요청 구조 | 요청 / 반복 pair | Input tokens 범위 | Intra−edge E TX 중앙값 (MiB) | D HTTP 중앙값 차이 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Single image | 2 / 4 | 4,831–7,823 | +43.033 | +40.004 |
| Few images | 5 / 10 | 2,961–21,193 | +32.813 | +24.970 |
| Long document | 2 / 4 | 25,084–30,195 | +3.878 | +4.665 |

측정된 TX 증가가 가장 큰 pair는 7,823-token 단일 이미지 요청이었다. Suffix 19 대 18 tokens에서 intra의 E TX는 +53.253 MiB, D HTTP는 +37.307 ms였다. D HTTP가 가장 낮아진 pair는 25,084-token 44-image 문서였다. Suffix 2,312 대 2,311에서 intra는 E TX +3.874 MiB를 보냈지만 D HTTP는 0.947 ms 더 짧았다. 이 사례 선택 규칙은 각각 측정된 최대 byte 차이와 최소 시간 차이다. 표의 문맥 길이 구간은 구조 간에 다르므로 같은 입력 길이에서의 image-count 효과는 여전히 INCONCLUSIVE다.
