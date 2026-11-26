# Modality 정보를 검사하는 이유와 선행 범위

Vision Encoder를 지난 Visual/Text embedding에는 동일한 dense LLM 경로가 적용된다. 모달리티 비용은 실제 전송, storage, readiness, scheduling 메커니즘에서 검증한다. Visual이 입력의 99.37%라는 기존 통계만으로는 추가 feature의 성능 효과를 판정할 수 없다. [동결 workload와 Counter 범위](mllm_epd_prefill_heavy_token_split_final_summary.md)

| 가설 | 직접 검사 | 범위 |
| --- | --- | --- |
| H-M1 | 비슷한 suffix work에서 intra-image cut이 추가 descriptor, 실제 materialization 또는 storage lifetime을 만드는가 | MODALITY_EFFECT_MEASURED. 각 transport의 18/18 pair에서 whole-image clone 하나를 더 실행했다. Inline E TX +3.874–53.253 MiB |
| H-M2 | 동일 predictor에서 image/encode feature가 heldout suffix 비용과 false-safe를 낮추는가 | NO_INCREMENTAL_MODALITY_EFFECT. 동일 Ridge의 full-feature fit은 두 transport에서 tail 과소예측을 줄였고 MAE를 높였다. Inline의 boundary-only MAE 차이는 0.009 ms였다 |
| H-M3 | 동일 budget과 실제 cut 탐색에서 image-aware 제약이 실행 성공률과 D/P 비용을 개선하는가 | NOT_MEASURED |
| H-M4 | 비슷한 문맥 길이와 출력 조건에서도 image count와 span 분포가 비용 차이를 설명하는가 | INCONCLUSIVE. 구조와 중복 payload 크기는 확인했고 matched online strata의 효과는 NOT_MEASURED |

Whole-request encoding에서는 개별 이미지 ready time이 runtime에 전달되는지 확인한 뒤 사용한다. 입력의 image 전체 embedding storage 참조, 실제 slice, P/D 복제, IPC descriptor, bytes와 storage lifetime을 분리한다. 기존 Native EC descriptor 통제에서도 inline 경로는 중요한 교란 변수였다. [Native EC causal control](mllm_epd_native_ec_descriptor_causal_control_v1.md)

## 이번 메타데이터 검사

원본 NPZ SHA, modality mask와 Qwen2.5-VL 2×2 merge grid를 Main100 및 Calibration32의 132개 요청에서 검증했다. Main100은 1,251개 이미지와 1,102,959개 입력 토큰을 포함하며 그중 1,095,976개는 Visual이다. Calibration32는 405개 이미지다. Whole-request E 응답의 완료는 관측 가능하다. Runtime에서 개별 이미지의 GPU-ready 시점과 encoder 실행 시간은 NOT_AVAILABLE이다.

| Nominal suffix | Main100의 intra-image cut | Calibration32의 intra-image cut |
| ---: | ---: | ---: |
| 5% | 98/100 | 31/32 |
| 10% | 99/100 | 31/32 |
| 20% | 99/100 | 32/32 |
| 30% | 100/100 | 32/32 |

5–30% suffix 범위 안에서 image span의 시작 또는 끝이 하나라도 있는 경우는 Main100 55건과 Calibration32 16건이다. 이는 image boundary hard constraint가 후보를 제한할 수 있다는 구조적 증거다. 이 값에는 실제 P chunk와 D budget 검사를 적용하지 않았다.

| 구조 | Main100 건수 | 입력 토큰 중앙값 | 5–30% true boundary 후보 건수 | Nominal 10% cut과 겹치는 이미지 payload 중앙값 (MiB) | Calibration32 건수 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single image | 13 | 4,847 | 0 | 32.94 | 4 |
| Few images (2–4) | 45 | 4,651 | 13 | 13.60 | 15 |
| Many images (5+) | 5 | 28,845 | 5 | 32.94 | 1 |
| Long document | 37 | 14,869 | 37 | 4.21 | 12 |

Long-document 원본 stratum을 먼저 분리했다. Payload는 실제 cut 위치에서 겹치는 이미지의 BF16 크기이며 겹치는 이미지가 없으면 0이다. 이는 실제 online transfer가 아닌 구조적 크기다. 단일 이미지 요청에서는 후보 부족이 예상되는 반면 긴 문서에서는 boundary 선택 가능성이 더 높았다. Long-document에서 겹친 이미지 한 장의 payload 중앙값은 오히려 작았다. 해당 구조와 길이의 연관을 matched online 효과와 분리한다. [Frozen image span inventory](../mllm_epd_h200_modality_aware_dynamic_split/results/modality_inventory.json) · [Figure 6](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig06_frozen_workload_structure.pdf)

## 현재 H200 P-heavy 재검사

| NativeCommon QPS5 | E/P/D kernel busy | P 대기 표본 | P 처리량/이전 isolated tps | ≥160 ms quiet 겹침/P busy | Profile Decode SLO |
| --- | --- | ---: | ---: | ---: | ---: |
| B8 | 87.36% / 94.91% / 59.51% | 87.34% | 103.43% | 28.35%, 8개 | 99% |
| Free | 86.78% / 94.34% / 69.92% | 100% | 105.09% | 15.17%, 6개 | 95% |

두 capture에서 중앙 p20–p80 구간의 P 압력 및 D quiet가 관측됐다. 두 capture 모두 5% Clean/Profile/Clean gate에 실패해 DIAGNOSTIC이다. 공식 판정은 P_HEAVY_D_IDLE_NOT_MEASURED다. P 대기는 최근 Scheduler 표본으로 보간된 시간의 비율이다. D quiet 겹침의 분모는 P busy 시간이다. [현재 Motivation 영수증](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_motivation.json)

B8의 clean 후 TTFT p50은 clean 전보다 24.51% 높았고 ITL p99는 9.14% 낮았다. Profile TPOT p95는 clean 평균보다 15.92% 높았다. Free의 clean 후 TTFT p50은 14.32% 높았고 profile TPOT p95는 clean 평균보다 6.03% 높았다. 시작 내부의 clean drift와 profile 편차가 모두 관측돼 정확한 원인 귀속은 미확정이다. [5% 기준별 observer 결과](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_observer.csv)

## 읽은 선행 연구

- FluidPD의 v1 §IV-B·Fig.6은 SGLang P/D 흐름에서 P prefix와 D suffix를 분리하고 D의 진행 중 decode를 일시 지연한다. PPI/DPI 및 batch/context 비용 predictor를 사용하며 v1 정책은 stream-average TPOT의 누적 여유를 허용한다. v2 §4.3·Fig.5–6은 예측한 step slack에 η=1을 적용하고 suffix를 나누며 최소 진행량 256 tokens 예외를 둔다. 완료 요청은 기존 decode batch에 합류한다. 이번 실험은 active가 없는 D만 예약하며 실제 client ITL≤34 ms 전체를 검사한다. 사용한 원문과 연결된 공식 SGLang PD 구현을 확인했다. FluidToken의 독립 공개 코드와 실험 D cap 숫자는 확인되지 않았다. [FluidPD v1](https://arxiv.org/html/2610.06917v1) · [FluidPD v2](https://arxiv.org/html/2610.06917v2) · [SGLang PD multiplexing](https://github.com/sgl-project/sglang/pull/11592)
- DuetServe는 실행 간섭이 지연을 위협할 때 GPU 내부에서 SM을 P/D에 분할한다. Kairos는 요청 전체의 prefill을 D로 보내고 in-flight decode 사이에 SLO를 고려해 prefill chunk를 삽입한다. 두 설계의 범위를 논의하며 재현 arms를 사용하지 않는다. [DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) · [Kairos](https://arxiv.org/html/2607.02043v1)
- EPD disaggregation은 E/P/D 배치, multimedia token caching, 요청 내부 encoding 병렬화 및 역할 전환을 다뤘다. HydraInfer는 E/P/D 인스턴스에서 stage별 batch와 migration을 스케줄한다. EAServe는 Encode 출구에서 microbatch 및 전송률을 제어하고 co-resident prefill과 SM 공간 분할을 사용한다. [EPD](https://proceedings.mlr.press/v267/singh25d.html) · [EPD code](https://github.com/vbdi/epdserve) · [HydraInfer](https://arxiv.org/html/2505.12658v2) · [EAServe](https://arxiv.org/html/2609.31551v1)
- RServe/REDServe는 image embedding readiness를 추적해 요청 내 encode와 prefill을 겹친다. ModServe는 modality/stage에 따라 배치와 자원 배분을 조정한다. vLLM V1의 disable-chunked-mm-input은 chunked prefill에서 image 같은 하나의 multimodal item을 나눠 스케줄하는 것을 막는 옵션이다. 그 옵션의 의미는 cross-GPU cut의 전송 비용이나 exactness를 보장하지 않는다. [RServe](https://arxiv.org/html/2509.24381v1) · [ModServe](https://arxiv.org/html/2502.00937v3) · [ModServe code](https://github.com/Azure/ModServe) · [vLLM engine options](https://docs.vllm.ai/en/latest/configuration/engine_args/#disable-chunked-mm-input-enable-chunked-mm-input)

Generic dynamic ratio, adaptive placement, image readiness 및 stage-aware resource allocation은 각각 선행 범위에 있다. 논문 주장은 동일 실행기에서 확인한 모달리티 feature의 추가 효과가 있을 때 그 메커니즘과 조건으로 제한한다.
