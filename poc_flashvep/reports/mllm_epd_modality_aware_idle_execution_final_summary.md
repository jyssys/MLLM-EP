# Vision-heavy EPD의 Decode 유휴 자원 탐색

## 결론

**투입한 H200 세 대와 Qwen2.5-VL-7B에서 온라인으로 승격할 신규 Modality-aware 방식은 확보하지 못했다.** 원본 장문 문서 부하에서 P에 대기 작업이 쌓이고 D는 상대적으로 가벼운 상태를 재현했다. D는 기존 가중치로 실제 Vision 작업을 수행할 수 있었다. 같은 여섯 장을 E/D로 나누니 GPU 간 전달과 ACK를 포함한 Vision 경로가 **274.46→186.08 ms, 32.2% 감소**했고 원래 E 결과와 **4/4 bitwise** 일치했다. 그러나 주 부하에서 빠른 E 완료가 P handoff를 당길 근거가 부족했고, 활성 Decode 중 실행하자 최대 ITL이 **6.99→66.20/123.26 ms**가 됐다. 모델의 첫 전역 Vision attention 앞에서 분할하는 새 경로도 **21/21 cut에서 E 단독보다 느렸다**. [물리 결과](mllm_epd_idle_candidate_physical_screening.md) · [Decode 간섭 실험](mllm_epd_d_side_encoder_sharing.md)

실제 Online N64에서 기존 Generic E→D Isolation은 Native EPD 대비 TTFT p95를 **16.129→11.283 s, 30.0% 감소**시켰고 Joint Goodput은 **0.144→0.390 req/s**였다. TPOT p95 **4.536→4.847 ms, +6.86%**, ITL p99 **12.470→13.294 ms, +6.61%**로 각각 +5% 한도를 넘었다. 절대 Decode SLO는 양쪽 모두 64/64였다. 두 우선순위를 적용하면 이 실행은 서비스 수준 절충의 실측 사례다. Strict Decode Protection은 실패했다. [온라인 대조](mllm_epd_decode_protected_online_serving.md) · [TTFT와 Decode Pareto](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_pareto.svg)

별도 재시작의 비용 추정 Generic은 같은 N64 요청 중 11개를 D에서 직접 Prefill했다. TTFT p95는 10.016 s, Goodput은 0.449 req/s였다. TPOT p95 6.674 ms와 ITL p99 23.658 ms는 Native 대비 각각 +47.14%, +89.72%였고 절대 Decode SLO는 63/64였다. 이 비교군은 현재 런타임에서 관측 가능한 HTTP 상태와 추정 비용을 사용했다. [비용 기반 기준선](mllm_epd_decode_protected_online_serving.md)

## 동일 모델로 재현한 범위

H200 GPU4/5/6을 각각 E/P/D로 고정했다. Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 원본 BF16과 MRoPE, B8192/S128, P/D 각 42 GiB KV, FA3, CUDA Graph, encoder CUDA IPC 및 P→D NIXL을 사용했다. APC와 요청 간 Embedding 재사용을 끈 상태에서 원본 문서 24개를 반복했다. 중앙값은 24,790 LM 토큰, 42.5장, 약 99.8% Visual이며 출력은 Fixed16, offered QPS 1.465다. N은 독립 요청 제출 수다. 이전 AR 또는 다른 모델의 속도를 현재 H200 측정으로 환산하지 않았다. [실행 계약](mllm_epd_modality_aware_idle_execution_poc_spec.md) · [데이터 및 Geometry](mllm_epd_idle_execution_workloads_and_vision_geometry.md)

새 독립 Stage 실험의 E/P/D 처리율은 각각 1.639/1.221/6.539 req/s였다. E는 원본 Encoder HTTP와 embedding export를 수행했다. P는 준비한 embedding으로 B8192에서 실제 Prefill하고 KV를 해제했다. D는 B1로 이미 준비한 원본 KV를 NIXL로 받아 Fixed16을 생성했다. 각 값은 유한 요청의 별도 물리 Calibration이다. 과거 동일 부하의 B16384 P 대조는 1.256 req/s였다. [Stage 및 Online 표](mllm_epd_decode_protected_online_serving.md)

### 동일 관측 구간과 Observer 판정

**Motivation은 `CONDITIONAL_PREFILL_HEAVY_EPD`다.** 새 Exact CUPTI N128의 세 GPU 공통 36.580 s에서 E/P/D Kernel Busy가 **88.83/92.51/9.84%**, E와 P가 동시에 바쁘고 D Kernel이 없는 비율이 **73.98%**였다. P의 Embedding-ready 대기 표본은 96.72%였고, P throughput 32,107 tok/s는 같은 부하의 최적 B16384 독립 대조 대비 97.90%였다. P KV 할당 거절은 0/111, D Capacity Wait 표본은 0/129였다. D Kernel idle과 proxy request-free 교집합은 80.35%, 연속 구간 p50/p95/max는 625/882/1,117 ms다. Native NIXL READ 및 copy 구간의 독립 계측은 이 에폭에 없었다. [공통 구간 및 동기화](mllm_epd_idle_motivation_qualification.md)

Exact 계측의 clean/profile/clean은 각각 128/128 완료했고 출력 토큰도 일치했다. Clean 중앙값 대비 Profile의 절대 변화는 TTFT p50 5.39%, TPOT p95 15.64%, ITL p99 6.58%였다. 두 clean 실행의 ITL p99 자체 차이도 11.18%였다. 5% Observer 가드는 실패했으며, 계측 구간 내 절대 Decode 통과는 44/45였다. 따라서 공동 GPU 수치는 `DIAGNOSTIC`이다. 앞선 graph-only 시도는 D의 CUDA graph replay 중 SIGSEGV로 끝났고, 위 수치는 그 뒤의 별도 exact-kernel 에폭에서 얻었다. [Observer 검증](mllm_epd_idle_motivation_qualification.md) · [Graph 실패 영수증](../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility/graph_failure_receipt.json)

자연 EOS 확인용으로 별도의 원본 장문 문서 요청 24개를 사전에 동결했다. 원래 소스·문서·이미지 해시가 주 코호트와 겹치지 않는다. 이미지와 질문은 그대로 유지했으며 출력 token budget에 Fixed16을 적용하지 않았다. 서로 다른 모델에서의 재현은 `NOT_RUN`이다. [워크로드 동결 영수증](../mllm_epd_modality_aware_idle_execution_discovery/results/cohort_receipts.json)

Native Fixed16의 QPS1.465에서 N128은 128/128 완료, Decode SLO 127/128, TTFT p95 29.944 s였다. N256은 256/256 완료, Decode SLO 254/256, TTFT p95 61.866 s였다. Joint SLO는 N256에서 6/256이었다. 같은 offered load에서 코호트를 늘리며 TTFT가 증가했으므로 이 지점의 finite-cohort drain 결과를 지속 가능한 서비스 용량으로 해석할 수 없다. N256에서도 D가 HTTP request-free인 시간은 82.19%였고, P outstanding ≥2인 의사결정 시점에서 다음 native P-ready까지 최대 1,060.5 ms였다. 일정한 1,150 ms 작업에 25 ms 여유를 더하면 사후에도 적합한 시작점이 없었다. [지속성 도표](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_persistence.svg) · [Idle 분석](mllm_epd_idle_motivation_qualification.md)

별도 자연 EOS N24의 Native→Generic Isolation은 모든 요청을 EOS로 완료했다. 실제 길이는 2–24 tokens, 중앙값 15였다. TTFT p95는 7.207→7.311 s, TPOT p95는 +20.68%, ITL p99는 10.953→15.002 ms, Joint Goodput은 0.967→0.922 req/s였다. 각 prompt와 생성 token은 24/24 일치했다. 절대 Decode SLO는 양쪽 24/24였고 Strict는 실패했다. 일치 결과는 해당 24개 출력에 한정된다. [Natural EOS 실측](mllm_epd_decode_protected_online_serving.md)

같은 물리 GPU 세 대와 동일 도착열의 DP3 B2048은 N128에서 TTFT p95 3.852 s, TPOT p95 112.257 ms, ITL p99 163.032 ms, 절대 Decode SLO 43/128, Joint Goodput 0.569 req/s였다. 별도 Natural EOS N24는 3.820 s, 108.507 ms, 146.341 ms, 8/24, 0.583 req/s였다. 두 쌍 모두 prompt identity를 확인했다. DP3의 실제 낮은 TTFT가 strict Decode 성능과 함께 유지되는 지점은 이 대조에서 확인되지 않았다. [동일 GPU 예산 대조](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_pareto.svg)

같은 언어 모델의 24,576-token Text-only 두 대조는 동일한 입력·출력과 Fixed16을 사용했다. Poisson QPS1.465에서 실제 이미지 E-ready 시각 64개 재생으로 바꾸자 TTFT p95는 7.777→7.400 s, ITL p99는 10.815→13.126 ms였다. 두 경로 모두 절대 Decode SLO 64/64였다. 재생한 도착률은 관측 구간 기준 1.617 req/s였다. 텍스트와 멀티모달 사이의 입력 내용 및 처리 경로 차이, 그리고 두 텍스트 실행의 도착률 차이가 남아 있으므로 modality 고유 효과 크기를 이 결과에서 구하지 않았다. [Text-only 대조](mllm_epd_decode_protected_online_serving.md)

## 24개 연구 질문

| 질문 | 증거와 결론 |
| --- | --- |
| 1–3. 부하 재현, 적격 계측, 지속성 | 실제 Native Online, 고정 코호트와 독립 재시작을 사용했다. 아래 계측 판정과 [동시 타임라인](mllm_epd_idle_motivation_qualification.md)에 clean, kernel, 실패한 graph 순서와 별도 관측 경계를 기록했다. |
| 4. Deadline-safe D idle | Clean N64/N128의 D proxy request-free 비율은 각각 약 82%였다. E-ready 시 P outstanding ≥2인 후보의 다음 실제 P-ready까지 중앙값 364/315 ms, 최대 644/858 ms였다. 250 ms 작업, 25 ms reserve의 고정 p10 인과 예측은 사후 점수에서 6/6과 10/10 fit이었다. 실제 deadline-safe로 완료한 온라인 후보 시간은 아직 미측정이다. |
| 5–6. D의 Vision 실행과 비용 | D에 1,353,100,288 B의 원본 Vision 가중치가 상주했고 전용 RPC로 물리 실행했다. 각 한 장의 D Vision GPU 중앙값은 70.15–117.89 ms, 전체 복사·검증·ACK 포함 표본 p95는 101.60–255.75 ms였다. 전체 7개 fixture에서 5개가 물리 p95 250 ms 안에 포함됐다. 원래 embedding과 28/28 bitwise 일치했다. |
| 7. E 보조가 P 병목을 푸는가 | 주 부하에서 40/40, 74/74 중앙 E 완료 시 더 오래된 P HTTP 작업이 있었다. 비용 없는 88.38 ms E 가속을 준 관측 순서 고정 replay는 새 P-ready를 0/40, 0/74 앞당겼다. 저부하 replay에서는 22/40이 앞당겨졌다. P 순서 재편을 실행한 실험은 없다. |
| 8. Joint Encode–Prefill | 전체 D-local LM Prefill만 약 1.15 s로 관측한 주 부하 idle 후보보다 길었다. E와 P 일을 함께 제거하는 작은 exact joint quantum은 `NOT_RUN`이다. |
| 9–11. Modality 실행 단위와 Slice | B의 이전 Slice 대조에서 인접 interior cut은 image boundary보다 중앙값 7 KiB를 더 보냈다. F의 새 Vision window 단위는 독립성이 있으나 이번 양방향 실행은 최선도 5.9% 느렸다. E-only split control도 같은 수치 차이를 관측했다. E의 Merger/다른 구조의 DeepStack 전용 실행은 `NOT_RUN`이다. |
| 12. Encoder readiness | 실제 24개 요청, 873개 이미지 IPC와 P 사용 기록을 정확히 대조했다. ≥25 ms 먼저 준비된 연속 causal prefix가 7/24, median 12,532행이었다. 조건 ≥50 ms에는 6/24, median 7,548행이었다. 이 diagnostic trace에서 실제 P 제출은 24/24 모두 전체 E ACK 이후였고 조기 LM 실행은 미측정이다. |
| 13. 자체 발견 Mechanism | Qwen의 첫 full-attention 앞 정확한 window frontier를 물리 검증했다. 21개 모두 native E보다 느렸다. 다음으로 구조적 구분이 가능한 방향은 실제 in-flight 이미지 준비 시점의 recipient 선택이지만 이번엔 역산된 시간과 causal prefix의 존재까지만 검증했다. |
| 14–15. D에서 실제 완료 및 기존 Work 감소 | A와 F에서 D의 실제 GPU 작업과 전달을 완료했다. A는 고립된 E batch의 critical path를 줄였다. 살아 있는 Native Decode와 겹칠 때 A는 절대 SLO를 0/8로 낮췄다. P의 실제 온라인 Work 또는 전체 Critical Path가 줄었다는 증거는 없다. |
| 16–20. TTFT, TPOT, ITL, 절대 SLO, Generic 및 DP3 | 같은 3 GPU의 Generic Isolation은 위 N64 성능과 tail trade-off를 보였다. 새 modality 후보가 같은 강한 Generic을 개선한 온라인 결과는 없다. DP3와 Native/Generic을 동일 목표의 Pareto에 두고 절대·상대 SLO를 각각 표시했다. |
| 21. Natural EOS | 별도 원본 장문 QA로 Native/Generic 및 가능한 DP3를 평가했다. EOS와 실제 길이, 응답 ID 및 지원되는 짝의 정규화 출력은 성능 보고서에 있다. 별도의 채점 비열등성은 검증하지 않았다. |
| 22–24. Modality ablation, 모델 일반성, 논문 기여 | 최종 후보를 승격하지 않아 온라인 modality 제거 실험과 다른 MLLM 검증은 `NOT_RUN`이다. 같은 backbone의 Text-only 및 실제 E-ready 도착열 재생은 독립 대조로 유지했다. 이번에 확인한 비전 병렬성만으로 기존 EPD/ModServe/SpaceServe와 구별되는 논문 기여를 입증하지 못했다. |

관련 증거와 판정 범위를 [E-only Critical Path](mllm_epd_encoder_critical_path_bound.md), [정확한 준비 순서](mllm_epd_readiness_aware_recipient_execution.md), [A–G 조사](mllm_epd_autonomous_candidate_discovery.md), [기존 연구 및 공개 구현](mllm_epd_idle_execution_prior_work.md)에서 확인할 수 있다.

## 승격 판정과 우선순위

| 물리 게이트 | 통과 범위 |
| --- | --- |
| P0 · 실행 Mechanism | A의 이미지 독립 실행과 F의 pre-global window를 원본 모델에서 확인했다. |
| P1 · 물리 정합성 | A의 embedding 경로는 통과했다. F의 일부 split shape가 불일치했다. C–E·G의 새 LM, state 및 모델별 실행은 검증 범위 밖이다. |
| P2 · Useful Work | A는 고립된 Vision critical path에서 통과했다. F는 E/D로 실제 실행했으며 같은 경로의 총 시간이 증가했다. |
| P3 · Decode 보호 | 활성 Decode의 A 개입은 실패했다. 오프라인에서 fit한 유휴퀀텀을 인과적으로 실행하는 실측은 남았다. |
| P4 · 같은 예산의 추가 이점 | 승격 조건을 충족한 경로는 없다. |
| P5 · Online 적용 기회 | 이미 도착한 사건을 이용한 causal replay와 사후 상한까지만 확인했다. 신규 후보의 Native Online 데이터는 없다. |

| 후보 | 판정 | 이유 |
| --- | --- | --- |
| A · 완전한 이미지 단위 D Vision | `CONDITIONAL_PHYSICAL_SIGNAL` | 고립된 실제 E/D 경로는 32.2% 빨랐다. 활성 Decode RPC는 `NO_GO_DECODE_INTERFERENCE`다. 주 부하에서의 P Critical Path 이득은 관측 순서 고정 replay에서 확인되지 않았다. |
| B · 이미지 경계에 따른 LM 실행 단위 | `NO_GO_INCREMENTAL` | 같은 Slice transport의 기존 비교가 약한 경계 차이를 보였고, Hard cut은 실제 Generic offload를 제외했다. 이번에 LM chunk 자체가 빨라졌다는 새 결과는 없다. |
| C · D에서 Encode와 Prefill 결합 | `NO_GO_IDLE_OPPORTUNITY` | 전체 요청 단위에 적용한다. 더 짧은 Prefix 방식은 미구현이다. |
| D · 실제 readiness 기반 P/D 소비자 배치 | `NOT_RUN` | 정확한 causal prefix를 발견했다. IPC readiness와 LM 실행 사이의 실제 독립 스케줄 경계가 필요하다. |
| E · Merger 및 모델별 주입 구조 | `NOT_RUN` | 원본의 Vision merger 전송량은 BF16 기준 동일 merged row당 10,240 B이며 native embedding은 7,168 B다. 다른 구조의 비교는 미실행이다. |
| F · 첫 Vision global block까지의 window 분할 | `NO_GO_COMMUNICATION` | 21개 cut 모두 느렸다. Native 대비 28/84 bitwise와 최대 상대 L2 0.09559도 별도로 확인했다. |
| G · 이미 준비된 visual 행의 첫 LM QKV | `NO_GO_INCREMENTAL` | 설계의 초벌 계산으로 QKV 9,216 B/row와 기존 7,168 B/row를 비교했다. 남은 Residual과 causal attention을 P에서 수행해야 한다. 물리 실험은 미실행이다. |

A의 물리적 이점은 실제 P 대기와 일정 경쟁이 낮은 조건에서만 재검토할 가치가 있다. 다음 실행 단위를 찾는다면 D의 **실제 in-flight Vision readiness**와 P 대비 **독립적인 causal LM 실행**을 먼저 대조한다. 최소 반증 조건은 이미지 여러 장이 E에서 준비된 정확한 시점에 P 또는 D가 원래 위치와 KV를 유지한 1개 chunk를 실제 완료하고, 같은 frontier를 제공한 generic scheduler 대비 최초 head start가 전달 및 admission 비용을 넘는지 비교하는 것이다. 모델 전반의 재현이나 논문 기여를 입증하려면 그 뒤 동일 예산의 엄격한 온라인 검증이 필요하다. [판정 도표](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_candidate_verdict_matrix.svg) · [실행 DAG](mllm_epd_execution_dag_and_resource_analysis.md)

## 재현 및 정리

고정 실행 명령과 환경, 원본 자산 및 런타임 해시, 에폭별 실측 요청·스케줄·CUPTI·Vision 기록을 남겼다. 커밋용 아카이브에는 코호트/타임스탬프, 지표 재계산용 값, 출력 토큰 및 정규화 텍스트 digest를 담았다. 지연 재생의 입력은 exact cohort ID로 수정하고 워밍업을 제외했다. GPU가 지정된 실험에 참여하는 동안 승인된 나머지 다섯 대의 `train_lm.py`를 유지했고, 실험 종료 후 GPU0–7의 해당 프로세스와 GPU 활동을 확인했다. 같은 자원 조건을 쌍마다 적용했다. 외부 burner의 공유 Host/NVSwitch 영향은 별도로 정량화하지 않았다. 관련 CPU 회귀 검사 18개를 통과했다. [재현 명령](mllm_epd_modality_aware_idle_execution_poc_spec.md) · [아카이브와 해시](../mllm_epd_modality_aware_idle_execution_discovery/evidence/) · [GPU 복원](../mllm_epd_modality_aware_idle_execution_discovery/results/burner_restore.json)
