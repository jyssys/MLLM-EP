# Architecture-first MLLM Disaggregation · 실행 계약

이번 탐색은 실제 Weight, Module, State에 도달하는 경계를 먼저 확인한다. Dense Qwen2.5-VL에서 발견한 의미적 민감도와 범용 Shape 효과는 구조적 근거의 대조로 사용한다. Dynamic Token Split, Image Boundary와 매 층 Remote Visual KV는 이전 결과에 보존한다. [Discovery200 결론](mllm_specific_disaggregation_discovery_final_summary.md) · [Dynamic/Slice-aware](mllm_epd_h200_dynamic_slice_aware_final_summary.md) · [Notion의 연구 방향 결정](https://app.notion.com/p/4220ee8277e48153a020e374317777f8)

## 환경과 실험 순서

- H200 8대의 기존 환경과 NV18를 재사용한다. 같은 실행군은 같은 GPU pair에서 측정한다. 실험 GPU의 확인된 `train_lm.py`만 중단하고 종료 후 복원한다.
- 시작 SHA `b4947a556134afc5214bdb1dd0412c7e961f989d`에서 새 worktree 및 branch를 사용한다.
- 실제 Source Screening과 Native 최적화 확인 뒤 공개 safetensors 두 Family를 확보한다. 우선 후보는 CogVLM2-19B와 IDEFICS-9B이다. IDEFICS-1과 IDEFICS3은 별개의 실행 구조다.
- 세 번째 구조 대조는 installed vLLM의 ERNIE-4.5-VL-28B-A3B-Thinking이다. Shared attention을 유지하면서 FFN expert pool을 row role로 선택하는지를 확인한다. 공식 체크포인트와 로컬 image-only 전처리를 사용한다. 별도 PD 후보 성능은 최초 골격이 현실적인 native smoke를 통과할 때만 확장한다.
- 기존 Discovery200 grouped-v2의 150/50 분리를 유지한다. CogVLM2의 공식 builder는 intact single-image를 사용한다. IDEFICS 프로파일에는 사전 Resource Envelope로 16 images까지 포함한다. 각 모델의 실제 Processor Token, Mask 및 입력 길이를 측정한다. 영상과 modality 진단용으로 질문 의미를 바꾼 데이터는 성능 cohort에서 제외한다. [표집 목록 및 제외 내역](../mllm_architecture_first_disaggregation_discovery/results/cohorts.json)
- Phase별 GPU 비용, 실제 전송량, 바이트 및 호출된 Weight를 별도 Raw에 남긴다. 불가능한 수집은 근거 범위를 적는다. Python 및 Host wrapper의 wall time은 GPU-exclusive kernel과 분리한다.

## 후보별 1차 결과와 대조군

| Family | Primary Screening Metric | 일반 대조 |
|---|---|---|
| CogVLM2 | 같은 최대 VRAM 예산에서 실제 온디바이스 KV 할당 가능한 요청 수. 원래 BF16 Decode의 Teacher 및 Greedy 일치. | D-resident fast language path, modality label에서 독립적인 Source reachability; Native와 같은 전체 KV 및 같은 준비 Batch. |
| IDEFICS-1 | 실제 Mask로 필요한 Decode Visual State의 byte와 TTFT/Decode transport-inclusive time. K/V norm과 전체 Self KV 정확성. | 동일 Raw State 후 D-side projection cache, 동일 Mask 제약의 일반 packing; Stock HF reprojection은 별도 대조. |
| ERNIE-VL | Native가 실제로 호출한 vision/text FFN bank와 phase별 residency. | 같은 vLLM의 text-only 입력 및 image+text에서 생성한 각 text token의 실제 호출 경로. |

기준선 설정과 실제 Toy Smoke에서 실패한 Runtime은 기록한다. 타깃 비용의 실측 약 15% 절감 또는 의미 있는 Weight/KV capacity 개선이 후속 기준이다. 통신과 추가 packing을 포함한 net 효과를 평가한다. 동일 Runtime의 강한 일반 대조에 해당 최적화가 이미 포함되면 독립적인 Modality Contribution 판정을 유보한다.

## 온라인 검증 계약

같은 모델과 실제 GPU 수에서 Capacity와 정확성 후보를 먼저 선택한다. Cognition model의 upstream 여부가 공식 vLLM에 없으면 upstream BF16 연산의 격리 타이밍과 실제 in-process open-loop를 구분한다. 역사적 요청 SLO를 TTFT ≤5,000 ms, 모든 ITL ≤34 ms, TPOT ≤33 ms로 동결한다. 부하 screening의 실제 완료율은 finite-cohort rate이다. 양의 후보에 한해 같은 Arrival에서 독립 N3와 Validation을 확장한다. 품질은 원래 질문과 점수 범위를 명시하고 완전한 외부 심사 없이 새 점수 기준으로 일반화하지 않는다.
