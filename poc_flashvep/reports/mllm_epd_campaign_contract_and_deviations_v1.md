# MLLM EPD: 실행 계약·최신 결정·편차 감사

이 문서는 실행 범위를 확인한다. 성능이나 품질 PASS를 부여하는 문서가 아니다.
이전 보고서와 Raw Evidence는 수정하지 않았다.

## Notion 결정의 적용 범위

2026-11-25에 [연구 방향 문서](https://app.notion.com/p/4220ee8277e48153a020e374317777f8)를 다시 조회했다.
반환 문서의 명시된 As-of는 `2026-11-24T08:13:32.709Z`다.
조회 시각만으로 그 이후의 모든 변경을 확인했다고 주장하지 않는다.
상단의 초기 MLLM100 변경과 Phase1 범위를 현재 결정으로 적용했다.
하단의 Video·4×4 Quota·다른 모델 계획은 역사적 설계로 구분했다.

실제 Main400과 Natural400 출력은 저장소에 이미 존재한다.
문서 하단의 역사적인 Manifest 미생성 문장을 현재 실행 상태로 사용하지 않았다.
Main100은 Main400의 원본 요청에서 선정했다. Main400을 삭제하거나 덮어쓰지 않았다.
분포 보존을 위해 이미 측정한 Natural 길이는 선정에 사용했다.
Method 성능·SLO·정답 점수는 선정 입력에 사용하지 않았다.
따라서 Main100은 출력 길이를 보기 전의 무작위 표본이 아니다.

Main100의 Mean과 중앙 분포는 원본에 가깝다. 모든 Tail이 동일하지는 않다.
출력 Max는 Main400의 2048에서 Main100의 609 Tokens로 달라졌다.
Cap Hit는 각각 1/400과 0/100이다. 원래 Cap 요청은 Main400에 남아 있다.
이 차이를 숨기거나 Main100을 Production 분포로 주장하지 않았다.

[실험 아카이브](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489)도 다시 조회했다.
연결된 Schema의 MLLM·PD 분류에서 EPD·Prefill 관련 제목과 요약을 검색했다.
지정한 Query의 결과는 9건이었다. 전체 아카이브를 모두 재검토한 것은 아니다.
과거 Looped Model·MoE·Wavefront 결과를 현재 Qwen2.5-VL의 실측으로 옮기지 않았다.

## 유지한 계약

| 계약 | 실제 적용과 증거 범위 |
|---|---|
| 모델 | Qwen2.5-VL-7B-Instruct, Original BF16, 원래 Vision·Processor·Full28·Causal Attention·MRoPE |
| 모델 Revision | cc594898137f460bfe9f0759e9844b3ce807cfb5 |
| Workload | Image-only Main400 보존. 사용자 변경 후 Main100을 후속 주 실험에 사용 |
| 제외 범위 | Video·Pruning·Modality-aware Learning·Layer/Hybrid·RL·근사 KV 없음 |
| 자원 | 주 비교는 같은 H100 3개. GPU4/5/6을 E/P/D로 사용 |
| 강한 기준선 | Agg DP3와 E+PD DP2의 Continuous Scheduler·Chunked Prefill 유지. NativeStrong의 Stock Compiled 경로 유지 |
| Sampling | Greedy Natural EOS와 Cap2048. 요청별 Equal-output Replay는 별도 |
| SLO | TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token |
| 상대 Gate | NativeStrong·NativeCommon 각각 ITL p99·TPOT p95의 Paired CI 상한≤1.05 |
| 실험 단위 | 독립 Server Start. 요청·Token·ABBA Period를 독립 Start로 취급하지 않음 |
| 근거 구분 | C3·C0·TRACE_ANALYSIS·OFFLINE_ORACLE·CORRECTNESS_ONLY·DIAGNOSTIC을 분리 |
| 정합성 | 원래 Assert·KV/Embedding Lifetime·Clone 완료 ACK 유지. Tensor·출력 덮어쓰기 없음 |

정합성 문제를 해결하기 위한 Processor Metadata·Transport·Compiler 제어는 별도 Runtime 변경이다.
이를 모델 Weight나 Vision 구조의 최적화로 부르지 않았다.
원래 Stock 경로의 수치 실패는 보존했다. 공통 Compiled 경로를 Stock과 동일하다고 표현하지 않았다.

## 중요한 Phase Gate 편차

원래 계획은 Native 정합성과 Prefill-heavy·안전한 여유를 먼저 확인한 뒤 Token Serving을 평가하도록 정했다.
Native의 일부 조건부 정합성과 Runtime Motivation은 확보했다.
하지만 E 비병목·P 포화·Service Proximity·D Safe Budget의 동시 조건은 확인하지 못했다.

그 상태에서 C34 Discovery와 C37의 보수적 Token C3를 실행했다.
추가 Suffix는 기존 D HTTP Stream이 없을 때만 시작했다.
검증되지 않은 Active-decode Window에 Prefill을 강제로 넣지는 않았다.
그래도 이 순서는 원래의 전체 Motivation Gate가 통과한 후에만 진행한다는 계획과 다르다.
이 편차를 삭제하거나 P-heavy 기준을 사후 완화하지 않는다.

따라서 C37은 실제 C3 실행이지만, 요청한 Prefill-heavy 영역의 성공 Confirmation은 아니다.
미확인 영역에서 보수적 Admission의 비용과 한계를 검사한 실제 Runtime 근거다.
낮은 Kernel Busy나 일부 Output 일치를 전체 Gate 통과로 사용하지 않았다.
Static10·Idle10의 Runtime NO-GO와 전체 품질 HOLD를 별도로 유지한다.

후속 Whole-request Deflection도 별도 대조군이다. Generic Token Split 성공을 대신하지 않는다.
E/D Prefix Probe는 실제 추가 계산이지만 Client 요청의 완료에 기여하지 않는다.
Probe의 폐기한 Output을 Throughput이나 Goodput에 더하지 않았다.
HTTP 보호 검사를 GPU-exclusive Encoder Service나 미래 Safe Window 검증으로 바꾸지 않는다.

## 계측·품질·반복의 미충족 항목

고정한 5% Gate를 통과한 Full CUPTI 근거는 저부하 C25다.
고부하·Short ROI·문서 Stratum의 실패는 DIAGNOSTIC으로 보존했다.
Clean Spread 실패를 Profiler의 인과 효과로 확정하지 않았다.
SM Active·Tensor Core Activity·HBM Bandwidth·독립 P Service Proximity는 미측정이다.

BI0의 다른 Encoder Batch에서 Embedding 수치 차이가 확인됐다.
Same-E 또는 Same-geometry 정합성을 Cross-geometry 품질 PASS로 확대하지 않았다.
MMLongBench-Doc의 공식 외부 Answer Extraction은 승인되지 않았다. 외부 유료 호출은 하지 않았다.
Raw Task Score와 일부 Task의 동일 점수는 전체 품질 검증을 대신하지 않는다.

Main100의 Source별 비율 보존이 Discovery50과 Heldout50 각각의 Source 균형을 보장하지 않는다.
원래 Split을 유지했고 Source 불균형을 보고했다.
Whole-request의 Fresh Context50은 별도로 동결했다. Main100을 결과에 따라 다시 선정한 것이 아니다.
N3 CI에는 제한된 해상도가 있다. N5 Confirmation을 수행한 것으로 표현하지 않았다.

## 아직 확장하지 않은 후속 연구

Notion의 Inter-request Pipeline·Layer-wise Wavefront·Elastic Context Parallelism·AEPP는 조건부 후속 계획이다.
이번 캠페인에서는 세 GPU의 새로운 Token Partition이나 Dynamic Modality Selector를 구현하지 않았다.
물리적 동시 Idle·Memory Fit·Offline Window Fit만으로 해당 확장의 GO를 선언하지 않는다.
실제 후보의 E2E 이득·품질·Encode/Decode 보호가 필요하다.

## 재현 기록

Evidence Namespace는 `mllm_epd_prefill_heavy_token_split_20261124/`다.
Notion 원시 재조회는 `references/notion_final_direction_refresh_v0.json`과 `references/notion_final_archive_refresh_v0.json`에 있다.
Schema 기반 Query는 `references/notion_final_archive_query_v0.json`에 있다.
Main100 선택 계약은 `configs/main100_selection_plan.json`이다.
원래 Manifest와 수치 한계는 [Main100 계약](mllm_prefill_heavy100_dataset_contract.md)에 있다.
실제 C37 결과와 Gate는 [Online Serving 보고서](mllm_epd_generic_token_split_online_serving.md)에 있다.
