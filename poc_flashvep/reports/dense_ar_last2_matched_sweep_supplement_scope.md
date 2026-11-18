# Last2 Matched Sweep 추가 범위

## 사용자 취소

사용자는 추가 시간을 피하기 위해 Last2 Sweep 요청을 철회했다.
추가 Controller를 GPU 실행 전에 중지했다.
이 추가 범위에서 실행한 성능 Start는 0회이다.
추가 State 정합성 Start도 0회이다.
아래 계획과 예상 시간은 취소된 설계 기록으로만 보존한다.
준비된 설정의 상태는 `NOT_RUN`이다.
기존 Last4/8/12 Matrix와 GPU 실행은 유지한다.
이전에 완료한 별도의 Serial Last2 Diagnostic은 변경하지 않는다.
취소 전에 생성한 Last2 제목의 Preview는 Baseline Curve만 포함했다.
이 Preview는 `cancelled_last2_preparation_previews/`에 별도로 보존한다.
이를 Last2 Serving 성능으로 사용하지 않는다.

## 요청과 기존 범위

사용자는 Static Last2/4/8을 Sweep에 추가하도록 요청했다.
기존 Discovery 232회에는 Last4/8/12와 각 Matched Token 비교가 이미 포함됐다.
기존 Confirmation 120회에도 세 Pair가 포함됐다.
따라서 Last4와 Last8을 중복 실행하지 않는다.

이번 추가 범위는 Last2와 Matched Token432/448이다.
사용자는 Last2/Last4의 Serial Diagnostic 결과를 확인한 후 이 범위를 요청했다.
따라서 이 범위를 최초 사전 계획으로 표시하지 않는다.
기존 설정, 원시 Evidence, SLO 및 Inference Source Freeze를 유지한다.

## 실행 계약

모델은 Qwen/Qwen3-8B이다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16과 Full36 Decode를 유지한다.
GPU0/1과 기존 KV42GiB/GPU 계약을 사용한다.
두 Worker에 B8192/S1024를 적용한다.
공통 D Admission 상한은 8이다.
실제 Layer 준비 Group은 Query Budget 때문에 1일 수 있다.
Token Group은 같은 Admission 상한 안에서 최대 8이다.
이 차이를 숨기거나 같은 실제 Batch라고 표시하지 않는다.

N144에서 Layer Last2는 D에 2,304,000 GEMM Token-Layer Rows를 이전한다.
Matched Token은 432 Tokens를 32개 요청에서 사용한다.
나머지 112개 요청은 448 Tokens를 사용한다.
Token의 전체 D 작업량도 2,304,000 Rows이다.
동일 Rows는 동일 Attention FLOPs나 동일 GPU 실행시간을 의미하지 않는다.

## 실행 수

| 구분 | 조건 | 추가 Start 수 |
|---|---|---:|
| Discovery | Output16/32/64/128/200, QPS2/4/6/8/10, 두 방법 | 50 |
| Output200 추가 점 | QPS3/5/7/9, 두 방법 | 8 |
| Confirmation | 다섯 Output, QPS6, 세 Arrival Seed, 두 방법 | 30 |
| 합계 | GPU Serving Performance Start | 88 |

Discovery Seed는 2026111780이다.
Confirmation Seed는 2026111790/2026111791/2026111792이다.
기존 Native 및 Agg의 같은 조건을 재사용한다.
N144의 기존 실제 문서 Prompt Pool을 사용한다.
새로운 미관측 Prompt Pool이라고 주장하지 않는다.

추가 성능 실행 전에 N8 → N32 → N128 State 정합성을 검사한다.
Output200에서는 N8 정합성을 별도로 검사한다.
각 검사에서 Native, Layer2, Token2를 실행한다.
새 Native는 Token432/448의 Hidden Partition Hash를 보존한다.
이 12개 Correctness Start는 위 88개 성능 Start에 포함하지 않는다.

## 예상 시간과 증거 수준

시간 예측에는 완료된 Last4 Pair의 Dispatcher 기록을 사용했다.
Output별 평균 Start 시간과 추가 Start 수를 곱했다.
88개 성능 Start의 합계는 약 6.98시간이다.
이 값은 `MODELED_ESTIMATE`이다.
Last2 실측 소요시간은 아니다.
정합성 검사와 운영 여유를 포함한 추가 시간은 약 7–9시간으로 예상한다.
기존 대기 작업과 공유하는 시간은 개별 첫 결과의 도착을 늦출 수 있다.
이 예측은 전체 캠페인의 잔여시간을 의미하지 않는다.

완료한 Start만 `MEASURED`로 표시한다.
정합성 검사는 `CORRECTNESS_ONLY`로 표시한다.
아직 실행하지 않은 성능 조건은 `NOT_RUN`이다.
Serial Diagnostic의 D Forward 이득을 Online TTFT 이득으로 대체하지 않는다.
독립 Start 기반 CI와 Native-relative Decode 비열등성을 별도로 검사한다.

## 재개와 보존

추가 Controller는 `last2_supplement.py`이다.
설정과 진행 Manifest는 기존 Evidence Namespace의 `last2_supplement_*` 파일에 저장한다.
기존 GPU 작업을 종료하지 않는다.
추가 작업도 기존 Quiet-lane Lease를 획득한 뒤 실행한다.
기존 Matrix를 수정하지 않는다.
최종 분석은 추가 88개 Start와 정합성 Gate의 완료도 검사한다.
Figure는 Last2/4/8/12를 구분한다.
Last12 기본 Figure Alias는 유지한다.
