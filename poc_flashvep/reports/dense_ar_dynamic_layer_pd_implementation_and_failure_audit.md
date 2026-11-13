# Dense AR Dynamic Layer PD — 구현 및 실패 감사

상태 기준: 2026-11-13 05:37 KST.
이 문서는 중간 감사 기록이다.
최종 성능 판정은 별도 Final Summary에서 제공한다.

## 1. 보존한 실행 계약

모델은 Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
BF16과 36개 Layer를 유지했다.
Attention 근사, Layer 생략, Weight 변경은 적용하지 않았다.

기존 `dense_ar_layerwise_pd`의 Python Source 82개를 보존했다.
기존 최종 보고서도 보존했다.
각 GPU 실행 전에 이 83개 파일의 SHA256을 검사한다.
사용자 Checkout, Staged Index, External Runtime의 보존 Signature도 검사한다.

새 구현은 `poc_flashvep/dense_ar_dynamic_layer_pd/`에 있다.
Evidence는 `dense_ar_dynamic_layer_pd_crossover_20261113/`에 있다.
기존 Source와 Evidence는 수정하지 않았다.

## 2. 실제 요청별 Cut Protocol

Coordinator는 P Prefix 제출 전에 Cut을 선택한다.
Source Batch 안의 요청은 같은 Cut을 사용한다.
서로 다른 Source Batch는 다른 Cut을 사용할 수 있다.
D는 KV Credit 예약을 승인한다.
Coordinator는 Commit 이후에 P Prefix를 제출한다.

거절이나 Timeout은 Prefix 제출 전에 Native로 복귀한다.
Timeout 취소는 D의 취소 ACK를 기다린다.
Commit 이후에는 Prefix를 버리거나 전체 Forward를 다시 실행하지 않는다.
예약 Generation과 Cut은 요청 ID에 연결한다.
Native KV Allocator는 예약된 Future Credit을 제외한 자원만 할당한다.

P는 실제 Prefix Layer만 실행한다.
NIXL은 Prefix KV와 Hidden/Residual을 전달한다.
D는 실제 Suffix Layer만 실행한다.
이후 Decode는 항상 Full36 경로를 사용한다.
Native Fallback은 Hidden Buffer를 생성하거나 전송하지 않는다.

D의 실행 Guard는 기존 Decode Cohort가 끝날 때까지 새 Prefill을 기다린다.
현재 구현은 실행 중 Kernel을 선점하지 않는다.
이 Guard만을 새로운 Dynamic 기여로 주장하지 않는다.
새 정책은 Cut 크기, Cut Ready, D 완료 예측, Transfer와 P Saving을 함께 평가한다.
이 값들은 예측이다.
물리적인 Idle 측정값은 아니다.

## 3. 정합성 Gate

증거 수준은 `CORRECTNESS_ONLY`다.
성능 결과로 재사용하지 않는다.

| 검사 | 결과 | 범위 |
|---|---|---|
| Static Last2/4/8/12 | PASS | 각 N8, 출력 및 독립 Full Reference |
| 최초 Forced Mixed N8 | PASS | 요청별 Cut 및 Native Fallback |
| 최초 Forced Mixed N32 | PASS, 일부 Cut 미실행 | 2s 협상 Timeout으로 일부 요청이 Native로 복귀 |
| 전체 Cut Forced Mixed N32 v2 | FAILED | Native KV Lease 만료를 검출 |
| 전체 Cut Forced Mixed N32 v3 | PASS | 실제 Last0/2/4/8/12 모두 실행 |

v3는 32개 요청의 Greedy Output을 Native와 비교했다.
모든 Output이 일치했다.
실제 Attention 호출 범위와 Full36 Decode Graph도 검사했다.
P Prefix와 D Suffix는 각 요청의 선택한 Frontier를 따랐다.

측정 요청의 Suffix Reference 비교는 6개 Batch였다.
모든 수치 Gate가 통과했다.
Final Hidden의 최대 절대 오차는 0이었다.
Native Fallback 8개 요청의 유효 Prompt KV Identity도 통과했다.
Hidden Transfer의 Byte Hash, 정상 ACK, KV 및 Reservation 종료 Gate를 별도로 검사했다.

이 검사는 Forced Cut을 사용했다.
실제 State-aware Online Cut 선택의 성공 증거는 아니다.
그 검사는 아직 `NOT_RUN`이다.

## 4. Native KV Lease 실패와 수정

v2는 독립 Full Reference와 Byte Hash를 수행했다.
이 진단 작업은 실제 Serving보다 훨씬 느리다.
D가 진단 작업을 수행하는 동안 다음 Prefix의 KV가 Source에 대기했다.

설치된 NIXL Connector의 기본 `kv_lease_duration`은 30s다.
Source는 만료된 KV를 Remote Read 이전에 해제했다.
로그는 `q8`부터 만료와 이후의 미인식 Remote Read를 기록했다.
D의 독립 Reference 검사는 Prefix KV 변화 Assertion으로 실행을 중단했다.
해당 실행은 무효다.
허용 오차를 늘리지 않았다.
실패 로그와 Source Snapshot을 보존했다.

새 선택적 보호 경로는 Source KV를 Native ACK까지 유지한다.
기존 Native Completion Notification이 실제 해제를 수행한다.
보호 경로는 만료 직전에 Lease를 갱신한다.
ACK가 설정한 최대 대기시간 안에 오지 않으면 실험을 실패 처리한다.
살아 있는 KV를 조용히 해제하거나 재사용하지 않는다.
Source 종료와 Process Watchdog도 별도로 유지한다.

v3에서 Lease 갱신은 28회였다.
종료 시 미해제 추적 항목은 0개였다.
동일한 전체 Cut N32 검사는 통과했다.
이 수정은 Generic Transport Safety다.
Layer-wise 성능 이득으로 계산하지 않는다.
이 실패로 과거 모든 Native 결과가 무효였다고 주장하지 않는다.

## 5. Strong Colocated의 미해결 Gate

HND Layout을 맞춘 Streaming Chunk512/1024/2048은 N64 Output16에서 출력 Gate에 실패했다.
실패 요청 수는 각각 8/64, 6/64, 3/64였다.
각 Prompt Query의 Token Range는 누락이나 중복 없이 전체 8000 Token을 덮었다.
그러나 Query Coverage만으로 수치 정합성을 통과시킬 수는 없다.
정확한 Kernel 수준 원인은 아직 `NOT_IDENTIFIED`다.

Full-prompt Streaming도 별도로 실행했다.
N64 Output16에서 TTFT p95는 520.595ms였다.
ITL p99는 299.423ms였다.
그러나 출력 Gate가 실패했다.
이 값들은 실패 설정의 진단 결과다.
정상적인 Crossover나 Primary 성능 승리로 사용하지 않는다.

Cohort4/8/16은 현재 Output16 출력 검사를 통과했다.
더 긴 Output과 QPS Sweep은 진행 중이다.
이들을 모든 최적화된 Chunked Scheduler보다 강하다고 주장하지 않는다.

## 6. Kernel 계측의 범위

두 번의 짧은 CUPTI Capture에서 P/D Kernel 기록을 확보했다.
Drop과 Invalid Record는 모두 0개였다.
D의 실제 CUDA Graph Kernel 기록도 포함됐다.
여러 Stream의 Kernel Interval은 Union으로 계산했다.
Memcpy 기록은 별도로 보존했다.

두 Capture 모두 5% 간섭 기준을 통과하지 못했다.
따라서 Clean Physical Idle Capacity는 `NOT_MEASURED`다.
진단용 Kernel Inactivity를 Decode-safe Budget으로 바꾸지 않는다.
NVML Utilization의 보수값도 물리적 Idle로 사용하지 않는다.

계측된 Timeline과 CUDA Event Enclosure는 별도 증거다.
Host Scheduling Interval도 별도 증거다.
이 세 시간을 서로 대체하지 않는다.

## 7. 사전에 고정한 SLO

각 Output Cap에서 Serial Low-load N16을 독립 Start 3회 실행했다.
총 Calibration Start는 15회다.
Dynamic GPU 평가 전에 SLO를 고정했다.
Freeze SHA256은 다음과 같다.

`c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`

TTFT-priority, Balanced, Decode-priority를 모두 평가한다.
이 Regime들은 연구 설계값이다.
실제 Production Standard라고 주장하지 않는다.
Native-relative ITL p99와 TPOT p95의 1.05 기준은 별도로 유지한다.
현재까지 Candidate 성능에 맞춰 SLO를 변경하지 않았다.

## 8. 다음 Gate

1. Strong Colocated의 Output 및 Load Sweep을 완료한다.
2. 동일 HND 경로에서 비용 Lookup Table을 Calibration한다.
3. 실제 State-aware N32를 검사한다.
4. Best Static Fraction과 Dynamic을 비교한다.
5. Held-out N128 독립 Start로 성능과 비열등성을 확인한다.

실측하지 않은 Speedup이나 Decode-safe Capacity는 보고하지 않는다.
