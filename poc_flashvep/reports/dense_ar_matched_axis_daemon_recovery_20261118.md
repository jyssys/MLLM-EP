# 제어 세션 재시작과 실험 재개

## 확인한 상태

2026-11-18 08:22 KST에 재시작 후 상태를 검사했다.
Host의 마지막 Boot 시각은2026-07-29 17:11:37 KST이다.
이번 중단을 Host OS 재부팅으로 분류하지 않는다.
기존 실험 Controller와 GPU Worker는 종료한 상태였다.
GPU0–7에는 승인된 Burner만 남아 있었다.

Discovery155/232회와 Admission 통제52회는 완료 기록을 유지했다.
Primary Confirmation120회는 아직 시작하지 않았다.
사용자의 기존23–24시간 완료 목표를 초기화하지 않았다.
Last2 Full Sweep 취소도 유지했다.

## 중단한 실행 — FAILED

원래 실행 이름은 다음과 같다.

`discovery_Layer_Last12_N144_O128_q4_s2026111780_A8_v3`

설정은 Input8000, Output128, QPS4, Last12, N144이다.
두 Worker의 B8192/S1024와 공통 D Admission 상한8을 유지했다.
반환 코드는−15이며 SIGTERM 종료를 나타낸다.
Dispatcher가 기록한 실행시간은260.5671753883362초이다.
설정한 Wall Timeout은1650초이다.
따라서 Wall Timeout에 도달한 종료가 아니다.
제어 세션 재시작 시점과 외부 SIGTERM 기록이 일치한다.
CUDA/OOM 또는 모델 정합성 오류가 중단 원인이라는 증거는 확인하지 못했다.

일부 요청의 중간 기록은 존재한다.
정상 완료 상태, 최종 Request 파일 및 Worker Exit Gate는 없다.
이 실행을 Clean 성능이나 독립 반복 표본에 포함하지 않는다.
원래 Dispatcher Receipt, Source ZIP, 로그 및 설정은 보존한다.
Controller의 후속 `ProcessLookupError` 기록도 보존한다.

## 재개 방법

중단한 조건의 대체 이름은 다음과 같다.

`discovery_Layer_Last12_N144_O128_q4_s2026111780_A8_v3_resume1`

원래 설정과 대체 설정은 `name`만 다르다.
Prompt, Arrival Seed, Arrival Trace, Output, Layer Cut 및 SLO는 같다.
모델 Revision과 동결한8개 Inference Source Hash도 같다.
완료된155개 Discovery 조건을 GPU에서 다시 실행하지 않는다.
기존 완료 Metric은 성공 Receipt와 Gate를 확인한 뒤 재사용한다.

원래232개 조건의 Manifest는 유지한다.
별도 복구 기록이 중단한 Slot과 대체 이름을 연결한다.
최종 검증은 같은 Slot을 두 개의 독립 Start로 세지 않는다.
원래 실패 기록의 Hash가 달라지면 복구 예외를 적용하지 않는다.
새로운 실패는 기존의 Critical Failure 규칙으로 처리한다.

Controller는 기록한 PID를 가진 분리 Session에서 실행한다.
실행 로그와 PID를 별도로 보존한다.
이 방법이 Host 재부팅까지 생존한다는 뜻은 아니다.
Quiet-lane Lock, GPU Ownership 및 승인 Burner Guard는 유지한다.

## 검증

재개 경로의 CPU 회귀 시험을 먼저 실패 상태로 확인했다.
수정 후 관련10개 CPU 시험이 통과했다.
기록한 실패 Hash만 예외로 인정하는 시험을 추가한 뒤11개 시험이 통과했다.
완료한207개 성능 Start의 무결성 검사도 통과했다.
207회는 Discovery155회와 Admission 통제52회이다.
중단한 실행은 이 수에 포함하지 않는다.
이 검증은 새 대체 실행의 성공을 미리 선언하지 않는다.

이후 대체 실행은144개 요청을 모두 완료했다.
Dispatcher 실행시간은519.1236343383789초이며 반환 코드는0이다.
Normal Worker Exit와 Timing Isolation Gate가 통과했다.
Greedy Output, Query Coverage, KV 예약 수명 및 실제 양 Worker B/S 검사도 통과했다.
후속208개 Start의 무결성 검사에 이 대체 실행이 포함됐다.
이는 Serving 성능 우위나 Decode 비열등성의 판정이 아니다.

분리 Session의 Controller4개가 Parent PID1로 재배치된 상태도 확인했다.
원래155개 GPU 실행을 반복하지 않고 대체 실행의 시작 기록을 확인했다.
Controller PID와 실행 명령은 [분리 실행 기록](dense_ar_matched_axis_crossover_20261117/state/recovery_detached_controllers_20261118.json)에 보존한다.

복구 기록은 [daemon_recovery_20261118.json](dense_ar_matched_axis_crossover_20261117/state/daemon_recovery_20261118.json)에 보존한다.
원래 실패 Receipt는 [dispatcher_result.json](dense_ar_matched_axis_crossover_20261117/raw/discovery_Layer_Last12_N144_O128_q4_s2026111780_A8_v3/dispatcher_result.json)에 보존한다.
