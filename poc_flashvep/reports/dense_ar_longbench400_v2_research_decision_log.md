# LongBench400-v2 연구 결정 기록

상태: `ACTIVE`. 최종 성능 판정 문서가 아니다.

연구 시작은 2026-11-22 16:06:50 KST다. 최대 연구 시간은 20시간이다.
기준 Commit은 `5ebed239073b4fa3709361038cbec026d1c32b78`이다.
새 Evidence Namespace는 `dense_ar_longbench400_v2_20261122`다.
기존 Checkout, Index, v1 코드와 보고서는 보존한다.

## 1. 데이터 후보 부족

관측: 최초 후보만으로 일부 Input × Output Cell의 25개 할당을 충족하지 못했다.

가설: 다른 공식 LongBench v1 Task에 적격 원문이 존재할 수 있다.
검사: 공식 21개 non-E Task의 3,974개 후보를 모두 감사했다.
후보의 Natural EOS Output은 C1 Mode0 Prompt와 Native Full36 Decode 경로에서 생성했다.
Output Cap은 2,048이다. 후보 Pool에서 Cap Hit는 110개다.

실제 적격 후보 수는 다음과 같다. Output 구간은 열 순서대로 1–10, 11–50, 51–100, 101–200이다.

| Input 구간 | 1–10 | 11–50 | 51–100 | 101–200 |
|---|---:|---:|---:|---:|
| 3K–6K | 430 | 144 | 44 | 54 |
| 6K–10K | 278 | 126 | 64 | 79 |
| 10K–20K | 759 | 177 | 131 | 86 |
| 20K–36K | 86 | 31 | 17 | 6 |

판정: 20K–36K의 마지막 두 Output 구간은 목표에 미달한다.
같은 Input 구간의 인접 Output 구간에 부족분을 재배분했다.
최종 마지막 행은 `[46, 31, 17, 6]`이다. 다른 세 행은 각각 `[25, 25, 25, 25]`다.
원문 수정, 복제, Padding 또는 EOS 강제는 사용하지 않았다.
400개 선정 이후에는 성능 결과에 따라 요청을 교체하지 않는다.

자료: `analysis/post_expansion_candidate_availability.json`, `data/balanced400_v2_freeze_manifest.json`.

## 2. 원래 Compiled C1 경로의 독립 확인

관측: BI0에서는 실행 Geometry가 바뀌면 Natural Output도 바뀔 수 있다.

검사: 선정된 400개와 정합성 대조 요청을 원래 Compiled Full36 C1 경로에서 다시 생성했다.
Primary C3 품질 기준과 2 percentage-point 비열등성 한계는 새 C3 품질 결과 이전에 동결했다.
기존 Mode0 C1 품질은 Secondary 기준으로 유지한다.

결과: 선정 기준 Output 평균은 49.96 Tokens다.
Compiled C1의 실제 Main400 Output 평균은 53.0325 Tokens다.
Compiled C1에서는 2/400개가 200 Tokens를 초과했다. 최댓값은 1,324 Tokens다.
두 경로의 Output 차이를 숨기거나 400개를 다시 선정하지 않았다.

판정: 선정 시점의 Output 구간은 고정 Metadata다.
각 Serving 실행의 실제 Output 길이는 별도로 기록한다.
Natural EOS 작업량 차이가 5%를 넘으면 별도 Controlled Replay를 검사한다.
이 Replay는 Natural Serving 결과를 대체하지 않는다.

## 3. Partial Prefill Compiler

원래 증상: 이전 C3 Hybrid는 Stock Token보다 느렸다.
이전 세 Paired Start의 TTFT p95 비율은 Start-mean 기준 1.121429432다.
이 값은 현재 v2 성능 결과가 아니다.

첫 가설: Partial Prefix의 Eager 실행이 원인 일부일 수 있다.
경쟁 가설: Attention 배치, Handoff, Admission 또는 Queue가 더 큰 원인일 수 있다.

첫 Prototype은 Inductor 입력 ABI 오류로 실패했다.
두 번째 Prototype은 Config와 Cache를 분리했다. ABI 오류는 사라졌다.
그러나 Full KV와 Final Hidden의 Strict 검사는 실패했다.
16개 Greedy Token의 일치만으로 정합성 통과를 선언하지 않았다.

후속 수정은 원래 CUDA Prompt 연산과 독립 Cache를 유지했다.
D Decode는 기본 Compiled 경로를 유지했다.
Layer와 Genuine Hybrid는 N1, N8 Strict 검사를 통과했다.
후속 N32와 N400 State 및 Task-quality 검사도 통과했다.

판정: 구현 가능성과 정합성은 확인했다.
Compiler의 실제 Serving 개선은 동일 Source의 On/Off Intervention에서 별도로 측정한다.
이 단계의 정합성 통과는 TTFT 또는 Goodput 개선을 의미하지 않는다.

## 4. Agg의 정상 Preemption과 관측 코드

관측: Agg N400에서 q90의 Prompt 관측 Cursor가 25,840 이후 0으로 재시작했다.

경쟁 가설은 다음과 같다.

1. 실제 중복 연산 또는 잘못된 Request State가 발생했다.
2. 정상 Preemption 이후 관측 코드만 이전 Cursor를 유지했다.
3. Query Metadata가 잘못된 Request와 연결됐다.

실제 Scheduler 기록에는 q90의 같은 Cursor에서 Preemption이 존재했다.
실제 Hook과 Query Trace를 재생한 CPU 검사는 기존 관측 오류를 재현했다.
수정한 Hook은 해당 Request의 미사용 Preemption 증거가 있을 때만 관측 Buffer를 초기화한다.
증거가 없는 Cursor 재시작은 계속 실패한다.
모델, Scheduler, Tensor 또는 Sampling 결과를 수정하지 않았다.

결과: 수정 후 새로운 Agg N32와 N400 검사가 통과했다.
실제 Preemption 재계산 Query는 계산량 기록에 남긴다.
이 수정은 정합성 관측 코드의 수정이다. GPU 속도 개선으로 주장하지 않는다.

## 5. Stock Token의 오래된 Suffix Map

관측: 이전 Config를 복사한 후보에 v1 Request별 Suffix Map이 남았다.
새 Prompt 길이와 일치하지 않는 Partition은 유효한 기준선이 아니다.

검사: 실제 새 Dataset과 Config를 입력하는 CPU 검사를 만들었다.
기존 Map은 실패했다. 새 길이와 Cut으로 재생성한 Map은 통과했다.
Producer와 Consumer의 실제 Partition 기록도 검사한다.

조치: 아직 실행하지 않은 잘못된 Config는 `NOT_RUN`으로 보존했다.
새 Config 이름으로 정합성 검사를 다시 실행했다.
원래 Compiled Full36 Stock Token 경로는 유지했다.
Mode0 Token으로 기준선을 대체하지 않았다.

결과: 수정된 Stock Token Last8과 Last12는 N8 Strict 및 N32/N400 검사를 통과했다.
작은 Suffix의 Last2와 Last4는 N8 KV·Hidden Strict 검사에서 실패했다.
실패 로그와 Config는 보존한다.

## 6. 작은 Suffix의 Attention 경계 진단

현재 상태: `ROOT_CAUSE_NOT_IDENTIFIED`.

동일 q0의 Input은 3,987 Tokens다.
Last2의 Prefix는 3,763 Tokens다. Suffix는 224 Tokens다.
Teacher와 PD 모두 Compile3, Native RMSNorm, BF16 및 동일 KV Layout을 사용했다.
모든 GPU의 SM 수는 132다. Compute Capability는 9.0이다.

관측: Layer0의 전체 유효 KV는 일치했다. 후속 Layer의 KV와 Final Hidden은 달랐다.
N8 중 q0의 Greedy Token도 달랐다. 다른 7개는 Greedy Token만 일치했다.
따라서 Token 일치만으로 Full State 통과를 선언하지 않는다.

반증: 단순한 Compile Mode 또는 RMSNorm 설정 차이라는 설명은 현재 기록과 일치하지 않는다.
다음 검사는 원래 FA3 호출 직후 Q, 유효 KV, Attention Output 및 Plan Metadata를 저장한다.
Snapshot은 모델 입력과 출력을 교정하지 않는다.
CPU 복사가 Timing을 교란하므로 해당 실행은 `DIAGNOSTIC`이다.
정합성 기준을 낮추거나 실패 후보를 Primary 성능 표에 넣지 않는다.

## 7. 통계와 Decode 보호

Point Winner는 Paired Start의 Mean TTFT p95 비율로 정한다.
CI가 1을 포함해도 Point Winner 색은 평균을 따른다.
통계적 확정과 Decode Gate는 별도 Column과 Figure에 기록한다.
N1은 Start-level CI를 제공하지 않는다.

C0의 Mode0 Token은 메커니즘 대조군이다.
C3의 Strong Stock Token 성능으로 재사용하지 않는다.
C3의 Decode 보호는 NativeStrong과 동일 D 정책 Native를 모두 요구한다.
두 기준에서 ITL p99와 TPOT p95 비율의 Paired 95% CI 상한이 각각 1.05 이하여야 한다.
Absolute ITL 34ms 및 TPOT 33ms의 요청 통과율도 별도로 기록한다.

## 8. 아직 실행 결과를 주장하지 않는 항목

- C3 Main QPS 성능과 Held-out Serving 개선은 미확정이다.
- C3 Operator별 GPU-exclusive GEMM·Attention 비중은 미측정이다.
- 실제 Decode-safe Executable Window는 미확정이다.
- Active-decode Dynamic Insertion은 검증 전까지 `NOT_RUN`이다.
- Output Prior의 예측 정확도는 Axis Selection 또는 Serving 개선이 아니다.
- Offline Oracle은 실제 Online Speedup으로 표현하지 않는다.

이 기록은 이후 실측과 반증 결과를 추가하는 문서다.

## 9. Native Capacity 확인 후 추가한 부하 지점

64개 Pilot의 실제 Poisson 도착률은 Offered QPS의 약 0.728배였다.
Native Discovery208의 실제 도착률은 QPS2.6에서 2.6242 Requests/s였다.
완료율은 2.1292 Requests/s였다. TTFT p95는 17,853.4ms였다.
QPS4.8의 실제 도착률은 4.8447 Requests/s였다.
완료율은 2.1523 Requests/s였다. TTFT p95는 50,628.7ms였다.
두 결과는 N1의 `MEASURED_ONLINE` Discovery다. Sustainable Capacity가 아니다.

판정: 원래 두 Discovery 부하는 모두 과부하 영역이다.
기존 8개 QPS와 결과는 삭제하지 않는다.
새 Layer·Stock Token·Hybrid의 Clean 성능을 보기 전에 QPS2.0을 추가로 동결했다.
Geometry와 Policy 튜닝은 2.0/2.6/4.8을 사용한다.
중요 Confirmation 지점은 2.0/4.0/4.8이다.
근거와 동결 시각은 `configs/c3_confirmation_knee_refinement_v3.json`에 있다.

## 10. 시간 예산과 정합성 관측 비용

새 Compiled C3 실행은 매번 Actor를 새로 초기화한다.
초기화와 Full-state 관측은 별도 연구 시간을 요구한다.
Primary Main400, Held-out192 및 8개 Main QPS는 변경하지 않는다.

Policy Timing을 보기 전에 다음 실행 우선순위를 동결했다.

1. Geometry 튜닝은 기존 Discovery208을 유지한다.
2. Policy 튜닝은 Discovery에서 원문 96개를 결정론적으로 추출한다.
3. 네 Input 구간에서 각각 24개를 사용한다.
4. 모든 Policy는 같은 96개와 도착 Trace를 사용한다.
5. Main400은 8개 QPS를 먼저 N1로 실행한다.
6. QPS2.0의 Main400 및 Held-out192를 독립 N3로 확인한다.
7. 기존 Overload Confirmation도 남은 예산에서 실행한다.

96개 추출은 성능 또는 EOS를 사용하지 않는다.
이 데이터는 Calibration Subset이다. Main400을 대체하지 않는다.
Budget으로 실행하지 못한 조건은 `NOT_RUN`으로 남긴다.
다른 요청 수나 GPU Pair를 하나의 성능 CI로 합치지 않는다.
동결 근거는 `configs/c3_budget_priority_amendment_v4.json`에 있다.

모든 새 Policy의 Strict N8 검사는 기존 Raw CPU SHA256을 유지한다.
Natural N32는 전체 Byte의 GPU SHA256 Leaf Tree를 CPU 결과와 비교한다.
Natural N400은 같은 전체 Byte 알고리즘을 사용한다.
이 알고리즘은 Raw Byte Stream SHA256과 구분한 Schema를 사용한다.
일부 위치만 검사하는 Sample Hash가 아니다.
새 실행의 Self-test와 실제 GPU 검사는 통과 여부를 별도로 기록한다.
Clean 성능 실행에서는 State Hash 및 Self-test 관측을 비활성화한다.

## 11. 관측 의존성의 추가 반증

같은 q0의 N1은 추가 FA3 및 Actual Numerical Observer 없이도 Strict 검사를 통과했다.
같은 GPU2/3의 N8은 Numerical Observer를 추가하면 전체 8개 요청이 통과했다.
N8 Observer는 Forward 이후 동기화와 Tensor Snapshot을 추가한다.
따라서 이 결과만으로 동기화가 원인이라고 확정하지 않는다.

다음 두 검사를 같은 N8, Pair2/3, Cut2 및 Source5 기준으로 준비했다.

1. 추가 Observer 없는 원래 실행을 반복한다.
2. Tensor 복사 없이 Forward 이후 완료 Fence만 추가한다.

첫 검사는 기존 불일치의 재현성을 확인한다.
두 번째 검사는 완료 순서의 영향을 검사한다.
두 검사 모두 `DIAGNOSTIC`이다.
통과해도 별도의 N32/N400 검증 없이 Primary 성능 기준선으로 승격하지 않는다.

## 12. 작은 Suffix의 원래 N8 실행을 반복한 결과

추가 Observer 없는 원래 N8 Config를 새 Actor로 반복했다.
모든 요청의 Strict KV, Hidden 및 Greedy 검사가 통과했다.
실패 Config와 반복 Config의 차이는 실행 이름뿐이다.
기존 실패 Gate는 실패 상태로 유지한다.

실패 실행과 반복 실행의 P/D FX 계산 그래프는 각각 동일했다.
두 실행 모두 해당 Compile 구간을 새로 컴파일했다.
따라서 단순한 Cache 재사용 여부로 결과 차이를 설명하지 않는다.
현재 판정은 `NON_REPRODUCED_STRICT_MISMATCH`다.
근본 원인은 `ROOT_CAUSE_NOT_IDENTIFIED`다.

Fence-only 실행은 모델 검사가 시작되기 전에 Harness Import에서 실패했다.
이 결과는 `FAILED_HARNESS_IMPORT`다. Fence 효과의 반증 또는 입증이 아니다.
실패한 Source ZIP과 로그를 보존한다.
수정된 Fence Harness의 GPU 재실행은 `NOT_RUN`이다.
원래 실행도 반복에서 통과했으므로 Fence의 인과 효과를 주장할 근거가 없다.

## 13. D Partial Prefill Compiler의 정합성 검사

Layer와 Hybrid의 D Partial Prefill에 선택적 Compiled Interval 경로를 추가했다.
원래 Weight, Causal Attention, Position 및 KV 의존성은 유지한다.
Decode-only Forward는 기존 Native Compiled 경로를 유지한다.
Interval별 Compilation Config와 Cache는 분리한다.

Layer와 Hybrid 모두 새 N1 및 N8 Strict 검사를 통과했다.
관측된 Compiled Interval 실행도 확인했다.
현재 증거 수준은 `CORRECTNESS_ONLY`다.
N400 정합성, Clean Online TTFT 및 Decode Tail 이득은 아직 확인하지 않았다.

다음 검사는 동일 Source의 D Eager와 D Compiled를 비교한다.
모든 새 Signature는 Strict N8 및 Natural N32/N400 검사를 먼저 수행한다.
Hybrid의 Fused-short 조합도 별도 정합성 Gate를 적용한다.
Compiler 개선이 Host Admission 또는 P Queue 병목을 제거한다고 가정하지 않는다.

## 14. Primary Confirmation 실행 순서

Main 또는 Held-out 성능을 보기 전에 실행 순서를 추가로 동결했다.
기존 Main400, Held-out192, QPS Grid 및 모든 예정 실행은 유지한다.
먼저 사전 선정한 QPS2.0에서 Main400과 Held-out192를 각각 독립 N3로 확인한다.
두 Request 집합은 서로 다른 고정 Pair를 사용한다.
서로 다른 집합이나 Pair를 하나의 CI로 합치지 않는다.

그다음 나머지 Main QPS를 실행한다.
가장 오래 걸리는 Low-load QPS0.8은 Primary Confirmation 이후에 실행한다.
Natural EOS의 Output 작업량 차이가 5%를 넘으면 사전 정의한 Replay를 준비한다.
Replay는 독립 Compiled C1의 요청별 길이를 사용한 `DIAGNOSTIC`이다.
실제 EOS를 Online 입력으로 사용하지 않는다.
Natural 결과를 Replay 결과로 대체하지 않는다.

CPU 대기 Controller는 새 Policy 계획을 만들기 전에 최신 Module로 재시작했다.
실행 중인 GPU 작업은 중단하지 않았다.
기록은 `state/c3_policy_controller_reload_v5.json`에 있다.

## 15. Decode Tail의 Worker와 Client 경계

새 Host 분석은 모든 Client ITL을 기존 Metric과 다시 일치시켰다.
Client ITL은 Worker Emission 간격과 Delivery Delay 변화의 합이다.
Co-delivery의 0ms 간격도 기존 정의대로 유지한다.

QPS2.0 Discovery208의 NativeStrong에서 ITL p99는 18.5355ms였다.
상위 99개 간격 중 73개는 NIXL Host API와 겹쳤다.
같은 73개 간격에는 새 D Request 등록이 있었다.
시간 중첩만으로 두 원인의 독립적 기여율을 계산하지 않는다.

Common-policy Native의 ITL p99는 8.7293ms였다.
하지만 Max ITL은 125.0473ms였다.
해당 간격의 Worker Emission 간격은 7.8404ms였다.
Delivery Delay 변화는 117.2068ms였다.
이 Max ITL을 느린 GPU Decode로 해석하지 않는다.
원래 Client Timestamp는 수정하지 않는다.

Hybrid2의 ITL p99는 8.5550ms였다. Max ITL은 41.2455ms였다.
그 Max 간격의 Worker Emission 간격은 3.8628ms였다.
Delivery Delay 변화는 37.3827ms였다.
이 실행들은 N1 Discovery다. Paired CI 기반 Decode 통과를 선언하지 않는다.

## 16. 원문 Dataset의 재구성 검사

400개 원문의 ID, 순서, Metadata 및 Hash Roster를 따로 저장했다.
공식 Dataset Snapshot, 기존 Prompt Template 및 동일 Tokenizer로 Main400을 재구성했다.
재구성한 파일의 SHA256은 동결 Main400의 SHA256과 일치했다.
원문, Token 및 Token 순서를 바꾸지 않았다.

LongBench Code의 MIT License만으로 모든 제3자 Context의 재배포 권리를 가정하지 않는다.
전체 원문 Dataset과 무손실 Archive는 로컬 Evidence에 보존한다.
공개 Commit에는 재구성 Code와 ID·Hash Roster를 포함한다.
이 구분은 Dataset 선정이나 성능 결과를 변경하지 않는다.

## 17. Calibration 부하와 Main Confirmation 예산

기존 Geometry 실행의 평균 Actor 실행시간은 QPS2.0에서 260.73초였다.
QPS4.8에서는 254.71초였다. 각 값은 15개 실행의 평균이다.
초기화 비용 때문에 Policy 36개를 세 부하에서 모두 측정하면 Main 확인 시간이 줄어든다.

Policy 성능과 Held-out 성능을 보기 전에 Calibration 우선순위를 변경했다.
모든 Policy를 QPS2.0과 QPS4.8에서 같은 방식으로 측정한다.
두 부하는 각각 Native Knee 후보와 Overload 대조군이다.
중간 QPS2.6의 Policy Sensitivity는 `NOT_RUN`으로 남긴다.
이미 측정한 QPS2.6 Geometry 결과는 그대로 유지한다.

31개 새 Policy의 N8/N32/N400 정합성 검사는 모두 유지한다.
원문 96개, 모든 후보, Strong Stock Token 및 SLO는 변경하지 않는다.
Main400과 Held-out192의 독립 Confirmation을 우선한다.
Main의 기존 8개 QPS도 변경하지 않는다.
동결 근거는 `configs/c3_policy_load_budget_v7.json`에 있다.

진행 중인 두 GPU Lane Controller는 계속 실행했다.
기존 CPU 대기 Parent만 새 Orchestrator로 교체했다.
GPU Actor, 실행 중인 Request 및 Raw Log는 중단하거나 초기화하지 않았다.

## 18. Stock Token의 추가 Strict 실패

`StockToken_Cap2_Credit16`의 N8 실행은 정상 종료했다.
하지만 8개 요청 모두 Full KV 및 Final Hidden의 Strict 비교에서 실패했다.
8개 요청의 Greedy Output은 모두 일치했다.
해당 후보의 N32/N400 및 성능 실행은 정합성 Gate로 차단한다.

이 실패를 Source Credit16의 인과 효과라고 표현하지 않는다.
Strict Probe는 Coordinator Credit을 0으로 설정한다.
통과한 Credit8 Probe와 실패한 Probe의 실제 Config 차이는 이름과 GPU Pair뿐이다.
통과 Probe는 GPU2/3을 사용했다. 실패 Probe는 GPU0/1을 사용했다.

전체 P Prefix KV는 두 Probe에서 일치했다.
유효 Prompt Query Geometry와 P/D FX 계산 그래프도 일치했다.
D Full KV의 첫 불일치는 8개 요청 모두 Layer1에서 관측됐다.
보존된 Final Hidden의 상대 RMS 차이는 약 0.00910–0.01827이었다.
이 수치는 전체 Suffix KV의 허용 오차를 검증한 결과가 아니다.

현재 판정은 `STRICT_STATE_MISMATCH_ROOT_CAUSE_NOT_IDENTIFIED`다.
Compiler, GPU Pair 또는 실행 순서 중 하나를 원인으로 확정하지 않는다.
실패 Gate와 Raw Evidence는 그대로 유지한다.
Source Credit0의 무제한 Window와 통과한 Credit8 후보는 Strong Stock Token 탐색에 계속 포함한다.

## 19. Main 실행 순서와 Motivation Point

Main 성능을 보기 전에 실행 순서를 구체화했다.
먼저 QPS2.0의 Main400 및 Held-out192 독립 N3를 완료한다.
그다음 기존 8개 QPS의 Agg/Native Motivation Point를 우선한다.
나머지 Axis Point와 기존 Overload 반복도 그대로 계획에 남긴다.
데이터, 설정, Pair, Seed 및 QPS는 변경하지 않는다.
실행하지 못한 Point는 `NOT_RUN`으로 표시한다.

## 20. 추가 정합성 진단의 범위

Stock Token 실패에 대해 Control → Pre-forward Fence → Control 검사를 사전에 고정했다.
세 실행은 같은 Source와 GPU0/1을 사용한다.
Fence는 Tensor를 복사하거나 교정하지 않는다.
기존 정합성 Lane이 모두 끝난 뒤 실행한다.
검사 결과와 무관하게 기존 실패 Gate를 성공으로 바꾸지 않는다.

추가 Cache 감사에서 P/D FX Graph는 같았지만 일부 Autotune 선택은 달랐다.
한 Reduction의 `R0_BLOCK`은 2048 또는 4096이었다.
Generated Code의 Kernel Path Comment를 제외하면 해당 Source는 일치했다.
이 관측은 Reduction 실행 구성의 차이를 보여준다.
그러나 그것이 Full State 불일치의 원인임을 입증하지는 않는다.
Cache를 실행하거나 원래 Runtime Package를 수정하지 않았다.

## 21. Fused Hybrid의 독립 Teacher 경로

Fused Hybrid의 첫 N8 비교는 Full KV와 Hidden에서 실패했다.
8개 중 1개 요청은 Greedy Output도 달랐다.
이 결과를 성능에 사용하지 않는다.

Source 감사에서 Teacher의 필수 Dispatch Flag가 빠진 것을 확인했다.
`teacher_eager_all_prompt=True`만으로는 Eager Wrapper가 활성화되지 않는다.
Installer는 `teacher_eager_prefix_compiled_suffix`가 없으면 즉시 반환한다.
실제 실패 Teacher에도 해당 실행 경로의 Log가 없었다.

새 Version은 이 Dispatch Flag만 복구한다.
독립 Teacher는 원래 Qwen Layer와 전체 로컬 KV를 그대로 사용한다.
Teacher는 Partition Adapter 또는 NIXL을 사용하지 않는다.
Candidate의 모델 실행, Sampling 및 허용 오차는 바꾸지 않는다.
기존 실패와 잘못된 Teacher 경로도 그대로 보존한다.

Fused Eager 및 Fused D-Compiled 조합에 새 N8/N32/N400 Gate를 적용한다.
새 Teacher의 Eager 경로가 실제로 실행됐는지도 검사한다.
GPU 재검증 전에는 이 수정이 정합성을 해결했다고 선언하지 않는다.

## 22. C0 전체 Output Map 완료

C0의 720개 실행을 완료했다. Discovery 80개 Cell과 Held-out 80개 Cell을 확보했다.
Held-out의 각 Cell은 독립 Server Start 3회를 사용한다.
실제 Batch1/1/1, Source Credit4 및 정확한 Aggregate GEMM Rows 검사를 통과했다.

Output8/16/64/128/192에서 Layer Point Win은 각각 3/5/8/10/13개였다.
각 Output의 분모는 같은 16개 Input×Cut Cell이다.
Layer의 CI 확정 Cell은 각각 1/2/5/8/11개였다.
양쪽 Axis의 C0 Decode Gate 통과 Cell은 각각 16/16/9/4/7개였다.
따라서 긴 Output의 Point Win 증가를 Decode-safe Serving 성공으로 표현하지 않는다.

같은 Prompt와 Arrival에서 Output8→192의 Endpoint Contrast도 계산했다.
이 Intervention은 D Residency, Queue 및 Capacity-relative Load를 함께 바꾼다.
Host Stage는 Mean TTFT에 더해지는 경계다. TTFT p95의 Operator 분해가 아니다.
C0의 Mode0 Token과 C3의 Compiled Stock Token도 계속 구분한다.

새 C3 Bridge는 원래 Held-out Prompt와 실제 Continuous Scheduler를 사용한다.
C0의 결과를 C3 결과로 재사용하지 않는다.

## 23. Fused Teacher 재검증과 Stock Fence 반증

ContextFixV15의 Fused Hybrid 두 후보는 새 N8/N32/N400 Gate를 통과했다.
새 Teacher의 실제 Eager Dispatch Log도 확인했다.
이 결과는 Candidate의 Tensor 교정이 아니다. 독립 Teacher의 누락된 실행 Flag를 복구한 결과다.
원래 실패 Gate는 계속 실패 상태로 보존한다.

Stock의 Control → Pre-forward Fence → Control 세 실행은 모두 Strict State를 통과했다.
따라서 Fence가 기존 오류를 해결했다고 결론 내리지 않는다.
Compiler Autotune 구성의 차이는 남아 있다. Strict 불일치의 인과적 원인은 미확정이다.

## 24. C0의 절대 ITL 위반 위치

Held-out의 고유 C0 실행 540개를 모두 조사했다.
Native가 여러 Cut에 재사용된 경우 한 번만 계산했다.
Client ITL>34ms 위반은 총 66개 구간이었다.
그중 65개는 같은 구간의 Worker Emission Gap이 34ms 이하였다.
가장 큰 Client ITL은 604.143ms였다. Worker Gap은 7.149ms였다.
해당 Delivery Delay 변화는 596.994ms였다.

이 수치는 일부 절대 SLO 실패가 GPU Decode Step의 지속적 악화와 다르다는 근거다.
하지만 GC, Socket, CPU Scheduling 또는 File I/O 중 하나를 원인으로 확정하지 않는다.
Client-visible Timestamp와 원래 실패 판정은 변경하지 않았다.
C3의 Tail과 동일한 원인이라고 가정하지 않는다.

## 25. 고정 Knob 확인의 조기 실행

독립 Knob 실험은 이미 정해진 Compiler, Credit 및 Cap만 비교한다.
Policy Timing 결과나 최종 후보 선택을 입력으로 사용하지 않는다.
모든 Composition Gate와 C3 Bridge가 완료됐으므로 불필요한 대기 Barrier를 줄였다.

GPU4/5에서 기존 N3 Fixed-length Replay를 먼저 실행한다.
Source, 데이터, Pair, 순서, Arrival, QPS 및 SLO는 바꾸지 않는다.
기존 CPU 대기 Process에 Child가 없는 것을 확인한 뒤 교체했다.
실행 중인 GPU 작업은 중단하지 않았다.
최종 Static 선택을 동결하기 전에는 이 Held-out Timing 결과를 읽지 않는다.

## 26. Native CPC의 조기 실행

GPU6/7의 유휴 시간에 기존 NativeStrong QPS2.0/4.8 CPC를 먼저 실행한다.
새 QPS 또는 다른 계측을 추가하지 않는다.
최종 Activity Plan은 같은 Config와 Receipt를 재사용한다.
같은 실행을 두 개의 독립 Start로 계산하지 않는다.

Data, Seed, Pair 및 전체 Trace/ROI의 5% Observer Gate는 유지한다.
이 Timing은 Static 후보 선택에 들어가지 않는다.
최종 후보 선택은 동결된 Discovery96 규칙만 사용한다.

## 27. 최종 Static 후보 동결

QPS2.0/4.8의 동일한 Discovery96에서 각 Family의 정책을 비교했다.
Layer는 Last2/Cap2/Credit16을 선택했다.
Stock Token은 Work Cut8/Cap0/Credit0을 선택했다.
Hybrid는 Last2/Suffix1/8의 Fused Short+D Compiled/Cap2/Credit8을 선택했다.
Deflection은 Cap1/Credit8을 선택했다.

선택은 사전에 정한 품질 및 Decode Point Screen을 먼저 사용했다.
그다음 Native-normalized TTFT와 Goodput 규칙을 사용했다.
이 N1 Screen은 최종 N3 CI Gate가 아니다.
동결 전에는 C3 Bridge 및 조기 Knob의 Held-out Timing을 읽지 않았다.
최종 후보를 바꾸지 않고 Main400과 Held-out192의 QPS2.0 N3를 먼저 확인한다.

## 28. Native 계측의 제한과 한 번의 Recovery Bundle

Native QPS2.0/4.8의 두 CPC는 품질과 Output Work 조건을 통과했다.
하지만 ROI 노출 Request의 ITL p99 비율은 각각 1.1101과 1.2304였다.
허용 변화 5%를 초과했으므로 Kernel Busy는 `DIAGNOSTIC`으로 남긴다.
Kernel Busy가 낮다는 관측만으로 Decode-safe Headroom을 선언하지 않는다.

경쟁 가설은 양쪽 GPU 동시 수집, 추가 상태 Observer 및 Natural Batch/Output 변동이다.
단일 Role, 5초 ROI 및 Quiescent Witness 비활성화의 저부하 Bundle을 고정했다.
P와 D 각각 한 CPC만 추가한다. 통과할 때까지 반복하지 않는다.
같은 5% 전체 Trace/ROI Gate와 Output Work/품질 조건을 유지한다.
여러 Observer 설정을 함께 바꾸므로 개별 설정의 인과 효과로 표현하지 않는다.

GPU4/5의 Primary Equal-output Replay를 이 선택적 Recovery보다 우선한다.
따라서 Knob 및 Primary Replay가 끝난 후에만 실행한다.
예산이 끝나면 해당 Capture는 `NOT_RUN`으로 남긴다.

## 29. Natural Primary의 독립 확인

동결한 QPS2.0에서 Main400과 Held-out192를 각각 독립 Start3회 실행했다.
두 요청 집합과 GPU Pair의 CI는 합치지 않았다.
Layer, Stock Token, Hybrid 및 Deflection은 두 Native 대조군의 Decode Gate를 통과했다.
그러나 일부 개별 Client ITL은 34ms를 초과했다.
Max ITL과 모든-ITL 요청 통과율을 원래 값으로 보존했다.

Natural Held-out192의 Layer/Stock TTFT p95 비율은 1.0471이었다.
95% CI는 [0.7719,1.5230]이었다.
Hybrid/Stock은 1.1270 [0.6096,1.9694]였다.
두 비교 모두 Layer/Hybrid 우위를 확정하지 못했다.
Natural Output 작업량 차이가 사전 기준 5%를 초과했다.
따라서 이미 선언한 Equal-output Replay를 실행했다.

## 30. Output 작업량을 통제한 반증

Replay는 요청별 독립 Compiled C1의 길이를 사용했다.
동일한 Held-out 원문, Arrival 및 고정 정책을 유지했다.
이 실험은 Fixed-length Diagnostic이다. Natural Serving을 대체하지 않는다.

Layer, Stock Token 및 Hybrid는 각각 독립 Start3회를 완료했다.
Layer/Stock TTFT p95 비율은 1.1588 [1.1379,1.1892]였다.
Hybrid/Stock은 1.3042 [1.2040,1.4201]였다.
SLO-goodput 비율은 각각 0.8574 [0.8004,0.9224]와 0.7880 [0.7381,0.8332]였다.
이 세 Arm은 두 Native 대조군의 Decode Gate를 통과했다.
이 조건에서 Output 작업량 차이는 Layer/Hybrid의 패배를 설명하는 유일한 원인이 아니었다.

Replay의 전체 27개 계획 중 26개 실행을 완료했다.
마지막 Common-policy Native Cap1 Start는 신규 Dispatch 예산이 끝나 미실행으로 남겼다.
따라서 Deflection Replay의 완전한 N3 Common-policy Decode Gate를 주장하지 않는다.
선택적 단일 Role 계측 Recovery도 미실행으로 남겼다.

## 31. Kernel Activity와 실행 가능한 Window의 분리

일부 Native 및 Axis CPC는 고정된 5% Observer Gate를 통과했다.
NativeStrong QPS3.2의 약20초 ROI에서 P Kernel Busy는 99.55%였다.
D Kernel Busy는 48.42%였고 P Waiting 평균은 27.31 Requests였다.
이는 해당 ROI의 P/D 불균형 근거다.
하지만 이 CPC의 모든-ITL 요청 통과율은 Clean/Profile/Clean에서 90.75%/93.00%/96.50%였다.

Physical Kernel Gap은 안전한 Prefill 실행 가능 Window가 아니다.
Partition 준비, KV Credit, Stream 가용성 및 Tail Cost를 함께 검증하지 못했다.
따라서 Active-decode Dynamic Insertion은 실행하지 않았다.
실패한 Observer Gate의 Capture는 Diagnostic으로 보존했다.

## 32. 예산 종료와 최종 판정 범위

시작 후 18시간에 신규 GPU Dispatch를 종료했다.
이미 실행 중인 소유 작업은 정상 종료까지 기다렸다.
Agg/Native Motivation은 공통 Offered QPS 0.8/1.4/2.0/2.6/3.2/4.0/4.8/6.0을 확보했다.
모든 Method의 전체 QPS Grid는 완료하지 못했다.
나머지 계획은 `NOT_RUN`으로 보존했다. 누락된 값을 보간하지 않았다.

Raw Metric 재계산은 1303개 Client Metric 실행에서 통과했다.
Client Metric이 없는 Teacher/Diagnostic 14개는 별도로 남겼다.
실행 실패 4개 및 미실행 Receipt 20개를 성공으로 변경하지 않았다.
전체 Configuration Inventory에는 미실행 238개가 있다.
Configuration과 Receipt의 분모는 서로 다르다.

C0의 긴 Output Point Win 증가는 확인했다.
하지만 테스트한 강한 C3 고정 후보에서 Layer/Hybrid의 추가 Serving 이득은 확인하지 못했다.
최종 Serving 판정은 `NO-GO_ON_TESTED_GRID`다.
이는 테스트하지 않은 전역 설계 공간의 불가능성을 뜻하지 않는다.

## 33. 게시용 Private Index의 지연 수정

최종 게시에서 파일마다 Private Index를 다시 쓰는 경로가 느렸다.
다른 후보 원인은 파일 Hash 계산과 저장장치 지연이었다.
소유한 CPU Publisher만 Commit/Ref/Push 이전에 종료했다.
Branch Ref와 사용자 Checkout/Index가 바뀌지 않은 것을 확인했다.
GPU 측정, Raw Evidence 및 Burner는 변경하지 않았다.

작은 실제 Git Fixture에서 파일 3개가 Index 갱신 3회를 발생시켰다.
Index 갱신 1회를 요구한 회귀 검사는 수정 전 실패했다.
수정은 Hash와 Index Entry를 일괄 제출한다.
수정 후 검사는 Index 갱신 1회와 Serial 경로의 동일한 Git Tree를 확인했다.
공백 및 한글 파일명도 검사했다.
이 수정은 게시 경로만 변경한다. GPU Runtime의 성능 수정이 아니다.
