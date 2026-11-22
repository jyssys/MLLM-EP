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
