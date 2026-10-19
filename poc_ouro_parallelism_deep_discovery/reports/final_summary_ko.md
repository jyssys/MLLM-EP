# Ouro 병렬화 심층 탐색 요약

## 판정

`NO_QUALIFYING_CANDIDATE_IN_SEARCHED_SPACE`.

기록된 실제 coordinator 연구 시간은 36008.456322초, 약10.002349시간이다.
명시적인 사용자 일시정지 구간은 제외했다.
동시에 수행한 worker 시간을 중복 합산하지 않았다.

이번 탐색에서 후속 아키텍처 PoC로 선정한 후보는 0개다.
이는 Ouro에 새로운 병렬화가 불가능하다는 뜻이 아니다.
19개 후보 모두를 실제 엔진으로 만들어 실패했다는 뜻도 아니다.

후보의 구조적 차별성, 실행 의존성, 상태 소유권, 전체 비용, 정확성 경로를 함께 검토했다.
현재 증거로는 이 조건을 함께 충족한 후보가 없다.
새로운 대형 엔진은 구현하지 않았다.
후보의 실제 E2E 가속과 자유 생성 품질은 모두 `NOT_RUN`이다.
원래 후보 검증용 held-out은 사용하지 않았다.

## 수행 범위

대상은 Dense Base `ByteDance/Ouro-1.4B`와 `ByteDance/Ouro-2.6B`다.
Thinking 모델이나 dLLM으로 대체하지 않았다.
BF16과 pretrained weights를 유지했다.
Fixed R4와 기존 gate의 Adaptive R2–R4를 구분했다.
Adaptive는 q=0.5, `last_exited`, full-R4 prefill을 사용했다.

물리 GPU는 H100 4/5/6/7만 사용했다.
기존 CDB factorial 실험과 다른 GPU 작업은 보존했다.
P8 실험은 수행하지 않았다.
P8의 후보별 critical-path 예측도 완성되지 않았다.

## 확인한 물리적 결과

기존 TP1/TP2/TP4, DP4, TP2×DP2 구성에서 400개 clean trial을 수집했다.
별도 context sweep은 144개 trial이다.
이들은 소수의 고정 prompt pattern과 128-token 강제 출력을 사용한 하드웨어 비교다.
자연 reasoning 성능으로 해석하면 안 된다.

동일한 API timing boundary의 C1/N1024 중앙값은 다음과 같다.

| Base | TP1 | TP2 | TP4 | TP1/TP4 |
|---|---:|---:|---:|---:|
| 1.4B | 1089.762 ms | 929.606 ms | 813.429 ms | 1.340× |
| 2.6B | 2062.765 ms | 1826.399 ms | 1628.599 ms | 1.267× |

32K context의 TP1/TP4 평균 paired ratio는 각각 2.1883×와 2.2098×다.
이는 두 prompt pattern의 결과다.
새 method의 가속이나 attention만의 가속이 아니다.
DP/hybrid의 parent-wall과 TP의 API-wall은 다르다.
그 차이를 무시한 전체 method 순위는 제시하지 않는다.

## 품질과 실제 Adaptive 실행

추가 baseline cohort는 domain당 32문제다.
같은 64개 문제를 두 모델과 각 실행 방식에서 사용했다.
최대 생성 길이는 8192 token이다.
길이 제한에 도달한 응답은 정답 numerator에서 제외하고 N에는 포함했다.

| Base / Runtime | GSM8K 정답/N | MATH-500 정답/N |
|---|---:|---:|
| 1.4B staged Fixed | 24/32 | 17/32 |
| 1.4B staged Adaptive | 24/32 | 21/32 |
| 2.6B staged Fixed | 28/32 | 25/32 |
| 2.6B staged Adaptive | 28/32 | 24/32 |
| 1.4B native TP4 Fixed | 25/32 | 16/32 |
| 2.6B native TP4 Fixed | 28/32 | 25/32 |

상세 보고서는 request-bootstrap 95% CI, parsing failure, truncation을 함께 제공한다.
이 표본으로 1–2 pp의 작은 품질 차이가 없음을 입증할 수는 없다.
이 결과는 Thinking 모델의 공식 reasoning 평가 재현도 아니다.

Adaptive의 기록된 평균 decode depth는 2.809121–3.059609다.
이 수치는 full-R4 prefill과 첫 출력 token을 제외한다.
일부 recurrent core 실행이 실제로 생략됨을 확인했다.
이를 전체 FLOPs, HBM traffic, E2E 가속으로 환산하지 않았다.

Native와 staged Fixed의 첫 출력 token은 128쌍 모두 같았다.
전체 출력 token ID는 49/128쌍에서만 같았다.
다른 79쌍 중 78쌍은 종료 전에 처음 달랐다.
나머지 1쌍은 native EOS에서 처음 달랐다.
BF16 backend, TP 구성, cache, stop 처리 중 어떤 요인이 원인인지는 확인하지 못했다.

## 왜 후보를 선정하지 않았는가

- Shared weights는 다음 recurrent depth의 실제 hidden state나 KV를 미리 제공하지 않는다.
- 일부 exact 변형은 generic pooling, TP/SP, PP, memoization과 겹친다.
- 실제로 측정한 remote-MLP component는 해당 fixture에서 손해였다.
  완전한 fused native MLP나 모든 pooling 구조를 반증한 것은 아니다.
- C07의 bitwise qualification은 실패했다.
  첫 번째로 다른 연산은 아직 확인되지 않았다.
  유효한 service timing은 0개다.
- CDB observer 4건은 기능 검사에는 통과했지만 timing validity 검사에는 모두 실패했다.
  해당 시간으로 E2E 비중이나 Amdahl 가속을 계산하지 않았다.
- Exact certificate와 lossless KV는 제한적인 수학·구조 증거만 있다.
  실제 decoder, fallback, 상태 lifetime과 전체 물리 비용은 검증되지 않았다.
- Adaptive terminal alias는 이상적인 row 단위 payload 모델에서 평균 5.478–25.336%를 줄일 가능성이 있다.
  Whole-page alias만 허용한 page16 모델에서는 1.472–6.599%다.
  이는 실제 HBM 절감이나 가속이 아니다.
- C14의 모델상 API-wall 감소는 4.13–10.14%다.
  이는 실제 pipeline 가속이나 latency bound가 아니다.

## 다음 행동

현재 증거로 새 Loop Parallelism 엔진을 바로 구현하는 것은 권장하지 않는다.
재개하려면 먼저 하나의 좁은 전제만 검증하는 것이 낫다.
예를 들어 native backend에서 유효한 certificate 비용과 hit rate를 확보하거나, 정확한 state ownership 변경의 실제 exposed cost를 측정해야 한다.
이들은 선정된 아키텍처가 아니라 추가 연구 전에 필요한 증거다.

자세한 근거는 [영문 최종 보고서](final_report.md)에 있다.
재현 명령은 [reproduce_final_v1.md](reproduce_final_v1.md)에 있다.
실제 시간, 자원 확인, 검증 범위는 [완료 의무표](completion_obligations_v1.md)와 완료 receipt에 남긴다.
