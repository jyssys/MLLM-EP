# LongBench400: Layer × Token 2D PD 종합

작성 상태: 과학적 검증과 보고서 검토를 완료했다.
GPU 자원·Archive·Publication은 별도 Receipt로 검증한다. 이 문서만으로 Remote Push 완료를 판정하지 않는다.

## 현재 확인한 결론

실제 LongBench 원문 400개와 Genuine 2D Continuation을 구현했다.
C0 Batch1/1/1에서는 Layer Winner와 Token Winner를 모두 독립 반복으로 확인했다.
그러나 그 결과를 강한 Continuous Token 대비 Serving 우위로 확대할 수는 없다.
검증한 Interior Hybrid의 작은 C0 이득도 C3에서 그대로 유지되지 않았다.

Decode Tail은 GPU 계산뿐 아니라 Remote READ 준비와 새 Request 등록의 영향을 받았다.
따라서 큰 Prefill 계산만 Active Decode 뒤로 미루는 정책은 충분하지 않았다.
원래 실패 결과와 별도 Runtime 수정 결과를 모두 보존한다.
등록 수정 후 QPS1.8과 2.8에서는 네 방법의 Decode 보호를 각각 독립 Start3회로 확인했다.
수정 후 Controlled Bridge에서도 모든 조건이 안전해진 것은 아니다.
Output64/Cap2의 Layer는 두 Native의 Decode Gate를 통과했다. 강한 Token 대비 TTFT 차이는 불확실했다.
같은 조건의 Hybrid는 두 Native의 상대 Gate와 Start별 Absolute Gate를 모두 통과하지 못했다.
마지막 Prefill/Decode Kernel 정책 검사도 독립 Start3회씩, 전체21회를 완료했다.
이 수정은 Output16/Cap1 Hybrid의 두 Native Decode Gate를 통과했다. Layer의 Common Native ITL Gate는 남았다.
수정한 Hybrid도 강한 Token보다 TTFT에서 느리고 SLO-goodput이 낮았다.

## 공통 계약과 증거 구분

| 항목 | 계약 |
|---|---|
| Model | Qwen/Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218` |
| 계산 | BI0, Original BF16, Exact Causal Attention, Full36 Decode |
| Sampling | Greedy, Thinking Disabled, Natural EOS, Cap2048 |
| GPU | H100 SXM80GB 2개/Start. 실제 UUID와 Pair는 Raw에 기록 |
| Scheduler | B8192/S1024, Chunked Prefill, Prefix Cache Disabled |
| KV | GPU당 요청 Budget42GiB, Block128 |
| Absolute SLO | TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms |
| 상대 Decode 보호 | ITL p99와 TPOT p95의 Paired 95% CI 상한≤1.05 |
| 독립 단위 | 별도 Server 초기화. Request와 Token은 독립 Start가 아님 |

C0는 실제 Batch1/1/1 Mechanism Control이다.
C3는 실제 Continuous Scheduler다. C3의 Source Credit, Compile Family와 Admission 정책은 Arm별로 명시한다.
같은 B/S를 같은 실제 Batch로 해석하지 않는다.
Primary Latency는 Coordinator Client-visible 값이다. HTTP Serving Latency는 아니다.
모든 Throughput은 유한 요청 집합의 Arrival부터 Drain까지 측정했다.
이를 Sustainable Capacity로 표현하지 않는다.

## 1. Dataset400은 실제로 균형을 이루었는가?

`DATASET PASS`다. 다만 16개 Cell의 정확한 균등 배정은 부분 달성이다.
Input 3K–6K, 6K–10K, 10K–20K, 20K–36K에 각각 100개를 배정했다.
앞의 세 Input 구간은 Output1–8/9–24/25–49/50–99에 각각 25개다.
마지막 Input 구간은 52/29/3/16개다.
선언한 13-task 후보 Pool에는 마지막 두 Output Cell의 원문이 부족했다.
이 부족분은 Serving 결과를 보기 전에 정한 기준으로 재배분했다.

| C1 Natural Reference 통계 | Input Tokens | Output Tokens |
|---|---:|---:|
| Mean | 13839.600 | 27.420 |
| Median | 9965 | 16 |
| Min–Max | 3088–32988 | 2–99 |
| p95 | 31446.250 | 84 |
| 합계 | 5535840 | 10968 |

400개 Request와 272개 고유 Context를 보존했다.
새 Truncation, Filler 또는 Request 복제는 사용하지 않았다.
선정 집합의 C1 Reference는 400/400 EOS, Cap Hit0개다.
선정 집합은 짧은 Natural Output에 조건을 둔다. 일반 Production 분포를 대표하지 않는다.
후보 전체의 Output 평균은 111.879 Tokens였다.
실제 C3 Output은 이 C1 Reference와 다를 수 있다. Start별 실제 생성 수를 사용한다.

Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다.
Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.
선정 Seed는 `2026112101`이다.
Context-disjoint Discovery200과 Heldout200의 평균 Input은 각각 11808.345와 15870.855 Tokens다.
Held-out에는 20K–36K 요청이67개다. Discovery에는33개다.
분할 뒤에 분포를 바꾸지 않았다. 같은 Offered QPS를 같은 부하로 해석하지 않는다.
근거: [Dataset 계약](dense_ar_longbench400_dataset_contract.md).

## 2. 짧은 Input이면 Layer가 이기는가?

일부 조건에서는 그렇다. 모든 짧은 Input 또는 작은 Cut에 성립하지는 않는다.
C0의 64개 Cell 판정은 Layer22, Token16, Tie21, Invalid5개다. 미실행 Cell은 없다.
각 Cell은 독립 Start3회다. 사전 지정 Core Cell3개는 5회다.
CI는 Pointwise Start-level Empirical 95% CI다. 다중 비교 보정은 하지 않았다.
원래 Label은 이긴 Arm의 Safety를 요구한다.
두 Arm 모두 안전해야 한다는 더 엄격한 추가 감사에서는 47/64개 Cell이 남았다.
그 안의 판정은 Layer15, Token16, Tie16개다. 원래 Label과 실패 기록은 변경하지 않았다.
같은 안전 부분집합에서 Winner가 Native TTFT도 개선한 Cell은 Layer4개, Token15개다.
이는 C0의 Pointwise 판정이다. Natural C3의 강한 Static 대비 우위를 뜻하지 않는다.

3K–6K/Output8/Last2의 Layer/Token TTFT p95 비율은 0.9437 [0.9241,0.9653]이었다.
하지만 Layer/Native는 1.0029 [0.9696,1.0324]였다.
이는 Axis Win이다. Native 대비 Serving Win은 아니다.
같은 Input/Output의 Last8은 Token Winner였다.
근거: [전체 64-cell Map](dense_ar_longbench400_input_output_axis_winner_map.md).
별도 [양축 모두 Decode를 보호한 Winner Map](dense_ar_longbench400_2d_20261121/figures/axis400_both_axes_decode_safe_winner_map.png)은 미적격 Arm도 회색으로 남긴다.
회색 Cell의 실측 비율을 삭제하거나 성공으로 처리하지 않았다.

| 실제 Input 구간 | Layer 확인 | Token 확인 | Tie 또는 CI 불확실 | 한쪽 이상 Decode 미적격 | 전체 Cell |
|---|---:|---:|---:|---:|---:|
| 3K–6K | 2 | 1 | 7 | 6 | 16 |
| 6K–10K | 6 | 2 | 4 | 4 | 16 |
| 10K–20K | 3 | 4 | 3 | 6 | 16 |
| 20K–36K | 4 | 9 | 2 | 1 | 16 |
| 합계 | 15 | 16 | 16 | 17 | 64 |

각 행은 Controlled Output4개×Cut4개다. 두 Axis 모두 Native Decode Gate를 통과한 Cell만 Winner로 분류했다.
같은 Cell의 실측 비율과 CI는 모두 보존했다. 이 표를 Continuous Serving의 Winner 표로 해석하지 않는다.

## 3. Output Length는 Axis Winner에 독립적으로 영향을 주는가?

정확한 Controlled Counterfactual 하나에서 영향이 재현됐다.
같은 20K–36K Prompt16개, QPS1.253873, Source, Pair와 Arrival에서 Output만 바꿨다.
Hybrid/Token TTFT p95 비율은 Output16에서 1.070249 [1.054657,1.081369]였다.
Output64에서는 0.980561 [0.971238,0.990277]였다.
두 조건은 각각 독립 Start3회다.
Natural EOS의 보편적 Output 임계값을 찾았다는 뜻은 아니다.
Output이 바뀌면 Decode Residency와 Queue Regime도 달라진다.

## 4. 어떤 Cut과 QPS에서 Crossover가 발생하는가?

C0의 16개 Input×Cut 조합 중 9개에서 Controlled Output에 따른 Winner 방향 변화가 나타났다.
전체 위치와 CI는 64-cell 표에 있다.
QPS만 바꾼 짧은 Input/Output8/Last2 Control에서는 확정된 역전이 없었다.
낮은 QPS의 Layer/Token 비율은 0.943712 [0.924118,0.965350]였다.
높은 QPS에서는 1.055811 [0.948638,1.139639]였다.
높은 부하의 CI는 1을 포함한다. Point 방향만으로 QPS Crossover를 선언하지 않는다.
별도 Pipeline-load N1 Supplement도 Winner Confirmation으로 세지 않는다.

## 5. Pure Layer와 Pure Token의 최적 영역은 무엇인가?

시험한 C0 Grid에서는 Input, Output, Cut과 P/D Queue Regime이 함께 Winner를 바꿨다.
긴 Input의 짧은 Output에서는 Token 우세가 자주 나타났다.
길어진 Output의 일부 Cell에서는 Layer 우세가 나타났다.
그러나 Cut 하나나 Input 하나로 전체 Map을 설명하지 못했다.
Best C3 Static 후보는 별도 Discovery에서 선택하고 Test 전에 동결했다.
그 후보를 C0의 같은 Rows 비교와 혼합하지 않는다.

## 6. Genuine 2D Hybrid가 Pure Axis보다 좋은가?

정합한 Interior Partition을 구현했다.
Prefix Length를 `p=L-s`로 두면 P는 Lower36−m Layers의 Prefix를 계산한다.
D는 Prefix Upper-m Layers와 Suffix Full36 Layers를 의존 순서대로 계산한다.
D Rows는 `pm+36s=Lm+(36-m)s`다. `Lm+36s`는 중복 계산이다.
모든 Partition의 P/D Rows 합은 `36L`이다.
Attention Pair, KV와 Hidden Payload는 별도로 계산한다.
근거: [DAG와 State 계약](dense_ar_layer_token_2d_hybrid_design.md), [정합성](dense_ar_layer_token_2d_hybrid_correctness.md).

Iso-work 8개 Cell 중 I3/Output64에서 작은 Hybrid 이득을 확인했다.
Hybrid는 Last4+정확한 L/8 Suffix다. Pure 대조군은 Last8과 Rows-matched Token이다.
각 Arm은 독립 Start3회다.
Hybrid/Token TTFT는 0.980561 [0.971238,0.990277]였다.
Hybrid/Layer는 0.984181 [0.972197,0.996592]였다.
Hybrid/Native는 0.716077 [0.698106,0.730487]였다.
이 Cell의 Decode 및 Absolute Gate는 통과했다.
Hybrid/Token SLO-goodput은 1.004523 [0.998270,1.010060]이었다. 추가 Goodput 우위는 확정하지 못했다.
나머지 7개 Cell에는 확정한 Hybrid Win이 없었다.

## 7. 가장 강한 Static Token보다 좋은가?

위 Iso-work 이득만으로는 그렇다고 답할 수 없다.
강한 C3 Stock Token은 원래 Full-model Compile과 Continuous Scheduler를 유지한다.
Mode0 Token Control로 강한 Stock Token을 대체하지 않았다.
C3 Controlled Output16/Cap1에서는 Hybrid가 Stock Token보다 느렸다.
Output64/Cap1도 Hybrid/Stock Token TTFT가 1.052899 [1.016506,1.088930]이었다.
이 Cell의 Hybrid/Mode0 Token은 0.984218 [0.967055,1.000025]였다.
후자의 CI도 1을 포함한다. 반올림으로 Win을 만들지 않는다.
원래 Output64/Cap2 Bridge도 3회 반복을 완료했다.
Hybrid/Stock Token은 1.036561 [0.999455,1.084776]이었다. 우위가 확정되지 않았다.
반면 Hybrid/Mode0 Token은 0.962634 [0.950580,0.979656]이었다.
느린 Mode0 Control만 사용하면 강한 Token 대비 결론을 잘못 낼 수 있다.
원래 세 Bridge 조건은 모두 남긴다.
등록 수정 후에도 같은 세 조건과 대조군을 별도로 검사한다.

## 8. Native/Agg 대비 Serving 순이익이 있는가?

수정 전 Agg/Native Motivation에서는 실제 Trade-off가 있었다.
QPS1.8의 독립 Start3회 평균은 다음과 같다.

| 방법 | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | Raw Output Tokens/s | SLO-goodput(Output Tokens/s) |
|---|---:|---:|---:|---:|---:|
| Agg DP2 | 1554 | 278.15 | 163.41 | 50.28 | 28.77 |
| Native PD | 4886 | 15.914 | 8.448 | 50.087 | 44.468 |

이는 LongBench400의 새 Benchmark다. DuetServe Figure2의 수치 재현은 아니다.
Native의 낮은 Decode Tail과 Agg의 낮은 TTFT를 구분한다.
새 Offload Runtime의 순이익은 최종 Both-Native Gate와 함께 판정한다.
Whole-prefill Deflection도 강한 대조군으로 유지한다.
근거: [Motivation](dense_ar_longbench400_motivation_agg_native.md).

별도 동결한 최종 Main400/QPS1.8도 독립 Start3회를 완료했다.
아래 값은 원래 Source `a2d5...`의 결과다. 등록 수정 후 결과가 아니다.
Motivation과 다른 Arrival 반복이므로 두 표의 값을 같은 Start로 합치지 않는다.

| 최종 Main400/QPS1.8 Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) |
|---|---:|---:|---:|---:|
| NativeStrong | 4307.191876 | 14.619585 | 8.292451 | 47.911229 |
| Best Layer | 3254.888262 | 9.076652 | 8.288349 | 48.741613 |
| Best Token | 2839.826040 | 8.949966 | 8.239344 | 50.024480 |
| Best Hybrid | 3018.807822 | 11.707069 | 9.418748 | 46.381191 |
| Whole-prefill Deflection | 3068.499336 | 8.592625 | 7.945489 | 49.236632 |

Layer, Token과 Deflection은 두 Native 기준의 Decode 보호를 통과했다. Hybrid는 실패했다.
Layer/NativeStrong TTFT는 0.755687 [0.719199,0.794162]였다.
그러나 Layer/강한 Token TTFT는 1.146158 [1.139623,1.155887]로 Layer가 느렸다.
Layer의 Common-policy Native 대비 Goodput CI는 1을 포함했다.
Token은 두 Native 대비 TTFT와 Goodput CI 우위를 모두 확인했다.
Token/Common-policy Native Goodput은 1.034983 [1.017478,1.059480]이었다.
Deflection도 두 Native 대비 TTFT와 Goodput CI 우위를 확인했다.
이는 동결한 후보의 유한 Main400 결과다. Dynamic 또는 Layer-specific 승리가 아니다.

원래 Main140은 모두 완료했다. Main400/QPS2.8도 각 Arm의 독립 Start3회다.

| 최종 Main400/QPS2.8 Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) |
|---|---:|---:|---:|---:|
| NativeStrong | 34932.248686 | 14.844949 | 8.327879 | 7.021338 |
| Best Layer | 11017.682784 | 9.779264 | 8.449652 | 25.421326 |
| Best Token | 5579.425988 | 8.919498 | 8.193915 | 66.922743 |
| Best Hybrid | 9215.782076 | 9.125977 | 8.335438 | 35.503208 |
| Whole-prefill Deflection | 7119.071647 | 8.702015 | 7.941099 | 58.279573 |

이 수정 전 Source의 Token과 Deflection은 두 Native의 Safety와 TTFT·Goodput Gate를 모두 통과했다.
Layer와 Hybrid 및 별도 TokenCap2 Control은 상대 Decode Gate 하나 이상을 실패했다.
Layer의 Common-policy Native 대비 ITL 및 TPOT CI 상한은 각각 1.092160과1.062689였다.
Hybrid의 TPOT CI 상한은 1.050312였다. 반올림해서 1.05 통과로 만들지 않았다.
Layer/강한 Token TTFT는 1.974698 [1.505425,2.495660]이었다.
Hybrid/강한 Token TTFT는 1.651744 [1.482651,1.868662]이었다.
Token/Common-policy Native Goodput은 8.952670 [8.039871,10.135779]이었다.
큰 Goodput 비율은 Baseline의 TTFT SLO 위반도 반영한다. Sustainable Capacity의 같은 배수 증가가 아니다.
등록 수정 후의 별도 QPS2.8 결과로 이 원래 실패를 교체하지 않는다.

## 9. Decode SLO를 보호하는가?

모든 Offload 실행이 자동으로 보호하지는 않았다.
원래 동결한 Heldout200/QPS1.8의 Layer, Token, Hybrid는 같은 정책 Native의 5% Gate를 모두 통과하지 못했다.
그 실패를 빠른 TTFT로 상쇄하지 않았다.
C3 Controlled Output64/Cap1과 Cap2의 모든 Offload Arm도 Native-relative Gate 하나 이상을 실패했다.

READ-only 진단에서는 Hybrid ITL p99가 14.449714ms에서 9.176005ms로 줄었다.
TPOT p95는 9.572242ms에서 8.310944ms로 줄었다.
이 값은 같은 Trace의 N1 진단이다. 독립 N3 Safety PASS가 아니다.
최대 ITL은 오히려 62.165ms에서 71.288ms로 증가했다.
개선된 p99만으로 모든 Interval의 상한을 보장하지 않는다.

별도 READ-only Confirmation도 독립 Start3회를 완료했다.
Hybrid의 ITL p99 평균은 11.007910→9.158424ms, TPOT p95는 9.209552→8.319373ms였다.
그러나 Hybrid/NativeREAD의 TPOT CI 상한은 1.053354였다. 1.05 기준을 통과하지 못했다.
별도 Layer Control도 같은 TPOT Gate의 CI 상한이 1.058397이었다.
Stock Token은 이 후속 검사에서 두 Native Gate를 모두 통과했다.
Hybrid/Stock Token TTFT 비율은 1.230969 [1.132285,1.326660]이었다.
READ 준비 개선만으로 Hybrid의 강한 Token 대비 Serving 우위를 확인하지 못했다.

후속 Source는 Active Decode 동안 새 Request 등록도 미룬다.
첫 Token 이후의 Host Gap을 줄이려는 별도 개입이다.
IPC 수신, 기존 Decode, ACK와 Retirement는 계속 처리한다.
Admission 시점과 Batch도 바뀐다. Python 비용만의 인과 효과라고 단정하지 않는다.
QPS1.8 및 2.8의 새 Seed3회에서 NativeStrong과 정확한 Common-policy Native를 모두 검사한다.
이 검사는 원래 Main 결과를 교체하지 않는다.
QPS1.8의 새 Seed2026112771/2772/2773은 모두 완료했다.
세 Offload Arm은 NativeStrong과 정확한 Common-policy Native의 상대 Decode Gate를 모두 통과했다.
각 Start의 Request별 Absolute Decode Attainment도 95% 이상이었다.

| QPS1.8 등록 수정 후 Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) |
|---|---:|---:|---:|---:|
| NativeStrong | 3682.195197 | 14.685411 | 8.402706 | 46.173476 |
| NativeRegistration | 3759.741055 | 8.742812 | 7.946945 | 46.539887 |
| NativeCompletionRegistration | 3759.331107 | 8.746956 | 7.968561 | 46.729328 |
| LayerRegistration | 2640.257311 | 8.873075 | 8.225188 | 45.652340 |
| TokenRegistration | 2432.918372 | 8.629237 | 7.982386 | 46.938302 |
| HybridRegistration | 2668.047908 | 8.888570 | 8.246027 | 45.958081 |

표는 독립 Start3회 Metric의 평균이다. 모든 Request를 합친 p95는 아니다.
Layer/Common Native의 ITL 및 TPOT 비율 CI 상한은 각각 1.017283과1.036229였다.
Hybrid/Common Native의 상한은 각각 1.019108과1.039027이었다.
Token/Common Completion Native의 상한은 각각 0.989522와1.005423이었다.
이는 5% 비열등성이다. 모든 Client Interval의 동일성이나 최대 ITL 보호를 보장하지 않는다.
관측 전체 Max ITL은 Layer641.203169ms, Token66.864716ms, Hybrid201.539109ms였다.
큰 Interval은 Primary Metric과 Absolute SLO 검사에서 제거하지 않았다.

안전성 회복은 Hybrid 우위와 다르다.
Layer/Token TTFT도 1.085222 [1.034622,1.138262]로 Layer가 느렸다.
Hybrid/Token TTFT는 1.096645 [1.036991,1.147475]로 Hybrid가 느렸다.
Hybrid/Token Goodput은 0.979117 [0.967172,0.988945]였다.
Layer/Common Native Goodput도 0.980929 [0.976739,0.986437]로 낮았다.
Token/Common Native Goodput은 1.004472 [0.995422,1.015951]로 차이가 불확실했다.
이 후속 검사는 같은 Main400의 새 Arrival 검사다. 새로운 Request Population 검사는 아니다.
Actual Batch와 Credit은 Discovery에서 동결한 Best-versus-best 설정이다. Iso-work Control이 아니다.
근거: [등록 수정의 N3 결과와 전체 CI](dense_ar_longbench400_request_registration_tail.md).

같은 QPS1.8 검사에서 Layer−Token의 Mean TTFT 차이는 +127.338ms [94.976,166.502]였다.
Arrival→P Prepare는 +148.842ms, P Prepare→Ready는 +39.828ms였다.
D Submit→Prepare는 −52.686ms, D Prepare→First Worker는 −8.731ms였다.
즉, 이 Host 경로에서 D 구간의 절감은 P 앞과 P 경로의 추가 지연을 만회하지 못했다.
이는 같은 ID와 Arrival을 비교한 3회 평균의 분해다. TTFT p95의 분해가 아니다.
Host Prepare 경계는 GPU Kernel 시작이 아니다. C3 구간에는 다른 Request도 들어간다.
Credit와 Compile 및 실제 Batch도 다르다. 이를 Attention만의 인과 효과로 단정하지 않는다.
고립 Last4의 Operator 차이와 이 Last8 Natural Online 경계를 한 수식에 합산하지 않았다.

등록 수정 후 QPS2.8도 Seed2026112781/2782/2783의 독립 Start3회를 완료했다.
Layer, Token과 Hybrid가 두 Native 기준의 ITL·TPOT 5% CI Gate를 모두 통과했다.
각 Start의 Absolute Decode Attainment도 95% 이상이었다.

| QPS2.8 등록 수정 후 Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | Raw Output Tokens/s | SLO-goodput(Output Tokens/s) | Mean Joint SLO |
|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 26484.191104 | 16.938246 | 8.475937 | 61.686933 | 9.755980 | 0.177500 |
| NativeRegistration | 26649.548378 | 8.925276 | 8.004097 | 61.867057 | 10.172318 | 0.175833 |
| NativeCap2Registration | 26709.308993 | 8.779906 | 7.957054 | 61.677295 | 10.083014 | 0.175000 |
| LayerRegistration | 6583.601082 | 8.948487 | 8.239755 | 70.345654 | 52.488352 | 0.761667 |
| TokenRegistration | 4390.895747 | 8.947635 | 8.050413 | 71.526013 | 67.709592 | 0.959167 |
| HybridRegistration | 5902.597826 | 8.943290 | 8.231843 | 70.387073 | 57.243633 | 0.825833 |

표는 Start별 통계량의 평균이다. Mean Joint≥95%는 모든 Start의 Joint≥95%를 보장하지 않는다.
Decode 보호를 회복해도 TTFT 위반은 남을 수 있다. Layer와 Hybrid의 p95는 5000ms를 넘었다.
Layer/Common Native의 ITL·TPOT 비율 CI 상한은 1.026533/1.036656이었다.
Token/Common Native는 1.031733/1.011719, Hybrid/Common Native는 1.009701/1.035706이었다.
Layer/Token TTFT는 1.499375 [1.437462,1.573933]였다.
Hybrid/Token은 1.344281 [1.312721,1.378300]이었다.
Hybrid는 Layer보다 빨랐지만 강한 Token보다 느렸다.
Hybrid/Token Goodput도 0.845429 [0.698429,0.953074]였다.
이 결과는 유한400개 실행이다. Raw Throughput의 증가와 Sustainable Capacity를 구분한다.

같은 QPS2.8의 Layer−Token Mean TTFT 차이는 +1389.413ms [1224.063,1598.965]였다.
Arrival→P Prepare 차이는 +1278.653ms, P Prepare→Ready는 +44.002ms였다.
D Submit→Prepare는 +75.529ms였다. D Prepare→First Worker는 −9.694ms였으나 CI는 0을 포함했다.
같은 ID의 이 Host 경로에서도 Layer의 고립 D 효율만으로 Online TTFT를 설명할 수 없었다.
Credit, Compile과 실제 Batch가 다른 비교다. 순수 Attention 효과로 단정하지 않는다.
QPS1.8과 2.8은 Credit, Cap과 Fusion 설정도 다르다. 두 표를 QPS-only 개입으로 취급하지 않는다.
근거: [QPS2.8의 전체 CI와 Host 경계](dense_ar_longbench400_request_registration_tail_q28.md).

별도 Controlled Bridge도 원래 GPU4/5, Request, Arrival, Cut과 Output을 유지했다.
원래 세 조건과 여섯 Arm의 전체54회를 완료했다. 각 Arm은 독립 Start3회다.
수정한 Natural400의 성공을 이 Controlled Cell의 성공으로 옮겨 쓰지 않았다.

| 등록 수정 Controlled 조건 | Layer: 두 Native 상대 Gate | Hybrid: 두 Native 상대 Gate | Layer/강한 Token TTFT p95 [95% CI] | Hybrid/강한 Token TTFT p95 [95% CI] |
|---|---|---|---|---|
| Output16/Cap1 | FAIL | FAIL | 1.240270 [1.144866,1.328268] | 1.128765 [1.051812,1.190227] |
| Output64/Cap1 | FAIL | FAIL | 1.036029 [1.013767,1.058513] | 1.051327 [1.026377,1.075733] |
| Output64/Cap2 | PASS | FAIL | 0.991571 [0.978190,1.004531] | 1.026522 [1.012591,1.041044] |

Output16/Cap1의 Layer와 Hybrid는 Common Native를 통과했지만 빠른 NativeStrong 기준은 통과하지 못했다.
Layer/NativeStrong ITL CI 상한은1.063700, Hybrid는1.077669였다.
Output64/Cap1에서는 두 Arm 모두 ITL Gate를 통과하지 못했다.
Output64/Cap2의 Layer/NativeStrong ITL·TPOT CI 상한은1.035747/1.031948이었다.
Layer/Common Native의 상한은1.033846/1.033424였다. Absolute Decode Gate도 통과했다.
그러나 Layer/강한 Token TTFT CI는1을 포함했다. Layer 우위로 선언하지 않는다.
같은 조건의 Hybrid는 두 Native의 ITL Gate와 Absolute Decode Gate를 실패했다.
Admission Cap 변화는 Layer의 적격성을 바꿨다. 모든 Axis의 Tail을 제거하는 공통 해결책은 아니었다.
강한 Stock Token은 세 조건에서 두 Native와 Absolute Decode Gate를 모두 통과했다.
근거: [등록 수정 후 전체54회와 CI](dense_ar_longbench400_registration_controlled_bridge.md).

### 마지막 제한적 Kernel 정책 검사

모든 Prompt Stage는 원래 Eager/CUDA Norm을 사용한다. Decode-only는 원래 Native Compiled Model을 유지한다.
두 Axis의 Strict N1/N8, 실제 Mixed32, Natural32와 Natural400 정합성을 먼저 통과했다.
Cold Dispatch의 CPU 실패와 이전 Partial-only Hybrid의 Strict 실패도 보존했다.
Tensor, Greedy Token과 Timestamp를 보정하지 않았다.

성능은 같은 I3 원문16개, Output16, QPS1.253873, D Cap1과 Source Credit8을 유지했다.
Seed2026112551/2552/2553의 독립 Start3회다. 모든 새 Paired Arm은 GPU0/1에서 실행했다.
NativeStrong은 원래 Scheduler다. Common Native와 Offload에만 D Cap1을 적용했다.
원래 GPU4/5 결과와 CI를 합치지 않는다. 이는 Adaptive 같은 조건 검사이며 새로운 Request 검증은 아니다.

| 최종 Source의 Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) | 두 Native Decode Gate 및 Absolute Gate |
|---|---:|---:|---:|---:|---|
| NativeStrong | 6109.482200 | 8.975284 | 7.966278 | 9.701630 | Reference |
| NativeCommon | 6109.127819 | 8.800160 | 7.964016 | 9.696048 | Reference |
| Strong Stock Token | 3137.411254 | 8.923986 | 8.062524 | 19.928978 | PASS |
| Layer, Eager Prefill/Compiled Decode | 3896.066958 | 9.239217 | 8.164296 | 18.711462 | FAIL: Common ITL CI |
| Hybrid, Eager Prefill/Compiled Decode | 3518.385320 | 9.044181 | 8.142871 | 19.249358 | PASS |

표는 Start별 통계량의 평균이다. 같은 Source의 변경하지 않은 기존 Layer·Hybrid도 대조군으로 남겼다.
Hybrid/Common Native의 ITL·TPOT CI 상한은1.037632/1.023752였다.
Hybrid/NativeStrong의 상한은1.009377/1.025767이었다. 모든 Start의 Absolute Decode도 통과했다.
반면 Layer/Common Native의 ITL Point Ratio는1.049892였지만 CI 상한은1.065921이었다.
따라서 Point 값만으로 Layer를 5% 비열등성 PASS로 만들지 않았다.
이 실패는 모집단의 실제 악화가 반드시5%를 넘는다는 증명과는 다르다.

같은 새 Source의 기존 Hybrid 대비 ITL 비율은0.979221 [0.965802,0.991001]이었다.
TPOT 비율은0.980672 [0.976983,0.985258]이었다. 그러나 TTFT는0.996012 [0.985239,1.006252]로 불확실했다.
Goodput 개선도1.004634 [1.002212,1.007159]로 작았다. 이를5% Dynamic 이득으로 확대하지 않는다.
이 개입은 Compiler와 Kernel 정책 및 이후 Batch Feedback을 함께 바꾼다. GPU Operator 하나의 순수 효과가 아니다.

수정 Hybrid/강한 Token의 TTFT는1.121429 [1.050122,1.179215]였다.
Goodput은0.965898 [0.949395,0.988077]이었다. Decode를 보호해도 강한 Token보다 느렸다.
Hybrid/Common Native TTFT는0.575923 [0.542633,0.618146]로 개선됐다.
하지만 Native 대비 개선과 Best Pure-axis 대비 독립 기여는 다른 판정이다.
유한16개 Throughput과 SLO-goodput을 Sustainable Capacity로 표현하지 않는다.

최종 Source SHA256은 `2b004c881d5c2f7caf66d6c467957d43079e27a15694cae54b8ee90971504a9f`다.
근거: [모든21회, 실제 Batch·Compile 설정과 전체 CI](dense_ar_longbench400_eager_prefill_native_decode_results.md).

## 10. Dynamic Scheduler가 독립 이득을 제공하는가?

현재 `HOLD_DECODE_SAFE_WINDOW_NOT_IDENTIFIED`다.
적격 공통 ROI에서 P Backlog와 낮은 D Kernel Busy가 함께 관측됐다.
QPS2.8의 평균 Kernel Busy는 P99.520%, D49.028%였다.
같은 ROI의 P Waiting 평균은 20.921개, D Decode-ready Waiting 평균은 0개였다.
그러나 Kernel Inactivity는 준비된 Partial State가 들어갈 미래 Safe Window가 아니다.
실제 Safe-fit Rate는 `NOT_MEASURED`다. Online Dynamic Size/Axis는 `NOT_RUN`이다.

제한된 State-updating Offline Oracle의 최대 Workload-mean 예측 Goodput 비율은 약1.04418이었다.
이는 Model이 선택한 Fixed 후보 대비 값이다. 가장 강한 실측 Static 대비 상한이 아니다.
전역 최적 상한도 아니며 Online Speedup도 아니다.
검증된 평균5% 추가 Headroom은 확보하지 못했다.
근거: [Dynamic Gate와 Oracle](dense_ar_layer_token_2d_dynamic_scheduler.md).

## 11. 가장 중요한 1–3개 Knob은 무엇인가?

현재 유력한 설명 변수는 P/D Effective Service Balance, Source Credit/Backpressure, D Query Chunk 수다.
검증된 Natural C3 Online Selector는 아직 아니다.
Serial의 P+D 합이 작아도 여러 요청이 겹치는 Pipeline의 Critical Path가 짧다는 뜻은 아니다.
P에서 생긴 작은 추가 Service와 앞선 요청의 체류가 Queue를 증폭할 수 있다.
D의 짧은 Continuation만으로 그 대기를 만회하지 못하는 조건을 실제 Host 경계에서 확인했다.
위 QPS2.8의 Layer−Token 평균 TTFT 차이1389.413ms 중1278.653ms는 Arrival→P Prepare에서 나타났다.
이를 Request별 GPU Attention만의 기여율로 해석하지 않는다.

Source Credit만 바꾼 N1 Control도 대기를 바꿨다. Hidden Pool16과 D 정책은 유지했다.
Layer8의 Credit4→8에서 TTFT p95는7779.430→6227.499ms였다.
그러나 Credit16에서는6362.531ms로 다시 늘었다. Credit 증가가 항상 이득은 아니었다.
Chunk 사이 Host Gap을 줄인 별도 Admission 개입도 Layer·Hybrid TTFT를 악화시켰다.
따라서 Queue 원인을 단일 Gap 또는 Credit 값 하나로 환원하지 않았다.
이 N1 개입과 다른 Cut의 고립 Kernel 차이를 같은 수식으로 합산하지 않았다.

탐색적 `max(P,D)` Rule은 명확한 C0 Winner38개 중 34개를 맞췄다.
Serial `P+D` Rule은 16/38개였다.
하지만 Bottleneck Rule의 최대 TTFT Regret은 13.863%였다.
알려진 Controlled Output과 여러 Calibration 입력을 하나의 Score로 압축한 결과다.
이를 직접 관측 가능한 단일 Online Knob이라고 주장하지 않는다.

최종 로그에서는 DECODE로 표시한 Host Step이 첫 Worker 출력 전에 시작하는 경우도 있었다.
따라서 DECODE Label만으로 순수한 후속 Decode 비용을 만들지 않았다.
첫 출력 이후의1-query Host Step을 별도 분리했다. 기존 첫 출력 전·후 Table과 모든 Client Interval도 보존했다.
그 부분집합의 Host p99는 NativeStrong8.247261ms, Stock Token8.336188ms, 수정 Hybrid8.523289ms였다.
각 Arm의 분모는3개 Start의 Host Step672개다. Request48개를48개 독립 Start로 사용하지 않았다.
이 값도 GPU-exclusive Kernel 시간이 아니다. Primary Client ITL·TPOT와 안전성 판정은 바뀌지 않았다.

적격 긴 Input/Last4 계측에서 P Kernel Union은 Layer791.907ms, Token763.103ms였다.
P Attention은 각각 287.453ms와 258.186ms였다.
D Kernel Union은 Layer94.329ms, Token126.338ms였다.
D Attention은 각각 32.926ms와 65.107ms였다.
이 조건의 차이는 GEMM만이 아니라 Attention 배치가 크게 설명했다.
이는 고립 C0 계측이다. Natural C3의 Operator 기여율은 `NOT_MEASURED`다.
같은 적격 Capture에서 D Projection Kernel Launch는 Request당 Layer58개, Token144개였다.
그러나 Projection GPU Union은 각각 약50.534ms와52.069ms였다.
Launch 수 차이를 같은 배수의 계산시간 이득으로 바꾸지 않는다.
실제 D GEMM의 M 범위는 Layer1036–8192, Token2224–3456이었다.
전체 Input 길이를 GEMM의 M으로 대입하지 않는다. B8192가 긴 Layer Query도 나눈다.
이는 긴 Input/Last4 결과다. 짧은 Input의 미적격 Capture를 대신하지 않는다.
근거: [Cost와 Rule의 반례](dense_ar_longbench400_axis_cost_model.md).

## 12. 실패한 조건과 반복 확인한 반례는 무엇인가?

- 작은 Cut이면 항상 Layer가 이긴다는 Rule은 64-cell Map에서 반증됐다.
- C0의 작은 Hybrid TTFT 이득은 강한 C3 Stock Token 대비 우위로 이어지지 않았다.
- 큰 Prefill 계산만 미룬 정책은 Remote READ 및 Host 등록 Tail을 제거하지 못했다.
- Completion Gap을 줄인 개입은 일부 Arm에서 TTFT를 악화시켰다.
- Source Credit1은 강한 Serving 기본값이 아니다. Credit32는 동결한 Memory Reserve Preflight에서 실패했다.
- Short-input Operator Capture는 Observer5% Gate를 실패했다. 그 비중은 `DIAGNOSTIC`이다.
- 초기 Preserved-compile Hybrid의 Strict Hidden 검사는 실패했다. 해당 Arm의 성능은 실행하지 않았다.
- 초기 D-only Compiled Model 경로도 Strict KV·Hidden Gate를 실패했다.
- 같은 Harness의 기존 Mode0 대조군은 KV·Hidden·Greedy16이 모두 일치했다.
- D의 IR Norm Provider Bundle을 Mode0 우선순위로 맞추면 두 Axis의 Strict N1/N8이 모두 통과했다.
- 이 Global IR 경로는 Mixed32·Natural32·Natural400도 통과했다. 적격 Serving 속도 향상은 미측정이다.
- Partial-prefill만 원래 Norm으로 제한한 다음 Source의 Hybrid는 Strict State를 실패했다. 예정 성능21회는 모두 NOT_RUN이다.
- 드문 384.483039ms Client ITL에 같은 Worker 간격은 8.083349ms였다. 구체적인 전달 지연 원인은 `NOT_IDENTIFIED`다.
- Cancellation, Remote Failure 복구, 전체 Forced Decode Batch2/4 Grid는 `NOT_RUN`이다.

모든 실패 Raw와 불리한 Cell을 보존한다.
BI0의 다른 Batch/Compile Geometry는 완전 Token 일치를 보장하지 않는다.
같은 Numerical 경로의 Strict Teacher 검사와 다른 경로의 State/Task-quality 검사를 구분한다.
FP32 진단을 BF16 성능 Baseline으로 사용하지 않았다.
근거: [Numerical 범위](dense_ar_longbench400_bi0_numerical_reference_audit.md).
추가 근거: [D Compile State 검사와 Negative Control](dense_ar_longbench400_decode_compile_residual.md).

## 최종 과학적 판정

| Gate | 판정 | 범위 |
|---|---|---|
| Dataset400 | PASS, Cell 균형 부분 달성 | 실제 원문400개. Input100/bin. Long-input Output Cell 부족을 보존 |
| C0 Axis Crossover | 조건부 확인 | 양축 모두 안전한47개 중 Layer15/Token16/Tie16. Pointwise CI |
| Genuine 2D Correctness | PASS | 실제 Causal DAG, Pure Boundary와 Interior State·Lifetime |
| C0 Iso-work Hybrid | 제한적 TTFT 이득 | 8개 Cell 중1개. 강한 전체 Static 대비 추가 Goodput은 미확정 |
| C3 Layer/Hybrid-specific Strong GO | NO-GO ON TESTED GRID | 강한 Stock Token 대비 재현 가능한 추가 Serving 이득 미확보 |
| Decode 보호 | 조건별 PASS/FAIL | Natural400 두 Load의 공통 수정은 PASS. 모든 Fixed-output 조건은 아님 |
| Dynamic Size/Axis/2D | HOLD | 실제 준비된 State의 Decode-safe Window와 Online Predictor 미검증 |

모든 판정은 시험한 Qwen3-8B, BI0, Workload, Runtime과 GPU에 한정한다.
다른 Partial-prefill Compiler나 Transport 구현까지 불가능하다는 주장은 아니다.
P의 Mode0 Partial Forward와 Stock Token의 원래 Compiler 차이는 현재 C3 구현의 제한으로 남는다.
향후에는 Partial Forward의 적격 Compiler 경로와 실제 Safe-fit을 먼저 검증해야 한다.
빠른 D Forward만으로 Online TTFT 개선을 예측하는 단순 가설은 이 캠페인에서 충분하지 않았다.

## 검증과 Operational Closeout

원래 Main140, C3 Bridge54, READ Repair와 두 Load의 등록 정책 검사는 모두 완료했다.
원래 Main의 CPU Metric·Host 경계·Figure 재구성도 완료했다.
등록 수정 후의 별도 Controlled Bridge54도 완료했다. 모든 예정 조건과 실패를 유지한다.
최종 제한적 Kernel 정책도 전체21회와 새 Source의 모든 정합성 Gate를 완료했다.
Raw Metric, 실제 B/S, Source, Model, GPU UUID와 CPU 검사를 다시 수행했다.
Model 파일13개와 설치 vLLM Python Source2234개의 초기 Hash가 유지됐다. 모든 GPU Binary의 Upstream 검증은 아니다.
CPU 회귀검사18개를 통과했다. GPU 정합성이나 Decode Safety의 대체 증거로 사용하지 않는다.
최종 Figure와12개 답변은 별도 검토 Ledger의 SHA256으로 묶는다.

GPU Closeout, GPU0–7 Burner 복원과 무손실 Archive는 각각 Operational Receipt로 확인한다.
Remote SHA는 Commit 후에 생긴다. 자기 Commit SHA를 포함한 Report의 순환 참조는 만들지 않는다.
Commit/Push와 Remote SHA는 최종 응답 및 로컬 `state/publication_receipt.json`에 기록한다.
중간 Checkpoint 파일의 진행 상태는 해당 작성 시점의 기록이다. 최종 과학적 판정은 이 문서를 따른다.

## 재현 자료

- [Figure와 재계산 목록](dense_ar_longbench400_figures_and_reproduction.md).
- [Continuous 실행 계약](dense_ar_layer_token_2d_continuous_serving.md).
- [Source Credit와 Queue](dense_ar_longbench400_source_credit_chunk_causality.md).
- 전체 Raw, Config, Command, Source Capsule, 실제 Batch와 Metric은 `dense_ar_longbench400_2d_20261121`에 있다.

원래 최종 Source Capsule SHA256은 `a2d5ebc4cb39eff66a8fbba3b4c0b83a8957d79e50c11ff94a529b2cdb2ca6e3`다.
등록 정책 Overlay SHA256은 `07cdfb84be94fc4ae3b1e778d58c8c1d03ff7a99452b0cd3ee667966600447cd`다.
기존 사용자 Checkout과 Index를 보존한다. 두 Source의 결과도 혼합하지 않는다.
