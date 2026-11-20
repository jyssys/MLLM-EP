# BI0 Layer/Token Sweet Spot와 최소 Knob 분석

## 서로 다른 두 질문

Compute Sweet Spot은 같은 유효 작업을 어느 Axis가 더 빨리 계산하는지 묻는다. Online Sweet Spot은 Decode를 보호하면서 어느 Axis가 TTFT와 Joint SLO-goodput을 더 개선하는지 묻는다. 첫 질문의 승자가 둘째 질문에서도 이긴다고 가정하지 않는다.

고립 Forward의 Exact Matched-work Grid에서는 Layer가 더 빨랐다. Input은4608/8064다. Actual Batch는1이다. Last1/2/4/6/8/12를 검사했다. 각 Cell은 독립 Start3회다. 이 Grid에서 큰 Cut의 Token 우위를 관측했다고 주장하지 않는다.

Projection-only에서는 작은 Token M의 비용이 컸다. 큰 M의 비용비는 대체로1에 가까웠다. 이 결과는 Compute Sweet Spot의 일부다. Attention, Transfer, Queue와 Decode Residency를 제외한 결과다. 이를 단일 Token 길이의 보편적인 Online 임계값으로 사용하지 않는다.

## Attention 배치의 경쟁 효과

`STRUCTURAL_ACCOUNTING`. Input Length를 L, Layer Cut을 m, Token Suffix를 s로 둔다. 이상적인 Exact GEMM Work는 `L*m=36*s`다. Layer의 D Attention Pair는 `m*L*(L+1)/2`다. Token의 D Attention Pair는 `36*(L*(L+1)−(L−s)*(L−s+1))/2`다.

긴 L에서 Token/Layer D Attention Pair 비율은 대략 `2−m/36`이다. Token은 D에 더 많은 Attention을 배치한다. 그 대신 P에서는 더 많은 Attention을 줄인다. 전체 P+D의 유효 Causal Attention Pair는 같다. 이 배치 차이는 전체 Attention을 추가했다는 뜻이 아니다.

따라서 D의 실행시간이 짧은 Layer와 P의 실행시간을 더 줄이는 Token 사이에 경쟁 효과가 있다. P가 지배적인 병목이면 Token의 P 절감이 중요할 수 있다. D가 제한적인 시간 예산만 제공하면 Layer의 짧은 D 비용이 중요할 수 있다. 이 설명은 경쟁 가설이다. 실제 Serial Service와 Online Queue로 검증해야 한다.

## 작은 Cut의 제한

Last1은 P Layer Rows의1/36만 이전한다. Layer는 Input Token당 Hidden/Residual16384Bytes도 전송한다. 이는 KV4개 Layer에 해당한다. Last1의 유효 Payload는 Native36 대비39개 Layer 단위다. Exact Matched-work Token의 Payload는35개 단위다. 이 값은 Byte Accounting이다. 실제 DMA Time이나 HBM Read가 아니다.

유효 Prompt KV Byte는 Exact Work에서 Layer와 Token이 같다. Layer의 Hidden/Residual은 별도다. KV Block Rounding과 Runtime Reservation도 실제 전송량·Memory와 다를 수 있다. Layer Source는 전체36 Layer KV를 예약한다. 계산한 Prefix Layer만 유효한 KV로 전송한다.

## 후보 Knob

첫 후보는 보정된 P/D Service Balance다. P Service를 `p_a`, D First-output Service를 `f_a`, 후속 Decode Residency를 `d`로 두면 단순 직렬 Pipeline의 병목 후보는 `max(p_a, f_a+d)`다. 이 식은 실제 Runtime의 정확한 Queueing 법칙이 아니다. Source Credit과 Arrival을 갱신하는 Replay에서 오차를 검사해야 한다.

둘째 후보는 Candidate Fit Ratio다. `FitRatio_a=(Predicted Tail Cost_a+Margin)/Observed Safe Budget`으로 정의한다.1보다 크면 후보를 거절한다. 그러나 현재 Host Quiescence를 미래 Safe Budget으로 바꾸어 넣지 않는다. 실측 가능한 Ready State와 유효 Deadline이 없으면 이 Knob은 `NOT_IDENTIFIED`다.

실시간 P Backlog와 알려진 D Pending Work는 관측할 수 있다. 반면 Natural EOS의 남은 길이와 미래 Arrival은 알 수 없다. 이를 사용한 Replay의 좋은 예측을 Online Observability로 표현하지 않는다.

## 검증 제한

Full200은16-Token Prefix 정렬과 다양한 원문 길이를 사용한다. 전체 GEMM Rows가 수학적으로 완전히 일치하지 않는 Cut이 있다. 실제 차이를 모두 기록한다. Exact Work 주장은 고립4608/8064 Grid에 한정한다.

이1–2개 Knob의 Held-out Online Winner 예측은 최종적으로 `NOT_RUN`이다. 후속 Conditional Replay의 State Ablation은 완료했다. 추가 Queue State의 이득은 확인하지 못했다. Accuracy, TTFT Regret와 Goodput Regret를 실제 Online에서 검증하지 않았으므로 Minimal Knob Discovery PASS로 표시하지 않는다.

## 작은 Cut의 Online 반례

`MEASURED_ONLINE` Discovery와 `TRACE_ANALYSIS`다. QPS2.35, Seed2026112061, Actual Batch1/1/1에서 Last4의 TTFT p95는 Layer15.148초, Token10.049초였다. 각1개 독립 Start다. 같은 Cut의 D 경로가 짧다는 사실만으로 Layer의 TTFT 우위를 예측할 수 없었다.

Mean TTFT의 Layer−Token 차이는2673.86ms였다. 이 중 P Queue 차이는2711.09ms였다. P Prepare→Ready 차이는22.36ms였다. Ready→D Prepare 차이는−44.75ms였다. D Prepare→First Worker 차이는−14.86ms였다. First Worker→Client 차이는0.02ms였다. 각 요청의 경계를 합한 값이다. TTFT p95의 Additive 분해는 아니다.

Last1/2에서도 Layer의 D 경로는 짧았지만 P Queue가 길었다. 이 관측은 작은 요청당 Source 비용 차이가 Backlog에서 증폭된다는 가설과 일치한다. Operator별 GPU 인과 기여를 확정한 결과는 아니다. 모든 Cut의 Discovery와 선정한 Cut8의 Held-out3Start를 별도로 보존했다.

## Metadata와 State의 분리

Conditional Replay는 현재 Input Length, Task별 Output Prior와 보정된 Action Cost를 공통 Metadata로 사용한다. Metadata-only Rule은 P/D State를 무시한다. D-state-only Rule은 추정 D Remaining Work만 사용한다. 두 상태 Rule은 P Backlog도 사용한다.

이 Ablation은 Model Fit 전에 선언했다. Horizon은 Discovery Seed61에서 동결했다. Seed62/63에서 TTFT-qualified Token Proxy와 Regret를 비교했다. 이 Proxy는 Full Joint SLO, Task-quality 변화와 물리적 Safe Window를 포함하지 않는다.

## Offline Rule의 Held-out 결과

`OFFLINE_REPLAY`. Discovery에서 선택한 Horizon은2였다. Held-out은 QPS1.2/1.65/2.35와 Seed62/63의6개 Cell이다. Two-state Rule의 Discovery-frozen Fixed 대비 Proxy 비율은0.9272–1.0076이었다.5% 이상의 반복 가능한 개선은 확인하지 못했다.

동일 Metadata-only Rule 대비 비율은0.96174–1.00366이었다. 추가 State가 항상 개선하지 않았다. Low-load 일부 Cell은 모든 Rule이 같은 Arrival-limited Proxy를 얻었다. 이 동률을 같은 Action을 선택했다거나 Sweet Spot을 완전히 설명했다는 근거로 사용하지 않는다.

최소 Knob의 Held-out Online 검증은 `NOT_RUN`이다. 두 숫자로 Serving 우위를 예측할 수 있다고 주장하지 않는다. Model의 Queue 오차와 Task Output Prior의 작은 Calibration 표본도 남은 제한이다.

## 최종 Online 반례와 조건

QPS2.35에서 Discovery-frozen Layer8/Token8을 각각3개 독립 Held-out Start로 확인했다. Layer/Token TTFT p95 비율은1.2502[1.1871,1.3448]였다. Goodput 비율은0.7416[0.5128,0.8233]였다. Actual Batch1/1/1을 맞춰도 Layer의 고립 D 효율이 Online 승리로 연결되지 않았다.

QPS1.65의1회 확인과 동일 Output4723Tokens의 별도1회 Control에서도 Token TTFT가 더 낮았다. Natural Output 수 하나만이 순위의 원인은 아니다. Last12의 Discovery는 Layer 우위였지만 각1회다. 독립 반복은 예산 규칙에 따라 `NOT_RUN`이다. Cut-dependent Online Crossover의 통계적 확정은 보류한다.

현재 근거는 단일 Token M 임계값보다 P/D Service 배치와 Queue를 함께 봐야 함을 보여준다. 이 관측을 검증된 최소 제어식으로 표현하지 않는다. 실제 Safe Budget과 Online Tail Model이 없으므로 동적 Axis 선택의 충분조건도 아직 제시하지 못한다.

## 같은 Prompt의 Serial/Loaded 비교

`TRACE_ANALYSIS`. 각13개 Action에서 Calibration20과 Validation15를 Loaded Discovery의 같은 Prompt와 비교했다. 모든26개 Action/Split 비교에서 실제 Prompt Query Shape가 같았다. Greedy Output도 각20/20 또는15/15 일치했다. 핵심 Runtime8개 파일 Hash도 같았다.

P Worker Accept→Ready의 Loaded/Serial 평균비는1.0079–1.1746이었다. 다수 조건은 약1–3.5% 증가했다. Layer12의 Validation15가17.46% 증가한 최대 사례였다. D Worker Accept→First Output의 비율은0.6812–1.1001이었다.

같은 Input 분포와 Query Shape에서도 Service Envelope가 달랐다. Serial 비용을 Loaded 실행에 그대로 대입하는 데 한계가 있다. 이 결과만으로 GPU HBM 간섭, Clock 변화 또는 Host Control 중 하나를 원인으로 확정하지 않는다. Online Operator-exclusive GPU 비중은 여전히 Primary에서 미측정이다.

현재 Root-cause 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다. Host 구간과 Backpressure의 위치는 확인했다. 정확한 GPU Operator별 인과 기여와 일반화된1–2 Knob은 미확정이다. 근거는 `analysis/BI0_same_prompt_serial_loaded_service_v1.json`에 보존한다.

## 관련 연구와 비교 범위

[DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)는 Adaptive GPU Multiplexing을 다룬다. 본 실험은 Dense Qwen3-8B의 P→D Partial Continuation을 검증한다. 같은 Dataset이나 같은 실행 구조라고 주장하지 않는다.

[Kairos](https://arxiv.org/html/2607.02043v1)는 Prefill Deflection을 다룬다. 긴 D Quiescent Epoch가 존재한다면 작은 Suffix뿐 아니라 Whole-prefill Deflection도 강한 대조군 후보가 된다. 이번 구현에서 실행하지 않은 Whole-prefill 후보를 이겼다고 주장하지 않는다.

[LayeredPrefill](https://arxiv.org/html/2510.08055v1)은 다른 모델·Memory 실행 조건의 Layer-wise Scheduling을 다룬다. 해당 결과를 이 Dense2GPU 실험의 측정값으로 재사용하지 않는다.
