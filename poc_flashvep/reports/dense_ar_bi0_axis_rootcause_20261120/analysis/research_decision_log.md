# BI0 Axis 원인 분석 — 결정 기록

상태는 `CLOSED`다. 기준 시각은2026-11-20 KST다. 새 결과와 기존 결과를 분리한다.

## 우선순위

사용자는 Serial과 Online의 순위 차이를 최우선으로 지정했다. 먼저 기존 Last4의2711.089ms P Queue 차이를 분해했다. 이후 같은 Request, Cut, 도착 Trace와 출력 계약으로 새 Control을 실행했다.

기존 Last4의 직접 Credit Block 차이는−1.041ms였다. 이전 Source 체인의 차이는+2706.135ms였다. 따라서 직접 Credit Block만으로 추가 Queue를 설명하는 가설은 지지받지 못했다. Credit의 간접 효과는 실제 Intervention으로 별도 검사한다.

## Serial 계약의 반증

기존25.351ms 이점은 Last8/Calibration20의 Host 경로에서 나왔다. Last8/Validation15의 부호는 반대였다. 두 부분집합을 Full200 Last4의 Online 차이에 합산하지 않는다.

새 Fixed-output Full200 Control은 요청별 길이를 고정했다. 총 출력은4723 Tokens다. Last4와 Last8 모두 Serial P+D Host 합계에서 Token이 빨랐다. 이 Control은 Natural EOS Main 결과가 아니다.

Natural EOS도 별도로 검사했다. Last4 Serial P+D Host 합계는 Layer582.848ms, Token566.913ms였다. 각 Axis의 Serial/Online Greedy Sequence는200개 모두 일치했다. 따라서 이 Full200 조건에서는 Serial Layer 우위라는 전제가 재현되지 않았다.

## 계측 수정

첫 DRIVER-only 계측은 Source Kernel 시간의58.22%만 Request에 연결했다. 이 계측을 Primary로 사용하지 않았다. CUDA Runtime API Correlation을 추가한 별도 설정을 검증했다. Model과 Scheduler는 바꾸지 않았다.

첫 Runtime+Driver Last4/Online Fixed-output Source 계측은200개 Request를 모두 연결했다. Kernel 시간의 Mapping 비율은100%였다. 여섯 Observer 지표의 최대 절대 변화는3.145%였다. Strict Query/Output 검사도 통과했다. 이 결과는 Profile1회와 Clean Control2회의 Gate다. 반복 성능 우위의 CI가 아니다.

첫 Destination 계측은 Graph Timestamp가0인 Record2개를 포함했다. Global Integrity Gate는 실패했다. Observer Gate만 통과했다는 이유로 Primary Kernel 시간으로 승격하지 않았다.

별도 Kernel-only Destination Protocol을 사전 정의했다. 이 Protocol은 Graph Enclosure Record를 수집하지 않는다. 모든 Model Graph Replay가 Decode Step에 속하는지 별도로 검사한다. Suffix Step에 Replay가 있으면 실패한다. Acceptance Threshold는 유지한다.

## 모델의 한계

Loaded Service Vector Replay는 다른 Arrival Seed에서 TTFT p95/MAE Gate를 통과했다. 그러나 Layer의 SLO-goodput 예측 오차는+15.388%였다. 추가 Goodput 검사의10% 기준을 통과하지 못했다. 이 모델로 검증된5% Dynamic Headroom을 주장하지 않는다.

Source/Destination Service Vector의 혼합은 민감도 분석이다. 실제 Transformer Partition이 아니다. 이를 구현 가능한 Method나 GPU Service-time Intervention으로 표현하지 않는다.

## 반복 검증

Last12의 독립 Start3회에서 TTFT p95 Layer/Token 비율은0.9598이었다.95% 경험적 CI는[0.8844,1.0683]이었다. Layer 우위는 미확정이다. 두 Axis는 같은 Native 대비 Decode5% 비열등성을 통과했다.

나머지 Cut, Credit, Input Bucket과 Continuous Bridge의 실행은 별도로 진행한다. 미실행 Cell은 `NOT_RUN`으로 남긴다. C0 결과로 C3나 Active-decode Safe Window를 자동 적격 처리하지 않는다.

## Historical Winner 추가 요청

사용자는 기존 GPU 실행을 유지하면서 Input8000/Output16·Last8의 과거 승리를 재검증하도록 요청했다.
과거3개 Paired Start의 Raw Source ZIP과 Config Hash를 다시 검사했다.
다섯 핵심 실행 파일은4595080c와 일치했다. A Family는 검증된 과거 실측을 재사용한다.
Current Runtime의 같은72개 입력에서 BI0/BI1의 N8 State·Greedy Gate도 새로 통과했다.
새 B/C Online Bridge는 같은 Input, Arrival, Cut과 Output 계약을 사용한다.
BI1 진단을 BI0 Primary와 합치지 않는다.

## Ready→D 가설 수정

과거 Ready→D 차이−824.402ms는 D Submit 시각−480.601ms와 P Ready 지연에 따른 대기 감소−343.801ms로 나뉜다.
D FIFO 체인의 정확한 누적에서 이전 D 준비 구간은−537.264ms를 설명한다.
Decode Residency 누적은+43.140ms였다. 따라서 Decode 겹침 감소를 Decode GPU Service 개선으로 해석하지 않는다.
이는 동일 Trace의 항등식이다. Guard 제거 Intervention이나 Exclusive GEMM 증거가 아니다.

## Current P Queue 가설 수정

원래 Natural Last4의 추가 Queue2711.089ms를 Source FIFO 체인으로 풀었다.
이전 Source Host Service 누적은+2876.210ms였다. Credit Extension 누적은−62.619ms였다.
실행 가능한 P의 Host Admission Gap 누적은+4.718ms였다.
Source Service 누적 중 Prepare→Ready는+2339.260ms다. Dispatch→Prepare는+538.177ms다.
Natural GPU Profile은 Observer Gate를 실패했다. 따라서 이 Host 항을 순수 GPU 비용으로 바꾸지 않는다.
별도 Fixed-output Profile의 적격 Attention 차이14.724ms는 지지 근거로만 사용한다.

## 새로운 최소 반증 실험

Current BI0의 Historical Workload에서 KV128은 첫 Paired Start에서 Token이 빨랐다.
KV16 One-factor Pair에서는 Layer가 빨랐다. 아직 독립 반복 결과는 없다.
같은 Block16에서 Native State Gate와 반복 확인을 추가했다.
유리했던 Seed20은 Discovery다. Seed21/22/23을 별도 Held-out Confirmation으로 동결했다.
모든 반복의 패배와 미실행도 보존한다. Primary Block128은 바꾸지 않는다.

Historical Layer/Token Start의 Norm Correction Counter는0이었다.
Prefill Graph 우위를 보여주는 기록도 확인하지 못했다.
따라서 이 두 가설을 원인으로 확정하지 않는다.
대신 이미 구현한 정확한 Byte Coalescing만 바꾸는 Current BI1 진단을 추가했다.
새 N8 State·Hidden·Greedy Gate 이후에만 Timing을 실행한다.
이 추가 실행도 원래10시간 예산을 넘기지 않는다.

## 우선순위 변경과 종료 결과

대기 중인 Full200 Controller를 감사한 뒤 P0 Confirmation을 먼저 배정했다.
해당 Controller에는 자식 GPU 작업이 없었다. 실행 중인 GPU Job은 중단하지 않았다.
Dataset, Seed, Cut, Credit, B/S와 SLO는 바꾸지 않았다.
Pilot 완료 Flag를 미리 만들지 않았다. 별도 Priority Amendment를 보존했다.

Full200/QPS2.35의 Last2/4/8/12와 양쪽 Axis를 독립3회씩 확인했다.
Last2/4/8의 TTFT CI는 Token을 지지했다. Last12 CI는1을 포함했다.
양쪽 Axis는 모든 Cut에서 Native-relative Decode5% Gate를 통과했다.
그러나 재현 가능한 Layer-specific TTFT·Goodput 우위는 확보하지 못했다.

N50 Discovery는27개 Cell을 완료했다. 나머지36개는 `NOT_RUN`이다.
측정한27개 모두 Task-quality CI Gate를 실패했다.
품질 Margin을 완화하지 않았다. 적격 Primary 결과로 승격하지 않았다.
다른 QPS의 Full200 Cell을 N50 데이터로 채우지 않았다.

KV16의 유리한 첫 Historical Start는 Held-out3회에서 확정되지 않았다.
BI1 Coalescing Flag의 상대 TTFT 차이도 별도 진단으로 남겼다.
후속 감사에서 Descriptor 감소는0회였다. 의도한 물리적 병합은 실현하지 못했다.
따라서 이 Flag 비교를 DMA 병합의 인과 효과로 사용하지 않는다.

C3 Layer는 Hidden Pool 확대 후에도 Mixed-row Assertion에서 실패했다.
정합성 Assertion을 제거하지 않았다. C3 Layer 성능은 적격 처리하지 않았다.
Token C3의 추가 Hidden·Greedy Teacher는 미실행으로 남겼다.
Destination Profile 실패와 Collector 종료 실패도 삭제하지 않았다.

GPU 작업은 약22:20 KST에 종료했다.
초기 Burner 사용률 확인은 모델 로딩 중40초 Poll 한도에서 실패했다.
로딩 후 같은 승인 Process의 소유권과8개 GPU 사용률100%를 다시 확인했다.
실험을 다시 실행하거나 다른 소유자의 Process를 종료하지 않았다.
CPU 검증, 무손실 Archive와 별도 Git Index 출판으로 마무리한다.
