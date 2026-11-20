# BI0 ShortAnswer200 연구 Decision Log

## 실행 범위

새 Branch는 `flashvep/dense-ar-bi0-shortanswer-axis-dynamic-20261120`다. Base는 `3a97151dd4e32b3c4e421de36363931f7ad62621`이다. 기존 보고서와 Raw Evidence는 변경하지 않는다. Primary Numerical Mode는 BI0다.

Phase A에서 입력을 동결했다. 기존160개 입력을 보존했다. Qasper40개를 사전 Seed로 선택했다. Natural Output을 확인하기 전에 대체 Task 정책도 기록했다.

## Loop1: 기존 Tail의 재현

먼저 과거 실제 Trace를 읽는 결정적 Reproduction Command를 실행했다.

`python -m poc_flashvep.dense_ar_bi0_shortanswer_pd.repro_tail --historical`

Return Code는1이었다. 이는 의도한 Red Signal이다. Client-visible ITL p99는120.433ms였고 Frozen Threshold는34ms였다. NIXL Transfer Host p95는250.132ms였다. 단일 과거 BI1 Start200개 요청의 Trace다. 신규 BI0 결과로 사용하지 않는다.

이 Loop는 API 구간과 Client Tail을 재현한다. 동시 발생만으로 원인을 확정하지 않는다. 새 BI0 N8/N32 Trace에서 같은 검사를 적용한 뒤 개입 결과를 비교한다.

## 경쟁 가설과 다음 최소 실험

1. Thread-pool 제출 대기가 Host Critical Path를 막는다. 같은 BI0 Workload에서 NIXL Thread 수4→1→0을 바꾸면 API Wall과 ITL Tail이 함께 감소할 수 있다.
2. Descriptor/등록/실제 Copy가 주요 비용이다. Thread 변경만으로 충분히 줄지 않고 Byte 수나 Descriptor Geometry에 더 강하게 반응할 수 있다.
3. Async Scheduler와 Host 제출 경계가 지연을 증폭한다. Thread 변경 효과가 작고 Sync/Async 변경에서 Host Gap이 변할 수 있다.

우선 Native의 기본4 Thread 경로와 단일 변수 변경을 비교한다. Model, Dataset, BI0, B/S, KV Budget과 Arrival Trace는 유지한다. 효과가 나타나도 전체 KV/Hidden과 Task 품질을 검증하기 전에는 Primary 개선으로 채택하지 않는다.

## 품질 기준의 사전 처리

같은 Geometry의 Strict Token Match와 다른 Geometry의 수치·Task 품질을 구분한다. 다른 Geometry의 Task 점수 허용폭은 사용자에게 비차단 질문으로 확인했다. 답이 없으면 연구용 기본2.0점 기준을 별도로 사전 등록한다.0–100점의 Task별 차이와 Paired CI를 모두 기록한다. Production 정확도 보장으로 표현하지 않는다.

## 보존과 시간 관리

새 Namespace만 사용한다. 설치된 vLLM/PyTorch/NIXL 파일은 수정하지 않는다. Scoring Dependency는 캠페인 전용 Tool Directory에만 설치한다. Resource Guard와 Quiet-lane Lock을 재사용한다. 마지막1시간에는 새 대형 GPU 실험을 시작하지 않는다.

Notion Archive의 설명은2026-11-10까지다. 이번 Native와 Axis 연구의 최신 근거는 Git Commit과 Raw Evidence다. Notion은 읽기 전용으로 확인했다.

## Loop2: NIXL Thread 수 개입

동일 N32/QPS2.35/Seed에서 Native4/1/0 Thread와 Agg를 측정했다. Native ITL p99는 각각27.250/17.317/125.882ms였다. Native TTFT p95는13.734/13.773/14.000초였다. 각 설정은 독립 Start1회다. 모든 설정의 작은 표본 Task 품질 검사는 통과했다. 이 검사로 전체200개 품질을 확정하지 않는다.

0 Thread가 가장 좋다는 가설은 기각했다. 1 Thread는 Tail 개선 후보로 남겼다. 기본4 Thread를 Common Main에서 유지한다. 1 Thread는 같은200개와3개 Arrival Seed의 별도 개입으로 확인한다. API Host 시간과 ITL 변화만으로 순수 DMA 원인을 확정하지 않는다.

SDK의 기본 Agent Sync Mode는 `NIXL_THREAD_SYNC_NONE`이다. 안전한 동시 호출 계약을 확인하기 전에는 Transfer API를 임의 Background Thread로 옮기지 않는다. 비동기 Posting 변경은 아직 구현하지 않았다.

첫 Thread-screen Controller는 GPU 작업 시작 전에 JSON 직렬화에서 실패했다. NumPy Boolean을 Python Boolean으로 변환했다. 실제 Quality 함수의 JSON Regression Test를 추가했다. 원래 실패 Log와 해결 상태를 보존했다. 이후4개 GPU Arm은 정상 완료했다.

## Loop3: Forward 효율과 Online Queue의 분리

공통 Main은 QPS0.4/1.2/2.35의200개 Natural 요청으로 시작한다. 기본4 Thread Native와 Agg를 비교한다. QPS2.35는3개 독립 Arrival Seed로 확인한다. 이 Grid는 N32 Discovery 뒤, 신규200개 성능을 보기 전에 동결했다.

별도 고립 실험은 실제 BI0 Decoder Layer API를 사용한다. 실제 원문 Token의4608/8064 Prefix를 사용한다. 두 길이는576의 배수다. 따라서 Last1/2/4/6/8/12의 모든 비교에서 `L*m=36*s`와16-Token 정렬을 정확히 만족한다. 이 진단 입력을 Main Dataset으로 대체하지 않는다.

Token 경로는 Prefix Cache로 정확한 Causal KV를 만든다. Layer 경로는 실제 Full-query Boundary Hidden/Residual에서 시작한다. 모든 Replay의 Hidden을 같은 Geometry의 원래 Forward와 비교한다. 전송·Queue·LM-head·Sampling은 이 비용에 포함하지 않는다. CUDA Event Enclosure와 CUPTI Kernel Union도 별도로 보고한다.

## Loop4: Transport Geometry와 초기화

KV Block16/128/256을 별도 개입으로 비교했다. N8의 전체 유효 KV, Hidden과 Import 검사는 모두 통과했다. N32의 측정 요청32개만 사용한 SDK Posting 중앙값은 Block128에서17.426ms, Block256에서10.946ms였다. Warmup2개를 포함한 초기 요약과 분모가 다르다. 최종 요약은 Measurement Window 밖의 호출을 제외한다.

이 개입은 Descriptor 수와 Memory Rounding을 바꾼다. Attention Kernel Geometry도 바뀔 수 있다. 따라서 Posting 감소를 순수 DMA 개선이나 단일 원인의 완전한 증명으로 표현하지 않는다. Shared Axis Control은 Online Axis 결과를 보기 전에 Block128로 동결했다. Block256 결과도 보존한다.

Wrapper Import 순서가 공식 UCX 초기화를 건너뛴 사실을 발견했다. 공통 Main의 다음 실행 전에 기존 Controller를 Between-job 경계에서 중단했다. GPU 작업은 없었다. 수정 후 N8 전체 상태 검사를 통과했다. 새 기준선은 공식 초기화를 명시한다. 초기화 A/B의 단일 Start는 성능 개선을 보장하지 않았다.

## Loop5: Observer 간섭

Native N32의 Clean/Profile/Clean은5% Observer Gate를 통과하지 못했다. Controlled Native에서도 실패했다. 따라서 이 두 Trace의 P/D Kernel Busy는`DIAGNOSTIC`이다. 낮은 D Kernel Busy만으로 Decode-safe Window를 선언하지 않는다.

다음 반증 실험은 D Collector만 제거한다. P Collector와 나머지 계약은 유지한다. 같은5% 기준을 적용한다. 이 개입도 실패하면 Primary Kernel Busy 판정을 보류한다. 통과할 때까지 같은 조건을 반복하는 실험은 하지 않는다.

## Loop6: 실제 Continuation과 시간 정렬

새 Adapter는 실제 BI0 Model Layer와 공식 NIXL Connector를 사용한다. Layer는 Prefix Layer KV와 Raw Hidden/Residual을 보낸다. Token은 완료한 Full36 Prefix KV를 보낸다. D는 정확한 Causal Continuation과 Full36 Decode를 수행한다. N8 검사는 순차 입력과 Burst 입력을 구분한다.

안전한 Mechanism Control은 P1/D-suffix1/D-decode1과 Source Credit4를 사용한다. D의 요청과 NIXL Pending이 없어야 한다. Device-wide Synchronization 후에만 D에 새 요청을 넣는다. 이 계약은 이미 실행 중인 Decode에 Suffix를 끼워 넣지 않는다. 일반 Continuous Native/Agg 기준선을 이 Control로 대체하지 않는다.

현재 상태만 보는 첫 Window Rule은 P 시작 시점의 D Active를 검사한다. 그러나 유효한 기회는 미래 P Cut-ready 시점에 생길 수 있다. 따라서 이 Rule은 이전 D가 방금 시작한 경우를 과도하게 거절할 수 있다. N8 Branch Control로 확인한다. 이 한계를 알면서 대형 성능 Sweep을 실행하지 않는다.

다음 단계는 실제 Serial Continuation 비용을 Input 길이별로 측정한다. Calibration20개와 Validation15개는 서로 다르다. Queue와 Source Credit을 갱신하는 Replay를 검증한다. 같은 Trace의 관측 비용을 다시 넣어 맞춘 결과는 Predictive Validation으로 사용하지 않는다.

## Loop7: 시작 실패와 복구

공식 초기화 후 첫 Main은 NIXL Listener 포트 충돌로 시작에 실패했다. 첫 측정 요청 전에 실패했다. Raw Log와 Source Snapshot을 보존했다. 포트 선택을 OS Ephemeral 범위 밖으로 옮겼다. Thread의 Bind 실패가 무한 Ready 대기를 만들지 않도록 실패를 Main에 전달한다. 두 CPU Regression Test를 추가했다.

세 개 새 Controller의 첫 Shell-background 실행은 Tool Process Group 정리 후 남지 않았다. GPU 작업은 시작하지 않았다. 별도 Session으로 다시 실행했다. 이후 PID와 Lease를 확인한다. 이 운영 실패를 GPU 성능 실패로 분류하지 않는다.

## Loop8: D 효율과 P Critical Path의 경쟁 가설

고립3-Start 실험에서 작은 Token M의 GEMM 비용이 컸다. 그러나 같은 GEMM Rows에서 Token은 Layer보다 D에 더 많은 Attention Pair를 배치한다. 그만큼 P Attention 작업을 더 많이 줄인다. 따라서 D Forward가 빠른 Axis와 P 병목을 더 줄이는 Axis는 다를 수 있다.

가설A는 Layer의 짧은 D Forward가 TTFT를 지배한다는 것이다. 가설B는 D 여유가 충분할 때 Token의 P Attention 절감이 더 중요하다는 것이다. 두 가설은 P Service와 D First-output Service를 분리한 실험에서 다른 예측을 한다. Serial Cost20/15와 실제 Batch1 Online 비교를 함께 사용한다. Queue 없는 P+D 경로와 Queue가 누적된 경로를 구분한다.

작은 Cut의 추가 Hidden/Residual도 반례 후보다. Last1의 전송량은 구조적으로 Native36 대비39개 Layer 단위다. 이 Byte 수는 실제 Transfer 시간이나 HBM Traffic이 아니다. 실제 Serial Transport 경계를 확인하기 전에는 이 원인을 확정하지 않는다.

## Loop9: Held-out Serving과 Oracle의 한계

실제 Static Discovery는 Seed2026112061을 사용한다. Full200 Task 품질 CI를 통과한 Last4/8/12 중 Axis별 후보를 선택한다. 선택 기준은 SLO-qualified Output tokens/s다. TTFT p95로 동률을 처리한다. 다른 Cut과 Fraction까지 전역 최적화했다고 주장하지 않는다.

Held-out Confirmation은 Seed2026112051/52/53을 사용한다. 이 Seed는 일반 Continuous Native/Agg Main과 같은 Arrival Trace를 만든다. Shared KV128의 일반 Native/Agg도 별도로 실행한다. C0 Batch1 Control과 C3 Continuous 실행을 같은 Batch 실험으로 표현하지 않는다.

Oracle은 FIFO Queue와 Source Credit을 갱신한다. 미래 Arrival과 Output Length를 사용하는 Oracle과 Online 관측을 구분한다. 현재 Model은 Client ITL Tail을 예측하지 않는다. 따라서 TTFT-qualified Token Proxy의 개선만으로 Joint-SLO Headroom이나 Online Axis Gate를 통과시키지 않는다. Beam Width24는 전역 최적 상한이 아니다.

CPU Controller의 Metrics 반환값 할당 오류를 실행 전에 발견했다. GPU 작업이 없는 대기 경계에서 해당 Controller만 다시 시작했다. Dataset과 동결한 실험 계약은 바꾸지 않았다. Raw Serving 결과는 아직 생성되지 않았으므로 덮어쓴 결과가 없다.

## Loop10: Token의 같은 Geometry Reference

Token은 Full-query Reference와 다른 Query Chunk 경계를 사용한다. 따라서 Import가 정확하다는 사실만으로 나머지 Causal 계산을 모두 검증하지 않는다. Single-GPU Teacher에 같은 Prefix/Suffix 경계를 적용한다. B8192 상한 안에서 해당 Step의 Budget만 낮춘다. 원래 Full36 Model과 KV를 유지한다.

각 Cut의 전체 유효 KV, Last Hidden, Greedy Sequence와 실제 Query Span을 비교한다. 이 검사는 성능 측정이 아니다. Token의 같은 Geometry Gate가 끝나기 전에는 Serial Cost와 Online Discovery Controller가 기다린다. 기존 실제 Query Geometry 차이와 이 Gate의 불일치를 구분한다.

## Loop11: C0/Async-off Control의 Decode 비용

새 Batch1 Native Control의 Mean TBT는10.998ms였다. 같은 Dataset/QPS/Seed의 일반 Continuous BI0 KV128 실행은7.491ms였다. 이 둘은 Compile/Async와 Batch 계약이 다르다. 그 차이를 Layer 또는 Token 효과로 귀속하지 않는다.

Control에도 Full Decode Graph Replay가4684회 있었다. Graph 미적용 하나만으로 설명되지 않는다. 일반 Continuous 실행은4302회 Replay했다. 서로 다른 Actual Batch와 Output 수를 포함하므로 두 Replay 횟수를 직접 속도비로 사용하지 않는다.

다음 개입은 C0, Batch1, KV128과 Trace를 고정한다. Async Scheduling만 False→True로 바꾼다. 차이가 남으면 Compiled Kernel 또는 다른 Host 경로를 조사한다. 빠른 후보가 나와도 KV/Hidden/Output과 Lifetime을 다시 검증한다. 현재 C0/Async-off의 개선만으로 일반 BI0 Native 대비 Decode5% 보호를 주장하지 않는다.

## Loop12: 실제 Decode Query 수

일반 Continuous Native의 QPS2.35/Seed2026112051에서 모든200개 요청은 출력에 나타나지 않은 Decode Query1개를 추가 실행했다. 실제 Query 기록으로 확인했다. 이는 Async EOS Lookahead다. P의 유효 Prompt Query에는 중복이 없었다.

따라서 Worker First→Last Output을 단순히 GPU Step 수로 나누지 않는다. Emitted Tokens, 실제 Decode Query와 Graph Replay 수를 분리한다. Client의 마지막 Token은 Device-wide Drain 완료 시각이 아니다. Batch1 Replay도 Post-EOS Drain 또는 그에 해당하는 D Control Delay를 포함해야 한다. 이 비용을 누락한 Oracle은5% Headroom Gate를 통과할 수 없다.

## Loop13: Async A/B와 Lookahead 시점

C0/B8192/S1024/KV128/Batch1을 고정한 N32에서 Async-off/on의 Mean TBT는10.9442/7.3798ms였다. TPOT p95는11.6257/8.0830ms였다. 각 설정은 독립 Start1회다. 두 실행의 Greedy Sequence는32/32 일치했다. 작은 표본 Task 품질 Point 검사도 통과했다. 이 결과만으로 전체200개 또는 Native-relative5% CI를 통과했다고 주장하지 않는다.

Async-off는 모든32개 요청에서 첫 Worker Output 전에 Query1개를 준비했다. Async-on은2개를 준비했다. Async-on의 추가 Generation Query1개를 단순히 마지막 Output 뒤의 GPU 비용으로 더하면 안 된다. 추가 Pipeline 작업의 일부는 측정한 D First-output Service 안에 들어간다. 두 실행 모두 마지막 Worker Output 뒤에 새 Query를 준비한 요청은0개였다.

N32에서 Device-wide Admission Sync의 최대 Wall은 Off0.1820ms, On0.1186ms였다. 이는 모든 미래 조건의 GPU Drain 상한이 아니다. GPU 종료시각과 Host Query 준비시각도 다르다. Model은 실측 First-output Service와 실제 Control Delay를 사용한다. 관측하지 않은7ms를 모든 요청에 임의로 추가하지 않는다.

빠른 후보의 Layer/Token 상태 검사를 새 Tag`fast128_v3`에서 반복한다. 기존 Async-off 비용과 성능은 새 후보의 Primary 표에 섞지 않는다. Token의 같은 Geometry Teacher도 다시 실행한다. 실제 Cost/Online Sweep은 이 Gate가 끝난 뒤에만 시작한다.

## Loop14: 유효 Tensor 범위의 계측 감사

초기 보조 Numerical Probe는 Layer Producer의36개 Layer Cache를 모두 읽었다. Producer가 계산하지 않은 Suffix Layer의 값은 유효 KV가 아니다. 이 보조 Tensor는 Layer 정합성 판정에 사용하지 않았다. 실제 판정은`axis_events`의 계산한 Prefix Layer KV와 D의 전체 유효 KV를 사용했다.

새 Probe는 Producer가 계산한 Layer만 읽는다. Native Fallback과 Token Producer는36개를 유지한다. 두 CPU Regression Test를 추가했다. 기존 Raw Tensor는 덮어쓰지 않는다. 유효 범위 밖의 값을 포함할 수 있는 Producer 보조 Tensor는 로컬에 보존한다. 게시 Manifest는 해당 제외 사유와 SHA256을 기록한다. 실제 유효 상태 검사와 실패 Log는 계속 게시한다.

## Loop15: 빠른 경로와 Context 경계

Async-on의 Qualification과 Discovery 후보를 Last1/2/4/6/8/12로 확장했다. 느린 경로의 Last4/8/12 선정 계획과 구분한다. 빠른 경로의 모든 후보는 같은 State Gate를 먼저 통과해야 한다. Held-out 선택 기준은 그대로 유지한다.

기존 N8의 Input 범위는1910–23204 Tokens였다. 동결 Dataset의 최대38912 Tokens를 포함하지 않았다. 추가 Boundary Control은 최단,8192에 가장 가까운 요청과 최장 요청을 선택한다. 성능이나 Output을 보고 선택하지 않는다. Last12의 Layer/Token과 각 Geometry의 Single-GPU Teacher를 검사한다. 이 추가 검사를 모든 Cut의 모든 Context 조합에 대한 전수 검사로 표현하지 않는다.

고립 Forward는 기본 KV Block16이었다. 후속 통제 Serving은 공통 Block128이다. Projection-only 실험에는 KV Access가 없다. 고립 Attention Cost Ratio를 Block128 Online 비율로 그대로 사용하지 않는다. 실제 Serial Request Cost는 Block128에서 다시 측정한다.

## Loop16: Metric와 Query Coverage 감사

대표 Main Native/Agg200개는 모든 Delivery Event에서 Token1개씩을 반환했다. 첫 Event에 여러 Token을 담은 요청은0개였다. 그래도 Metric Helper는 첫 Event의 동시 Token도 처리하도록 보강했다. Client-visible 동시 Delivery는0ms Interval이다. 이를 GPU 생성 간격이라고 해석하지 않는다. 전체 Interval 수는 각 요청의 Output Tokens−1과 일치해야 한다.

Work Accounting은 Token-Layer Rows의 합만 검사해서는 충분하지 않다. 같은 합에서도 중복과 누락이 상쇄될 수 있다. Source와 Destination의 Query Span이 각각 연속인지 별도로 검사한다. Native의 마지막 Prompt Query 재실행1개는 명시적인 예외다. CPU Regression Test를 추가했다. 최종 Analysis는 이 검사를 다시 적용한다.

## Loop17: 두 상태 변수의 Offline Rule

P Backlog와 추정 D Remaining Work를 사용하는 작은 Horizon Rule을 사전 정의했다. 현재 Input Length, Task별 Calibration Output Prior와 Action Cost Lookup도 사용한다. 따라서 모든 Metadata를 제외한 두 숫자만으로 문제가 해결됐다고 표현하지 않는다.

Horizon 후보는0/1/2/4다. Discovery Seed61의 QPS1.2/1.65/2.35에서 TTFT-qualified Token Proxy 합으로 선택한다. Seed62/63을 계산하기 전에 동결한다. Physical Simulator는 실제 미래 종료시각을 사용한다. Decision 함수는 이미 발생한 완료·Emission, 현재 Backlog와 Calibration Prior만 읽는다.

이 Rule은 CPU Offline Replay다. 실제 Online Controller가 아니다. Full Joint SLO, Task-quality 변화와 Physical Safe Budget을 모델링하지 않는다. 좋은 Proxy 결과만으로 복잡한 Axis Controller를 승인하지 않는다.

## Loop18: Descriptor 수의 직접 개입

KV Block 변경은 Attention Geometry와 Memory Rounding도 바꾼다. 따라서 Descriptor 수 자체의 기여를 분리할 추가 실험을 선택했다. 기존 Byte-exact Coalescing 옵션을 Native에서만 진단한다. Adjacent Local/Remote 주소가 모두 연속인 경우만 합친다. 등록한 Layer Region 경계를 넘지 않는다. Packing이나 추가 KV Copy는 하지 않는다.

KV Block16,4 Thread와 공식 UCX 초기화를 고정한다. N8 전체 KV/Hidden, 동일 Native Greedy와 Lifetime을 먼저 검사한다. 통과한 경우에만 동일 N32 Trace의 Off/On을 각각1회 실행한다. Descriptor 수가 실제로 감소했는지도 확인한다. 감소하지 않은 옵션을 유효한 개입으로 주장하지 않는다.

현재 KV128 Native/Layer/Token의 Primary 설정은 바꾸지 않는다. 이 진단이 성공해도 한 Method에만 조용히 적용하지 않는다. Common Runtime으로 채택하려면 모든 Baseline과 Axis를 같은 계약에서 재검증해야 한다.

## Loop19: 추가 QPS의 시간 규칙

Held-out Confirmation을 먼저 완료한다. 추가 QPS는 남은 시간으로만 결정한다. Large-job Cutoff까지45분 이상이면1.2/1.65를 실행한다.25–45분이면1.65만 실행한다.25분 미만이면 `NOT_RUN`이다. 이 규칙은 첫 Static Discovery 성능을 보기 전에 동결했다.

기존45분 규칙의 대기 CPU Controller는 GPU Child가 없는 경계에서 교체했다. 기존 설계는 보존했다. 결과를 보고 유리한 QPS를 선택하지 않는다. 마지막1시간의 검증·보고·Push·Burner 복원 예산은 유지한다.

## Loop20: Source Submission 비용의 누락 방지

Discovery의 작은 Cut에서 Layer D 경로는 더 짧았다. 그러나 P Queue 차이가 이를 상쇄했다. Source IPC와 Admission의 요청당 작은 차이도 Queue에 누적될 수 있다. P Prepare→Ready만 사용하는 Model은 이 경로를 빠뜨린다.

따라서 Serial P Submit→Ready를 Source Service로 사용하는 Model v2를 Fit 전에 동결했다. Serial 실행에는 다른 요청의 P Queue가 없다. 기존 Prepare→Ready도 보존한다. Calibration20만 Fit에 사용한다. 다른15개 요청의 Validation 경계는 바꾸지 않는다.

## Loop21: 추가 State가 필요한지 검사

두 Knob이 반드시 필요하다고 가정하지 않는다. Service Model Fit 전에 Metadata-only와 D-state-only Ablation을 추가 선언했다. 현재 Input Length, Task Output Prior와 Action Cost는 모든 Rule의 Metadata다. D-state-only는 P Backlog를 무시한다. Metadata-only는 두 State를 모두 무시한다.

기존 대기 CPU Rule Controller는 GPU Child가 없는 경계에서만 교체했다. 실제 GPU Sweep과 Frozen Static 선정 기준은 바꾸지 않았다. 결과는 Conditional Offline Proxy다. 실제 Online Winner 예측이나 Decode-safe Window의 검증을 대신하지 않는다.

## Loop22: Coalescing 진단의 중단

같은 Synchronized Probe의 Non-coalesced N8에서도 Cross-run Producer Tensor의 Bitwise Equality가 실패했다. 두 실행의 Within-run KV Import는8/8 정확했다. Greedy도8/8 일치했다. 전송 손상 또는 단순 BF16 차이로 원인을 확정하지 않는다.

Strict Tensor Gate를 유지했다. N32 Off/On Timing은 실행하지 않았다. 추가 Cache Key 감사에서 두 Namespace의 차이는 NIXL Side-channel Port였다. 생성 Computation Graph Hash는 같았다. Cache 경로 차이만으로 Numerical Kernel 차이를 원인으로 채택하지 않는다. Primary KV128 Axis 실험은 이 진단 옵션을 사용하지 않는다.

## Loop23: 전체 Cut의 Discovery와 선정 동결

Last1/2/4/6/8/12의 Layer/Token Full200 Discovery를 완료했다. 모든 Cell은 Task-quality CI를 통과했다. Source Runtime의 핵심8개 파일 Hash는 같았다. 가장 높은 Joint SLO-goodput 기준으로 두 Axis 모두 Cut8이 선정됐다. Seed2051/52/53의 결과를 보기 전에 동결했다.

Last12에서만 Discovery의 Axis 순위가 바뀌었다. Layer TTFT p95는9426.5ms였다. Token은9905.4ms였다. 각1개 Start다. 이 조건을 재현된 Crossover라고 확정하지 않는다. 가장 큰 사전 후보의 탐색적 후속 조건으로 기록한다.

후속 Cut12 반복은 모든 Primary 측정 뒤에만 허용한다. Large-job Cutoff까지20분 이상 남으면 독립 Seed3개를 실행한다. 그렇지 않으면 `NOT_RUN`이다. 이 후속 결과로 이미 고정한 Best Static Cut8을 다시 선정하지 않는다. Final-hour 예산도 늘리지 않는다.

## Loop24: Source Credit Backpressure의 위치

Source Dispatch와 D First Client Output으로 Source Ownership을 재구성했다. 같은 시점의 P Active, P Pending과 Source Credit4를 확인했다. P Pending이 있는데 P가 Active가 아니며 Credit4개가 모두 사용 중이면 Source Admission이 차단된다.

이 Host 상태의 전체 측정구간 비율은 Native0.18%, Layer8 2.58%, Token8 4.85%, Layer12 10.19%였다. 각1개 Discovery Start다. GPU Kernel Busy/Idle의 비율이 아니다. P Queue의 증가가 항상 P Compute 포화를 뜻하지 않음을 보여준다. 상세 Event와 Sample 경계는 `analysis/BI0_PD_host_queue_summaries_v1.json`에 보존한다.

## Loop25: 남은 시간의 Low-load 대체 실험

모든 Primary 측정 뒤에20분이 남으면 Cut12 독립 반복을 우선한다. 그 예산이 없지만14분 이상 남으면 QPS1.2의 Actual Batch1 Native/Layer/Token을 각각1회 측정한다. 기존 전체5개 Method의 QPS 선택 규칙은 바꾸지 않는다. QPS1.2가 이미 측정됐으면 중복 실행하지 않는다.

이 규칙은 첫 Held-out Layer/Token 결과 전에 선언했다. 시간만으로 추가 실행을 결정한다. Missing Continuous Baseline은 다른 KV Block의 과거 수치로 채우지 않는다. 최종1시간의 예산은 늘리지 않는다.

## Loop26: 빠른 Control의 Observer 실패

빠른 C0 Native P-only Profile의 Mean TTFT 간섭은5.2908%였다. 사전5% 기준을 넘었다. Mean TBT와 TPOT p95가 기준 안에 들어와도 전체 Gate는 실패다. 기준을 바꾸거나 동일 Profile을 반복해 PASS를 찾지 않는다.

해당 Kernel 수치는 Diagnostic으로만 보존한다. C3 Continuous Native에서 통과한 별도 P-only 계측과 혼합하지 않는다. Clean Held-out 성능은 그대로 측정한다. 현재 Online Operator-exclusive GPU 비중은 Primary에서 미확정이다.

## Loop27: Replay 검증과 중단 기준

Serial 비용 Model의 Discovery TTFT 예측 MAE는284.33–1897.95ms였다. p95를6.21–40.26% 낮게 예측했다. Model을 더 복잡하게 만든 뒤 같은 Validation에 맞춰 성공으로 표시하지 않는다. 실제 Online Safety에 사용하지 않았다.

Beam24의 Mixed 탐색이 알려진 Specialized 경로를 버리는 Cell1개를 발견했다. 기존 결과를 보존했다. 수정한 후보 집합은 계산된 유효 Subset 경로도 포함한다. Best Specialized 대비 추가 Proxy는9개 Cell 모두0%였다. 유한 탐색이 전역 최적 상한이라는 주장은 하지 않는다.

## Loop28: 두 상태 Rule의 반례

Frozen Horizon2의 Held-out Fixed 대비 Proxy는0.9272–1.0076배였다. Metadata-only 대비는0.96174–1.00366배였다. 추가 State의 독립적인 이득을 확인하지 못했다. Active Safe Window와 Full Joint SLO Model도 없으므로 복잡한 Online Axis Controller는 보류한다.

CPU JSON 저장에서 NumPy Integer와 JSON Dictionary Key 문제를 발견했다. 두 실패 로그와 상태 Snapshot을 보존했다. 숫자나 Frozen 선택은 바꾸지 않았다. 저장 형식만 수정한 재실행은 완료됐다. GPU 실험을 다시 실행하지 않았다.

## Loop29: 추가 QPS와 동일 Output Control

시간 규칙은 QPS1.65만 선택했다.5개 Arm을 각각1회 완료했다. Layer8/Token8의 TTFT p95는7153.97/5700.44ms였다. Full200 Task-quality CI도 모두 통과했다. 이를 독립 반복을 갖춘 다른 QPS의 결론으로 표현하지 않는다.

Seed2026112062의 동일 Output Control도 완료했다. 세 Arm 모두4723Tokens였다. Layer8/Token8의 TTFT p95는5784.64/4773.52ms였다. Natural Output 수의 차이만으로 Primary의 순위를 설명할 수 없었다. 고정 길이 진단을 Natural Task-quality 결과로 바꾸지 않았다.

Primary 종료 후 남은 Large-job 시간은 약665초였다. 사전20분/14분 조건을 만족하지 않았다. Cut12 반복과 추가 Low-load Control은 `NOT_RUN`으로 남겼다. Static Cut8을 Held-out 결과로 다시 선정하지 않았다.

## Loop30: 최종 분석과 종료

원시 Client Event에서 최종 Parser로 Main Metrics를 다시 계산했다. 기존 Metrics 파일은 바꾸지 않았다. 모든 완료 Controlled/Serial 실행의 Query Coverage를 다시 검사했다. 실제 Batch와 Source Provenance를 Main Table에 포함했다.

첫 CPU Test 명령은 Package Root 옵션을 빠뜨렸다. Relative Import22개가 실패했다. Package Root를 명시한 같은 Suite는59개 Test를 모두 통과했다. GPU 실행이나 수치 Model을 바꾼 실패가 아니다.

Runtime Wheel RECORD와 이전 보고서 Hash를 검사했다. 원래 Checkout과 Staged Index를 보존했다. 캠페인 Controller와 GPU Worker는 모두 종료됐다. 승인 Burner8개를 복원했다. Lossless Archive는 복원한 모든 Member의 SHA256을 검사했다.

최종 연구 판정은 `HOLD_LAYER_AWARE_DECODE_SAFE_DYNAMIC`이다. Native 대비 Quiescent Static 개선은 확인했다. 그러나 강한 Token 대비 Layer의 독립 이득과 Dynamic의 추가 이득은 확보하지 못했다. 안전 Gate를 완화하지 않았다.
