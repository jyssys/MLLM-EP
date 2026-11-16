# DuetServe Figure 2 및 Aggregated Decode 간섭 검증 계획

## 목적과 증거 수준

동일한 Aggregated 설정에서 QPS만 증가할 때 Decode Latency가 증가하는지 검사한다.
Mean TBT, ITL p99와 Request TPOT p95를 각각 판정한다.
Low-load 대비 2배 증가는 검증할 가설이다.
목표 결과를 만들기 위해 기준선을 약화하지 않는다.
현재 새 GPU 결과는 `NOT_RUN`이다.

## 고정 환경

Model은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16, Exact Attention과 Full36 계산을 유지한다.
Greedy Sampling과 Thinking Disabled를 유지한다.
GPU0/1의 H100 SXM 80GB를 사용한다.
GPU당 KV Budget은 42 GiB다.
Prefix Caching은 비활성이다.
Runtime은 vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129와 NIXL 1.4.1이다.
기존 Numerical Phase Correction을 재사용한다.
기존 Source, Raw Evidence와 사용자 Index를 변경하지 않는다.

## 논문과의 차이

Figure 2는 Qwen3-8B의 DP2다.
논문은 동일 Replica, Round-robin, Budget8192와 Input8000/Output200을 사용한다.
Figure 7은 Azure-Code 및 Qwen3-14B의 TP2다.
이번 Qwen3-8B TP2는 추가 비교다.
Figure 7의 정확한 재현으로 표현하지 않는다.
논문 본 평가의 vLLM은 0.10.1–v1이다.
현재 Runtime, 체크포인트와 Dataset은 동일하다고 가정하지 않는다.
근거는 [DuetServe 원문](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)이다.

## 실행 순서

1. Budget8192와 Budget512의 Continuous DP2를 N32로 검증한다.
2. Output200에서 QPS2–10을 같은 설정으로 Sweep한다.
3. Low/high 비율, 인접 변화와 Rank Correlation을 계산한다.
4. 같은 QPS에서 Budget256/512/1024/2048/4096/8192를 비교한다.
5. TP2의 출력 및 State를 검증한 뒤 Sweep한다.
6. Native PD를 같은 Prompt와 Arrival Trace로 비교한다.
7. 핵심 조건을 독립 Start 3회 이상 확인한다.
8. 유효한 조건에서 600초 이상의 Window와 Output16/64를 추가한다.
9. 실제 Migration 가능성이 있을 때만 Static/Dynamic을 추가한다.

Primary DP2의 초기 공통 설정은 Sequence Limit16, Async Scheduling과 Full Mixed CUDA Graph다.
각 Budget에서 QPS가 바뀌어도 이 설정을 유지한다.
Sequence Limit과 Graph Mode의 별도 민감도는 별도 실험군으로 기록한다.
논문의 본 평가 최대 Batch1024와 현재 S16은 차이다.
42 GiB에서 실제 Admission 가능한 동시 요청 수도 함께 보고한다.

S32와 S1024는 별도의 Concurrency 민감도 실험군이다.
각 실행군은 N32의 출력·KV·Hidden 검증을 먼저 수행한다.
S1024의 Budget8192 실행군은 논문의 최대 Sequence 설정에 더 가깝다.
추가 실행군의 Graph Capture Size는 현재 vLLM Throughput 기본 규칙을 따른다.
최대 Capture Size는 `min(2 × Sequence Limit, 512)`다.
작은 Shape는 8-token 간격으로 Capture한다.
256 이상은 16-token 간격을 사용한다.
실제 Constructed Configuration도 보존한다.
이 설정도 QPS Sweep 중에는 변경하지 않는다.
S16의 Budget-only 비교와 S1024의 Figure-2-adapted 비교를 합치지 않는다.
Strong Budget512/S16은 독립적인 Decode-priority 기준선으로 보존한다.

## Metric 정의

TTFT는 예정된 Open-loop Arrival부터 Coordinator의 첫 Token 수신까지다.
ITL은 연속 Token 수신 간격이다.
첫 ITL을 포함한다.
Mean TBT는 모든 유효 단일 Token ITL의 평균이다.
Request TPOT는 첫 Token부터 마지막 Token까지의 시간을 Output Count−1로 나눈다.
Worker Timestamp로 계산한 간격과 Delivery Lag 변화는 별도로 보존한다.
Raw Generation Throughput에는 Output Token만 포함한다.
논문 Total Tokens/s와 비교할 때 Input 포함 여부를 별도로 명시한다.
기존 Frozen SLO를 그대로 사용한다.
Request Joint Pass는 TTFT, TPOT와 모든 ITL의 통과를 요구한다.

독립 Server Start를 Bootstrap의 재표본 단위로 사용한다.
N64/N128 Finite-cohort 결과를 Sustainable Capacity로 표현하지 않는다.
과부하의 미완료 요청을 Latency 분모에서 제외하지 않는다.
Queue, Arrival, Completion과 Drain을 함께 보고한다.

## 원인 가설과 판별

가설 H1은 큰 Prefill Chunk의 긴 Mixed Iteration이다.
고정 QPS의 Budget 실험과 실제 Chunk/Batch를 비교한다.
가설 H2는 QPS에 따른 Decode Batch와 KV Context 증가다.
같은 Budget의 Batch별 비용과 Queue를 비교한다.
가설 H3는 Host Delivery 또는 Submission 지연이다.
Worker 간격과 Coordinator 간격의 차이를 분해한다.
가설 H4는 Graph/Eager 또는 TP Communication 비용이다.
실제 실행모드와 Collective를 별도 관측한다.

Host Enclosure와 CUDA Event Enclosure를 실제 Kernel Time으로 표현하지 않는다.
Profiler는 A/A 또는 A/B/A의 성능 간섭을 먼저 검사한다.
간섭이 5%를 넘으면 Clean Kernel Timeline으로 인정하지 않는다.
직접 계측하지 못한 Kernel Union, Spatial Slack과 배타적인 원인 기여도는 `NOT_MEASURED` 또는 `NOT_IDENTIFIED`로 남긴다.

## 보존과 종료

새 Evidence는 `duetserve_agg_dp2_tp2_tbt_crossover_20261116/`다.
모든 Config는 실행 전 저장하고 이후 변경하지 않는다.
Source Archive, Model Revision, Prompt Hash, Seed, GPU UUID와 오류를 보존한다.
실패한 실행은 새 이름으로 재시도한다.
Primary Timing 동안 Shared Quiet-lane Lease를 보유한다.
승인된 Burner만 중단한다.
작업 종료 시 GPU0–7의 Burner를 복원하고 실제 Activity를 확인한다.
Private Index로 새 Branch를 Commit/Push한다.
Remote SHA와 기존 사용자 상태의 보존을 확인한다.
