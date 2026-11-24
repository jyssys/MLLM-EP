# Native-relative TTFT·Joint SLO 조사 결정 기록

## D00 — 연구 계약

기준 Commit은 `0e2e062260103feb53bd0afbb0ac964ebb027cd9`다.
시작 시각은 2026-11-24 01:01:40 KST다.
계획 상한은 12시간이다. 마지막 90분은 검증·보고·자원 복원에 남긴다.
새 QPS Sweep, Partition 탐색, Attention Offloading과 Dynamic Insertion은 수행하지 않는다.
Main400, Heldout192, Discovery208과 기존 Numerical·SLO 계약을 유지한다.

## D01 — GPU 재측정 이전의 Raw 재현

먼저 주어진 Main400 Natural15개 실행을 요청 Raw에서 재계산했다.
일곱 주요 Metric과 Start별 값이 원래 표와 일치했다.
재현 명령의 Symptom 모드는 예상대로 종료 코드1을 반환했다.
이는 성능 회귀의 Red Signal이다. 정합성 실패가 아니다.
최소 비교는 Start2의 NativeStrong·Stock Token·Layer와 400개 요청을 유지한다.
동결 Workload의 요청을 삭제해서 최소 재현을 만들지 않았다.

추가로 기존 209개 실행의 25개 Client Metric과 요청별 Gate를 독립 재계산했다.
기존 Raw와 기존 Cached Metric은 수정하지 않았다.
과거 Family의 결과는 새 실측과 평균하지 않는다.

## D02 — 경쟁 가설

다음 예측을 GPU 정책 실험 전에 정했다.

1. First-token 경로 차이가 원인이면 Native에서만 P 첫 Sample 재사용이 관측되어야 한다.
2. D 추가 경로가 P 절약을 상쇄하면 Native 대비 지연은 P Ready 이후에 집중되어야 한다.
3. Admission·Credit가 원인이면 한 정책만 변경해도 Queue와 TTFT가 변해야 한다.
4. Metric 경계와 Token 가중이 원인이면 p95보다 5초 근처 CDF와 실패 Token 수가 Joint 차이를 설명해야 한다.
5. GPU Efficiency가 주요 원인이면 같은 Serving API의 Kernel 비용에서도 차이가 관측되어야 한다.

첫 가설은 기존 Main400에서 반증됐다. 검사한 Native와 Split은 모두 D 첫 출력 경로다.
두 번째 가설은 Host 경로의 위치로 지지된다. GPU-exclusive 기여율은 아직 별도 측정이 필요하다.
네 번째 가설은 실제 Count와 정확한 Metric 항등식으로 확인했다.
세 번째와 다섯 번째 가설을 구분하는 추가 실측을 선택했다.

## D03 — 최소 정책 개입

기존 Strong Stock Token의 Full-model Compiled 경로는 그대로 둔다.
기존 NativeStrong도 그대로 둔다.
다음 한 Factor 비교를 별도로 둔다.

- Stock Cap0→2.
- Cap2 Stock의 Request Registration 지연 ON→OFF.
- Common2 Native의 Request Registration 지연 ON→OFF.
- Layer의 Request Registration 지연 ON→OFF.
- Hybrid의 Request Registration 지연 ON→OFF.
- Layer의 Source Credit 반환을 첫 Client Token에서 안전한 Source Retirement로 변경.
- Hybrid의 같은 Source Retirement 변경.

Registration 지연 해제는 Remote READ 지연과 Decode Compute Priority를 유지한다.
안전한 Source Retirement는 원래 KV ACK, GPU Fence와 Hidden ACK를 유지한다.
필요한 State를 조기에 덮어쓰지 않는다.
각 비교의 실제 Runtime Signature에서 변경 Field가 하나인지 검사했다.

## D04 — Common0 대조군 추가

기존 Stock은 D Cap0이다. Common2만으로는 정확한 Common-D 정책 비교가 되지 않는다.
따라서 NativeCommon0을 사전 추가했다.
Common2에서 D Cap만 2→0으로 변경했다.
초기 Qualification12개가 끝난 뒤 Common0을 같은 순서로 검증했다.
Single1, Prime8, Warm Strict8A/B, Natural32, Natural400을 통과했다.
초기 12개와 Common0을 합친 13개 조합이 모두 정합성 Gate를 통과했다.
Qualification 시간은 Primary 성능 결과로 사용하지 않는다.

## D05 — 미실행 Timing 계획의 사전 수정

처음 작성한 V1 Timing CPU Controller 두 개는 아직 GPU 작업을 제출하지 않았다.
PID·명령·GPU 소유권과 Raw 미존재를 확인했다.
이 대기 Controller만 종료하고 V1 계획을 보존했다.
실행 중인 Qualification GPU 작업은 중단하지 않았다.
V2 Timing은 Common0을 추가하고 순서를 Forward·Reverse·고정 Shuffle로 동결했다.
근거는 `state/unlaunched_timing_controllers_design_checkpoint_v1.json`이다.

## D06 — Kernel 계측

Serial32, 실제 Batch1/1/1, Output16으로 같은 Serving API를 사용한다.
NativeStrong·Stock·Layer·Hybrid마다 사전 지정한 CPC 세 묶음을 계획했다.
각 묶음은 독립 Clean/Profile/Clean 실행이다.
P와 D의 CUDA Kernel Activity만 수집한다.
DMA, Runtime API와 Online 요청별 GPU Operator Attribution은 측정하지 않는다.
다섯 지연 Metric의 계측 간섭이 모두 5% 이내여야 적격이다.
한 Arm의 CPC가 실패하면 그 Arm의 남은 Profile을 중단한다.
실패 묶음을 통과할 때까지 반복하지 않는다.

## D07 — Main400 Natural 확인

13개 Arm의 독립 Start3회를 계획했다.
기존 Main400의 Prompt, 순서, Seed2271/2272/2273과 Poisson Offset을 그대로 사용한다.
여기서 Seed의 전체 값은 `2026112271/2026112272/2026112273`이다.
모든 Timing은 GPU0/1 한 Pair에서 순차 실행한다.
나머지 GPU는 승인된 Burner를 유지한다.
Qualification의 병렬 실행을 성능 증거로 쓰지 않는다.
각 Timing 전에 해당 정책과 Model Seed의 별도 Prime8을 실행한다.
Timed Log에서 Full·Partial AOT/Graph 재사용을 검사한다.

## D08 — Equal-output 확인의 사전 선정

Main 성능을 보기 전에 Heldout192의 10개 Arm을 동결했다.
NativeStrong, Common0, Common2, Stock, StockCap2와 세 Registration 대조를 포함한다.
Layer와 Hybrid의 원래 Arm도 유지한다.
기존 Heldout Trace Seed는 `2026112281/2026112282/2026112283`이다.
Frozen C1 길이는 실행당 총11208개 출력 Token이다.
Natural 출력 작업량 변화와 정책 효과를 구분하는 목적이다.
과거 이미 측정한 Retirement Heldout 반복은 추가하지 않는다.
전체 N3의 보수적인 Wall-time 추정이 예산에 들어갈 때만 후속 전체를 시작한다.
선정과 예산 결정은 성능 효과의 부호나 유의성을 사용하지 않는다.

## D09 — 계측 한계

Host Prepare는 GPU 시작이 아니다.
READ Host Enclosure는 DMA-exclusive 시간이 아니다.
Source Window Full과 P 대기의 겹침은 독립적인 인과 비용으로 합산하지 않는다.
모든 요청의 Output이 같은지 검사한 Replay와 Natural을 구분한다.
Gate 제거와 Frozen-path 제거는 Offline Metric Counterfactual이다.
Queue 갱신 없는 제거 값을 구현 가능한 Speedup으로 보고하지 않는다.

## D10 — CPU Reader 수정

무제한 Credit 실행에는 Credit Event 파일이 없을 수 있다.
이 경우를 `UNBOUNDED_COORDINATOR_WINDOW`로 구분했다.
파일이 없는 유한 Credit은 `NOT_INSTRUMENTED`로 남긴다.
Agent 전체 NIXL Event의 Request ID `None`은 특정 요청에 배정하지 않는다.
Source Capsule의 새 출력 Directory 생성은 첫 GPU 제출 전에 수정했다.
Figure 코드의 Syntax 오류도 CPU에서 수정했다.
이 문제들은 GPU 정합성 실패가 아니다. 기존 Raw를 수정하지 않았다.

## D11 — Kernel Clock Reader의 수치 오차

CUPTI의 약10^18ns Epoch를 Float와 혼합하면 수백 ns의 Partition 오차가 생겼다.
큰 Epoch를 사용한 Unit Test에서 먼저 실패를 확인했다.
Clock Offset, 경계와 Interval을 Integer ns로 계산하도록 Reader만 수정했다.
같은 Raw에서 Partition 합계와 측정 경계가 일치했다.
GPU 실행과 Raw Timestamp는 수정하지 않았다. CPC Gate도 완화하지 않았다.

## D12 — 계측 실패의 보존

NativeStrong, Layer, Hybrid의 prospective CPC는 각각 세 묶음이 통과했다.
Stock의 첫 Profile은 ITL p99가 Clean 평균의1.201861배였다.
해당 Capture를 `DIAGNOSTIC`으로 분리했다.
Stock의 후속 두 묶음은 `NOT_RUN`으로 남겼다.
적격 N3 없이 Stock의 GPU 비용을 Primary로 확정하지 않는다.

## D13 — Scheduler와 Queue Reader의 검증

Hybrid의 Scheduler에는 기존의 두 Chunk 경계 AST 변환이 있다.
초기 Reader는 모든 Arm의 Source가 Byte-for-byte 같다고 잘못 가정했다.
초기 실패 Audit를 보존했다. 알려진 변환의 원본 Hash와 두 삽입 지점을 별도 검사했다.
이 문제는 CPU Reader 가정의 실패다. 예기치 않은 Runtime 변경이 아니다.

Queue의 첫 Snapshot 이전 구간은 `UNKNOWN`으로 남긴다.
관측하지 않은 상태를 빈 Queue나 Free KV0으로 채우지 않는다.
Field별 Coverage와 관측 구간의 조건부 평균을 저장한다.

## D14 — Client ITL과 Worker Emission 구분

기존 Layer의 ITL-only 실패5건은 총3498개 출력 Token을 포함한다.
이 요청들의 Worker Gap은 모두34ms 이하였다.
Client 전달 지연 변화가 경계 초과에 필요했음을 항등식으로 확인했다.
이는 특정 GC나 GPU Kernel의 원인을 확정하지 않는다.
기존 v2와 새 Warm Runtime의 공통 GC·I/O 차이를 One-factor 효과로 해석하지 않는다.

## D15 — 추가적인 Coordinator 경계

기존 `offered_requests`의 Enqueue는 Coordinator Loop가 예정 Arrival을 발견한 시각이다.
HTTP Ingress로 이름을 바꾸지 않는다.
Arrival→P Submit을 Loop 발견 지연과 발견 이후 대기로 세분했다.
Credit Acquire가 있는 요청은 Acquire 이전·이후 경계도 보존한다.
이 내부 구간을 바깥 아홉 Stage에 중복 합산하지 않는다.
Loop 발견 이후의 전체 대기를 전용 Credit 인과 비용으로 확정하지 않는다.

## D16 — Best-family와 Iso-work의 구분

Main400의 실제 Partition·Model Visit·Query Rows를 재검사했다.
Stock은 Last8-equivalent, Layer는 Last2, Hybrid는 Last2+Suffix1/8이다.
각 Action은 기존 Best-family 후보다. 서로 같은 Offloaded Rows가 아니다.
일부 Config의 `target_D_rows`는 이전 Workload에서 상속한 Metadata다.
이 Field를 Work Matching의 근거로 사용하지 않는다.
실제 Raw는 수정하지 않고, 명목·실행 Rows와 상속 Field를 별도 Audit에 보존했다.
Layer의 짧은 D 경계를 전부 GEMM Efficiency로 해석하지 않는다.

## D17 — Equal-output N3의 시간 기반 Fallback

Main N3가 끝나기 전에 후속 일정의 시간 위험을 검토했다.
Heldout GPU 결과는 아직 없었다.
전체30개가 예산에 들어가면 원래 계획을 그대로 실행한다.
그렇지 않으면 사전 지정한 핵심7개 Arm의 전체 N3, 총21개가 들어가는지 검사한다.
핵심 Arm은 NativeStrong, Common0, Common2, Stock, StockCap2, Layer, Hybrid다.
Registration-open9개 Cell은 이 Fallback에서만 `NOT_RUN`으로 남긴다.
그 정책의 Main400 N3는 그대로 유지한다.

계산은 관측한 최대 Main Wall Time과 최대 Prime Wall Time에5% 여유를 준다.
선정에는 TTFT·Goodput 효과의 부호나 유의성을 사용하지 않는다.
12시간 상한과 마지막90분의 Closeout Reserve는 유지한다.
Dataset, Arrival, 출력 길이, SLO와 품질 Gate도 바꾸지 않는다.

대기 중인 Heldout CPU Controller의 PID·UID·명령과 GPU 미제출을 확인했다.
원래 Source Text, Hash와 Controller 기록을 Snapshot으로 보존했다.
이 대기 Controller만 새 Budget 분기로 교체했다.
진행 중인 Main GPU 실험은 중단하지 않았다.
원래10개 Arm의 계획과 모든 Config도 그대로 남겼다.

## 후속 기록

기존 Heldout192의15개 paired Control도 Source·GPU·Arrival·출력 계약으로 재검사했다.
GC1개와 Credit2개의 Mode별 총5개 N3 Contrast다.
Signature 변경이 한 Field라는 것을 확인했다.
기존 Credit Heldout을 다시 GPU 실행하지 않는 근거로 사용한다.
Main400의 새 Credit 확인은 Workload Family가 다르므로 별도로 유지한다.

실측 결과, 중단 사유와 최종 판정은 각 Controller의 Progress·Complete JSON에 보존한다.
모든 계획 Cell의 `COMPLETE / FAILED / DIAGNOSTIC / NOT_RUN` 상태를 최종 Inventory에서 확인한다.
최종 결론은 다섯 보고서와 Science Completion Audit에서 정리한다.

## D18. 새 Natural·Equal-output N3 완료와 반증

Main400의13개 Arm×N3, 총39개 Timing을 완료했다.
Heldout192의10개 Arm×N3, 총30개 Timing도 완료했다.
D17의 전체30개 일정이 시간 검사에 통과했다. 핵심7개 Fallback은 사용하지 않았다.
모든 Timing의 Prime8 및 Frozen Task Quality Gate가 통과했다.
GPU Timing을 더 실행하지 않고 Raw 재계산과 Closeout으로 전환한다.

새 Stock/NativeStrong Main TTFT 비율은0.94331 [0.75864, 1.35027]였다.
기존 회귀를 새 Runtime의 보편적 사실로 유지하지 않는다.
기존 원시 결과는 그대로 재현된다. 두 Family를 합쳐 평균내지 않는다.
Layer·Hybrid의 TTFT 회귀는 새 Main과 Replay에서도 확인됐다.

같은 Stock의 D Cap0→2는 Main TTFT를1.41442배, Replay를1.17105배로 늘렸다.
각 paired CI 하한은1보다 컸다.
ITL·TPOT는 좋아졌지만 Goodput은 두 Family에서 낮아졌다.
Replay의 출력 길이도 같았다.
따라서 이 정책의 총 효과와 단순 출력 작업량 차이를 구분할 수 있다.

Registration 해제는 D Submit→Accept를 줄이고 Accept→Prepare를 늘렸다.
이 Queue 이동을 E2E 개선으로 보고하지 않는다.
안전한 Credit 조기 반환도 TTFT를 일관되게 줄이지 않았다.
Hybrid Main에서는 오히려 TTFT가 증가했다.
더 많은 Credit이나 Guard 제거를 후속 Sweep으로 사용하지 않는다.

## D19. 최종 과학적 범위와 CPU 검증

전체278개 Timing의67104개 요청 관측값과3710695개 ITL을 재계산했다.
독립 CPU Pipeline의18개 단계가 모두 통과했다.
17개 Figure의 PNG·PDF·CSV를 두 번 생성해 Byte 일치를 확인했다.
역사적 Figure에 Historical Raw Recalculation 표기를 추가했다.
이 표기 변경 후 CPU Test와 Figure 재생성 검사를 다시 수행한다.

실패 Mask·Token Weight·Gate Counterfactual은 `SLO_FAILURE_CONFIRMED`다.
검사한 D 정책의 총 효과는 확인했다.
Main의 전용 GPU·DMA 기여율은 미측정이므로 전체 판정은 `PARTIAL_ROOT_CAUSE`다.
Stock Serial의 간섭 실패와 두 `NOT_RUN` Capture도 유지한다.
검증한 Decode-safe Window가 없어 위험한 Dynamic Insertion은 수행하지 않았다.
다음 단계는 승인된 Burner 복원, 무손실 Archive와 별도 Branch Push다.

## D20. GPU Closeout과 Archive 검증

모든 연구 GPU Actor와 Pair Lease가 종료됐음을 검사했다.
승인된 Burner만 교체했다. 다른 사용자 Process는 종료하지 않았다.
`bash run_utilize.sh` 경로에 GPU0–7을 명시해 Burner를 복원했다.
각 GPU의 소유권과 연속 두 번70% 이상 사용률을 확인했다.
이 Utilization은 Resource 확인용이다. 연구의 Kernel Busy가 아니다.

로컬 Full Archive의62661개 Member를 무손실 검증했다.
공개 Archive는12493개 Member와6개 Part다.
원래 Raw·Tensor·Full Log는 삭제하지 않았다.
공개본에서는 원문 Dataset Context와 Reference·Weight를 제외했다.
큰 파생 CSV4개도 결정적 Gzip과 원본 SHA256을 비교했다.
이제 최종 Report Link를 공개 대상 또는 Base Tree와 대조한 뒤 Push한다.
