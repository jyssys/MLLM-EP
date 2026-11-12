# Dense AR Prefill-heavy PD 및 Layer Continuation PoC Spec

## 1. 범위와 시간

목표는 일반 Dense AR의 정상적인 Native 1P1D부터 확보하는 것이다.
그다음 같은 실행에서 P Compute 압력과 D의 추가 계산 수용 가능성을 확인한다.
마지막으로 정확한 Partial Layer Continuation과 제한된 Serving 순이익을 검증한다.
성능 이득이나 Novelty를 사전에 가정하지 않는다.

시작: 2026-11-12 05:50:17 UTC, 14:50:17 Asia/Seoul.
전체 마감: 2026-11-12 15:50:17 UTC.
GPU 실험 마감: 14:50:17 UTC.
긴 신규 Job 시작 마감: 14:30:17 UTC.
최종 분석, Commit/Push와 자원 복원은 전체 예산에 포함한다.

Branch: `flashvep/dense-ar-layerwise-pd-feasibility-20261112`.
Publication Parent: `99af71b9f8da513dd368c2214ea67b6e15449658`.
Source: `poc_flashvep/dense_ar_layerwise_pd/`.
Evidence: `poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112/`.
기존 Checkout, Staged Index와 외부 Runtime Patch는 유지한다.

## 2. 모델과 실행 환경

Primary는 `Qwen/Qwen3-8B`다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
실제 Config는 Layer 36개, Hidden4096, Q Head32, KV Head8, Head Dim128이다.
기존 예시의 32개 Layer를 실제 모델의 Layer 수로 사용하지 않는다.
전체 Context 한계는 40960 Tokens다.
Weight는 Original BF16이다.

초기 Runtime은 기존에 검증된 vLLM 0.26.0+cu129와 NIXL 1.4.1이다.
Python 환경은 `/home/esjung/.venvs/looped-pd-abc/bin/python`이다.
실제 Torch는 2.11.0+cu129다.
설치 Wheel의 출처와 주요 Runtime 파일 Hash를 기록한다.
기존 Python 경로의 이름을 Looped 모델 사용의 증거로 해석하지 않는다.

첫 Native 후보는 `VLLM_BATCH_INVARIANT=1`, Compile 활성, CUDA Graph NONE이다.
이 설정은 이전에 PD/DP2 및 서로 다른 Prompt Chunk에서 Output 정합성을 통과했다.
이번에도 새 Process Start와 짧은 Output에서 다시 검증한다.
Default Runtime과 해당 실행 Family의 비용은 섞지 않는다.

Secondary Qwen2.5-14B는 Primary 원인 분석에 필요할 때만 검토한다.
모델을 추가한다는 이유로 Primary 정합성이나 반복 예산을 줄이지 않는다.

## 3. 변경할 수 없는 계산 규약

- Original Weight와 BF16을 유지한다.
- Quantization, Early Exit, Layer Skipping을 사용하지 않는다.
- Attention의 수학적 의미와 Position/RoPE를 유지한다.
- 모든 Prompt Token은 모든 Layer를 정확히 한 번 통과한다.
- P와 D의 전체 Decode 모델 Weight를 유지한다.
- Native ACK와 Source Buffer Lifetime을 유지한다.
- 실패한 설정과 로그를 삭제하지 않는다.

Greedy Output은 동일 Prompt, Seed, Sampling 설정에서 일치해야 한다.
출력 차이를 허용 오차로 자동 통과시키지 않는다.
초기 State Gate는 가능한 곳에서 Bitwise Equality를 먼저 사용한다.
진단용 수치 비교는 BF16 변환 후 FP32에서 `atol=0.02`, `rtol=0.01`을 사전에 기록한다.
이 허용 오차는 Output Token 불일치나 잘못된 KV Ownership을 승인하지 않는다.
새 Numerical Fix는 별도 Namespace와 재검증을 필요로 한다.

## 4. 선행연구와 구현 경계

Kairos는 요청 전체 Prefill을 D에 배정하고 Chunked Prefill을 사용한다.
그 방식은 이미 P에서 계산한 Layer의 Continuation과 다르다.
공식 문서의 Repository 링크는 실제 접근 가능 여부를 확인한다.
접근 실패를 공식 코드 확보로 기록하지 않는다.

FluidToken은 P의 Token Prefix와 D의 Token Suffix를 연결한다.
PPI/DPI와 Decode Budget은 참고하되 공식 실행과 재현 구현을 구분한다.

Layered Prefill은 Layer Group 사이의 Hidden/Residual을 유지하며 Prefill과 Decode를 Interleave한다.
논문의 Dense Qwen3-8B 결과는 Co-located 환경의 음성 결과다.
그 결과를 Partial Layer Forward의 불가능성으로 해석하지 않는다.

vLLM RFC #62689는 PP Forward와 IntermediateTensors를 재사용한다.
현재 RFC의 KV Connector 비지원 제한을 확인했다.
이를 완성된 P→D Compute Migration 구현으로 표현하지 않는다.

PP2는 Weight를 영구 분할할 수 있다.
이번 1P1D는 양쪽에 전체 모델이 있고, D가 모든 Layer의 Decode를 처리한다.
Layer-wise KV 전송만 수행한 결과는 Layer Compute Continuation으로 집계하지 않는다.

선행연구의 Hardware, Workload, SLO와 수치는 우리 실측과 분리한다.
추가로 발견한 유사 시스템도 중복 가능성 목록에 남긴다.

## 5. Phase A: Native 정상화

기존 로그의 원래 Default 경로를 재생한 CPU Gate는 실제 Token 불일치를 검출한다.
Default Rate-change 비교는 7/24개 Sequence만 일치한다.
별도 BI Compile/Graph-NONE Family의 과거 N32 비교는 통과했다.
이것은 새 GPU 정합성 결과가 아니다.

새 Native 검증 순서는 다음과 같다.

1. N8에서 독립 Start 재현성, Prompt/Seed/Template 일치를 확인한다.
2. Arrival Trace 또는 Chunk가 달라져도 Output이 같은지 확인한다.
3. N32에서 PD Repeat와 DP2/Colocated 대조를 확인한다.
4. 실제 독립 GPU, Model Parameter 수, BF16, KV Block 해제를 확인한다.
5. NIXL 완료와 정상 Worker 종료를 기록한다.

Native 자체가 불안정하면 Candidate 성능 실험으로 진행하지 않는다.
기존 성공 Family의 재사용은 Runtime 선택이다.
그 효과를 Layer Split 이득으로 집계하지 않는다.

## 6. Phase B: Workload Discovery

초기 Input은 4096, 8192, 12288, 16384 Tokens다.
Output은 16/32를 우선하고 64를 부담 대조로 사용한다.
Output4/8은 진단용으로만 허용한다.
모든 입력은 실제 Context 한계를 확인한다.

기존 공개 원문 Prefix Pool을 우선 사용한다.
부족한 길이는 실제 긴 원문에서 새로 Tokenize한다.
무의미한 Token 반복으로 입력을 늘리지 않는다.
원문 ID, Source SHA, Tokenized Length와 Truncation 규칙을 보존한다.
N128은 128개 서로 다른 Prompt ID를 사용한다.
반복 Pool의 N512는 512개 독립 원문으로 표현하지 않는다.
Prefix Caching은 꺼 둔다.

독립 P Capacity와 Prepared D Capacity를 먼저 측정한다.
서비스 QPS는 실측한 P Capacity 주변에서 선택한다.
초기 비율 0.7/0.9/1.0/1.1/1.3은 탐색 출발점이며 고정 Grid가 아니다.
실패 원인이 KV, Decode, Host, Graph인지에 따라 다음 조건을 바꾼다.

Condition P는 같은 관측 구간에서 다음을 요구한다.

- 실제 P Kernel Union Busy가 80% 이상이다.
- Ownership을 확인한 Compute-ready Queue가 증가한다.
- Compute-ready Request가 반복적으로 존재한다.
- P Service Rate가 같은 실행 설정의 독립 Capacity Reference 대비 85% 이상이다.
- KV/Transfer 대기가 그 Queue의 주된 원인이 아니다.

Condition D는 Decode SLO, 적법한 KV Credit과 실제 추가 계산 수용 근거를 요구한다.
Temporal Idle과 Spatial Compute Slack을 따로 기록한다.
NVML 사용률만으로 둘 중 어느 것도 확정하지 않는다.
SM/Tensor Core/HBM Counter가 없으면 해당 값은 NOT_MEASURED다.

작은 실제 Layer/Prefill 연산을 D에 투입하는 Intervention을 허용한다.
실행 전후의 동일 Arrival Trace에서 Decode Tail을 직접 비교한다.
이는 미래 모든 Deadline의 안전을 보장하는 증거가 아니다.

유망 조건은 독립 N128 Start 3회로 확인한다.
가능하면 별도 Persistence Control을 수행한다.
Finite-cohort Rate와 Sustainable Capacity는 구분한다.

## 7. 계측과 SLO

Marker-free Output Delivery 시간을 Primary 성능으로 사용한다.
TTFT는 Arrival→First Output, E2E는 Arrival→Last Output이다.
TPOT는 요청별 첫 Token 이후 평균이다.
ITL/TBT는 연속 Token Delivery 간격이며 Coalescing 여부를 기록한다.
Request-level 값을 먼저 계산한 뒤 Percentile을 구한다.
구성요소 p95를 더하지 않는다.

Service TTFT는 실제 첫 Prefill 실행 시각이 있어야 계산한다.
없으면 NOT_MEASURED로 남긴다.
Host Submit 또는 Admission을 GPU 실행 시작으로 바꾸어 쓰지 않는다.

기존 Qualified BI Family의 Absolute SLO를 먼저 확인한다.
새 Context/Output에 별도 Calibration이 필요하면 Native Low-load에서 사전에 동결한다.
동결 규칙은 Serial TTFT/TPOT p95와 ITL p99의 두 배다.
동일 모델의 비교 Arm은 같은 Threshold를 사용한다.
선택 결과를 보고 Threshold를 완화하지 않는다.

후보는 Absolute Decode SLO와 Native-relative 두 조건을 모두 평가한다.
ITL p99와 TPOT p95 비율의 설계 목표는 1.05 이하다.
Maximum ITL과 개별 Threshold Exceedance를 따로 보고한다.
최종 주장은 독립 Start 단위 Paired CI를 포함한다.

계측의 Whole-run 및 Local Tail 간섭 목표도 5%다.
Native A/A Variation을 함께 측정한다.
CUPTI Full Trace가 실패하면 짧은 Window와 CUDA Event를 비교한다.
계측 실패에 전체 개발 예산을 쓰지 않는다.
Event Enclosure는 Kernel Union이나 Wire Time으로 표시하지 않는다.

## 8. Phase C: Layer Continuation

실제 Qwen3-8B의 Last 1/2/4/8 Layer Cut은 k=35/34/32/28이다.
단일 GPU에서 Full Forward와 두 Layer Span을 먼저 비교한다.

### 구현 세부 규약

첫 State 검증은 CPU Staging을 사용한다.
그 경로의 실행시간은 최적화된 전송 비용으로 사용하지 않는다.
후속 GPU 경로는 별도 NIXL Agent와 사전 등록한 Hidden/Residual Buffer를 사용한다.
KV는 기존 Native NIXL Connector를 유지한다.
Prefix Layer의 Descriptor만 선택하고, 남은 Layer의 KV는 D에서 생성한다.
Packed Cross-layer Region처럼 검증하지 않은 Layout은 거절한다.

Source Hidden Buffer는 Destination READ 완료 ACK 이전에 재사용하지 않는다.
Destination Buffer는 CUDA Event가 사용 완료를 확인한 뒤 재사용한다.
Native KV ACK와 Hidden ACK는 서로 다른 Buffer Lifetime을 관리한다.
서로 다른 Native Internal Request ID는 Source ID를 통해 명시적으로 연결한다.
Request ID Randomization을 비활성화하지 않는다.

N32는 32개 요청의 동시 제출이다.
32개 요청의 동시 GPU 실행을 뜻하지 않는다.
초기 검증에서는 D 실행을 직렬화한다.
후속 Cohort 검증에서는 최대 4개 요청을 함께 진행한다.

### 제한된 Cohort Admission

초기 Serving 후보는 기존 Decode가 모두 끝난 시점에만 새 Prefill Cohort를 D에 넣는다.
Cohort는 최대 4개 요청으로 제한한다.
새 Cohort는 모든 Native KV Import가 끝난 뒤 Layer Continuation을 시작한다.
이미 출력 중인 Decode를 중단하거나 Kernel 중간 선점을 가정하지 않는다.
이 Guard는 현재 관측 상태만 사용한다.
미래 Arrival이나 Offline Timeline은 Admission에 사용하지 않는다.

해당 방식은 완전한 Adaptive Predictor가 아니다.
Static Tail Layer Cut을 위한 제한된 Admission이다.
동일 Guard를 사용하는 No-migration Control을 별도로 둔다.
기존 Native와 이 Control을 모두 비교해 Batching 효과와 Layer 이전 효과를 구분한다.

### Compile 및 Warm-up

정확한 상태 비교는 모든 유효 Query KV와 최종 Hidden을 포함한다.
Native 경로의 기존 재계산과 Candidate의 Layer 실행을 별도로 집계한다.
수치 차이가 발생하면 Hash 실패를 보존한다.
실제 Tensor 오차를 확인하기 전에는 수치 정합성을 통과시키지 않는다.

실제 Prefill과 Decode의 B1/B2/B3/B4 Geometry를 같은 정책으로 Warm-up한다.
Warm-up은 성능 통계에서 제외하고 시간과 메모리 비용을 기록한다.
성능 비교는 같은 KV Budget과 전송 Buffer 예산을 사용한다.
Mode/Cut을 Closure로 사용하는 Compile Variant는 Cache 식별자를 명시한다.

### 계측 대안

전체 D Kernel Trace가 Decode Tail 간섭 기준을 넘으면 Clean Capacity로 사용하지 않는다.
대안으로 D Engine이 빈 구간을 Fence한 뒤 그 구간만 계측한다.
다음 Model Input 준비 전에 계측을 중단한다.
이 방식이 기록하지 않은 구간은 UNKNOWN이다.
이 결과로 D의 전체 Busy 비율이나 전체 Idle 분포를 추정하지 않는다.
필요하면 관측된 Inactivity의 하한만 보고한다.
작은 실제 Continuation의 Decode 간섭은 별도의 Marker-free 실행으로 확인한다.
그다음 실제 2GPU에서 Prefix KV와 Intermediate State를 전달한다.

vLLM의 PP 규약은 Hidden State와 Residual을 함께 전달한다.
Position, Attention Metadata, Block Table, Sequence Length와 Request ID도 유지한다.
P가 Prefix Layer만 실행하면 Suffix Layer KV는 아직 유효하지 않다.
D가 그 Suffix를 실행한 뒤에만 전체 KV가 완성됐다고 기록한다.

Protocol은 다음 상태를 구분한다.

```text
P_PREFIX_READY → D_RESERVED → CUT_COMMITTED
→ STATE_TRANSFERRED → D_IMPORTED → D_SUFFIX_COMPLETE
→ FULL_DECODE → ACKED → SOURCE_RELEASED
```

Commit 전 Reservation 거절은 Native Fallback이 가능해야 한다.
Commit 뒤 임의 State Return이나 중복 실행을 허용하지 않는다.
Source Buffer는 전송/ACK 규약 전에 재사용하지 않는다.
Timeout, Cancellation, GPU Failure는 소유권과 Block 반환을 확인한다.

N8→N32에서 Output, Final Hidden, Valid KV, Full Layer Work를 확인한다.
단지 Native가 마지막 Layer의 KV를 보내는 결과로 통과시키지 않는다.
가능한 최적화와 별개로 실제 Wire Payload의 유효·불필요 구간을 구분한다.

비용은 P Saving, State 준비, Partial KV, Hidden/Residual, Import, D Suffix, First Output으로 나눈다.
Native도 전송하는 전체 KV를 모두 추가 비용으로 집계하지 않는다.
Serial Median 합산을 Online TTFT 개선으로 표현하지 않는다.

## 9. Phase D: 제한된 Serving

Native, Phase B, N8/N32 State, 비용과 Decode 보호 가능성이 먼저 확인돼야 한다.
조건이 부족하면 소규모 Intervention은 진단으로만 수행한다.

Primary Arm은 Native 1P1D, 강한 2GPU Chunked/Colocated, Static Layer Continuation이다.
가능하면 Kairos-style, FluidToken-style, PP2를 추가한다.
미구현 Arm은 NOT_RUN이다.
Token Baseline에 불리한 작은 Chunk를 강제하지 않는다.

후보 Migration은 적은 Request와 Last 1~2 Layer부터 시작한다.
Current Ready Queue, KV Credit, 관측 Cost와 Safety Margin으로 Admission한다.
미래 Arrival을 Online 정책에 제공하지 않는다.
실제로 완료한 D Prefill Layer와 Migrated Request만 집계한다.
Decode를 지키더라도 Continuation이 장기 체류하면 성공이 아니다.

## 10. 판정과 산출물

Phase B 성공은 반복된 Same-window P Compute 압력과 D Extra-work 수용 가능성이다.
Phase C 성공은 실제 2GPU의 정확한 Layer Continuation이다.
Phase D 성공은 Native 대비 순이익과 Decode 보호를 함께 만족하는 것이다.
각 단계의 부분 성공과 미확정을 분리한다.
테스트하지 않은 조건의 구조적 불가능성을 주장하지 않는다.

모든 결과는 MEASURED, TRACE_ANALYSIS, STRUCTURAL_ACCOUNTING, OFFLINE_ORACLE을 구분한다.
CORRECTNESS_ONLY, NOT_MEASURED, NOT_IDENTIFIED, NOT_RUN, FAILED도 보존한다.
Figure는 실제 GPU Trace와 개념도를 구분한다.

각 Job은 Immutable Config, Source Hash, Model Revision, GPU Map, Seed와 Timeout을 기록한다.
Resume Manifest는 완료한 Job과 실패한 Job을 유지한다.
최종 보고서는 12개 연구 질문에 직접 답한다.
파일명은 `dense_ar_prefill_heavy_layerwise_pd_feasibility_final_summary.md`다.

최종 단계에서 관련 변경만 Private Index로 Commit/Push한다.
Remote SHA와 기존 Index/Runtime 보존을 확인한다.
GPU 0~7의 승인된 Burner를 복구하고 실제 생성 및 사용률을 확인한다.

## 11. 강한 Native 전송 Control

기존 Wrapper는 P가 생성한 첫 Token을 버리고 D에서 다시 생성한다.
이 비용을 Layer 방식의 고유 이득으로 집계하지 않는다.
추가 Native Control은 실제 P Token을 보존한다.
D는 원래 Prompt KV를 Import하고 그 P Token을 다음 입력으로 소비한다.
P Token은 Native Import와 다음 D Model의 Host 제출 이후에만 사용자에게 전달한다.
이 시점에 GPU 계산 완료를 가정하지 않는다.
이 보류 시간은 TTFT에 포함한다.
D는 나머지 15개 Token만 생성하고 전체 출력은 16개를 유지한다.
첫 ITL, 전체 Output Sequence와 ACK/KV 해제를 새 N8/N32 Gate에서 확인한다.
기존 Resampling Native도 별도 대조군으로 보존한다.
SLO와 수치 허용 오차는 변경하지 않는다.

## 12. 실행 중 발견한 Reference 문제와 재검증

이 절은 기존 실패 기록을 변경하지 않고 추가한다.
기존 Compiled Optional-input 경로는 실제 Suffix 대신 전체 Model을 실행한 사례가 있었다.
출력이 같더라도 이 경로는 Layer 이전 성공으로 인정하지 않는다.
독립 Suffix Module은 전달된 Hidden과 Residual을 필수 입력으로 받는다.
실제 Attention Backend 호출 Layer를 검사해 Prefix 중복 실행을 차단한다.

Eager BF16 경로에서는 네 Cut 모두 Prompt KV와 Hidden이 원래 Full Forward와 일치했다.
그러나 기존 PD Reference는 N32 중 한 요청에서 다른 Token을 생성했다.
독립적인 2GPU Full Forward는 네 Cut의 출력을 모두 재현했다.
기존 PD Reference의 마지막 Prompt Query 재실행에서 유효한 KV 변경도 확인했다.
이 결과는 Reference의 재실행 경로 문제를 가리킨다.
정확한 산술 Kernel 원인은 아직 NOT_IDENTIFIED다.

수정한 Native는 P의 실제 첫 Token과 원래 Prompt의 유효한 KV만 사용한다.
새 Token용 Destination Block은 예약하되 Source에서 읽지 않는다.
Native READ 검사, ACK와 Buffer Lifetime은 유지한다.
수정한 Native N32는 독립 Full Forward의 32개 출력을 모두 재현했다.
State Reference와 Output Reference는 역할을 분리해 기록한다.
이전 실패 Receipt는 덮어쓰지 않는다.

Compiled Native와 Eager BF16 Native는 별도 Execution Family다.
Family 사이에서 출력 일치를 가정하지 않는다.
Layer 방식의 성능은 같은 Eager Family의 검증된 Native와 비교한다.
Compiled Native의 물리적 측정값을 Eager Serving의 Same-window 증거로 사용하지 않는다.

## 13. Resume 단계의 제한된 확인 실험

N64에서 네 Cut 모두 출력 검사를 통과했다.
그러나 각 Cut의 Decode 비열등성 Gate는 별도로 평가한다.
TTFT 개선만으로 성공을 선언하지 않는다.
같은 2GPU의 Full-replica Colocated 기준선도 유지한다.

같은 Eager Family에서 P-only Capacity를 다시 측정한다.
Native A/Quiet/B를 독립 Arrival Seed 세 개에서 수행한다.
Quiet-D 방식은 관측된 Empty-engine Epoch만 측정한다.
전체 D Idle 비율이나 Spatial Slack은 그 결과로 추정하지 않는다.

Batch 제한을 4에서 2로 변경하는 진단은 Native와 Colocated에도 동일하게 제공한다.
이 실험은 Selection이다.
Held-out Arrival Seed 11/12/13의 N128은 균형 잡힌 순서로 비교한다.
128개 Prompt는 서로 다르다.
그중 64개는 Selection에도 사용했으므로 새 Prompt 일반화를 주장하지 않는다.

Selection에서 Decode Gate를 통과한 후보가 없을 수도 있다.
그 경우 N128은 Absolute Decode SLO를 통과한 후보의 제한된 진단 확인으로만 수행한다.
이 실행을 Qualified Phase D 성공으로 표현하지 않는다.
실제 Kernel Trace, CUDA Event Enclosure와 Host Timestamp는 계속 구분한다.

## 14. Source Export Credit과 Decode Graph 확인

Batch 제한 2의 첫 실행은 Source Hidden Export Credit 고갈로 실패했다.
이 실패 Namespace와 정상 해제에 실패한 기록을 보존한다.
Buffer 용량 증가로 문제를 숨기지 않는다.
Source는 새 Atomic Prefix 계산 전에 빈 Export Slot을 확인한다.
Credit이 없으면 아직 시작하지 않은 요청을 Queue에 유지한다.
이미 전달한 State를 버리거나 다시 계산하지 않는다.
4개 Source Slot과 D Batch 제한 2의 N8/N32에서 이 Backpressure를 검증한다.
정확한 출력, Prompt KV, Hidden, 실제 Layer 호출과 정상 ACK 해제가 모두 필요하다.

Eager 실행에서 큰 Decode 제출 비용이 남으면 Native Decode CUDA Graph를 별도로 검토한다.
이 후보는 Compiler Mode 0과 원래 BF16 CUDA Operator를 사용한다.
Inductor Fusion으로 중간 State 의미를 바꾸지 않는다.
Source Prefill은 Eager로 유지하고 D의 전체 36-Layer Decode만 Graph로 실행한다.
각 Arm은 같은 Graph Shape 정책과 KV Budget을 사용한다.
Native Graph의 실제 36-Layer Capture와 Replay를 검사한다.
Native와 Layer 방법 모두 독립 Full Forward 출력과 다시 비교한다.
새 Execution Family의 N8/N32를 통과하기 전에는 성능 이득을 주장하지 않는다.

첫 P Token의 전달 정책은 유지한다.
Native Import와 다음 전체 Decode Model 또는 Graph의 Host 제출 뒤에 전달한다.
Model 제출과 Graph Replay 제출은 각각 실제 경로에서 확인한다.
GPU 완료를 미리 가정하지 않는다.
Generic Graph 최적화의 이득을 Layer Split의 이득으로 집계하지 않는다.

## 15. Text-enabled 최종 대조와 보수적인 후보 선택

Graph Native의 첫 ITL에 Prompt Text Decoder 초기화 비용이 포함된 사례를 확인했다.
Text를 끈 진단은 원인 분리에만 사용한다.
Primary Serving은 Text 출력을 유지한다.
원래 Tokenizer Decoder를 Request 준비 단계에서 초기화하는 후보를 검증한다.
CPU Incremental String 비교와 GPU Runtime String 비교를 모두 요구한다.
초기화 비용을 제거한 것처럼 보고하지 않는다.
초기화 시점 이동에 따른 TTFT 비용도 측정에 포함한다.

같은 사전 준비와 비동기 RPC 송신을 Native, Layer와 Colocated에 제공한다.
Model Weight, GPU 산술, Output Cap과 SLO는 유지한다.
Native A/Layer/Colocated/Native B의 N64 순서로 같은 Arrival Seed를 비교한다.
Last 1/4/8 Layer를 후보로 사용한다.
새 Cut은 먼저 N8/N32의 State와 Work Gate를 통과해야 한다.
후보는 두 Native Control 각각에 대해 ITL과 TPOT 5% 목표를 평가한다.

통과 후보가 있으면 그중 TTFT가 가장 낮은 후보를 선택한다.
통과 후보가 없으면 Absolute Decode SLO를 통과한 후보만 제한적으로 재확인한다.
이 경우 Decode 보호를 확인하기 전에는 성공한 Online Method로 표현하지 않는다.
최종 Arrival Seed는 2026111231/32/33이다.
각 Seed의 Native/Layer/Colocated 순서를 균형 있게 배치한다.
Selection과 Confirmation의 같은 원문 사용 범위를 명시한다.

실제 Cross-GPU Kernel 중첩은 별도 Full Capture로 진단할 수 있다.
그 Trace는 실제 계측 실행의 증거다.
Clean Serving Capacity나 안전한 미래 Budget의 증거로 자동 승격하지 않는다.
GPU 실험 종료 후 원본 Evidence를 Lossless Archive로 보존한다.
최종 보고, Remote SHA 확인과 GPU0~7 Burner 복원은 전체 시간 예산 안에 완료한다.
