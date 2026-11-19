# Native Decode 지연 원인 분석

현재 상태: `PARTIAL_ROOT_CAUSE_IDENTIFICATION`. Numerical 경로의 비용 차이는 재현했다. N8 전체 상태, N32와 Main200 Output 검증도 통과했다. Concurrent NIXL 지연의 내부 분해는 아직 남아 있다.

## 같은 Standard vLLM 실행에서 재현한 차이

설치된 vLLM0.26.0+cu129의 `LLM` API를 직접 실행했다. Repository의 Model, RMSNorm, KV 또는 Sampling Adapter는 설치하지 않았다. Standard Wheel의 Python Source2234개는 RECORD Hash와 일치했다. 이 검사는 Binary 전체의 무변경을 증명하지 않는다.

두 Arm은 Batch-invariant Flag만 다르다. BF16, Model Revision, GPU1, Prompt, Output16, Concurrency1, B8192/S1024, KV42GiB와 실행 옵션을 맞췄다. Compile0과 Full Decode CUDA Graph를 사용했다. 각 요청 전과 첫 Sample 및 최종 Output 뒤에 동기화했다.

| Input Tokens | Standard BI0 Mean TBT | Standard BI1 Mean TBT | BI0 첫 Sample→종료 | BI1 첫 Sample→종료 |
|---|---:|---:|---:|---:|
| 4096 | 7.1205ms | 20.9115ms | 106.9009ms | 313.7733ms |
| 8192 | 7.3243ms | 22.2404ms | 109.9622ms | 333.7099ms |
| 16384 | 7.6779ms | 24.6510ms | 115.2705ms | 369.8780ms |

BI0은 Batch-invariant 비활성화다. BI1은 활성화다. 각 Arm은 독립 Server Start1회다. 길이별 측정 요청은 6개다. 그 앞의 2개는 Warmup이다. 서로 다른 길이는 같은 Start를 공유한다. 6개 요청을 독립 Start6회로 취급하지 않는다.

모든 요청에서 후속 Decode Graph Replay15회를 관측했다. 따라서 이 실행의 22–25ms를 CUDA Graph 미사용만으로 설명할 수 없다. 이 Standard 실행에는 NIXL 전송이나 Custom PD Coordinator가 없다.

기존 Custom 측정의 Input8192 후속15Token335.6ms와 Input16384의370.2ms는 Standard BI1의333.71ms와369.88ms에 가깝다. 이는 Batch-invariant 실행 경로가 지연의 주요 원인이라는 근거다. 아래의 별도 Profile은 Kernel별 Duration 합도 기록했다. 서로 다른 실행의 전체 차이를 특정 Kernel의 배타적 비용으로 확정하지 않는다.

vLLM은 Batch-invariant 모드에서 선형 연산 등에 결정적 Reduction 경로를 적용한다. 공식 문서는 이 모드의 성능 비용을 명시한다. [vLLM0.26 Batch Invariance](https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/).

## 출력 정합성과 성능을 구분

BI0/BI1 사이의 Greedy Output 완전 일치는 Input4096에서4/6, Input8192에서6/6, Input16384에서5/6이었다. 성능을 얻으려고 BI1을 단순 해제한 결과를 기존 정합성 계약과 동일하다고 취급하지 않는다.

기본 Compile3 실행의 BI0 Reference도 별도로 측정했다. 이 Arm은 Compile/Graph 옵션도 다르다. 따라서 Flag만 변경한 인과 비교에 포함하지 않는다.

## Native 경로의 추가 검증

새 Native Worker는 Standard NIXL Connector와 Continuous Scheduler를 사용한다. 기존 Custom Reservation ACK와 Whole-cohort Barrier는 이 경로에 없다. 두 경로의 차이는 별도 Runtime 변경으로 기록한다. 이를 Layer/Token Method 개선으로 표현하지 않는다.

초기 N8 시도는 P 첫 Token을 D Prompt 뒤에 추가했다. Input38912의 Warmup에서 D가 요청한 KV Block 수가 P의 전송 Block 수보다 커졌다. Standard NIXL의 Block 수 Assertion으로 실패했다. 이 실행은 정합성 또는 성능 통과로 사용하지 않는다.

후속 검증은 공식 Toy Proxy와 같은 원본 Prompt 전달 경로를 사용한다. P의 첫 Sample은 버리고 D에서 첫 Output을 생성한다. 이는 마지막 Prompt Query의 재실행을 포함할 수 있다. 실제 Query Span을 기록한다. [vLLM0.26 공식 NIXL Proxy](https://github.com/vllm-project/vllm/blob/v0.26.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py).

## 초기 감사에서 지정한 Gate

- Standard Native의 Full Prompt KV, Hidden과 Greedy Output 검증.
- Standard Agg DP2의 Mixed Batch 출력 검증.
- GPU-exclusive Kernel Time, Stream Enclosure와 Host Gap 분해.
- Clean/Profile/Clean Observer Overhead5% Gate.
- Natural LongBench200에서 실제 Continuous 실행과 Queue 검증.

아래 추가 검증은 이 초기 목록의 일부를 통과했다. 전체 End-to-end Reproduction과 Decode-safe Headroom은 별도로 판정한다.

## 2026-11-19 N8 추가 검증

공식 Batch-invariant 모드에서 Standard Single-GPU와 Native PD의 Greedy Output8/8이 일치했다. 전체 Prompt KV와 Final Hidden도8/8이 일치했다. D의 Import 직후 전체 Prompt KV는 P 및 Single-GPU와 일치했다. P/D의 KV Credit도 초기값으로 복원됐다.

이 실행은 공식 NIXL Connector의 원본 Prompt 전달 경로다. 별도의 Token Bound Adapter나 기존 Repository의 RMSNorm Correction은 적용하지 않았다. D는 각 요청의 마지막 Prompt Query를 한 번 다시 실행했다. 실제 최초 D Query Position은 `Input Length−1`이었다.

이는 `CORRECTNESS_ONLY / N8` 결과다. CPU로 상태를 복사한 시간이 포함되므로 이 실행의 TTFT를 정상 성능으로 사용하지 않는다. 후속 Agg DP2 N8도 같은 검증을 통과했다. 두 방법의 N32 Continuous Output은32/32개가 일치했다.

BI0에서는 첫 Sample 재사용과 Source의 실제 Remote Token 수를 제한하는 작은 Adapter도 검증했다. 이 경로는 Output, 전체 KV, Hidden과 Credit 검증8/8을 통과했다. 공식 Proxy 그대로의 실행과 구분한다. 현재 Main 후보는 양쪽 방법에 같은 공식 Batch-invariant 모드를 적용하는 경로다. 빠른 Native BI0와 느린 Agg BI1을 섞어서 성능을 비교하지 않는다.

초기 BI0 Native의 Import Witness는 Block Table을 Gather하기 전에 읽었다. 따라서 해당 Witness의 Hash 불일치는 KV 손상의 증거가 아니다. 원시 결과는 보존했다. Read-only Witness를 Gather 뒤로 이동하여 새 실행에서 검증했다. [Witness 정정 기록](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/N8_BI0_state_witness_erratum.json).

## GPU-exclusive Time과 Host Time

Input8192/Output16의 별도 진단에서 Clean/Profile/Clean을 실행했다. 같은 Compile0과 동기식 Scheduler를 사용했다. 각 Arm은 독립 Start1회다. 측정 요청은 Start당6개이며, 각 요청의 후속 Decode는15 Steps다.

| 같은 BI1 Profile 실행의 측정값 | Step당 평균 |
|---|---:|
| GPU Kernel Union | 23.5624ms |
| CUDA Stream-event Enclosure | 27.6548ms |
| Engine Step Wall | 27.5747ms |
| Inter-step Host Gap | 0.1466ms |
| Client-visible Mean TBT | 27.7180ms |

Kernel Union은 같은 GPU의 모든 Stream에서 중첩 Kernel 구간을 합친 시간이다. Stream Enclosure는 대기와 제출 Gap을 포함한다. 이 다섯 값을 서로 더하거나 서로 다른 실행의 값에서 빼지 않는다.

BI1 Profile의 Mean TBT 변화는 두 Clean Start 평균 대비+4.9747%였다. Decode ROI Wall 변화는+4.9808%였다. 사전에 정한5% 점추정 Gate는 통과했다. 경계에 매우 가깝다. 작은 표본의 이 결과를 엄밀한 간섭 상한으로 주장하지 않는다.

BI0 Profile의 Mean TBT 변화는+18.5181%였다. 이 Arm의 Kernel 결과는 `DIAGNOSTIC`으로만 사용한다. 두 Arm 모두 Dropped/Invalid Activity Record는0개였다.

BI1의 `matmul_kernel_persistent` Duration 합은 Step당19.7444ms였다. FlashAttention Forward Duration 합은2.8309ms였다. 이는 Kernel 이름별 Duration 합이다. 배타적인 비용 비율이나 순수 DMA Time으로 표현하지 않는다. Step당 Matmul145회와 FlashAttention Forward36회를 관측했다. Source의 Full36 실행과 일치하는 패턴이다.

이 Profile은 Main의 기본 Compile3/Async 실행과 다르다. Main200의 Kernel Busy나 Headroom을 이 수치로 대체하지 않는다.

## 200개 Natural Output의 Numerical 경로 비교

같은200개 입력에서 BI0/BI1의 전체 Greedy Sequence는133/200개가 일치했다. Task별 일치는 HotpotQA40/40, MuSiQue37/40, NarrativeQA37/40, GovReport0/40, QMSum19/40이었다.

두 경로의 평균 Output Length는128.675와129.325 Token으로 가까웠다. 평균 길이가 가깝다는 사실은 Token Sequence가 같다는 뜻이 아니다. Main은 같은 BI1 Reference를 두 방법에 적용한다. [전체 Numerical 비교](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/BI0_BI1_Natural200_output_comparison.json).

## Raw Evidence

- [BI0 진단 실행](dense_ar_native_pd_real_workload_reproduction_20261119/raw/standard_matched_BI0_L4096_8192_16384_O16_C1_v2/single_gpu_results.json)
- [BI1 진단 실행](dense_ar_native_pd_real_workload_reproduction_20261119/raw/standard_matched_BI1_L4096_8192_16384_O16_C1_v2/single_gpu_results.json)
- [Wheel Source 검사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/installed_vllm_wheel_source_integrity.json)
- [새 Native의 실패 실행](dense_ar_native_pd_real_workload_reproduction_20261119/raw/qual_Native_LongBench_N8_B8192_S1024_HND_BI0_v1/status.json)

## 이전 수치의 원시 기록 확인

다음 값은 이번 GPU 실행 결과가 아니다. 기존 `local_capacity.json`을 다시 읽고 SHA256을 저장했다. [이전 Native Local Control 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/historical_Native_local_capacity_audit.json).

| 기존 Input | P-only Prefill 평균 | Prepared D 첫 Sample→종료 평균 | 측정 요청/Arm |
|---|---:|---:|---:|
| 8192 | 295.7612ms | 335.5842ms | 12 |
| 16384 | 659.0229ms | 370.2387ms | 12 |

각 Prepared D 요청은 누적 출력1~16의 Event16개를 기록했다. 따라서 첫 Sample 뒤의 추가 Token Emission은15개다. 전체 `engine.step()` 호출은 각각17회와18회였다. 추가 호출을 모두 Decode Kernel Step으로 계산하지 않는다. 이 진단에는 NIXL, ACK와 Coordinator가 없다.

기존 Input8192/Output16/QPS4의 별도 Online Profile은 P/D Kernel Busy82.1181%/90.69%를 보고했다. 같은 Host Trace의 Source-ready→D Host Submit 평균은1250.9623ms였다. 이 값은 Kernel Busy 또는 순수 전송시간이 아니다.

기존 Input16384의 P/D Kernel Busy는96.9523%/58.2035%였다. 두 Profile은 실제 Prefill/Decode Batch1과 Whole-cohort 실행 계약을 사용했다. 새 Mixed200 Continuous Batching과 동일한 계약이 아니다. 이 차이와 가변 Output을 제외하고 Busy 수치만 직접 비교하지 않는다.

## Concurrent SDK 경계의 추가 Source 근거

NIXL1.4.1의 Thread-pool Backend는 Sub-request 제출 뒤 `future.wait()`를 호출한다. 현재 기본 `num_threads=4`는 이 Backend를 선택한다. 이는 호출 Thread의 낮은 CPU Time과 긴 Wall Enclosure가 함께 나올 수 있는 Source 근거다. [NIXL1.4.1 UCX Backend](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_backend.cpp).

이 Source만으로 실제 지연 전체를 해당 Wait에 귀속하지 않는다. Worker의 제출 비용, UCX 경로와 GPU 의존성은 추가 식별이 필요하다. SDK의 `postDuration`도 순수 DMA Time이 아니다. Primary 설정은 이 조사 결과에 맞춰 변경하지 않았다.

## Natural EOS의 비동기 Query 준비

첫 Main Native 실행은200개 Output Sequence가 모두 Reference와 일치했다. D의 준비된 Query Rows는26065개였다. 실제 반환한 Output은25865 Token이었다. 각 요청에서 원래 Prompt의 마지막 Query1개를 다시 준비했다. 그 뒤의 Query Rows는 반환 Output의 첫 Token 이후 수보다 각각1개 많았다.

이 값은 `prepare_inputs()`에서 관측한 유효 Query Rows다. 배타적인 GPU 실행시간으로 바꾸지 않는다. 공식 Async Scheduler는 Output Placeholder를 사용해 다음 작업을 미리 준비한다. 이 Source와 관측은 EOS 확인 전의 한 단계 선행 준비와 일치한다. [vLLM0.26 Async Scheduler](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/async_scheduler.py).

이는 전체 Prompt의 추가 재계산이 아니다. 또한 반환 Token을 덮어쓰는 보정도 아니다. Extra Query의 배타적 시간과 이를 제거했을 때의 Serving 효과는 `NOT_MEASURED`다.

## Main 설정의 Kernel Activity와 전송 호출

Natural Mixed200에서 Clean/Profile/Clean을 QPS1과2.35에 실행했다. 각 QPS는 Clean Start2회와 Profiled Start1회다. Profile은 사전에 정한 첫90초를 수집했다. 공통 구간의 양끝2초를 제외한 약86초를 분석했다.

두 조건 모두 Observer 점추정 Gate를 통과했다. QPS1의 최대 절대 변화는 ROI Mean TBT의+1.2814%였다. QPS2.35는 전체 TPOT p95의−1.5566%였다. 각 Profile의 Dropped/Invalid Kernel Record는0개였다. 이 값은 간섭의 엄밀한95% 상한이 아니다.

| Offered QPS | P Kernel Busy | D Kernel Busy | 같은 ROI의 P Waiting 평균 | D Active Decode 평균 | D Ready Waiting 평균 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 69.1322% | 89.1763% | 1.3734 | 3.5103 | 0 |
| 2.35 | 99.3415% | 85.2237% | 36.7721 | 5.9778 | 0 |

Queue는 같은 Profile의 Scheduler 상태를 시간 가중한 값이다. Ready Waiting은 첫 D Query가 준비된 요청도 포함한다. Active Decode는 Scheduler 상태이며 SM Occupancy가 아니다.

20ms 이상의 D Kernel 공백을 별도로 분석했다. 이20ms는 분석용 분류 기준이다. 안전한 실행 예산으로 사용하지 않았다.

| QPS | 공백 수 | 공백 총시간 | 전송 Host 호출과 겹친 시간 | 시간 비율 |
|---:|---:|---:|---:|---:|
| 1.0 | 66 | 8.7173초 | 8.4489초 | 96.9208% |
| 2.35 | 109 | 12.1207초 | 11.7842초 | 97.2231% |

두 조건의 모든 큰 공백이 전송 Host 호출과 일부 겹쳤다. 이는 같은 실행의 시간적 동시 발생이다. Backend를 바꾼 인과 개입 실험은 아니다. 순수 DMA, CPU-exclusive Time 또는 사용 가능한 Decode-safe Window로 표현하지 않는다.

QPS2.35는 높은 P Kernel Busy와 큰 P Queue를 함께 보였다. 그러나 D도85.22%의 Kernel Busy를 보였다. 나머지14.78%를 그대로 Prefill에 사용할 수 있다고 주장하지 않는다. 상당한 공백은 이미 전송 호출과 겹친다. Native의 절대 Decode SLO도 충분히 만족하지 못했다.

따라서 현재 식별 범위는 Model Numerical 경로와 Concurrent NIXL 호출 경계다. Backend 내부의 대기 및 실제 DMA 분해는 아직 남아 있다. Raw Scope 문구보다 실제 Timestamp와90초 Config를 사용했다. [Capture Scope 설명](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/activity_window_scope_clarification.json)을 보존했다.

- [QPS1 Kernel/Host 분석](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_online_kernel_activity_q1_v1.json)
- [QPS2.35 Kernel/Host 분석](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_online_kernel_activity_q2.35_v1.json)

## UCX Protocol 진단과 종료 오류

별도 N8 실행에 `UCX_PROTO_INFO=y`, `UCX_LOG_LEVEL=info`만 추가했다. Output8/8, KV Credit 복원과 Worker 정상 종료를 확인했다. 로그는 CUDA-to-CUDA `ucp_get*(multi)`에 `zero-copy | cuda_ipc/cuda` 선택을 보고했다. UCX Version은1.22.0이었다.

이는 `DIAGNOSTIC_PROTOCOL_SELECTION / TRACE_ANALYSIS`다. 실제 원격 Read에 대한 Protocol 선택 로그다. Self Endpoint의 `cuda_copy/cuda` 표와 구분했다. GPU별 Local Ordinal은 실제 GPU UUID Mapping과 함께 해석한다. Main 전체의 DMA 경로, 전송 Byte별 Timeline이나 NVLink Bandwidth 측정은 아니다.

따라서 진단 실행에서 Host-staged 경로만 선택됐다는 가설은 지지되지 않는다. 그러나 CUDA IPC를 선택했다는 사실만으로 긴 `postDuration`의 원인이 해결되지는 않는다. Thread-pool 제출, Stream 의존성과 DMA 비용의 완전한 분해는 여전히 `NOT_IDENTIFIED`다.

종료 로그에서는 `cuDevicePrimaryCtxGetState`의 Error4를 발견했다. NVIDIA는4를 `CUDA_ERROR_DEINITIALIZED`로 정의한다. [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html).

Main Native11회 모두36개씩 기록됐다. 첫 오류는 Coordinator `COMPLETE` 이후4.456–6.681초에 발생했다. 모든 Main 요청은 그 전에 측정을 끝냈다. 각 Worker의 Return Code는0이었다. KV Free Block과 Pending Request Gate도 이미 통과했다. Probe의 같은 오류도 완료 후5.254초에 발생했다.

관측 시점과 Error 정의는 종료 순서 문제와 일치한다. 정확한 Destructor 순서의 원인은 아직 분리하지 않았다. 따라서 Model Output 실패나 측정 중 Kernel 실패로 재분류하지 않는다. 반대로 SDK 종료가 완전히 깨끗했다고 주장하지도 않는다. Raw Error를 보존하고 `POST_RUN_SDK_CONTEXT_TEARDOWN_ERROR`로 기록했다.

[Protocol 선택과 Main 종료 오류 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/UCX_protocol_selection_and_teardown_audit.json)에 로그 SHA256, Line Number, 완료 시각과 Return Code를 저장했다. 이 제한도 End-to-end Reproduction `HOLD` 판정에 포함한다.
