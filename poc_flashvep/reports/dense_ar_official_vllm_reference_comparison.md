# 공식 vLLM과 Native PD 비교

증거 범위: `MEASURED / CORRECTNESS_ONLY / TRACE_ANALYSIS`.

## 핵심 결과

기존22–25ms Decode Step은 Standard vLLM의 공식 Batch-invariant 경로에서도 재현됐다. Batch-invariant를 끈 같은 실행은 약7–8ms였다. 따라서22–25ms를 H100의 불가피한 Decode 비용이라고 일반화하지 않는다.

Main 비교는 두 방법 모두 공식 Batch-invariant 모드를 사용한다. 이는 엄격한 Greedy Output 일치를 위한 공통 Numerical 계약이다. Stock BI0의 Production Throughput을 재현했다는 뜻은 아니다.

## 비교한 실행 경로

| 경로 | GPU 수 | 실행 계약 | 용도 |
|---|---:|---|---|
| Standard vLLM Single-GPU | 1 | 설치된 `LLM` API와 원본 Model | 독립 Model/Decode Reference |
| Standard NIXL Native PD | 2 | 공식 Connector와 기본 Continuous Scheduler | Main Native Baseline |
| Native + 수동 Descriptor 병합 | 2 | 같은 Model/Scheduler, Byte-preserving 전송 병합 | 진단 Ablation |
| 이전 Custom Native | 2 | Reservation ACK와 Whole-cohort Guard 포함 | `HISTORICAL` Source/Trace 감사 |

마지막 두 경로는 같지 않다. 수동 병합 Ablation을 이전 Runtime 전체의 재현이라고 부르지 않는다. 이전 Custom Native와 새 경로의 완전한 동일 조건 End-to-end A/B는 `NOT_MEASURED`다.

새 Native는 공식 Toy Proxy와 같은 Prompt 전달 계약을 사용한다. P에서 첫 Sample을 생성하지만 Client에 보내지 않는다. D는 원래 Prompt와 Remote KV 정보를 받는다. D의 첫 Forward는 마지막 Prompt Query1개를 다시 계산한다. [공식 NIXL Toy Proxy](https://github.com/vllm-project/vllm/blob/v0.26.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py).

이번 실행은 In-process `LLM` Worker와 로컬 Coordinator를 사용한다. 공식 HTTP Proxy와 완전히 동일한 배포 구성은 아니다. HTTP, Detokenization과 네트워크 Client 비용을 측정하지 않는다.

## 같은 조건의 C1 Control

Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다. Original BF16, Full36, Input8192/Output16, GPU1 Decode를 사용했다. B8192/S1024, KV42GiB, Prefix Cache 비활성화, FLASH_ATTN/HND와 공식 Batch-invariant 모드를 맞췄다.

| 경로 | Mean TTFT | Mean TBT | ITL p99 | TPOT p95 | Output |
|---|---:|---:|---:|---:|---:|
| Standard Single-GPU | 279.59ms | 23.30ms | 24.67ms | 23.74ms | 8/8 |
| Standard NIXL Native | 365.78ms | 22.58ms | 23.70ms | 22.64ms | 8/8 |
| Native + 병합 Ablation | 331.28ms | 22.83ms | 23.58ms | 22.89ms | 8/8 |

각 경로는 독립 Start1회다. 요청8개를 독립 Start8회로 취급하지 않는다. 이 표는 정식5% 비열등성 통과를 증명하지 않는다. Single-GPU와2GPU의 Sustainable Capacity를 비교하지 않는다.

기본 Standard/Native에는 이후2회씩 추가하여 Arm당3회의 Clean C1을 확보했다. Mean TBT 평균은 Standard23.192ms, Native22.697ms였다. Native/Standard ITL p99 비율은0.96785[CI95:0.95846,0.97676]이었다. TPOT p95 비율은0.96795[0.95602,0.98105]이었다. 두 상한 모두1.05 이하였다.

이 CI는 Unpaired Start-level Bootstrap729개 조합이다. 초기 Start의 Read-only Observer Source 차이를 포함한 진단이다. Concurrent Import나 Mixed200의 Decode 보호를 입증하지 않는다. [반복 결과와 Scope](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_clean_C1_confirmation_v1.json)를 참조한다.

## 지연 원인의 범위

Flag-only 진단은 Compile0과 Full Decode CUDA Graph를 사용했다. Input8192의 BI0/BI1 Mean TBT는7.3243/22.2404ms였다. Input16384는7.6779/24.6510ms였다. 이 실행에는 PD나 NIXL이 없다. [실제 측정과 조건](dense_ar_native_decode_latency_root_cause.md).

Standard BI1에서도 선형 연산 등은 별도의 결정적 Numerical 경로를 사용한다. 공식 문서는 이 모드의 성능 비용을 명시한다. [vLLM Batch Invariance](https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/).

Concurrent Native의 ITL Tail은 C1 Model 시간보다 길었다. Standard NIXL의 `transfer()` 호출에서 긴 Host Enclosure를 관측했다. SDK 내부 대기와 실제 DMA의 완전한 분해는 `NOT_IDENTIFIED`다. C1 Model 재현을 전체 Decode Isolation 성공으로 확대하지 않는다.

Main은 기본 NIXL 경로를 유지한다. 수동 병합은 정합성을 통과했지만 반복 가능한 SLO-goodput 우위를 입증하지 못했다. 이 Ablation을 Primary에 조용히 적용하지 않았다.

## 실제 Scheduler 설정

설치된 Runtime의 H100 기본값은 Usage Context에 따라 달랐다. `LLM_CLASS`는 B16384/S1024였다. `OPENAI_API_SERVER`는 B8192/S1024였다. 실제 `EngineArgs.get_batch_defaults(1)` 반환값을 Worker별로 저장했다.

Primary는 자동 기본값에 의존하지 않는다. 모든 Worker에 B8192/S1024를 명시한다. 실제 KV Block은19114개였으며, Null Block을 제외한 초기 Free Block은19113개였다. 실제 GPU Batch는 별도 Query Log로 기록한다.

## 재현성 자료

- [같은 C1 비교 Raw Statistics](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/clean_Native_standard_decode_comparison_v1.json)
- [Main 이전 Gate와 제한](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_reproduction_pre_main_gate.json)
- [N8 전체 State Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/official_invariant_N8_qualification_v1.json)
- [설치된 Wheel Python Source 검사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/installed_vllm_wheel_source_integrity.json)

Wheel의 Python Source2234개는 RECORD Hash와 일치했다. Binary와 모든 Dependency의 무변경을 이 검사만으로 주장하지 않는다. GPU UUID와 실행별 Source Archive SHA256은 각 Raw Job에 보존한다.
