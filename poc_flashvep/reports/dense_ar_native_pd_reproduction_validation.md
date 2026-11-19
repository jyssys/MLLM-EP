# Native PD 재현성 검증

현재 상태: `N8_STATE_PASS / N32_OUTPUT_PASS / MAIN200_OUTPUT_PASS / ONLINE_KERNEL_OBSERVER_PASS`. End-to-end 원인 분해와 Decode-safe Headroom은 `HOLD`다.

모델의 Decode 실행 경로는 공식 vLLM과 비교했다. Concurrent KV Import에서 발생하는 Client-visible Tail은 별도의 Runtime 제한으로 남아 있다. 이 두 결과를 구분한다.

## 공통 조건

- Qwen/Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Original BF16, Exact Causal Attention, Full36 Layers, Greedy, Thinking Disabled.
- vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- GPU0/P: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1/D: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- B8192/S1024, KV42GiB/Engine, Prefix Cache 비활성화, FLASH_ATTN, HND.
- 두 방법에 동일한 공식 Batch-invariant 모드를 적용한다.

## 검증 결과

| Gate | Standard Single-GPU | Native PD | Agg DP2 |
|---|---:|---:|---:|
| N8 Natural Output | 8/8 | 8/8 | 8/8 |
| N8 전체 Prompt KV | 8/8 | P8/8, D Import8/8 | 8/8 |
| N8 Final Prompt Hidden | 8/8 | P8/8 | 8/8 |
| N8 정상 KV Credit 복원 | PASS | PASS | PASS |
| N32 Continuous Natural Output | 전체200 Reference 사용 | 32/32 | 32/32 |
| Main200 Online Output | Reference 생성 완료 | 11 Starts×200/200 | 11 Starts×200/200 |

N8의 CPU 상태 복사는 TTFT를 크게 교란한다. 해당 실행의 지연을 정상 Serving 성능으로 사용하지 않는다. N32에서는 상태 복사를 사용하지 않았다. N32 두 방법 모두 같은 전체200 Natural Reference와 Output Token ID가 일치했다.

Native P는 전체 Prompt Query를 누락이나 중복 없이 처리했다. D의 원래 Prompt 재계산은 요청당 마지막 Query1개였다. 추가 Full Prompt 재계산은 관측하지 않았다. 공식 Proxy와 같은 경로에서 P 첫 Sample은 버리고 D 첫 Sample을 반환했다. 이 작은 재계산을 Layer/Token Offloading으로 부르지 않는다.

Agg는 독립 Full-model Replica2개에 Round-robin으로 요청을 보냈다. Standard Continuous Batching과 Chunked Prefill을 사용했다. Cohort-only Scheduler나 고정 Batch1 제한을 추가하지 않았다.

## 같은 C1 조건의 Decode Latency

다음 결과는 Input8192/Output16의 진단 Control이다. 실제 LongBench Main Workload와 구분한다. 같은 실제 Prompt8개, GPU1 Decode, Numerical 경로와 실행 옵션을 사용했다. 상태 복사는 사용하지 않았다.

| 실행 | Greedy Output | Mean TTFT | Mean TBT | ITL p99 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| Standard Single-GPU | 8/8 | 279.59ms | 23.30ms | 24.67ms | 23.74ms |
| Standard NIXL Native PD | 8/8 | 365.78ms | 22.58ms | 23.70ms | 22.64ms |
| Coalesced Native 진단 | 8/8 | 331.28ms | 22.83ms | 23.58ms | 22.89ms |

각 Arm은 독립 Start1회다. 요청8개를 독립 Start8회로 취급하지 않는다. 이 표는 정식5% 비열등성 검사가 아니다. Single-GPU의 Throughput을 2GPU Capacity와 비교하지 않는다.

이후 Standard와 기본 Native에 각각2회의 독립 Start를 추가했다. 위 초기 Start를 포함해 Arm당3회다. 모두 Output8/8과 KV Lifetime을 통과했다.

| Arm당3회 Clean C1 평균 | Standard Single-GPU | Native PD |
|---|---:|---:|
| Mean TTFT | 280.299ms | 362.717ms |
| Mean TBT | 23.192ms | 22.697ms |
| ITL p99 | 24.435ms | 23.649ms |
| TPOT p95 | 23.506ms | 22.753ms |

Native/Standard ITL p99 비율은0.96785[CI95:0.95846,0.97676]이었다. TPOT p95 비율은0.96795[0.95602,0.98105]이었다. 두 CI 상한은1.05 이하다. Unpaired Start-level Bootstrap729개 조합을 사용했다.

이 통과는 Clean C1 진단에 한정한다. 초기 Start에는 이전 Read-only Observer Source가 포함됐다. 공통 Numerical 경로와 Engine 설정은 유지했다. 이 결과를 Concurrent NIXL Import, Mixed200 또는 전체 Production의 비열등성으로 확장하지 않는다. [3회 반복의 Raw Statistics](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_clean_C1_confirmation_v1.json)를 보존했다.

Native의 후속 Decode는 같은 Numerical 경로의 Standard vLLM과 비슷한 범위였다. 기존22–25ms/Step이 Custom PD만의 비용이라는 가설은 이 Control에서 지지되지 않았다. Batch-invariant를 끈 Standard 경로는 더 빨랐지만, 일부 출력이 달랐다. 그 결과를 동일한 정확도 계약으로 섞지 않는다.

## Concurrent NIXL Import의 제한

N32 Native에서는 ITL p99가169.94ms였다. 정상 Model Decode의23ms와 다르다. N40/QPS2의 전송 API 계측에서는 `transfer()` Host Enclosure 평균122.89ms를 관측했다. 이는 순수 DMA Time이나 CPU-exclusive Time이 아니다.

Coalescing N40/QPS2에서는 같은 호출의 Wall 평균83.66ms, 호출 Thread CPU 평균0.71ms였다. 따라서 긴 Wall Time을 호출 Thread의 CPU 계산시간으로 해석하지 않는다. 내부 대기와 Backend 동작의 완전한 분해는 아직 하지 않았다.

Coalescing은 N8 전체 상태와 N40 출력 검증을 통과했다. 그러나 QPS0.5의 Online ITL p99 개선은 작았다. Primary에는 검증되지 않은 개선을 조용히 적용하지 않는다. 기본 Standard NIXL 경로를 유지하고 Coalescing은 별도 Ablation으로 보존한다.

추가 SDK Telemetry도 확인했다. 기본 경로의 최대 길이 Warmup은 Python 입력 Descriptor87552개를 받았지만, SDK는 실제 Descriptor36개를 보고했다. `postDuration=99071µs`, `xferDuration=111172µs`, `totalBytes=5737807872`였다. 이 기록은 SDK의 자동 병합을 보여준다. Python Descriptor 수만으로 지연 원인을 설명할 수 없다.

현재 식별한 것은 오래 걸리는 API 경계와 작은 호출 Thread CPU Time이다. SDK 내부 대기, 실제 전송과 GPU Stream 의존성의 완전한 인과 분해는 `NOT_IDENTIFIED`다. Hardware의 NV18 연결을 실제 전송 경로 또는 실효 Bandwidth 측정으로 해석하지 않는다.

후속 N8 Protocol 로그는 CUDA-to-CUDA Read에 CUDA IPC zero-copy 선택을 보고했다. 이는 Main의 DMA-only Time 측정이 아니다. Main Native11회에는 완료 후4.456–6.681초에 SDK Context 종료 오류가 있었다. Output, KV Credit과 Worker Return Code Gate는 통과했다. SDK Destructor 순서 문제는 미해결이며, 정상 Request Retirement와 구분한다. [종료 오류 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/UCX_protocol_selection_and_teardown_audit.json)를 보존했다.

Decode SLO와 실제 Safe Headroom은 이 재현성 검증과 별도로 판정한다. Native Model 경로가 재현됐다는 사실만으로 Motivation `PASS`를 선언하지 않는다.

## 증거와 판정 범위

- [N8 전체 State Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/official_invariant_N8_qualification_v1.json)
- [Native N32 Output Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/qual_Native_LongBench_N32_B8192_S1024_BI1_q2_v1_output_gate.json)
- [Agg N32 Output Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/qual_Agg_LongBench_N32_B8192_S1024_BI1_q2_v1_output_gate.json)
- [같은 C1 조건의 비교](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/clean_Native_standard_decode_comparison_v1.json)
- [Kernel/Host 분해](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/decode_kernel_host_breakdown_v2.json)
- [Native Transport A/B](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_transport_AB_v1.json)

각 Raw Job은 실제 Command, Source Archive SHA256, Worker Configuration, GPU UUID와 Input Hash를 보존한다. Main200의22회 QPS 실행은 완료했다. 별도 QPS1/2.35 Profile의 P/D Kernel Busy와 Observer Gate도 확인했다. Profile당 약86초의 같은 구간에서 Queue를 함께 분석했다. Layer/Token 실험은 재개하지 않는다.

QPS2.35의 P/D Kernel Busy는99.3415%/85.2237%였다. D Ready Waiting은0이었지만 Active Decode 평균은5.9778개였다. D의20ms 이상 Kernel Gap 시간 중97.2231%가 전송 Host 구간과 겹쳤다. 이 관측은 안전한 Offloading Budget을 뜻하지 않는다. [Prefill-heavy 최종 판정](dense_ar_longbench200_prefill_heavy_feasibility.md)을 참조한다.
