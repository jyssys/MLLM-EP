# BI0 NIXL Host Tail 진단

상태: `TRACE_ANALYSIS`와별도 설정 개입이다. 순수 DMA 비용이나 GPU Exclusive Time을 측정한 표가 아니다.

## 같은 N32 Trace의 KV Geometry 개입

ShortAnswer200-v2의 고정 N32/QPS2.35/Seed2026112033을 사용했다. 모든 Engine은 BI0, B8192/S1024, KV42GiB와 기본 NIXL4 Thread를 사용했다. KV Block만16/128/256으로 변경했다. 각 설정의 독립 Start는1회다. 아래 SDK 통계는 Measurement Window의32개 완료 호출만 포함한다. Warmup2개는 제외했다.

| KV Block | SDK Posting p50 | SDK Posting p95 | SDK Descriptor p50 | SDK Byte p50 | ITL p99 |
|---:|---:|---:|---:|---:|---:|
| 16 | 106.6295ms | 439.5219ms | 31212 | 2421817344Bytes | 27.250ms |
| 128 | 17.4260ms | 75.8625ms | 3942 | 2425356288Bytes | 9.230ms |
| 256 | 10.9460ms | 29.1628ms | 2052 | 2434793472Bytes | 11.181ms |

Block16의 Host Submit 중앙값은106.6503ms였다. 같은 호출 Thread의 CPU 중앙값은0.0490ms였다. 이 차이는 Caller가 그 시간 동안 CPU 계산을 독점했다는 설명과 맞지 않는다. 다른 UCX Worker의 CPU 시간이나 DMA 시간을 이 값으로 계산하지 않는다.

Block16의 Delivered Event Interval은414개였다.34ms 초과4개는 모두 Host Submit과 시간상 겹쳤다. 그4개 Interval의 합은469.5838ms였다. API 겹침의 합은431.8244ms였다. 이는 Request별 Interval의 합이다. GPU Busy Union이 아니다. Block128은431개 Interval 중34ms 초과0개였다. Block256은460개 중0개였다. BI0 Natural Output이 달라 Interval 분모도 달랐다.

유효 KV와 Hidden 및 D Import는 각 Block의 N8에서 통과했다. 그러나 Block 크기는 Memory Rounding과 Attention Geometry도 바꾼다. 따라서 Tail의 모든 차이를 Descriptor 수 하나에만 귀속하지 않는다. 이 진단은 Posting 경로의 기여를 지지한다. 전체 Causal 분해를 완료한 것은 아니다.

## Source 근거

NIXL1.4.1의 UCX Thread Pool 경로는 Chunk 작업을 Worker에 제출한다. 이후 Caller는 `future.wait()`에서 Posting 작업의 완료를 기다린다. 이는 전체 DMA 완료를 기다리는 것과 다르다. [공식 NIXL Source](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_backend.cpp)

현재 Wrapper는 공식 Connector의 `_read_blocks`, Transfer 생성, Submit과 SDK Telemetry를 관찰한다. Native의 Model, Sampling, Continuous Scheduler 우선순위는 변경하지 않는다. Default Agent의 `NIXL_THREAD_SYNC_NONE` 계약 때문에 Transfer 호출을 임의 Background Thread로 옮기지 않았다.

## Thread와 초기화의 반례

같은 N32에서 Thread4/1/0의 ITL p99는27.250/17.317/125.882ms였다. 따라서 Thread를 없애면 항상 개선된다는 가설은 기각했다. 후속 Full200의3개 Start는 Thread1의 개선을 재현하지 못했다. Common Main은4 Thread를 유지했다.

초기 Wrapper는 공식 UCX 기본값 설정 전에 SDK를 Import했다. 이 계약의 결과는`v1`로 보존한다. 초기화 복구 후 N8 전체 상태 검사를 통과했다. N32 초기화 A/B에서는 Tail 개선을 입증하지 못했다. 초기화 복구와 성능 개선을 같은 주장으로 취급하지 않는다.

## Scope와 다음 검사

이 표는 Old-init 진단 Family다. 최종 공식 초기화 Main의 성능 표와 섞지 않는다. 새 Shared Axis Control은 Online Axis 결과 확인 전에 Block128을 동결했다. Native/Layer/Token 모두 같은 Block을 사용한다. 기본 Block16과 Block256의 결과도 삭제하지 않는다.

공통 Main은 공식 초기화, Block16, 기본4 Thread로 재측정했다. Block128도 별도 공통 Runtime Family로 측정했다. Primary 비교의 전체200개 Task 품질과3개 Start CI를 확인했다. NIXL API와 GPU Gap의 시간상 겹침만으로 Decode-safe Window를 선언하지 않는다.

## 근거

- [Warmup을 제외한32개 호출](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_filtered_NIXL_telemetry_v1.json)
- [KV Geometry 개입](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_KV_geometry_screen_v1.json)
- [Thread 개입](dense_ar_bi0_shortanswer200_20261120/analysis/NIXL_thread_screen_v1.json)
- [Primary Source Hash](dense_ar_bi0_shortanswer200_20261120/references/primary_source_manifest.json)

## Full200 Thread Confirmation

공식 UCX 초기화와 KV Block16에서 Thread1/4를 다시 비교했다. QPS2.35의 동일한 Full200 Trace를 사용했다. 각 설정은 독립 Start3회다. Seed는2026112051/52/53이다. 모든 Start는 Full200 Task-quality CI 검사를 통과했다.

| NIXL Threads | Mean TBT | ITL p99 | TPOT p95 | TTFT p95 | Raw Output Tokens/s |
|---:|---:|---:|---:|---:|---:|
|4|8.5318ms|51.0895ms|10.3926ms/token|22370.22ms|44.8558|
|1|8.7012ms|70.5438ms|11.7996ms/token|22414.76ms|44.5939|

Thread1/4의 ITL p99 비율은1.3808이다. Paired-start95% CI는[1.1239,1.7155]다. TTFT p95 비율은1.0020이다. CI는[0.9967,1.0068]다. Mean TBT 비율은1.0199다. CI는[0.9902,1.0457]다. 이 실험은1 Thread의 전체 Serving 개선을 지지하지 않는다. 기본4 Thread를 유지한다.

앞선 N32와 이번 Full200은 Dataset 수와 초기화 Family도 다르다. 두 실험 사이의 차이를 Request 수 하나의 인과 효과로 해석하지 않는다. 이번 Full200 내부의 Thread1/4 비교는 같은 초기화와 같은 Trace를 사용했다. 원시 결과는 [Full200 Confirmation](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_NIXL_thread_full200_confirmation_v1.json)에 보존했다.

## Byte-exact Coalescing의 정합성 한계

Block16을 유지하고 인접한 Local/Remote Descriptor 쌍만 합치는 진단을 추가했다. Source와 Destination의 주소 대응을 바꾸지 않는다. 등록된 Layer Region의 경계도 넘지 않는다. CPU Byte-map Regression Test는 통과했다.

Coalesced N8 실행은 정상 종료했다. 각 요청의 Producer 전체 유효 KV와 실제 D Import는8/8 정확히 일치했다. Lifetime도 복원됐다. 기존 Native와 Greedy Sequence는8/8 일치했다. 그러나 다른 실행의 Producer KV/Hidden Hash는 모두 달랐다. 이 차이는 D 전송 전에 이미 존재했다.

기존 Reference의 계측 차이를 의심했다. 같은 Synchronized Probe를 사용한 별도 Non-coalesced N8을 실행했다. 두 실행의 Within-run Import는 각각8/8 정확했다. Greedy도8/8 일치했다. 그러나 Cross-run Producer KV/Hidden의 Bitwise Equality는 다시 실패했다. Source의 수치 경로 또는 계측 차이의 정확한 원인은 `NOT_IDENTIFIED`다.

q0의 이전 Synchronized Probe 비교에서 Sampled KV Relative L2 차이는0.007311이었다. Last Hidden은0.011795였다. 모든 비교 값은 유한했다. 이는 전체 Model 정확성 또는 단순 BF16 반올림 원인을 인증하지 않는다. 전송 손상으로 확정하는 근거도 아니다.

사전에 정한 Strict Tensor Gate를 완화하지 않았다. 따라서 이 Arm의 N32 Off/On 성능은 `NOT_RUN`이다. Coalescing의 Posting 또는 ITL Speedup을 보고하지 않는다. 현재 Native/Layer/Token Primary는 이 옵션을 사용하지 않는다. 관련 실패 판정과 두 N8 Raw Evidence는 보존했다.

근거는 [첫 Strict 비교](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_exact_coalescing_N8_state_v1.json)와 [같은 계측 조건의 후속 비교](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_exact_coalescing_N8_matched_instrumentation_v2.json)다. 성공한 Descriptor 축소와 실행 정확도 판정을 혼동하지 않는다.
## Compiled Cache 추가 감사

`TRACE_ANALYSIS`. 두 N8 진단의 C3 Cache Directory는 달랐다. Cache Key의 차이는 `VLLM_NIXL_SIDE_CHANNEL_PORT`였다. 생성된 `computation_graph.py`의 SHA256은 같았다. 다른 Serialized Artifact Hash만으로 서로 다른 Numerical Kernel을 실행했다고 확정할 수 없다.

따라서 Cache Directory 차이를 Producer Tensor 불일치의 원인으로 채택하지 않는다. 원인은 `NOT_IDENTIFIED`로 남긴다. Cache를 삭제하거나 외부 Runtime을 수정하지 않았다. 상세 Hash는 `analysis/BI0_coalescing_compile_cache_key_audit_v1.json`에 보존한다.
