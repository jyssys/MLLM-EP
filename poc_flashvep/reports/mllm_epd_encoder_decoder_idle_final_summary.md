# MLLM EPD: Encoder/Decoder Idle 추가 연구 종합

후속 연구 판정은 `HOLD`다. 저부하의 물리적 Idle은 확인했다. 안전한 추가 Prefill 실행은 확인하지 못했다.
이 보고서는 완료된 C25·C40·C41 추가 연구의 결과다. 전체 캠페인의 종료나 Serving GO를 선언하지 않는다.

## 첫 페이지: 연구 질문 8개

| 질문 | 답과 증거 수준 |
|---|---|
| 1. E에도 D와 비슷한 Idle이 있는가 | 저부하 C25에서 E87.2674%, D85.6284%의 Kernel Idle 관측. TRACE_ANALYSIS |
| 2. P가 포화되고 E/D가 동시에 Idle인가 | 동시 Idle은 관측. 그러나 적격 Trace의 P Busy는14.0682%다. P 포화와의 동시 조건은 미확인 |
| 3. 어느 쪽 Window가 긴가 | 같은 저부하 Trace의 E Gap Mean·p50·p99가 D보다 길었다. 실제 Safe Budget의 비교는 아님 |
| 4. 안전한 Quantum은 얼마나 다른가 | NOT_MEASURED. E/D128·512 Prefix Probe의 N3 전체 보호 Gate는 모두 미통과 |
| 5. E의 추가 LLM 메모리는 가능한가 | 추가 Full Model·중복 Vision·KV8GiB의 실제 공존 적재와 Main100 실행 완료. 최대 Batch Capacity는 미측정 |
| 6. Online Intervention은 안전한가 | 검증한 조건에서 전체 Encode/Decode/E2E 보호를 입증하지 못함. 실제 요청 Offload 이득도 아님 |
| 7. 다음 구조의 우선순위는 무엇인가 | 먼저 Runtime·고부하 병목·현재 상태 Admission을 규명. 그 후 Request-level부터 검토. Pipeline·Wavefront·Elastic CP는 NOT_RUN |
| 8. EAServe/HydraInfer와 차별화한 근거인가 | Image-level 전달 단위와 Exact Visual-span Continuation을 관측. 이 관측만으로 새로운 기여나 우위를 확정하지 않음 |

E Idle 활용 자체는 새로운 기여로 주장하지 않는다. 선행연구와 구현은 별도 Related-work 감사에 있다.
원래 Image-only Main400은 보존했다. 사용자 변경 후 후속 실험은 동결 Main100을 사용했다.

## 독립 판정

- E_IDLE_CONFIRMED: 적격 저부하의 물리적 Kernel Idle에 한정한다.
- E_D_SIMULTANEOUS_IDLE_CONFIRMED: 같은 저부하 Trace에 한정한다.
- ENCODE_SAFE_HEADROOM_NOT_CONFIRMED: 실제 Probe의 전체 보호 Gate를 통과하지 못했다.
- Encoder-exclusive Service Gate와 E/D 같은 Suffix의 Safe Quantum 비교는 NOT_MEASURED다.
- Prefill-heavy 및 D의 추가 Prefill 보호는 검사한 전체 계약에서 미확인이다.
- 후속 Dual-side 연구는 HOLD다. 모든 Workload에서 물리적으로 불가능하다는 판정은 아니다.

## 동일 구간의 Kernel·Queue 상태

C25는 QPS0.5, Main100 Replay5269 Output Tokens, 독립 초기화1회다. 중앙 구간은 157.172988초다.
예정 Arrival p10–p90를 사용했다. 이를 Sustainable Steady State라고 주장하지 않는다.

| Stage | Busy % | Idle % | No HTTP Pending + Idle % | HTTP Pending + Idle % | Waiting Mean/Observation | Waiting Max |
|---|---:|---:|---:|---:|---:|---:|
| encoder | 12.732604 | 87.267396 | 72.642880 | 14.624516 | 0.509677 | 1 |
| prefill | 14.068244 | 85.931756 | 77.030389 | 8.901367 | 0.041166 | 2 |
| decode | 14.371586 | 85.628414 | 82.180431 | 3.447983 | 0.019316 | 1 |

Waiting Mean은 Schedule 관측별 평균이다. 시간 가중 Queue가 아니다.
Kernel Busy는 SM Active·Tensor Core Activity·HBM Bandwidth가 아니다. 해당 Counter는 NOT_MEASURED다.
Kernel Gap이 있고 HTTP Pending이 없더라도 GPU Stream의 완전한 Quiescence를 보장하지 않는다.
TRUE_ENCODE_IDLE·HOST_SCHEDULER_WAIT·INPUT_OR_TRANSFER_WAIT·INTRA_ENCODE_KERNEL_GAP·DOWNSTREAM_BACKPRESSURE의 완전한 배타적 분해는 미완료다.
미확인 상태를 유리한 Idle Category로 채우지 않았다.

## E/D Kernel Gap의 분포

| Stage | Count | Total Idle s | Mean ms | p50 ms | p90 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| encoder | 72439 | 137.160774390 | 1.893465873 | 0.001565997 | 0.006792000 | 0.041539197 | 0.243080035 | 6769.806614451 |
| decode | 816866 | 134.584737473 | 0.164757423 | 0.000096006 | 0.000288004 | 0.001088003 | 0.023007997 | 6501.268365115 |

Count CDF와 Idle Duration 가중 Coverage는 다르다. 매우 짧은 Gap이 Count를 지배한다.
공통 Monotonic Clock으로 변환했다. 짧은 GPU 간 선후관계에는 원시 Clock Anchor 오차가 있다.
미래 Window의 길이를 알고 실행할 수 있다고 가정하지 않았다.

| Threshold ms | E Window Count % | E Idle Duration % | D Window Count % | D Idle Duration % |
|---:|---:|---:|---:|---:|
| 0.1 | 3.117105 | 99.786177 | 0.104546 | 99.662610 |
| 1 | 0.150471 | 99.449977 | 0.029625 | 99.510169 |
| 5 | 0.095253 | 99.406768 | 0.009059 | 99.298164 |
| 10 | 0.092492 | 99.398967 | 0.008814 | 99.286252 |
| 20 | 0.092492 | 99.398967 | 0.007835 | 99.208389 |
| 50 | 0.092492 | 99.398967 | 0.007712 | 99.189653 |
| 100 | 0.091111 | 99.339073 | 0.007100 | 98.922669 |

## P Busy와 동시 Idle

| Physical State | 전체 ROI 대비 % | P Busy 대비 % | Accounting |
|---|---:|---:|---|
| P_ACTIVE_E_IDLE | 11.532757 | 81.977234 | MARGINAL_OVERLAPS_OTHER_ROWS |
| P_ACTIVE_D_IDLE | 11.871610 | 84.385869 | MARGINAL_OVERLAPS_OTHER_ROWS |
| P_ACTIVE_E_IDLE_D_IDLE | 9.994642 | 71.043991 | DISJOINT_PARTITION |
| P_ACTIVE_E_ACTIVE_D_IDLE | 1.876968 | 13.341878 | DISJOINT_PARTITION |
| P_ACTIVE_E_IDLE_D_ACTIVE | 1.538115 | 10.933243 | DISJOINT_PARTITION |
| E_P_D_ALL_ACTIVE | 0.658519 | 4.680888 | DISJOINT_PARTITION |

첫 두 Marginal 행은 다른 행과 겹친다. Disjoint 네 행만 P Busy를 분할한다.
P Busy와 P Queue 존재는 다른 조건이다. 위 표는 지속적인 P Queue 시간의 분해가 아니다.
Idle Time Overlap을 Actionable Slack Overlap 또는 동일 요청의 3GPU 병렬화 가능성으로 해석하지 않는다.
고부하 C11·C27·문서 C39의 Observer Gate 실패는 DIAGNOSTIC으로 유지했다.
Runtime Source가 다른 점이나 Short ROI를 하나의 적격 QPS 곡선으로 연결하지 않았다.

## 실제 추가 Prefill과 메모리

C40은 실제 Online E/P/D Main100에 별도의 실제 Prefix Probe를 추가했다.
E/D128·512를 Natural과 Replay에서 각각 독립 N3로 실행했다. 새 모델 초기화와 실제 추가 작업을 확인했다.
같은 Block 안의 조건은 초기화를 공유했다. 조건별 독립 초기화 수는 3개다.
추가 작업은 613회,195200 Prefix Tokens,5465600 Token-layer Rows다. 버린 Sample은 Client Throughput에 포함하지 않았다.
8개 조건의 전체 보호 Gate는 모두 미통과다. 일부 Metric의 개선과 모든 부정적인 결과를 함께 보존했다.
추가 Probe가 없는 Control에서도 실패 Start가 있었다. 모든 Gap을 Probe의 GPU 간섭으로 단정하지 않는다.
E의 HTTP p95는 Encoder-exclusive GPU Service가 아니다. 원래 서비스 Gate를 대신하지 않았다.
E는 추가 Full Model과 중복 Vision·KV8GiB를 실제 적재했다. Main100 후 Final Device Free는 19,053,608,960–21,605,842,944 bytes였다.
Device 표본은 Process Allocator와 구분했다. 최대 Encoder Batch나 OOM 경계는 측정하지 않았다.

C41은 실제 Prefix KV Import가 있는 D Suffix128/256/512/1024/2048을 검사했다.
15개 Input×Quantum의 정합성과 작업량 보존은 통과했다. 전체 비용 Observer Gate는 실패했다.
비용은 DIAGNOSTIC이다. GPU-exclusive Kernel·DMA 또는 실제 Safe Budget으로 사용하지 않았다.
미래 Physical Gap과 비용을 비교한 Fit은 OFFLINE_ORACLE이다. Online Speedup이나 보호 성공이 아니다.

## 재현 자료와 전체 캠페인의 남은 작업

모델은 Original BF16 Qwen2.5-VL-7B-Instruct, Full28, 원래 Vision·Processor·MRoPE다.
Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6의 E/P/D를 사용했다.
P/D B8192/S1024, KV42GiB, Block128, Prefix Cache Off다. E의 B114688은 별도 Encoder Budget이다.
C25 Stock Compiled와 C40/C41 Common Compiled BI0의 Source·Compiler 계약을 섞지 않았다.
Main100 Hash는 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.
분포·원문·이미지·문서 Split은 그대로 유지했다. Full Task Quality는 PENDING이다.
analysis/idle_addendum_final_*_v0에 표·Source SHA·독립 판정을 저장했다.
상세 N3 Gate·메모리·비용은 mllm_epd_dual_side_prefill_headroom_report.md에 있다.
대응 Figure와 CSV는 idle_scope_*·low_load_C25_*·interference_PROBE_C40R4_*·suffix_QUANTA_C41R3_*에 있다.
후속 Native Transport·Actual Batch 감사와 전체 Resource·Git Closeout은 별도 작업이다.
현재 파일은 그 작업이 완료됐다고 주장하지 않는다.
