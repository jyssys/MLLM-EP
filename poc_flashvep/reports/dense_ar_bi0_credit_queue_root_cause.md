# BI0 Source Credit와 P Queue 원인 분석

상태는 `CLOSED`다. Source Host Queue를 부분 식별했다. Natural GPU 기여율은 미식별이다. 새 Evidence는 `dense_ar_bi0_axis_rootcause_20261120/`에 저장한다. Base Commit은 `12e0f370a26ec544af81a90577874527ce8d4c5b`다. 이전 Raw Evidence와 보고서는 수정하지 않았다.

## Source Credit의 의미

현재 Controlled Runtime은 P Dispatch 때 Source Credit을 획득한다. Coordinator가 D의 첫 실제 Token을 받으면 반환한다. Source Credit4는 이 구간의 Outstanding Request 상한이다. 실제 GPU Batch4가 아니다.

Source KV는 공식 NIXL Sending Completion과 Scheduler Retirement를 따른다. Hidden Slot은 D의 완료된 READ ACK 뒤에 재사용한다. Destination Hidden Slot은 Suffix CUDA Event 완료 뒤에 재사용한다. 세 반환 경계는 Source Credit 반환과 다르다.

D는 현재 요청이 끝난 뒤 다음 요청을 받는다. Device-wide Synchronization과 Pending NIXL 상태도 확인한다. 이 Whole-request Admission을 Active-decode 사이의 Safe Window로 표현하지 않는다.

## 기존 현상의 재현

`HISTORICAL_TRACE_ANALYSIS`다. 기존 Last4의 Layer/Token TTFT p95는15147.96/10048.87ms였다. 평균 P Queue는7141.51/4430.42ms였다. Ready→First Worker Host Envelope는272.57/332.18ms였다. 짧은 D 경로가 더 낮은 TTFT를 보장하지 않았다.

새 계측은 Credit 획득·반환 Event와 Coordinator 대기 상태의 변경을 기록한다. P Active 상태에는 IPC, 준비, GPU 계산과 응답이 함께 들어간다. 이를 GPU-exclusive 계산시간으로 사용하지 않는다. P가 비어 있는데 Credit이 꽉 찬 대기를 별도로 구분한다.

## 반증 실험

Last8/QPS2.35에서 Credit1/4/8/16/32를 검사했다. Native, Layer, Token에 같은 상한을 적용했다. 실제 Batch는1/1/1을 유지했다. Layer Hidden Pool은 메모리 사전 검사로 고정한32 Slots를 사용했다. Primary의4 Slots와 구분한다.

Credit 변경의 효과는 Credit-full 대기, Source Service, D Admission과 TTFT를 함께 비교한다. Credit1은 Backpressure Control이다. 첫 Token 뒤에는 다음 P와 이전 요청의 Decode가 겹칠 수 있다. 전체 P+D+Decode를 직렬화하는 별도 Admission Barrier와 다르다. 결과를 보고 Primary Credit1로 자동 변경하지 않는다.

## 기존 Last4의 정확한 Queue 분해

상태는 `HISTORICAL_TRACE_ANALYSIS`다. 두 방식은 같은200개 Request와 Seed2026112061의 QPS2.35 도착 Trace를 사용했다. 다음 값은 Request 평균이다. p95 구간을 합산한 값이 아니다.

| Arrival→첫 P Prepare 경계의 구성 | Layer | Token | Layer−Token |
|---|---:|---:|---:|
| 이전 Source 체인으로 인한 대기 | 7125.876ms | 4419.741ms | +2706.135ms |
| 해당 시점의 직접 Credit Block | 1.765ms | 2.806ms | −1.041ms |
| 실행 가능→Coordinator Dispatch | 1.122ms | 0.972ms | +0.150ms |
| Dispatch→첫 Prepare | 12.743ms | 6.898ms | +5.845ms |
| 합계 | 7141.506ms | 4430.417ms | +2711.089ms |

첫 Prepare는 Host 경계다. GPU 실행 시작 시각이 아니다. 이전 Source 체인에는 앞선 요청의 Service와 지연이 누적된다. 이 표만으로2706.135ms를 GPU 계산시간으로 확정하지 않는다.

직접 Credit Block은 이 조건의 추가2711ms를 설명하지 못했다. 그러나 Credit 변경의 전체 간접 효과가 없다는 뜻은 아니다. Credit 변경은 이후 Queue 상태도 바꾼다. 실제 Credit Intervention과 상태 갱신 Replay를 별도로 검사한다.

## 이전 Source 체인을 정확하게 푼 결과

같은 Natural200 원시 기록에서 각 이전 Source Service와 Admission Gap을 순서대로 누적했다.
이전 요청의 비용은 이후 여러 요청의 Queue에 반복해서 들어간다.
다음 값은 Mean P Queue 차이에 들어가는 실제 순서 가중 항이다.
모든 값은 Layer−Token이다. 독립적인 무작위 Intervention 효과는 아니다.

| Mean P Queue 차이의 정확한 구성 | 차이 |
|---|---:|
| 최초 P Dispatch Offset | −0.023ms |
| 이전 Source Host Service의 누적 | +2876.210ms |
| 이전 Arrival Gap의 누적 | −113.041ms |
| Source Credit Extension의 누적 | −62.619ms |
| 실행 가능한 P의 Host Admission Gap 누적 | +4.718ms |
| 현재 Dispatch→첫 Prepare 차이 | +5.845ms |
| 합계 | +2711.089ms |

이전 Source Service 누적2876.210ms도 세 구간으로 정확하게 나눴다.
Dispatch→Prepare가 +538.177ms다. Prepare→Worker Ready가 +2339.260ms다.
Worker Ready→Parent Ready가 −1.227ms다.
따라서 이 조건의 주된 항은 P Service의 누적이다.
Credit 대기는 이 비교에서62.619ms를 상쇄했다.
실행 가능한 P를 Host가 제출하지 않은 Gap의 누적은4.718ms였다.
다만 Prepare→Ready는 GPU-exclusive 시간이 아니다. Launch와 Host 진행도 포함한다.
Natural Profile의 Observer Gate가 실패했으므로2339.260ms를 전부 GPU 시간으로 확정하지 않는다.

[정확한 Source 체인 JSON](dense_ar_bi0_axis_rootcause_20261120/analysis/exact_source_chain_decomposition_v1.json)에 원시 실행명과 항등식 검사를 보존한다.

## 새로운 독립3회에서도 확인한 Source 체인

별도 Seed2071/2072/2073의 Full200 Confirmation도 같은 방법으로 풀었다.
QPS2.35, Credit4, Hidden Slots4, KV128과 실제 Batch1/1/1을 유지했다.
다음 값은 각 Start의 정확한 가중 항을 평균한 Layer−Token 차이다.

| Cut | 이전 P Host Service 누적 | Credit Extension 누적 | Arrival Gap 누적 | 나머지 항 | Mean P Queue 차이 |
|---|---:|---:|---:|---:|---:|
| Last2 | +1512.920ms | −9.961ms | −47.740ms | +7.484ms | +1462.703ms |
| Last4 | +2255.147ms | −103.152ms | −59.740ms | +13.042ms | +2105.297ms |
| Last8 | +3801.376ms | −514.676ms | −2100.266ms | +17.165ms | +1203.598ms |
| Last12 | +4371.059ms | −2424.919ms | −1635.372ms | +15.099ms | +325.867ms |

Arrival Trace가 다르다는 뜻은 아니다.
같은 다음 Arrival보다 앞선 P Ready가 빠르거나 늦으면 Source의 Arrival Gap이 달라진다.
이 Host Gap도 Physical GPU Idle을 뜻하지 않는다.
Credit Extension은 같은 Credit4 계약에서 실현된 추가 대기다.
다른 상한을 적용한 독립 Intervention 효과로 해석하지 않는다.
특히 Last12에서는 Token의 더 큰 Credit 대기가 P Service 이점 일부를 상쇄했다.
따라서 Last4의 결론을 모든 Cut의 Credit 효과가 없다는 주장으로 확대하지 않는다.

[Confirmation Host Service와 정확한 체인](dense_ar_bi0_axis_rootcause_20261120/analysis/full200_confirmation_host_service_demand.json)에 세 Start의 모든 항을 보존한다.

기존 평균 TTFT 차이는2673.857ms였다. 같은 경계의 차이는 다음과 같다.

- P Queue: +2711.089ms.
- P Prepare→Ready: +22.358ms.
- Ready→D Prepare: −44.751ms.
- D Prepare→첫 Worker Output: −14.861ms.
- 첫 Worker Output→Client: +0.022ms.

다섯 차이의 합은 평균 TTFT 차이와 일치한다. D 경로의 약59.612ms 절감보다 P Queue 증가가 컸다. 이 결과의 Queue 원인과 GPU 기여는 구분한다.

## Serial 이점의 비교 조건

기존25.351ms Layer 이점은 Last8/Calibration20의 Host First-output 경로에서 나왔다. 같은 Last8의 Validation15에서는 Layer가15.057ms 더 느렸다. 단일 Start와 서로 다른 Request 부분집합의 결과다. 이를 Last4/Full200의 Online2673.857ms 차이에 직접 합산하지 않는다.

새 Last4 Control은 동일200개 Request, 도착 Seed2026112082, QPS2.35와 요청별 출력 길이를 사용한다. 두 방식 모두4723 Tokens를 생성한다. 이 Fixed-output Control은 Natural EOS Main 결과가 아니다.

별도 Serial Admission에서 P Dispatch→Parent Ready와 D Submit→첫 Client Token의 합은 Layer577.066ms, Token570.692ms였다. 이 Full200 Last4 조건에서는 Token이 Serial에서도6.374ms 빨랐다. 따라서 이 조건 자체에서는 Serial 승자의 역전이 재현되지 않았다.

Online Credit4의 TTFT p95는 Layer16162.409ms, Token13301.587ms였다. Credit1에서는47767.331ms와46000.177ms였다. 각 조건은 독립 Start1회다. Credit1이 두 방식의 Queue를 크게 늘렸다. 이 결과를 성능 개선이나 통계적으로 확정된 승리로 표현하지 않는다.

## GPU 계측 상태

후속 Clean Control3회를 모두 사용하는 통계도 계산했다. Profile Start는 포함하지 않았다.
Fixed-output Serial의 P+D First Host 비율은1.005966이다.95% CI는 [0.989338, 1.020603]이다.
따라서 Serial에서 한 Axis가 확실히 빠르다고 판정하지 않는다.
Fixed-output Online TTFT p95 비율은1.212395다. CI는 [1.201986, 1.221838]이다.
Natural Online TTFT p95 비율은1.271760이다. CI는 [1.249572, 1.298296]이다.
Natural Online의 P+D First Host 비율은0.994929다. CI는 [0.978783, 1.015758]이다.
각 값은 Layer/Token 비율이다. 각 Group은 Server Start3회지만 Arrival Seed는1개다.
이 결과는 Process-start 반복성이다. 새 Arrival Seed 일반화로 표현하지 않는다.
합산 Service가 비슷해도 P/D Stage의 배치 위치와 Queue 누적은 다를 수 있다.

Runtime+Driver Correlation의 후속 Full200 Fixed-output Online Profile은 양측 Gate를 통과했다.
Layer의 P Kernel Union은462.531ms다. Token은449.767ms다.
식별한 Attention Kernel 차이는 Layer가14.724ms 더 크다.
같은 Profile의 P Host Stage 차이는15.725ms다.
이는 Token의 P Attention 절감이 실제 P Service 차이에 기여한다는 측정 근거다.
Profile은 각1회다. 전체 GEMM의 완전한 분류나 순수 GPU Service Intervention은 아니다.
Natural Source Profile의 Observer Gate는 실패했다. 해당 Profile은 `DIAGNOSTIC`으로 남긴다.
따라서 과거 Natural Last4의2711ms Queue 중 정확한 GPU 기여율은 아직 확정하지 않는다.

첫 DRIVER-only 계측은 Observer의6개 지표에서5% 이내였지만, Kernel 시간의58.22%만 Request에 연결했다. Attribution Gate를 통과하지 못했다. 해당 값은 `DIAGNOSTIC`으로 보존한다.

CUDA Runtime과 Driver의 Correlation을 함께 기록하는 별도 설정을 검증했다. Kernel 시간은 API Correlation ID로 Request에 연결했다. Host Envelope를 GPU-exclusive 시간으로 대체하지 않았다. D Decode Graph의 Kernel별 Attribution은 Primary에서 미검증이다.

Destination Kernel-only 재시도도 Primary Gate를 통과하지 못했다.
Layer Profile은 D 준비 구간의 Observer 변화가8.747%로5%를 넘었다.
Token Profile은 요청200개가 완료됐지만 Collector 종료 Assertion 뒤 Worker가 비정상 종료했다.
해당 실행을 정상 Dispatcher 완료로 판정하지 않았다.
실패한 Raw CSV와 Log는 보존한다. Collector 실패의 근본 원인은 `NOT_IDENTIFIED`다.
뒤따른 UCX 종료 Error를 앞선 Assertion의 확정 원인으로 표현하지 않는다.

API Correlation의 정의는 [NVIDIA CUPTI 문서](https://docs.nvidia.com/cupti/main/main.html#correlation-id-mechanism)를 따른다. 두 API Domain을 사용한 별도 Eager/Runtime Witness는 통과했다. 이 Witness 통과만으로 Online 계측의5% Overhead Gate를 대신하지 않는다.

근거는 새 Namespace의 `analysis/historical_Last4_source_eligibility_accounting.json`, `analysis/causal_historical_TTFT_partition.json`, `analysis/serial_online_service_vector_replay_v1.json`에 저장한다.
