# LongBench400: Source Credit와 Continuation Chunk의 인과 분석

상태: Discovery의 Credit·Queue·Chunk 분석을 완료했다. 아래 N1 수치를 최종 Confirmation으로 표시하지 않는다.

## 실행 계약

모델은 Qwen3-8B다. Numerical Mode는 BI0다. 모든 Worker는 B8192/S1024를 명시한다.
실제 GPU Batch는 별도로 기록한다. Primary Natural Generation은 EOS를 허용한다. Output Cap은 2048이다.

데이터셋은 LongBench-Balanced400-v1이다. Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.
이번 분석의 Discovery200은 동결한 400개 중 200개다. 최종 Held-out200과 Request Context가 겹치지 않는다.
다만 모든 요청은 이전 Natural Reference 생성에 사용됐다. 따라서 완전히 관측하지 않은 Prompt 집합은 아니다.

## 1. Credit의 의미

Source Credit은 실제 GPU Batch가 아니다. Coordinator는 P에 요청을 제출할 때 Credit을 획득한다.
Coordinator는 첫 Client 출력이 나올 때 Credit을 반환한다. Credit1에서도 이전 요청의 후속 Decode와 다음 P 작업은 겹칠 수 있다.
따라서 Credit1을 전체 시스템의 완전 직렬 실행이라고 부르지 않는다.

C3의 `continuous_source_credit_limit=0`은 별도 Coordinator Outstanding 제한을 끈다는 뜻이다.
요청을 0개 허용한다는 뜻이 아니다. Worker Scheduler, KV Capacity와 Admission 제한은 그대로 적용한다.
C0의 별도 `source_credit_limit`과 혼동하지 않는다. C0 Primary는 Credit4다.

Credit Sweep은 축별 Hidden Pool을 16 Slots로 고정한다. Credit만 1/4/8/16으로 바꾼다.
Native와 Token은 불필요한 Hidden Pool을 할당하지 않는다. Layer와 Hybrid는 실제 Hidden/Residual 전송 Pool을 사용한다.
KV Budget은 GPU당 42GiB다. Credit1은 직렬화 비용을 확인하는 Control이다. Serving 기본값 후보에서는 제외한다.

설계와 Source SHA는 [Credit 계약](dense_ar_longbench400_2d_20261121/configs/source_credit_sensitivity_Layer_v1.json)에 있다.
개별 실행의 Config와 Source SHA는 같은 Namespace의 `raw/<name>/configuration.json`에 있다.

## 2. Credit Intervention의 Discovery 결과

다음 값은 Layer Last8, Discovery200, QPS2.8의 실측이다. Decode 정책과 Hidden Pool16은 고정했다.
각 Cell은 독립 Server Start 1회다. 신뢰구간은 아직 없다.

| Source Credit | TTFT p95 | ITL p99 | TPOT p95 | SLO-goodput |
|---|---:|---:|---:|---:|
| 1 | 34,192.666ms | 8.701ms | 8.206ms/token | 11.360 output tokens/s |
| 4 | 7,779.430ms | 8.799ms | 8.231ms/token | 49.920 output tokens/s |
| 8 | 6,227.499ms | 9.002ms | 8.273ms/token | 54.309 output tokens/s |
| 16 | 6,362.531ms | 9.839ms | 8.454ms/token | 54.820 output tokens/s |

증거 수준은 `MEASURED_ONLINE`이다. 같은 축 안에서 Credit만 변경했다.
이 결과는 Source Outstanding Window가 Online TTFT에 영향을 줄 수 있음을 보인다.
그러나 Credit8이 일반적인 최적값이라는 결론은 아니다. 독립 반복은 별도로 남았다.
Layer와 Token의 순수 GPU 계산 차이도 이 표에서 판정하지 않는다.

전체 결과는 [Discovery Ledger](dense_ar_longbench400_2d_20261121/analysis/decode_tail_recovery_discovery_all_v1.json)에 있다.
불리한 Credit1 결과도 보존한다. 최종 Serving 비교는 별도의 동결된 선택과 새 Arrival Seed를 사용한다.

## 3. Queue 경계의 해석

Source Credit4인 Layer/Hybrid는 Coordinator 앞에서 기다릴 수 있다.
Credit 제한이 없는 Token은 P Worker 안에서 기다릴 수 있다.
따라서 `Arrival→P Submit`만 비교하면 실제 P 대기를 잘못 해석할 수 있다.

P 경로 비교는 적어도 `Arrival→첫 P Prepare`까지 포함한다.
P Prepare는 Host의 입력 준비 경계다. GPU Kernel 시작 시각이 아니다.
Continuous 실행의 `P Prepare→Ready`에는 다른 요청과 Chunk 사이의 대기도 들어갈 수 있다.
이를 Request별 GPU-exclusive Service Time으로 해석하지 않는다.

`service_accounting.py`는 동일 Request ID와 Prompt ID를 확인한다.
이 도구는 평균 TTFT를 관측 경계의 합으로 재구성한다.
평균 구간 차이의 합을 TTFT p95의 분해라고 부르지 않는다.
Natural EOS가 달라진 요청의 실제 Output Count도 별도로 기록한다.

## 4. B8192와 Token-Layer Rows는 다른 단위다

B8192는 Scheduler Iteration의 Query Token 상한이다. Token-Layer Rows 상한이 아니다.
같은 GEMM Rows에서도 Layer와 Hybrid의 D Query는 거의 전체 Prompt 길이다.
Token Split의 D Query는 Suffix 길이다. 따라서 필요한 Chunk 수가 다를 수 있다.

다음 값은 QPS2.8의 동일 Trace Discovery 실행이다. 각 Method는 독립 Start 1회다.
Pure Axis는 Last4다. Hybrid는 Last4와 Suffix 약 L/8을 함께 사용한다.
따라서 이 세 Method는 이 표에서 Iso-work 비교가 아니다.

| Method | D Query Tokens/request 평균 | D Prefill Calls/request 평균 | B8192 하한 평균 | Chunk 사이 Host Gap 평균 |
|---|---:|---:|---:|---:|
| Layer Last4 | 11,808.345 | 2.075 | 1.945 | 40.366ms |
| Token Last4-equivalent | 약 1,312 | 1.000 | 1.000 | 0ms |
| Hybrid Last4 + L/8 | 11,808.345 | 2.950 | 2.705 | 106.792ms |

Hybrid의 Host Gap 중 약 103.644ms는 다른 요청의 Decode Host Step과 겹쳤다.
Layer에서는 약 39.705ms가 겹쳤다. 이는 `TRACE_ANALYSIS`다.
이 값은 GPU-exclusive Decode Time이 아니다. 이 값만으로 TTFT 악화의 인과성을 확정하지 않는다.
먼저 시작한 Prepare가 기존 대기를 같은 Enclosure 안으로 이동시킬 수도 있다.

원시 Query 크기와 Request별 구간은 [Chunk 분석](dense_ar_longbench400_2d_20261121/analysis/continuation_chunk_residency_summary_v1.json)에 있다.
추가 Calls는 합법적인 Multi-request Packing으로 발생할 수 있다. 계산 누락이나 중복의 증거는 아니다.

## 5. 공통 Admission 반증 실험

추가 가설은 다음과 같다.

1. 큰 Partial Prefill의 끝부분에 새 Partial Prefill을 섞으면 첫 요청의 완료가 지연될 수 있다.
2. 작은 Token Suffix는 한 Iteration에 끝나므로 이 효과가 작을 수 있다.
3. 이 차이가 중요하지 않으면 공통 Admission 변경은 TTFT를 개선하지 않거나 악화시킬 수 있다.

실험은 Native/Layer8/Stock Token8/Hybrid4+L/8에 같은 Admission Rule을 적용한다.
현재 실행 중인 Prefill의 잔여 Query가 B8192보다 작을 때만 새 요청의 예상 완결 가능성을 검사한다.
Rule은 할당 전에 임시 Running Admission 상한을 낮춘다. 기존 상한은 예외 발생 시에도 복원한다.
작은 Token/Native 후보가 남은 Budget에 들어가면 수락할 수 있다.

Rule은 B/S를 변경하지 않는다. 이미 할당한 Query를 삭제하지 않는다.
Model Forward, KV Ownership, Output Token과 Client Timestamp도 변경하지 않는다.
Decode 중 Prefill 계산을 막는 기존 Rule과 READ Setup 지연 Rule은 그대로 유지한다.

현재 Rule은 완전한 미래 예측기가 아니다. Remote KV가 준비되지 않으면 추정이 보수적이거나 불완전할 수 있다.
따라서 Zero Fragmentation을 보장하지 않는다. 실제 Query와 완료 시각으로 효과를 확인한다.
각 축의 Natural32/400 State·Lifetime·Quality Gate를 먼저 실행한다. 정합성 실패 Arm은 성능 실행에서 제외한다.

QPS2.8의 공통 정책 A/B는 완료했다. 각 Arm은 독립 Start 1회다. 신뢰구간은 없다.

| Axis | Guard0 TTFT p95 | Guard1 TTFT p95 | Guard0→1 Chunk 사이 Host Gap |
|---|---:|---:|---:|
| Native | 10,519.082ms | 10,564.118ms | 큰 Continuation 없음 |
| Layer Last8 | 7,626.870ms | 9,399.778ms | 57.506→22.305ms/request |
| Token Last8-equivalent | 4,984.654ms | 5,151.242ms | 별도 Query Manifest 참조 |
| Hybrid Last4+L/8 | 9,086.934ms | 10,813.237ms | 104.498→43.321ms/request |

Guard는 Layer와 Hybrid의 Chunk 사이 Host Gap을 줄였다. 그러나 두 Axis의 TTFT p95는 악화됐다.
따라서 해당 Gap만 제거하면 TTFT가 개선된다는 가설은 이 A/B에서 지지되지 않았다.
Packing과 Request 완료 순서의 변화도 함께 확인해야 한다.
이는 Host 경계의 반증이다. GPU-exclusive Operator 원인을 확정한 결과가 아니다.

Token의 TPOT p95는 8.472→8.297ms/token으로 낮아졌다.
따라서 TTFT와 Decode Protection의 방향도 같지 않았다.
유리한 Safety 결과와 불리한 TTFT 결과를 모두 보존한다.
이를 미래 Window를 예측하는 Dynamic Scheduler 구현이라고 부르지 않는다.

근거: [공통 Guard A/B](dense_ar_longbench400_2d_20261121/analysis/completion_guard_common_policy_intervention_v1.json).

### Credit4→8의 Host 경계

아래 값은 같은 축의 Credit4→8 변화다. Hidden Pool16과 D 정책은 고정했다.
각 Cell은 독립 Start 1회다. 평균 경계와 TTFT p95는 서로 다른 통계량이다.

| Axis | Arrival→Credit 획득 평균 | Arrival→P Prepare 평균 | TTFT p95 |
|---|---:|---:|---:|
| Layer8 | 2613.147→1184.779ms | 2802.604→1837.882ms | 7779.430→6227.499ms |
| Token8 | 2151.588→600.614ms | 2281.515→1001.244ms | 6781.905→4794.173ms |
| Hybrid4+L/8 | 3222.866→1361.011ms | 3364.764→1813.156ms | 9302.088→6539.747ms |

Credit 증가에 따라 대기 위치와 전체 TTFT가 함께 바뀌었다.
Layer와 Hybrid의 평균 P Prepare→Ready는 각각 491.158→558.552ms와 452.127→551.641ms로 늘었다.
따라서 이 Enclosure만으로 Request별 GPU Service가 느려졌다고 판단하면 안 된다.
더 많은 Outstanding 요청과 Chunk Packing이 같은 Host 구간 안에 들어갈 수 있다.

Credit16에서는 Layer와 Hybrid의 TTFT p95가 Credit8보다 다시 나빠졌다.
Source Window를 크게 만들기만 하면 계속 빨라진다는 가설도 지지되지 않는다.
근거: [Request별 Credit Intervention](dense_ar_longbench400_2d_20261121/analysis/source_credit_host_intervention_v1.json).

## 6. 현재 한계

Natural C3의 Operator별 GPU-exclusive 기여율은 아직 완전히 분해하지 못했다.
Hidden/KV Bytes의 구조적 비율을 전송시간 비율로 바꾸지 않는다.
Source Credit32의 초기 메모리 검사는 Raw CUDA Free만 사용했다.
이 값은 반환 가능한 PyTorch Cache를 제외한다. 따라서 초기 실패를 물리적 실행 불가능성으로 확정하지 않는다.
별도 Cache 반환 Preflight도 실행했다. 약 2.898GB의 Cache를 반환했다.
이후 CUDA Free 21,802,516,480 Bytes는 Pool과 동결 Reserve의 합 22,548,578,304 Bytes보다 작았다.
따라서 Credit32의 유효 성능 결과는 없다. 이를 물리적 OOM 불가능성으로 확대 해석하지 않는다.

최종 판정은 새 Source의 Full400 정합성과 독립 Server Start 기반 ITL/TPOT 5% 비열등성을 요구한다.
이 문서의 Discovery 수치만으로 Layer, Token 또는 Hybrid의 Serving 승리를 선언하지 않는다.
독립 반복과 원래·수정 Source의 최종 판정은 [종합 보고서](dense_ar_longbench400_layer_token_2d_final_summary.md)에 있다.
