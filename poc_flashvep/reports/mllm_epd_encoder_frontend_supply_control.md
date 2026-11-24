# MLLM EPD Encoder Frontend Supply Control

판정은 DIAGNOSTIC이다. 전처리 Worker2는 P 포화나 Capacity 개선을 입증하지 못했다.
Generic Token Split은 실행하지 않았다. Main100과 보존한 Main400은 바꾸지 않았다.

## 변경한 요소

A는 Processor Cache1GiB와 Renderer Worker1을 사용했다.
B는 E의 Processor Cache만 0으로 바꿨다. P/D의 Cache1GiB는 유지했다.
C는 B에서 E의 Renderer Worker만 2로 바꿨다.
Cache를 켠 상태의 Worker2는 설치된 공식 Runtime이 금지한다. 이 검사는 제거하지 않았다.
각 E 재시작에는 새 EC·Metadata 경로를 사용했다. 이전 Cache0의 Engine-local Image ID는 재사용하지 않았다.
P/D는 E의 원래 Token·Hash·Image Grid를 전달받았다. 원본 이미지와 Vision Encoder 계산은 바꾸지 않았다.
B와 C의 N8 Natural EOS는 기존 같은 Geometry Native EPD와 토큰·입력 ID Strict Match를 통과했다.
세 Arm은 같은 GPU4/5/6을 순차 사용했다. Source Archive와 Boot별 UUID는 실행 Config에 보존했다.
Model·Processor Revision, BF16 BI0, Prefix Cache Off와 Natural 생성 상한2,048은 유지했다.
LLM의 상한은 B8192/S1024, KV42GiB/GPU, Block128이다.
E는 B114688/S1024이며 실제 Chunked Prefill은 꺼져 있다. E는 KV 없는 Encoder 전용 Scheduler를 사용한다.
CPU 기록 정책은 모든 Arm에서 Immediate였다. Source Credit이나 인위적인 D Cap은 추가하지 않았다.

## 동일 출력 C3 Screening

QPS3, 도착 Seed2026112402, 순서 Seed2026112404를 사용했다.
각 Arm은 같은100개 요청과 5,269개 출력 Tokens를 완료했다. 이는 Natural EOS 성능표가 아니다.
Cell당 독립 초기화는 1회다. 독립 Start3회 CI는 없다.

| Arm | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | All-ITL PASS % |
|---|---:|---:|---:|---:|---:|
| A | 2888.619 | 10697.458 | 18.506 | 7.101 | 100.00 |
| frontend_B4_cache0_T1 | 3002.565 | 10405.932 | 16.489 | 6.867 | 99.00 |
| frontend_B5_cache0_T2 | 2765.655 | 10766.497 | 17.282 | 6.986 | 98.00 |

Worker2는 Worker1보다 TTFT p95가 낮지 않았다. 따라서 단순 CPU Worker 수를 해결책으로 채택하지 않는다.
## GPU 활동과 관찰자 검사

C의 Clean/Profile/Clean은 각각100개 요청과 같은 출력 작업량을 사용했다.
Profile과 두 Clean 평균의 차이는 모든 동결 지표에서 5% 이하였다.
그러나 Clean 간 TPOT p95 차이는 7.73%였다. 동결한 관찰자 안정성 Gate를 통과하지 못했다.
이는 Profiler 자체의 간섭이 5%를 넘었다는 증거와는 다르다. 그래도 GPU 결과는 Diagnostic으로 제한한다.
중앙 도착 구간은 전체 요청의 예정 도착 시각 p10–p90이다. 지속 가능한 Steady-state를 입증한 구간은 아니다.

| Stage | Kernel Busy % | Pending HTTP + Kernel Idle % | No Pending HTTP + Kernel Idle % |
|---|---:|---:|---:|
| encoder | 62.325 | 31.696 | 5.979 |
| prefill | 58.285 | 8.201 | 33.514 |
| decode | 51.253 | 7.498 | 41.249 |

E/D 동시 Kernel Idle은 P Busy 시간의 15.315%였다.
이 값은 물리적 Kernel Inactivity의 겹침이다. 실행 가능한 추가 Prefill Budget은 아니다.
HTTP Pending에는 전처리·Transfer·Scheduler·Service가 모두 포함된다.
이 자료만으로 E Idle을 다섯 원인에 정확히 분배할 수 없다. 미분리 부분은 NOT_MEASURED다.
SM Active, Tensor Core Active, HBM 대역폭 포화율은 이 계측에서 측정하지 않았다.
P Busy가 80% 미만이며 지속적인 P Queue도 입증하지 못했다. PREFILL_HEAVY_NOT_CONFIRMED를 유지한다.
실제 추가 Prefill 삽입은 없었다. Encode-safe와 Decode-safe Headroom은 NOT_MEASURED다.

## 별도 Natural EOS 확인

C의 QPS2 Natural100은 5,149개 Tokens를 출력했다.
TTFT p95는 6821.131ms, ITL p99는 17.559ms였다.
All-ITL PASS는 97.00%, Joint PASS는 70.00%였다.
이는 N1 확인이다. 상대 비열등성 CI와 전체 Task Quality Gate는 아직 미완료다.
자연 출력 작업량이 다른 과거 실행과 섞어 평균을 계산하지 않는다.

## 메모리 범위

| Stage | Allocated GiB | Reserved GiB | Peak Allocated GiB | Logical Parameter GiB |
|---|---:|---:|---:|---:|
| encoder | 3.187 | 42.719 | 17.644 | 1.260 |
| prefill | 58.010 | 60.725 | 59.952 | 15.445 |
| decode | 57.901 | 60.723 | 59.952 | 15.445 |

이는 Quiescent 시점의 Allocator·Device Counter다. E에 추가 LLM을 올린 Co-residency 결과는 아니다.
현재 E는 Vision Weight만 보유했다. 추가 LLM Weight·KV·Workspace의 동시 Peak는 아직 측정하지 않았다.
논리 Weight 크기만으로 안전한 추가 Memory Budget을 확정하지 않는다.

## 다음 원인 검증

설치된 ECExampleConnector는 Debug 구현이다. E에서 이미지마다 동기 CPU 복사와 Safetensors 저장을 수행한다.
새 관측은 이 경로의 정확한 인과 기여율을 확정하지 않는다.
별도 GPU0/1에서 CUDA IPC의 Tensor·View·Lifetime 프로토콜을 먼저 검사했다.
이 프로토콜 검사는 Serving Connector나 Generic Token Split 구현을 뜻하지 않는다.
이후 Transport 변경을 진행하면 N1→N8, KV·MRoPE, Natural 품질과 독립 A/B 검사를 다시 통과해야 한다.
기존 Native의 Task Quality 문제와 Observer 실패는 보존한다. 이 보고서는 Gate를 완화하지 않는다.

CSV: `mllm_epd_prefill_heavy_token_split_20261124/analysis/frontend_*_v0.csv`.
PNG/PDF: `mllm_epd_prefill_heavy_token_split_20261124/figures/frontend_*_v0.*`.
