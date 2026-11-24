# Secondary Document37 — Native E/P/D 계측

Main100을 교체하지 않았다. 기존 Parent Discovery의 MMLongBench-Doc37개를 모두 사용했다.
Source Stratum은 성능 결과 전에 동결했다. Natural N1 Pilot과 Equal-output CPC를 구분했다.
전체 Task Quality는 PENDING이다. 이 실행은 Token Split 성능 비교가 아니다.

## Natural Online Pilot

| Offered QPS | Requests | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Joint % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 37 | 3496 | 2339.535 | 12.253 | 6.711 | 100.000 |
| 1 | 37 | 3350 | 3225.421 | 14.907 | 6.748 | 97.297 |
| 2 | 37 | 3309 | 5940.126 | 15.616 | 6.808 | 56.757 |
| 3 | 37 | 3314 | 9829.903 | 15.778 | 6.801 | 21.622 |

## CPC와 같은 시간 구간의 GPU 상태

각 QPS는 새 E/P/D 초기화를 사용했다. Worker당 Collector 시작·종료는 1회다.
Profile 차이와 두 Clean의 Spread가 모두 5% 이하여야 Primary 계측으로 인정했다.
QPS1/2/3 모두 이 Gate를 통과하지 못했다. 모든 Kernel 수치는 `DIAGNOSTIC`이다.
Clean Spread 실패만으로 Profiler가 그 차이를 일으켰다고 단정하지 않는다.
아래 분모는 예정 Arrival p10–p90 구간이다. Sustainable Steady State로 확정하지 않았다.

| QPS | E Busy % | P Busy % | D Busy % | E Waiting Max | P Waiting Max | D Waiting Max | E&D Idle % | E&D Idle/P Busy % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.769 | 58.444 | 82.397 | 2 | 3 | 1 | 5.071 | 6.116 |
| 2 | 84.568 | 75.902 | 52.196 | 6 | 6 | 1 | 8.243 | 1.290 |
| 3 | 84.231 | 67.180 | 30.566 | 13 | 4 | 1 | 12.075 | 7.677 |

Kernel Busy는 Kernel 구간 Union이다. SM Active·Tensor Core Activity·HBM Bandwidth는 미측정이다.
Waiting Max와 평균은 Scheduler 관측값이다. 50ms Sampling을 시간 가중 Queue로 바꾸지 않았다.
D Decode-ready, Remote-KV 대기와 Free KV Block은 별도 Resource CSV에 있다.
E의 Dummy KV Pool은 추가 LLM Weight의 KV Capacity가 아니다.
전체 Cohort와 중앙 Arrival 결과는 따로 보존했다. Warm-up은 모든 Main 결과에서 제외했다.

## 판정과 한계

이 Profile만으로 E 비병목·P 포화·D Decode-safe 여유를 동시에 확정할 수 없다.
독립 P Capacity를 측정하지 않았다. Service Proximity≥85%도 미측정이다.
큰 E/D Kernel Gap과 동시 Idle은 향후 실행 가능 Window 또는 3GPU 병렬화의 증거가 아니다.
No-pending HTTP Idle과 Pending HTTP Idle은 분리했다.
Host Scheduler·Input·Transfer·작업 내부 Gap·Downstream Backpressure의 완전한 인과 분류는 미측정이다.
해당 Label을 모든 Window에 임의로 부여하지 않았다. 실제 추가 Prefill 안전성은 별도 Intervention이 필요하다.
이 분석은 `PREFILL_HEAVY_NOT_CONFIRMED`다. 미확정을 확인된 부재로 해석하지 않는다.

## 재현 계약

Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.
Original BF16, BI0, 원본 Vision/Processor, Full28, B8192/S1024, KV42GiB, Block128이다.
P/D는 공통 Compiled Reduction 설정이다. GPU4/5/6은 E/P/D다.
Order Seed2026112509와 Arrival Seed2026112510을 사용했다.
Manifest는 `data/secondary_document37_discovery_v0.json`이다.
CPC의 Output 수는 기존 R0 Natural 결과로 고정했다. Natural Pilot과 평균을 합치지 않았다.
Raw Profile은 `raw/profiles/ec_DOCUMENT_C39R1_CPC_*`에 있다.
집계 CSV/JSON은 `analysis/document_C39_*_v0`에 있다. Figure는 `figures/document_C39_*_v0`에 있다.
