# 고부하 Short CUPTI Capture — 진단용

짧은 구간으로 줄인 계측도 고정한 5% Gate를 통과하지 못했다. 증거 수준은 `DIAGNOSTIC`이다.
Low-load C25의 통과 판정은 유지한다. QPS3의 이 Trace를 Primary GPU 비용으로 사용하지 않는다.

## 사전 고정한 실험

Original Qwen2.5-VL-7B BF16, BI0, Full28과 원본 Processor를 사용했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
동일 Main100, QPS3, Arrival Seed2026112402, Order Seed2026112404와 Output Replay5269 Tokens를 유지했다.
GPU4/5/6은 E/P/D다. B8192/S1024, KV42GiB, Block128과 원래 Compiled 모델은 변경하지 않았다.
Warm-up, Clean Before, Profile, Clean After 모두 실제 Online E/P/D를 실행했다. 독립 초기화는 1회다.

관측 구간만 완료 요청 30–70번으로 사전 고정했다. 성능 Metric은 요청 100개 전체로 계산했다.
미래 Window를 사용하는 Offloading Policy가 아니다. 추가 Prefill은 삽입하지 않았다.

## 계측 Gate

| Metric | Profile Difference % | Clean Spread % | Gate |
|---|---:|---:|---|
| duration_seconds | 0.8713 | 0.1757 | PASS |
| TTFT_p50_ms | 5.4396 | 5.2524 | FAIL |
| TTFT_p95_ms | 2.1054 | 2.4136 | PASS |
| ITL_p99_ms | 28.1761 | 2.5977 | FAIL |
| TPOT_p95_ms | 5.2074 | 0.1593 | FAIL |

ITL p99가 비계측 평균보다 28.1761% 높았다. TPOT p95도 5.2074% 높았다.
TTFT p50은 Profile 차이와 Clean Spread 모두 5%를 넘었다.
Collector의 Drop/Invalid 검사는 통과했다. 그것만으로 Observer Gate PASS를 부여하지 않았다.

## 실제 관측 범위만 사용한 진단

세 GPU의 실제 Collector Anchor가 겹치는 구간만 분석했다. 관측하지 않은 시간은 Idle로 채우지 않았다.
Central Arrival Window도 이 실제 공통 구간으로 제한했다. 이는 전체 실행의 Busy가 아니다.

| ROI | Duration s | Stage | Kernel Busy % | Observed Waiting Mean | Waiting Max | Scheduled Requests Max |
|---|---:|---|---:|---:|---:|---:|
| observed_common_ROI | 16.933630 | encoder | 76.9893 | 1.972222 | 15 | 10 |
| observed_common_ROI | 16.933630 | prefill | 73.1256 | 0.418605 | 9 | 6 |
| observed_common_ROI | 16.933630 | decode | 49.9453 | 0.028019 | 2 | 3 |
| central_arrival_window | 12.675497 | encoder | 73.8675 | 1.235294 | 12 | 10 |
| central_arrival_window | 12.675497 | prefill | 76.7569 | 0.268657 | 7 | 6 |
| central_arrival_window | 12.675497 | decode | 48.7706 | 0.029570 | 2 | 3 |

Waiting Mean은 관측별 평균이다. 정확한 시간 가중 Queue 평균이 아니다.
Scheduled Requests는 요청 수다. E의 이미지 수 또는 모든 GPU의 동일 Batch를 뜻하지 않는다.
E와 P 모두 Waiting이 관측됐다. 이 값으로 E 비병목을 확정하지 않는다.
독립 Capacity Reference와 Service Proximity는 미측정이다. P Compute 포화도 이 Trace로 확정하지 않는다.
Encode-safe / Decode-safe Window는 여전히 미검증이다. 낮은 Kernel Busy를 안전한 삽입 가능성으로 바꾸지 않았다.

## 재현과 검증

완료 경계, Collector 역순 종료와 관측 범위 교집합의 CPU 검사를 추가했다.
실제 CPC 제어 경로의 CPU 통합 검사도 요청 100개×3회와 동일 요청 순서 유지를 확인했다.
CPU Fixture의 가상 Metric은 성능 Raw에 포함하지 않았다.
원시 Trace는 `raw/profiles/ec_ROI_CUPTI_C27_CPC_q3_v0/`에 있다.
계측 비교, 완료 경계와 실제 ROI 분석은 대응 Qualification / Completion ROI / Idle Analysis JSON에 있다.
Figure와 Table은 `figures/short_capture_C27_*_v0.{png,pdf,csv}`다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
