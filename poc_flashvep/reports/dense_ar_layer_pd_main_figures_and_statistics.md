# Main Figure와 통계의 증거 범위

## 독립 확인 Figure

다음 Figure는 동결된 Operating Point의 독립 Server Start를 사용한다.
각 방법의 실제 Config와 결과는 Evidence의 CSV 및 JSON에 연결한다.
다른 QPS의 점을 임의의 연속 Curve로 보간하지 않는다.
전체 Frozen Four-arm QPS Grid를 실행하지 않았다면 그 한계를 남긴다.

| Figure | 내용 | 해석 |
|---|---|---|
| [1](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_1_TTFT_p95_ms.svg) | Offered QPS와 TTFT p95 | 독립 Start 평균과 경험적 95% CI |
| [2](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_2_TPOT_p95_ms.svg) | Offered QPS와 TPOT p95 | Output16만 비교 |
| [3](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_3_raw_completed_output_tokens_s.svg) | Raw Completed Output tokens/s | SLO 통과 여부와 별개 |
| [4](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_4_SLO_qualified_throughput.svg) | Balanced/Decode 유효 출력 처리량 | 속이 찬 점은 3회 모두 적격 |
| [5](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_5_confirmed_capacity_points.svg) | 선택한 Tested Capacity Point | 사선 막대는 확인 기준 미통과 |
| [6](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_6_actual_Dynamic_cuts.svg) | 실제 Last0/2/4/8/12 분포 | Measurement-arrival Cohort의 완료 Work |

Figure 5는 정확한 전역 최대를 뜻하지 않는다.
성공하지 않은 점도 그림에서 삭제하지 않는다.
선택한 QPS가 방법마다 다르면 같은 Arrival Rate의 Latency 비교가 아니다.
Native-relative 비열등성은 같은 QPS의 Pair에서 별도로 계산한다.

## Controlled Discovery Figure

- [7 — Bursty/Mixed Output](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_7_bursty_and_mixed_output.svg)는 각 Trace의 한 Finite Cohort를 비교한다.
- [8 — Static/Dynamic Pareto](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_8_static_Dynamic_Pareto.svg)는 N128 Matched-budget Discovery다.

한 Start의 그림에 독립 확인 CI를 붙이지 않는다.
Figure 8의 실제 Batch Geometry는 완전히 같지 않다.
따라서 이 그림만으로 독립적인 Dynamic 정책 이득을 주장하지 않는다.
실제 원문의 입력은 유지했다.
Controlled Output Cap과 인공적인 Bursty Arrival은 실제 Production Trace와 구분한다.

## P/D Timeline

[Matched Native/Dynamic Timeline](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_Host_P_D_timeline.svg)은 실제 Host 관측을 표시한다.
두 실행은 같은 QPS와 Arrival Seed를 사용한다.
선택 규칙은 첫 Dynamic 확인 실행의 180초 이후 첫 Migration Request다.
그 요청과 다음 5개 Request ID를 함께 표시한다.
빠른 요청을 골라 대표 Timeline으로 사용하지 않는다.

회색은 Admission을 포함한 P 대기다.
주황색은 최초 P Scheduling부터 Source-ready까지다.
파란색은 Handoff, D Admission과 Import의 관측 구간이다.
분홍색은 Import-ready부터 첫 Output까지다.
초록색은 첫 Token부터 마지막 Token 전달까지다.
수직 표시는 Source-ready다.
별 표시는 첫 Output이다.

이 구간들은 Exclusive GPU Kernel Time이 아니다.
Queue, Host Submission, Compute, 전송과 Delivery가 같은 Enclosure에 포함될 수 있다.
파란색 길이를 Wire Time으로 해석하지 않는다.
분홍색 길이를 순수 Suffix Kernel Time으로 해석하지 않는다.
여러 Request의 겹치는 Bar를 더해서 GPU Busy를 계산하지 않는다.

Physical Kernel Union, 실제 Kernel Idle과 SM/HBM Counter는 `NOT_MEASURED`다.
이번 캠페인에서는 새 CUPTI/Nsight Capture로 Clean Idle을 검증하지 않았다.
이전 캠페인의 Collector 실패를 이번의 새 측정 실패로 다시 집계하지 않는다.
NVML Utilization의 보수값도 Physical Idle로 사용하지 않는다.

## 통계 단위와 Metric 경계

Bootstrap 단위는 독립 Server Start다.
3개 Pair의 모든 경험적 재표본 27개를 사용한다.
Request와 Token은 독립 Start가 아니다.
따라서 수만 개 Token으로 표본 수를 부풀리지 않는다.

확인한 모든 실행을 통계에 남긴다.
실패한 Start를 제외해서 CI를 개선하지 않는다.
CI는 작은 표본의 경험적 Percentile Interval이다.
Production Guarantee나 WCET로 표현하지 않는다.

주요 ITL p99에는 첫 Interval을 포함한다.
First ITL, Steady ITL, Maximum ITL과 Threshold Exceedance도 별도로 저장한다.
Worker Emission Interval도 Host와 GPU Scheduling을 포함한다.
이를 Kernel Runtime으로 해석하지 않는다.
Coordinator Interval은 Worker Interval과 Receive-lag 변화의 합이다.
이 Identity는 Delivery 지연의 구체적인 원인을 자동으로 식별하지 않는다.

## 재현 파일

통계 코드는 [final_analysis.py](../dense_ar_slo_layer_pd/final_analysis.py)다.
Host Timeline 코드는 [final_visuals.py](../dense_ar_slo_layer_pd/final_visuals.py)다.
Discovery 산점도는 [figures.py](../dense_ar_slo_layer_pd/figures.py)로 생성한다.
원시 자료, Immutable Config와 Source Archive는 같은 Evidence Namespace에 보존한다.
최종 수치와 판정은 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 기록한다.

## 보충 Capacity 확인과 Figure 9

[Figure 9](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_9_confirmed_capacity_with_backoff.svg)는 별도로 선언한 낮은 Dynamic Rate도 포함한다.
원래 선택한 QPS4.8의 실패는 Figure 1~6과 원래 통계에 그대로 남는다.
보충 Dynamic QPS4.4는 새 Arrival Seed의 독립 Start 3회를 사용한다.
Prompt Pool과 Runtime·정책·SLO는 변경하지 않았다.

보충 Dynamic과 Colocated는 같은 Arrival Seed가 아니다.
각 3 Start를 따로 재표본화한 경험적 Bootstrap 729개를 사용한다.
이 비율을 Paired CI로 표시하지 않는다.
Dynamic/Colocated Qualified 비율은 0.888988 [0.872953, 0.906359]다.
이는 같은 SLO에서 각자의 확인된 Operating Point를 비교한 값이다.
같은 Offered QPS의 Latency 비교나 전역 Maximum의 증명은 아니다.

보충 처리 코드는 [confirmed_capacity_analysis.py](dense_ar_slo_throughput_dynamic_layer_pd_20261114/scripts/confirmed_capacity_analysis.py)다.
전체 분위수와 CI는 [확인점 통계](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/confirmed_capacity_with_backoff.json)에 있다.
Native처럼 3 Start 확인이 없는 방법은 `NOT_ESTABLISHED`로 표시한다.
이 표기를 0 tokens/s로 해석하지 않는다.
