# Sustainable Generation Throughput의 정의와 검증

## 판정 범위

이번 연구는 최대 지속 가능한 SLO-qualified Output tokens/s를 목표로 한다.
실험 결과에서는 가장 높은 적격 Tested Point를 보고한다.
희소한 QPS Grid로 정확한 전역 최대를 주장하지 않는다.
600초 또는 1200초의 Queue 안정성도 무한 시간의 수학적 안정성 증명은 아니다.

모델은 Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.
Primary Workload는 실제 공개 원문의 Input8000/Output16이다.
GPU0/1, 원본 BF16, 42 GiB KV/GPU와 Full36 Decode를 유지한다.
최종 방법별 설정은 Confirmation 전에 동결한다.

## 서로 다른 처리량 정의

| 지표 | 분자 | 분모 |
|---|---|---|
| Raw Completed Output tokens/s | 측정 구간에 완료한 요청의 모든 출력 Token | 고정 Measurement Seconds |
| Completed Requests/s | 같은 구간의 완료 요청 수 | 같은 Measurement Seconds |
| SLO-qualified Output tokens/s | 같은 완료 요청 중 Joint SLO를 통과한 출력 Token | 같은 Measurement Seconds |
| 실제 Token Event Rate | 구간 안에서 실제 관측한 Token 증가량 | 같은 Measurement Seconds |
| Finite-cohort Drain Rate | 전체 유한 Cohort의 완료 출력 Token | 최초 Arrival부터 마지막 완료까지 |

Raw Completed Output은 Warm-up에 도착한 요청도 포함한다.
실제 Token Event Rate는 경계 효과를 확인하는 별도 진단값이다.
두 수치를 섞지 않는다.
입력 Prefill Token은 Output Throughput에 포함하지 않는다.
Drain Rate를 Sustainable Capacity로 표현하지 않는다.

## 시간 구간과 미완료 요청

주요 장기 실행은 Warm-up 60초와 Measurement 600초를 사용한다.
Measurement 이후에도 90초 동안 요청을 계속 보낸다.
그다음 Arrival을 중단하고 모든 요청을 추적한다.
Persistence는 같은 구분에서 Measurement를 1200초로 늘린다.

측정 중 도착한 모든 요청은 별도의 Arrival Cohort를 이룬다.
그 요청이 Measurement 이후에 완료해도 Latency와 SLO 평가에 남긴다.
누락되거나 미완료인 요청은 성공 요청의 분모에서 제거하지 않는다.
Timeout과 Worker 실패도 별도로 보존한다.

Throughput은 Completion Window에서 계산한다.
TTFT/ITL/TPOT/E2E는 Measurement-arrival Cohort에서 계산한다.
Migration은 이 Cohort의 실제 완료 Work와 전체 Run의 Work를 구분한다.
CSV는 두 분모를 모두 제공한다.

## 사전 안정성 Gate

다음 조건을 모두 검사한다.

1. Completion Rate가 실제 Arrival Rate의 98% 이상이다.
2. Outstanding Queue Slope가 0.01 requests/s 이하다.
3. 해당 Slope의 경험적 상한이 0.025 requests/s 이하다.
4. 마지막 Outstanding 수의 증가가 `max(4, Measurement Arrivals의 1%)` 이하다.
5. Completion Window와 Arrival Cohort의 Joint SLO Attainment가 각각 95% 이상이다.
6. Output Exactness, 정상 Worker 종료와 KV/Hidden Lifetime이 통과한다.
7. 분리하지 못한 Coalesced Token Interval이 없다.

Queue 불확실성은 30초 Block을 사용한다.
인접한 두 Block의 Residual을 재표본화한다.
이것은 유한 Window의 경험적 판정이다.
Production Stability Guarantee로 해석하지 않는다.
임계값을 실패한 실행에 맞춰 변경하지 않는다.

코드는 [sustainable.py](../dense_ar_slo_layer_pd/sustainable.py)에 있다.
최종 Window 근거는 Evidence의 `analysis/*_sustainable_v2.json`이다.
전체 Run의 `*_metrics.json`과 Measurement Cohort의 분위수는 다를 수 있다.

## 독립 확인과 Confidence Interval

같은 방법과 Operating Point에서 독립 Server Start 3회를 사용한다.
Arrival Seed는 2026111511/2026111512/2026111513이다.
방법 순서는 Seed별로 회전하거나 뒤집는다.
평가 전 Source, Cost Model, Dataset, Config와 SLO Hash를 동결한다.

Start를 Bootstrap의 단위로 사용한다.
Token이나 Request를 독립 Start처럼 재표본화하지 않는다.
3개 Pair의 가능한 경험적 Bootstrap Sample은 27개다.
95% Percentile CI는 이 작은 표본의 경험적 구간이다.
이를 서비스 보장으로 표현하지 않는다.

방법별 Capacity 비교는 각자의 선택된 QPS를 사용한다.
Native-relative Decode 비열등성은 같은 QPS와 Arrival Trace에서 따로 검사한다.
Dynamic 운영 QPS의 Native가 과부하라면 이를 계속 추적한다.
그 Native 실행을 지속 가능한 점으로 집계하지 않는다.

ITL p99와 TPOT p95의 Ratio CI 상한이 각각 1.05 이하여야 비열등성 통과다.
Absolute SLO 통과만으로 이 조건을 통과했다고 표현하지 않는다.
한 Start의 좋은 Point Estimate만으로 GO를 판정하지 않는다.

## 현재 증거의 경계

Physical Kernel Union, 직접 측정한 D Idle과 Spatial Compute Counter는 `NOT_MEASURED`다.
Host Queue, 실제 완료율과 Lifetime은 별도로 관측했다.
CUDA Event Enclosure는 단독 Calibration에만 사용했다.
Event Enclosure를 Exclusive Kernel Time이나 물리적 Idle로 바꾸어 표시하지 않는다.

최종 Table과 판정은 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 기록한다.
적격점이 확인되지 않은 경우 해당 Capacity는 `NOT_ESTABLISHED`다.
미실행 값이나 미확정 값을 0으로 대체하지 않는다.

## 최종 확인점

Decode-priority에서 Colocated QPS5.0은 독립 Start 3회와 별도 1200초 Persistence를 통과했다.
Static Last12/D4 QPS4.4도 3회 통과했다.
원래 선택한 Dynamic QPS4.8은 3회 모두 실패했다.
실패 후 Source·정책·SLO를 변경하지 않은 낮은 QPS4.4를 사전에 추가 선언했다.
새 Arrival Seed의 보충 확인 3회는 통과했다.

최대 확인 Tested Point의 Qualified 평균은 C 80.071111, Static 70.417778, Dynamic 71.182222 tokens/s다.
Native QPS3.2는 1/3만 이 보수적 Queue Screen을 통과했다.
따라서 Native의 최대 확인 Capacity는 `NOT_ESTABLISHED`다.
실패한 두 Native Run은 끝 Queue가 감소했다.
그 결과를 무한 Queue 증가의 증명으로 사용하지 않는다.

보충 Dynamic과 Colocated의 비율은 0.888988 [0.872953, 0.906359]다.
다른 Seed의 독립 Start를 각각 재표본화했다.
서로 다른 QPS의 최대 확인점 비교이며, Paired 또는 같은 Arrival Rate의 Latency 비교가 아니다.
전체 그룹과 CI는 [보충 포함 통계](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/confirmed_capacity_with_backoff.json)에 있다.
