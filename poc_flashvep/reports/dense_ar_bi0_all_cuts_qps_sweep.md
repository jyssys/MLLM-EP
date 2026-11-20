# BI0 전체 Cut·QPS 비교

상태: `CLOSED / HOLD`. 완료한 Cell과 미실행 Cell을 구분한다.
기준 Dataset은 LongBench-ShortAnswer200-v2다.
Dataset SHA256은 `8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6`다.

## 실험 계약

Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16, BI0, Exact Causal Attention과 Full36 Decode를 유지한다.
GPU0/1의 H100 SXM80GB 두 대를 사용한다. KV Budget은 각42GiB다.
Scheduler는 B8192/S1024다. 실제 P/D Suffix/Decode Batch는1/1/1이다.
Primary는 Source Credit4, Hidden Source Slots4와 KV Block128이다.
Natural EOS와 Output Cap2048을 사용한다. Prefix Cache는 사용하지 않는다.

Frozen SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms다.
Relative Decode Gate는 Native 대비 ITL p99와 TPOT p95의 Paired95% CI 상한≤1.05다.
Absolute SLO와 Relative Gate를 별도로 보고한다.

## Discovery와 Confirmation

원래 요청한 전체200개×전체 Grid는 이 예산에서 모두 실행하지 않았다.
사용자가 Serial/Online 인과 분석을 최우선으로 추가한 뒤 Discovery를 N50으로 분리했다.
N50은 Task별10개를 Seed2084로 사전 선정했다. 원래 순서를 유지했다.
결과나 Output 길이로 샘플을 선택하지 않았다.

Discovery Grid는 QPS0.4/0.7/1.0/1.2/1.65/2.0/2.35다.
Cut은 Last2/4/8/12다. 각 Cell은 독립 Start1회다.
N50의 작은 차이를 통계적으로 확정한 Winner라고 표시하지 않는다.

실제로 완료한 Discovery는 QPS0.4/1.2/2.35의27개 Method Cell이다.
나머지 QPS0.7/1.0/1.65/2.0의36개 Cell은 `NOT_RUN`이다.
각 QPS는 Native1개와 Cut4개×Axis2개의9개 Method를 포함한다.
N50의27개 Cell은 모두 사전 Task-quality CI Gate를 통과하지 못했다.
특히 Task당10개에서는 NarrativeQA/QMSum의 CI가 −2점 Margin보다 낮아질 수 있었다.
이는 Token Count 누락이나 KV Lifetime 실패를 뜻하지 않는다.
그러나 해당 점들을 적격 Primary 성능으로 승격하지 않는다.
모든 Curve는 Hollow Marker와 Discovery 경고를 사용한다.
Quality Margin을 사후 완화하거나 유리한 Request로 교체하지 않았다.

Full200 Confirmation은 QPS2.35에서 모든 Cut을 대상으로 한다.
Arrival Seed2071/2072/2073과 각 Axis의 독립 Start3회를 사용한다.
Last12는 이번 캠페인 초기에 실행한 신규3회 결과를 재사용한다.
이전 Discovery의 유리한 Start를 Confirmation에 합치지 않는다.

## 완료한 Full200 독립3회 Confirmation

다음 값은 Start별 Metric의 평균이다. Request600개를 하나로 합친 p95가 아니다.
Actual Arrival Rate의 Start 평균은2.324886 requests/s였다.
모든 Axis의 Task-quality CI Gate와 실제 Batch1/1/1 검사를 통과했다.

| Method | TTFT p95 | ITL p99 | TPOT p95 | Raw Output tok/s | SLO-goodput tok/s | Joint SLO |
|---|---:|---:|---:|---:|---:|---:|
| Native | 22.460s | 8.922ms | 8.158ms | 43.180 | 1.155 | 6.50% |
| Layer2 | 18.963s | 8.942ms | 8.199ms | 45.964 | 1.241 | 6.67% |
| Token2 | 16.425s | 8.982ms | 8.208ms | 47.234 | 1.284 | 6.83% |
| Layer4 | 14.015s | 9.004ms | 8.238ms | 48.912 | 3.229 | 10.33% |
| Token4 | 11.128s | 9.077ms | 8.216ms | 50.670 | 17.302 | 32.67% |
| Layer8 | 8.532s | 9.023ms | 8.236ms | 52.485 | 30.437 | 54.00% |
| Token8 | 6.903s | 8.963ms | 8.216ms | 52.990 | 38.736 | 70.83% |
| Layer12 | 7.609s | 9.003ms | 8.238ms | 52.395 | 34.645 | 64.33% |
| Token12 | 7.929s | 9.014ms | 8.203ms | 52.074 | 35.863 | 68.33% |

Native의 독립3회 Reference는 모든 Cut에서 공유한다. 반복12회로 계산하지 않는다.
첫 Native Seed71의 초기 Snapshot은 Buffered Tracing 이전 Source였다.
Source 계약 때문에 같은 Trace의 Buffered 버전으로 교체했다. 성능으로 선택하지 않았다.
원래 실행도 Raw Evidence에 보존했다.

비율은 Layer/Token이다. 같은 Arrival Seed를 묶어 Start-level Bootstrap을 계산했다.
3개 Start의27개 재표본을 모두 열거했다.95% 경험적 CI의 Tail 정밀도는 제한적이다.

| Cut | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | TTFT 판정 |
|---|---|---|---|
| Last2 | 1.154544 [1.073808, 1.240517] | 0.966456 [0.930078, 0.996872] | Token |
| Last4 | 1.259430 [1.148548, 1.412117] | 0.186642 [0.090714, 0.285802] | Token |
| Last8 | 1.235989 [1.139479, 1.312967] | 0.785769 [0.743145, 0.874830] | Token |
| Last12 | 0.959758 [0.884393, 1.068346] | 0.966035 [0.831489, 1.043561] | 불확실 |

두 Axis의 Native-relative Decode5% Gate는 모든 Cut에서 통과했다.
그러나 Layer의 추가 TTFT·Goodput 우위는 확인하지 못했다.
Last12의 낮은 TTFT p95 Point와 더 높은 Goodput은 같은 주장이 아니다.
Output이 가변이므로 Joint SLO와 SLO-qualified Output tok/s도 비례하지 않는다.
모든 조건의 Joint SLO 평균은95% 미만이다.
이 결과를 Stable Queue의 Maximum Sustainable Goodput으로 표시하지 않는다.

## 같은 Confirmation의 Host 경로

다음 값도 독립3회에서 계산한 Request 평균의 평균이다.
P Service는 Coordinator Dispatch→Parent Ready다.
D First Service는 D Submit→첫 Client Token이다.
P Queue는 Arrival→첫 P Prepare다. GPU 시작 시각이 아니다.

| Cut | P Service Layer/Token | D First Service Layer/Token | P Queue Layer/Token | Ready→D Prepare Layer/Token |
|---|---:|---:|---:|---:|
| Last2 | 517.761/502.792ms | 80.248/94.212ms | 11326.523/9863.820ms | 177.550/199.958ms |
| Last4 | 485.449/464.188ms | 110.491/124.991ms | 8143.170/6037.874ms | 237.075/276.267ms |
| Last8 | 430.540/394.101ms | 162.289/180.053ms | 4046.810/2843.211ms | 373.949/435.716ms |
| Last12 | 366.310/325.315ms | 218.599/242.021ms | 2650.692/2324.825ms | 618.621/795.034ms |

Layer의 D First Host 경로는 모든 Cut에서 짧았다.
반대로 P Host Service와 P Queue는 모든 Cut에서 길었다.
따라서 짧은 D Forward/Host 경로만으로 낮은 Online TTFT를 보장하지 않았다.
이 표의 Host 비용을 Exclusive GPU GEMM·Attention 시간으로 바꾸지 않는다.
원래 Last4/Seed2061의2711ms 분해와 새 Seed3개의 평균을 하나의 식에 합산하지 않는다.

## 증거와 해석

[Discovery 전체 Cell JSON](dense_ar_bi0_axis_rootcause_20261120/analysis/pilot50_all_cuts_qps_v1.json),
[독립 반복 통계](dense_ar_bi0_axis_rootcause_20261120/analysis/full200_confirmation_statistics.json)를 참조한다.
각 Raw Directory는 Config, Source ZIP, GPU UUID와 Request별 Metric을 보존한다.
Aggregate GEMM Rows는 near-matched다. 정확한 오차는 별도 Work Audit에 있다.
Finite-cohort Drain Throughput은 Sustainable Capacity가 아니다.
BI0의 Query Geometry 차이는 Natural Output을 바꿀 수 있다.
같은 Geometry의 State·Greedy Gate와 전체200개의 Task-quality CI를 구분한다.
이번 Cut별 비교를 Best-vs-Best 전역 튜닝 결과라고 표시하지 않는다.
Confirmation 성능을 보고 Static Cut을 다시 고른 뒤 Held-out 결과라고 부르지 않는다.
