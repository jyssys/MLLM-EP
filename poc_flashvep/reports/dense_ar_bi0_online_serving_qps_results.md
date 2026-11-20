# BI0 ShortAnswer200 Online Serving — 기본 비교

## 측정 계약

`MEASURED_ONLINE`. Dataset은 `LongBench-ShortAnswer200-v2`다. Qwen3-8B Original BF16과 Natural EOS를 사용했다. Model Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다. Dataset SHA256은 `8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6`다.

GPU0/1의 H100 SXM 80GB 두 대를 사용했다. 모든 Engine은 B8192/S1024와 KV42GiB를 사용했다. 이 표의 KV Block은16이다. Prefix Cache는 비활성화했다. Chunked Prefill과 Async Scheduling을 활성화했다. 일반 Compile3 및 CUDA Graph 경로를 사용했다. 실제 Worker 설정과 Source ZIP SHA256은 각 Raw Job에 저장했다.

Agg는 전체 모델 Replica 두 개를 사용했다. 요청은 Round-robin으로 배정했다. Native는 공식 vLLM NIXL Connector의1P1D다. Custom Layer/Token Adapter는 Native에 설치하지 않았다. 두 방식의 같은 Seed는 같은 요청 순서와 Poisson Arrival Trace를 사용했다.

Primary 시간 경계는 예정 Arrival부터 Coordinator Token Delivery까지다. HTTP 경계가 아니다. Mean TBT는 실제 Client-visible Token Interval의 평균이다. ITL p99는 전체 유효 Interval의 p99다. TPOT p95는 요청별 `(마지막 Token−첫 Token)/(Output Tokens−1)`의 p95다. Output이1 Token인 요청의 TPOT는 정의하지 않는다.

SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms다. Main 결과를 보고 Threshold를 변경하지 않았다. Goodput은 이 세 조건을 모두 통과한 요청의 실제 Output Tokens/s다. 짧은 요청과 긴 요청의 Token 수가 다르므로 SLO Attainment와 Goodput은 비례하지 않는다.

## 기본 결과

| Offered QPS | 방식 | 독립 Start | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | Raw Output Tokens/s | SLO-qualified Tokens/s | Joint SLO |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0.4|Agg DP2|1|1630.12|9.51|8.37|9.05|8.67|95.5%|
|0.4|Native PD|1|2327.87|9.11|8.16|9.08|8.28|96.5%|
|1.2|Agg DP2|1|1706.90|286.39|209.07|26.74|19.08|77.0%|
|1.2|Native PD|1|7193.49|13.02|8.69|26.99|16.63|78.5%|
|2.35|Agg DP2|3|4622.66|389.45|292.61|55.06|15.97|40.83%|
|2.35|Native PD|3|22370.22|51.09|10.39|44.86|0.63|5.17%|

각 Cell은 모든 Start에서 동결된 Full200 Task-quality CI 검사를 통과했다. 이는 모든 Greedy Token이 C1 Reference와 같다는 뜻이 아니다. BI0의 다른 Batch Geometry는 수치적으로 다른 출력을 만들 수 있다. Task-quality Margin과 Strict Same-geometry 검사는 별도 계약이다.

QPS2.35에서 Agg의 TTFT p95 평균95% CI는[3334.47,5838.93]ms다. Native는[20742.13,24551.60]ms다. ITL p99의 CI는 각각[376.79,399.67]ms와[44.76,56.13]ms다. TPOT p95의 CI는 각각[290.23,294.71]ms와[9.85,10.91]ms다.

QPS2.35에서 Agg/Native Raw Throughput 비율은1.2274다. Paired-start95% CI는[1.1817,1.2822]다. Agg/Native ITL p99 비율은7.6229다. CI는[7.1207,8.4309]다. Mean TBT 비율은4.7777다. CI는[4.5790,4.9569]다. 이 값들은 전체 요청을 Drain한 Finite-cohort 결과다. Sustainable Capacity가 아니다.

CI는 독립 Server Start를 Resampling했다. QPS2.35의 Seed는2026112051/52/53이다. Start3개에서 가능한27개 Bootstrap 조합을 사용했다. 낮은 두 QPS는 Start1회이므로 성능 CI를 제시하지 않는다. 요청200개나 Token을 독립 Server Start로 세지 않았다.

## 실제 Batch와 Queue

대표 QPS2.35/Seed2026112051에서 Native P의 실제 Prefill Request Batch는1/2/3이었다. Iteration 수는 각각205/157/20이다. D Decode Request Batch는1/2/3이었다. 해당 Decode 포함 Iteration 수는3073/772/43이다. B8192/S1024는 실제 Batch가 아니다.

같은 실행에서 Native P Waiting Queue의 Host Sample 평균은26.97 Requests였다. 최대는46이었다. D Decode-ready Waiting은0이었다. D Remote-KV Waiting은 평균0.368, 최대3이었다. 이 Queue 통계는 불규칙한 Host Sampling이다. GPU 시간 가중 평균이나 Physical Idle은 아니다.

Agg Replica0/1의 Mixed Prefill–Decode Iteration은 각각146/131이었다. Prefill과 Decode는 정상적인 Continuous Batching에서 함께 실행됐다. Agg를 Cohort-only 실행으로 약화하지 않았다. Native의 높은 TTFT와 Agg의 긴 Decode Tail은 이 실행 계약에서 함께 관측됐다. 이 상관관계만으로 Kernel별 인과 비중을 확정하지 않는다.

## 범위와 제한

위 결과는 기본 KV Block16 비교다. NIXL Descriptor 감소를 위한 공통 KV Block128 비교는 별도 Family로 측정한다. Compile0/Actual Batch1의 Layer/Token Mechanism Control도 별도 Family다. 이 결과들을 하나의 동일 실행 계약으로 합치지 않는다.

이 Dataset은 고정 Input8000/Output200이 아니다. 따라서 DuetServe Figure2의 수치 재현으로 부르지 않는다. 실제 가변 길이 Short-answer Dataset에 같은 비교 방법을 적용한 결과다. 측정하지 않은 QPS와 Static/Dynamic 결과를 이 표에 채우지 않는다.

원시 통계는 `dense_ar_bi0_shortanswer200_20261120/analysis/BI0_continuous_baseline_statistics_v3.json`에 저장했다. 각 Job의 Metrics, Quality Gate, 실제 Batch, Config, Command와 Source Hash도 같은 Evidence Namespace에 저장했다.
## Shared KV128의 Held-out 고정 후보 비교

QPS2.35에서 Seed2026112051/52/53을 각각 새 Server Start로 실행했다. Discovery에서 두 Axis 모두 Cut8을 선정했다. 이 선택은 Held-out 전에 동결했다. 아래 값은 Start별 통계의 평균이다. Natural EOS의 실제 Output 수는 Method마다 다르다.

| Method | TTFT p95 | Mean TBT | ITL p99 | TPOT p95 | Raw Tokens/s | Good Tokens/s | Completed Requests/s | Joint SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Continuous Native C3|20068.02ms|7.517ms|16.443ms|8.093ms|46.251|0.943|1.937|5.67%|
|Agg DP2 C3|4209.34ms|38.394ms|372.783ms|283.808ms|55.036|16.085|2.320|42.50%|
|Controlled Native C0|24679.78ms|7.481ms|8.969ms|8.196ms|42.826|0.750|1.829|5.33%|
|Static Layer Last8 C0|12005.69ms|7.523ms|9.040ms|8.247ms|52.323|17.347|2.178|33.17%|
|Static Token L×8/36 C0|9603.00ms|7.494ms|8.974ms|8.208ms|53.547|23.393|2.237|44.67%|

모든 Method는 B8192/S1024와 KV42GiB/Block128을 사용했다. C3는 일반 Continuous Scheduler다. C0의 세 PD Control은 Actual Batch1/1/1과 Source Credit4를 사용했다. Agg까지 Actual Batch가 같다고 주장하지 않는다. 모든15개 Start는 Full200 Task-quality CI를 통과했다.

| 비교: Layer/Token | 비율 | Paired-start95% CI |
|---|---:|---:|
|TTFT p95|1.2502|[1.1871,1.3448]|
|ITL p99|1.0074|[1.0051,1.0103]|
|TPOT p95|1.0047|[1.0024,1.0069]|
|Raw Output Tokens/s|0.9772|[0.9522,0.9994]|
|SLO-qualified Output Tokens/s|0.7416|[0.5128,0.8233]|

Layer의 D 계산 비용 이점이 Best Static Token 대비 Online 우위로 이어지지 않았다. 두 방법 모두 Native-relative Decode5% Gate를 통과했다. Continuous Native 대비 Layer의 ITL/TPOT CI 상한은0.6043/1.0230이다. Token은0.6005/1.0162다.

Agg 대비 Goodput CI는 Layer[0.4744,1.5241], Token[0.8851,1.9147]이다. 점추정의 우위를 통계적으로 확정하지 않는다. 또한 이 값은 Finite-cohort Goodput이다. Maximum Sustainable SLO-qualified Throughput이 아니다.

Source Hash, 실제 GPU UUID, Trace의 일치 검사는 `analysis/BI0_shared_KV128_confirmation_statistics_v2.json`에 있다. 각 Start의 전체 Provenance는 `analysis/BI0_main_table_with_provenance.json`과 CSV에 있다. Confidence Interval은 Request나 Token이 아니라 독립 Start를 Resampling했다.

## 추가 QPS1.65: 독립 Start1회

Shared KV128과 같은 Frozen Cut8을 사용했다. Seed는2026112051이다. 추가 QPS는 사전에 동결한 남은 시간 규칙으로 선택했다. 성능 결과로 QPS를 선택하지 않았다. 각 Cell은1회이므로 성능 CI는 계산하지 않는다.

| Method | TTFT p95 | ITL p99 | TPOT p95 | Raw Tokens/s | Good Tokens/s | Joint SLO |
|---|---:|---:|---:|---:|---:|---:|
|Continuous Native C3|11274.96ms|16.917ms|8.132ms|37.050|26.141|67.0%|
|Agg DP2 C3|2143.17ms|321.899ms|250.457ms|36.993|22.540|66.0%|
|Controlled Native C0|12951.13ms|8.885ms|8.140ms|36.543|18.459|50.5%|
|Layer Last8 C0|7153.97ms|9.043ms|8.213ms|37.496|31.112|79.0%|
|Token Suffix≈L×8/36 C0|5700.44ms|9.036ms|8.218ms|37.389|33.969|90.0%|

Token의 더 낮은 TTFT 방향은 QPS2.35와 같았다. 이1회 확인을 새로운 QPS의 통계적 재현으로 표현하지 않는다. Full200 Task-quality Gate도 각 Arm에서 통과했다. 같은 KV128의 QPS1.2 전체5개 Arm과 Cut12의 독립 반복은 시간 규칙에 따라 `NOT_RUN`으로 남겼다.

## 동일 Output 작업량의 진단

`DIAGNOSTIC`이다. Seed2026112062, QPS2.35, Actual Batch1/1/1과 같은 KV128을 사용했다. 각 요청의 C1 Reference Output Length로 제한했다. EOS는 무시했다. 세 Arm 모두 정확히4723Tokens를 생성했다. 각 Arm은 독립 Start1회다. Natural Generation의 Task-quality 비교를 대체하지 않는다.

| Method | TTFT p95 | ITL p99 | TPOT p95 | Raw Tokens/s | Good Tokens/s |
|---|---:|---:|---:|---:|---:|
|Controlled Native|17233.39ms|8.916ms|8.101ms|43.450|0.865|
|Layer Last8|5784.64ms|9.001ms|8.100ms|49.697|44.783|
|Token Suffix≈L×8/36|4773.52ms|9.044ms|8.219ms|50.033|48.762|

Output Token 수를 같게 한 진단에서도 Token의 TTFT가 낮았다. 따라서 Natural Output 길이 차이 하나만으로 Primary의 순위를 설명할 수는 없다. 이는1회 Control이다. 모든 Kernel·Queue 원인을 식별한 결과는 아니다.

근거는 `analysis/BI0_static_QPS_extension_v1.json`과 `analysis/BI0_equal_output_work_control_v1.json`이다. Scheduled Arrival Rate, 실제 완료율, Task별 결과와 각 Request Metric도 보존했다.
