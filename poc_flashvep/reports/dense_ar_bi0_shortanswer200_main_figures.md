# BI0 ShortAnswer200 Figure 목록과 증거 범위

모든 Figure는 `poc_flashvep/reports/dense_ar_bi0_shortanswer200_20261120/figures/`에 PNG/PDF로 저장했다. 기본 Figure 재생성 명령은 `python -m poc_flashvep.dense_ar_bi0_shortanswer_pd.figures`다. Shared128와 Axis Figure는 `python -m poc_flashvep.dense_ar_bi0_shortanswer_pd.serving_figures`로 생성한다. 실제 Python 경로와 의존성은 Evidence의 Environment 및 Command에 저장했다.

## 기본 Online Figure

다음 독립 Figure는 Agg DP2와 Native PD를 비교한다.

1. `poc_flashvep/bi0_shortanswer200_TTFT_p95_vs_qps.png` 및 PDF.
2. `poc_flashvep/bi0_shortanswer200_ITL_p99_vs_qps.png` 및 PDF.
3. `poc_flashvep/bi0_shortanswer200_TPOT_p95_vs_qps.png` 및 PDF.
4. `poc_flashvep/bi0_shortanswer200_output_throughput_vs_qps.png` 및 PDF.

기존4-Panel 구성도 유지했다. `poc_flashvep/bi0_shortanswer200_four_metrics_confirmed.png`와 PDF다. Panel은 Joint SLO, Mean TTFT, Mean TBT, Generated Output Tokens/s다. 이 Figure에 Layer/Token이나 Dynamic 결과를 넣었다고 주장하지 않는다.

기본 공통 설정은 BI0, B8192/S1024, KV Block16, Natural EOS다. X축은 Offered QPS0.4/1.2/2.35다. 낮은 두 점은 각각 독립 Start1회다. QPS2.35는 Start3회다. Start3회인 점만 경험적95% CI를 표시한다. CI는 요청이나 Token의 Resampling이 아니다.

선은 측정점의 위치를 연결한다. 그 사이 QPS를 측정했다는 뜻이 아니다. 측정하지 않은 Static/Dynamic Cell과 QPS는 임의로 채우지 않는다. Raw Output Throughput은 Finite Cohort의 Arrival→Drain 범위다. Sustainable Capacity가 아니다.

## Dataset과 Mechanism Figure

Input/Output Distribution과 Joint Scatter는 실제 동결200개와 BI0 Natural Generation을 사용한다. Dataset Output은 정답 Text 길이가 아니다. Qasper는 Output 기준으로 사후 Resampling하지 않았다.

Forward Cost Ratio는 Actual Batch1의 고립 Serving API 측정이다. Input4608/8064, Last1/2/4/6/8/12를 사용했다. 독립 Start3회의 Clean CUDA Event p50을 요약했다. Queue와 Transport를 포함하지 않는다. TTFT Speedup으로 표시하지 않는다.

GEMM Ratio는 실제 Serving Linear API의 Projection-only CUDA Graph 측정이다. Shape Control용 Zero Tensor를 사용했다. Attention, Norm, KV, Transfer와 Sampling은 제외했다. Full Forward 또는 Online Speedup으로 표시하지 않는다.

P Activity/Queue Timeline은 Observer Gate를 통과한 P-only Profile을 사용한다. D 곡선은 Host Scheduler 상태다. D의 All-stream Physical Idle Timeline이 아니다. Host Quiescence Histogram도 미래 Decode-safe Budget Distribution이 아니다.

## Manifest

`figures/figure_manifest.json`과 `figures/serving_figure_manifest.json`은 생성 파일과 증거 범위를 기록한다. 기본 Metrics, Quality Gate, Source SHA와 Arrival Seed는 Raw 및 Analysis에 저장했다. 최종 Confirmation과 추가 QPS의 Manifest를 갱신했다.

## Shared KV128 Main Figure

새5곡선4-Panel은 [PNG](../bi0_shortanswer200_shared128_four_metrics_confirmed.png)와 [PDF](../bi0_shortanswer200_shared128_four_metrics_confirmed.pdf)다. Panel은 Joint SLO, Mean TTFT, Mean TBT와 Generated Output Tokens/s다. X축은 실측 QPS1.65/2.35다. 각각 독립 Start1/3회다. 측정하지 않은 Low-load 점을 KV16에서 가져와 채우지 않았다.

비교군은 C3 Agg DP2, C3 Continuous Native, C0 Controlled Native, C0 Layer Last8, C0 Token Suffix≈L×8/36이다. 모든 Engine은 B8192/S1024와 KV42GiB/Block128이다. C0 PD 세 Arm만 Actual Batch1/1/1과 Source Credit4가 같다. 이5개 곡선 전체를 Batch-matched 비교라고 부르지 않는다.

독립 Main Figure의 접두사는 `poc_flashvep/bi0_shortanswer200_shared128_external_`이다. Suffix는 아래와 같다. 각 파일에 PNG/PDF가 있다.

1. `TTFT_p95_vs_qps`
2. `ITL_p99_vs_qps`
3. `TPOT_p95_vs_qps`
4. `output_throughput_vs_qps`
5. `SLO_goodput_vs_qps`
6. `SLO_attainment_vs_qps`

Actual Batch를 맞춘 PD 세 Arm만 있는 Figure의 접두사는 `bi0_shortanswer200_shared128_controlled_`다. 같은6개 Suffix를 사용한다. Start-level95% CI는 n≥3인 점에만 표시한다. 이 Error Bar는 Method별 평균의 CI다. 별도 Table의 Layer/Token 비율에는 Paired-start CI를 사용한다.

## Axis와 Queue Figure

아래 파일은 Evidence Namespace의 `figures/`에 있다. 각 PNG/PDF를 함께 보존했다.

| 파일 Stem | 증거 수준 | 의미 |
|---|---|---|
|`bi0_fixed_cut_discovery`|`MEASURED_ONLINE`|Last1/2/4/6/8/12의 TTFT와 Goodput. 각1회 Discovery|
|`bi0_axis_cut_ratios_discovery`|`MEASURED_ONLINE`|같은 Cut의 Layer/Token 비율. 반복 확인된 Winner Map이 아님|
|`bi0_fixed_cut_host_queue_timelines`|`TRACE_ANALYSIS`|Last4/8/12의 P Queue와 D Admission. 각 Panel은 별도 실행|
|`bi0_controlled_host_queues_vs_qps`|`TRACE_ANALYSIS`|시간 가중 Host Queue. 물리 GPU Activity와 다름|
|`bi0_serial_candidate_latency_CDF`|`ISOLATED_SERIAL_SERVICE_MEASUREMENT`|실제 Prompt의 Ready→First Worker Envelope. Safe-fit Probability가 아님|
|`bi0_conditional_oracle_proxy_headroom`|`OFFLINE_ORACLE`|TTFT-qualified Token Proxy. 실제 Joint SLO-goodput이 아님|

## 미측정 Figure의 처리

`figures/requested_figure_evidence_status.json`은 사용자 요청10개 Figure의 실제 증거 상태를 기록한다. Active-decode Safe-window Distribution과 Safe-fit Rate는 `NOT_MEASURED` 또는 `NOT_RUN`이다. Host Quiescence와 Serial CDF를 그 대신 성공 Figure로 제시하지 않았다.

Online Dynamic Decision Distribution도 `NOT_RUN`이다. Static Cut이나 Offline Oracle 선택을 Online Dynamic 결과로 표시하지 않았다. 여러 Input/Output 조건에서 반복한 TTFT Gain Heatmap도 없다. 대신 측정한 Fixed-cut Discovery만 표시했다. 미완료 Gate를 빈칸이나 가상 곡선으로 숨기지 않았다.
