# Dense AR Layer/Token/Dynamic Figure와 통계

## 공통 경계

Figure의 Model은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16과 Exact Attention을 유지한다.
Input은 8000 Tokens다.
Output16과 Output64는 별도 Panel이다.
같은 QPS에서는 Prompt128개와 Arrival Trace를 일치시킨다.

검사한 Offered QPS는 2/4.8/6.4/10이다.
QPS2~10의 모든 정수를 실행했다고 주장하지 않는다.
실제 Trace QPS와 Offered QPS도 별도 기록한다.
독립 Start 수는 각 점 옆의 숫자로 표시한다.
Start 하나에는 CI를 만들지 않는다.
연결선은 측정하지 않은 QPS의 결과가 아니다.

95% CI의 재표본 단위는 독립 Server Start다.
같은 Arrival Seed를 짝지어 Method Ratio를 계산한다.
Request나 Token을 독립 Start로 취급하지 않는다.
작은 Start 수의 Bootstrap은 모집단 Tail을 정밀하게 보장하지 않는다.
여러 QPS·Regime의 CI는 Pointwise CI다.
동시에 모든 비교를 보장하는 Familywise CI가 아니다.

모든 Token Rate는 Output Token만 포함한다.
Window는 첫 Scheduled Arrival부터 마지막 Request 완료까지다.
Warmup과 Model Loading은 이 Window에 포함하지 않는다.
마지막 Drain은 포함한다.
Finite-cohort Rate를 Sustainable Capacity로 표시하지 않는다.

## Figure 목록

아래 파일은 [Figure Directory](dense_ar_layer_token_dynamic_headroom_20261117/figures/)에 있다.
각 Figure의 PNG와 PDF를 모두 저장한다.
정합성 Capture와 Offline Model은 성능 Curve와 구분한다.

| 번호 | 파일 이름 | 내용 | 증거 수준 |
|---|---|---|---|
| 1 | `01_mean_TTFT` | Offered QPS vs Mean TTFT | MEASURED |
| 2 | `02_mean_TBT` | Offered QPS vs Mean TBT | MEASURED |
| 3 | `03_ITL_p99` | Offered QPS vs ITL p99 | MEASURED |
| 4 | `04_raw_output_tokens_s` | Offered QPS vs Raw Output tokens/s | MEASURED |
| 5 | `05_joint_SLO` | Offered QPS vs Joint SLO Attainment | MEASURED |
| 6 | `06_SLO_output_tokens_s` | Offered QPS vs SLO-qualified Output tokens/s | MEASURED |
| 7 | `07_axis_Pareto` | TTFT p95 vs ITL p99 | MEASURED |
| 8 | `08_static_oracle_online` | 제약 Static·근사 Beam과 실제 Pilot을 분리한 Panel | OFFLINE_ORACLE / MEASURED |
| 9 | `09_host_budget_variance` | 실제 Decision의 예측 Deadline Residual 분포 | TRACE_ANALYSIS |
| 10 | `10_TP2_first_divergence` | Layer0 Projection Error와 첫 Argmax 분기 | CORRECTNESS_ONLY |
| 11 | `11_static_host_stream_timeline` | Layer의 같은 Run에서 Host와 CUDA Enclosure | TRACE_ANALYSIS |
| 12 | `11_token_host_stream_timeline` | Token의 같은 Run에서 Host와 CUDA Enclosure | TRACE_ANALYSIS, 간섭 기준 실패 |
| 13 | `13_actual_dynamic_cut_distribution` | 새 Prompt의 실제 Online Cut 분포, Start별 표시 | MEASURED |

Figure5/6은 Frozen Decode-priority SLO를 사용한다.
TTFT-priority와 Balanced 결과도 통계 JSON에 보존한다.
다른 Regime으로 최적화한 Static이나 Colocated Frontier를 측정했다고 주장하지 않는다.
Colocated의 Label은 B8192/S1024다.
이는 사용자가 선정한 기본 설정이다.
모든 Decode-priority 튜닝 중 최적 설정이라는 뜻은 아니다.
Figure13은 실제 완료한 Cut Commit을 사용한다.
강제 Cut 정합성 시험의 Histogram을 Online 선택으로 표시하지 않는다.
Cut이 다양하다는 사실만으로 Dynamic의 성능 이득을 주장하지 않는다.

## 타임라인 해석

Host Submission과 CUDA Stream Enclosure는 다른 Row에 표시한다.
Host의 Source-ready→D-submit Span은 Queue와 Handoff를 포함한다.
이를 Wire Duration이라고 표시하지 않는다.
CUDA Event의 Start/End는 Kernel 하나의 Start/End가 아니다.
Enclosure에는 같은 Stream의 Gap이 포함될 수 있다.
다른 Stream의 작업을 모두 관측한 것도 아니다.
Event 시간축은 같은 Run의 Host Anchor 중간값에 맞췄다.
Layer의 P/D Anchor Bracket 폭은 각각0.063824/0.077236ms였다.
Token은 각각0.071980/0.075240ms였다.
이 Bracket은 Kernel별 Timestamp 정확도나 Clock Drift의 전체 상한이 아니다.

Layer Event Capture는 비계측 Control 평균 대비 네 지표가 5% 이내였다.
Control2회와 Capture1회의 점 추정 Screen이다.
Profiler 간섭의 통계적 상한을 입증한 것은 아니다.
Token Capture의 TTFT p95는 8.1923% 변했다.
따라서 Token 타임라인은 Diagnostic으로만 사용한다.
서로 다른 Run을 같은 실제 시간축으로 합치지 않는다.
Physical Kernel Union, SM/HBM Counter와 Stream ID는 `NOT_MEASURED`다.

## 직접 저장한 네 가지 그래프

다음 파일은 `poc_flashvep/` 바로 아래에도 저장한다.
기존 DuetServe Figure를 덮어쓰지 않는다.

- [Mean TTFT](../dense_ar_layer_token_dynamic_mean_ttft.png).
- [Mean TBT](../dense_ar_layer_token_dynamic_mean_tbt.png).
- [Joint SLO](../dense_ar_layer_token_dynamic_slo.png).
- [Output tokens/s](../dense_ar_layer_token_dynamic_output_tokens_s.png).

같은 이름의 PDF도 제공한다.
이 Curve에서 빠진 Method나 QPS를 0 성능으로 해석하지 않는다.

## 근거와 재생성

- [Start별 CSV](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_per_start_metrics.csv).
- [독립 Start 통계와 전체 SLO Regime](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_start_statistics.json).
- [Figure Manifest](dense_ar_layer_token_dynamic_headroom_20261117/analysis/figure_manifest.json).
- [Host·Stream 원자료](dense_ar_layer_token_dynamic_headroom_20261117/analysis/host_stream_timeline_data.json).
- [Plot Script](../dense_ar_layer_token_pd/main_figures.py).
- [Timeline Script](../dense_ar_layer_token_pd/timeline_figures.py).

```bash
python -m poc_flashvep.dense_ar_layer_token_pd.statistics_report
python -m poc_flashvep.dense_ar_layer_token_pd.main_figures
python -m poc_flashvep.dense_ar_layer_token_pd.timeline_figures
```

Figure9는 물리적인 Safe Slack의 측정이 아니다.
변동성 때문에 Dynamic이 좋아졌다는 인과관계를 이 분포만으로 입증하지 않는다.
그 인과관계는 같은 Geometry와 이전량의 실제 Serving Control을 통과해야 한다.
