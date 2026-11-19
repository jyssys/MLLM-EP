# LongBench v1 Mixed200 QPS Sweep

상태: `MAIN_COMMON_B_S_COMPLETE / SENSITIVITY24_OUTPUT_PASS / ONLINE_KERNEL_OBSERVER_PASS`.

## 조건과 정합성

모델은 Qwen/Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`이다. Original BF16, Full36, Exact Causal Attention, Greedy와 Thinking Disabled를 사용했다. Runtime은 vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1이다.

GPU0/P UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`다. GPU1/D UUID는 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`다. 두 GPU는 H100 SXM80GB다. 모든44개 Primary Worker의 실제 설정은 B8192/S1024였다. GPU당 KV42GiB와 Prefix Cache 비활성화를 유지했다.

Main은22개 독립 Server Start를 사용했다. 총4400개 요청의 Output Sequence가 Reference와 일치했다. 각 Start는 동일한200개 요청과25865개 출력 Token을 포함했다. 전체 Main의 반환 출력은569030 Token이다. EOS Token도 출력 수에 포함한다.

N8에서는 전체 Prompt KV, Final Hidden, Greedy Output과 Credit 복원을 검증했다. N32와 Main200에서는 상태 복사 없이 Continuous Output과 정상 Retirement를 검증했다. Main4400개의 KV Tensor 전체를 CPU로 복사해 비교했다고 주장하지 않는다.

Input Dataset SHA256은 `1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553`이다. [Dataset 계약](dense_ar_longbench200_dataset_contract.md)에 Revision, Sampling Seed, 원문 ID, Truncation과 실제 길이를 기록했다.

## QPS와 반복

공통 Offered QPS는0.4/0.7/1/1.15/1.2/1.65/2.35다. N40 Pilot 이후 이 Grid를 동결했다. Pilot의 완료율은 장기 Capacity가 아니다. 긴 Output의 Drain과 Pilot/Main의 길이 분포 차이가 완료율에 영향을 준다.

QPS1과1.65는 각 Method당 독립 Start3회다. Arrival Seed는2026112021/2026112022/2026112023이다. 다른 QPS는 Seed2026112021의 Start1회다. 두 Method는 같은 Seed에서 동일한 Arrival Offsets를 사용했다.

QPS1의 실제 도착 Trace Rate 평균은1.0476requests/s였다. QPS1.65는1.7285requests/s였다. Offered QPS는 Poisson의 지정 Rate다. 유한한 Trace의 실현 Rate와 같다고 가정하지 않는다. 실제 Coordinator Dispatch Rate와 Dispatch Lag도 저장했다.

아래 값은 Start별 통계의 평균이다. 예를 들어 TTFT p95는 각 Start의200개 요청에서 먼저 계산했다. 그 뒤 Start 평균을 계산했다.600개 요청을 합친 p95가 아니다.

| QPS | Method | Starts | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Output tokens/s | Joint SLO % | Qualified tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | Native | 1 | 2458.268 | 49.472 | 32.666 | 49.761 | 63.00 | 4.571 |
| 0.4 | Agg | 1 | 1927.698 | 91.827 | 34.443 | 49.794 | 69.50 | 8.682 |
| 0.7 | Native | 1 | 2765.036 | 99.739 | 34.205 | 85.451 | 53.50 | 4.463 |
| 0.7 | Agg | 1 | 1938.268 | 320.877 | 72.496 | 85.484 | 60.00 | 7.496 |
| 1.0 | Native | 3 | 4229.046 | 131.420 | 36.556 | 128.698 | 45.83 | 3.605 |
| 1.0 | Agg | 3 | 1984.057 | 337.152 | 176.973 | 129.017 | 43.50 | 5.504 |
| 1.15 | Native | 1 | 4006.973 | 136.099 | 36.783 | 136.060 | 43.50 | 3.346 |
| 1.15 | Agg | 1 | 1973.140 | 335.913 | 119.096 | 136.314 | 46.50 | 6.298 |
| 1.2 | Native | 1 | 4611.834 | 132.134 | 38.256 | 141.497 | 41.50 | 3.414 |
| 1.2 | Agg | 1 | 1982.821 | 343.627 | 177.166 | 141.771 | 44.00 | 6.419 |
| 1.65 | Native | 3 | 20390.666 | 149.023 | 39.075 | 175.932 | 3.33 | 0.281 |
| 1.65 | Agg | 3 | 2507.581 | 416.878 | 298.673 | 202.239 | 23.33 | 2.748 |
| 2.35 | Native | 1 | 46554.458 | 138.617 | 38.888 | 176.833 | 3.00 | 0.191 |
| 2.35 | Agg | 1 | 5066.959 | 460.781 | 346.506 | 250.994 | 8.50 | 0.699 |

QPS0.4에는 Allocation Observer Source의 차이가 있다. Model Math와 설정은 같았지만 Wrapper Byte가 완전히 같지는 않았다. 해당 Cell의 한계를 [Crossover 분석](dense_ar_aggdp2_native_pd_crossover_analysis.md)에 기록했다. 다른 Pair의 Runtime 파일은 일치했다.

## 주요 Cell의95% CI

각 Cell은 독립 Start3회다. Empirical Bootstrap은 Start를 재표본화한27개 조합을 사용했다. 작은 Start 수의 CI이며 WCET 보장이 아니다. Start1회 Cell에는 CI를 만들지 않았다.

| QPS | Method | TTFT p95 ms, 평균[CI95] | ITL p99 ms, 평균[CI95] | TPOT p95 ms/token, 평균[CI95] | Output tokens/s, 평균[CI95] |
|---:|---|---|---|---|---|
| 1.0 | Native | 4229.05[3541.92,4900.90] | 131.42[123.79,137.47] | 36.56[35.76,37.18] | 128.70[120.94,137.51] |
| 1.0 | Agg | 1984.06[1976.64,1992.57] | 337.15[330.40,345.55] | 176.97[135.33,212.99] | 129.02[121.05,138.13] |
| 1.65 | Native | 20390.67[13567.22,27699.40] | 149.02[145.98,151.31] | 39.07[38.37,39.84] | 175.93[175.36,176.61] |
| 1.65 | Agg | 2507.58[2189.54,2841.76] | 416.88[414.75,419.16] | 298.67[265.67,330.28] | 202.24[191.12,214.80] |

동일한 Seed Pair를 재표본화한 Ratio CI도 별도로 저장했다. QPS1.65의 Native/Agg TTFT p95 비율은8.1316[6.1875,9.7377]이었다. ITL p99 비율은0.3575[0.3500,0.3639]이었다. TPOT p95 비율은0.1308[0.1162,0.1501]이었다. Raw Output 비율은0.8699[0.8220,0.9187]이었다.

## SLO 해석

Frozen SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms다. Joint SLO는 세 조건을 모두 통과한 요청 비율이다. Qualified Throughput은 통과 요청의 실제 Output Token 수를 전체 Window로 나눈 값이다.

QPS1에서는 Native의 Joint 요청 비율이45.83%로 Agg43.50%보다 높았다. 그러나 Qualified Output은3.605tokens/s로 Agg5.504tokens/s보다 낮았다. 가변 Output에서는 요청 비율과 유효 Token 처리율이 같은 지표가 아니다. [Task별 실제 결과](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_task_per_start_metrics.csv)를 보존했다.

QPS1.65 Native의 TTFT-only 통과율 평균은7.50%였다. Decode-only 통과율은39.17%였다. Joint 통과율은3.33%였다. 낮은 Joint SLO를 TTFT 문제만으로 설명하지 않는다. 실제 Decode 시간을 그대로 둔다면 TTFT만 개선해도 Joint 비율은 Decode-only 비율을 넘을 수 없다. 이는 조건부 논리 관계이며 Offloading Speedup 예측이 아니다.

## Figure4개

모든 Figure는 같은 Offered-QPS Grid를 사용한다. Raw 측정점만 표시한다. 평균과 CI의 표본 단위는 Server Start다. 파일은 Repository의 `poc_flashvep` 경로에도 저장했다.

1. TTFT p95: [PNG](../longbench200_ttft_p95_vs_qps.png), [PDF](../longbench200_ttft_p95_vs_qps.pdf).
2. Client-visible ITL p99: [PNG](../longbench200_itl_p99_vs_qps.png), [PDF](../longbench200_itl_p99_vs_qps.pdf).
3. Request TPOT p95: [PNG](../longbench200_tpot_p95_vs_qps.png), [PDF](../longbench200_tpot_p95_vs_qps.pdf).
4. Generated Output Throughput: [PNG](../longbench200_output_tokens_s_vs_qps.png), [PDF](../longbench200_output_tokens_s_vs_qps.pdf).

## 한계와 증거

측정 경계는 예정 Arrival부터 Coordinator Token 수신까지다. HTTP와 Detokenization은 포함하지 않는다. Mean TBT는 전체 Token Interval Pool의 평균이다. TPOT는 요청별 첫 Token부터 마지막 Token까지의 평균 Interval이다.

모든 요청을 끝까지 Drain했다. 미완료 요청을 분모에서 제외하지 않았다. 이 결과는 Finite-cohort Throughput이다. 장기 Sustainable Capacity는 `NOT_MEASURED`다. 인접 QPS의 Start 수와 실제 Arrival 실현값이 다르므로 완전한 단조성을 가정하지 않는다.

- [실행별 Config, Source와 Timeline 분석](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_full_analysis.json)
- [모든 Start 평균과 CI CSV](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_table.csv)
- [실제44개 Worker 설정](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_runtime_audit.json)
- [Figure 통계와 Start 수](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_figure_start_level_statistics.json)

## 제한적인 B/S Sensitivity

N40의 같은 Prompt와 Seed2026112031을 사용했다. QPS0.4/1/2.35에서 B8192/S1024를 기준으로 한 변수만 변경했다. 설정은 B4096/S1024, B16384/S1024, B8192/S256이다.24개 Cell이 모두 Output40/40과 정상 KV Lifetime을 통과했다. 실제 Worker의 B/S도 확인했다.

각 Cell은 독립 Start1회다. 이 결과에는 Start-level CI가 없다. 아래 표는 고부하 QPS2.35의 Discovery다. Main N200과 다른 N40 Dataset이다. 두 표의 Throughput을 같은 Capacity로 직접 비교하지 않는다.

| Method | B / S | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Output tokens/s | Joint SLO % |
|---|---|---:|---:|---:|---:|---:|
| Native | 8192 / 1024 | 12855.094 | 150.710 | 37.267 | 127.732 | 15.0 |
| Native | 4096 / 1024 | 13247.341 | 145.107 | 37.315 | 127.594 | 17.5 |
| Native | 16384 / 1024 | 12925.000 | 166.483 | 38.564 | 126.950 | 17.5 |
| Native | 8192 / 256 | 12918.896 | 149.129 | 38.108 | 128.485 | 15.0 |
| Agg | 8192 / 1024 | 4148.376 | 393.398 | 342.894 | 142.828 | 17.5 |
| Agg | 4096 / 1024 | 4331.232 | 217.421 | 171.766 | 145.606 | 15.0 |
| Agg | 16384 / 1024 | 4042.288 | 618.288 | 740.311 | 143.187 | 15.0 |
| Agg | 8192 / 256 | 4166.560 | 391.420 | 342.377 | 144.346 | 15.0 |

고부하 Agg의 Decode Tail은 B 변화에 민감했다. B4096은 B8192보다 낮은 ITL/TPOT Tail을 보였다. B16384는 더 높은 Tail을 보였다. Raw Throughput 차이는 이 표에서 상대적으로 작았다. 그러나 Low-load에서는 같은 단조 관계가 없었다. 예를 들어 QPS0.4 Agg의 ITL p99는 B8192에서235.58ms, B16384에서54.11ms였다. 단일 Start의 결과를 전체 부하의 법칙으로 일반화하지 않는다.

S256 변경은 실제 Active Request 수를 강제로256개로 만들지 않는다. 실제 Batch, Chunk Count와 KV Capacity는 [전체 Sensitivity 분석](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/B_S_final_discovery_summary.json)에 저장했다. [24개 Cell CSV](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/B_S_final_discovery_table.csv)도 보존했다.

Moderate Native B8192/S1024 Control은 확장 Queue Observer 추가 전에 실행됐다. 나머지 Moderate Control에는 해당 Read-only Observer가 있다. Model Math는 같지만 Wrapper Source는 완전히 같지 않다. 작은 차이를 순수 B/S 인과 효과로 확정하지 않는다. Low/High 설정은 Main Runtime Freeze 이후의 같은 Runtime 파일을 사용했다.

이 Sensitivity는 공통 설정의 유효성과 큰 변화의 방향을 확인하는 Discovery다. 각 목표에 맞춰 독립 Confirmation한 Best-tuned Frontier는 `NOT_RUN`이다. 가장 좋은 N40 관측값을 Production 최적값이나 Sustainable Capacity로 주장하지 않는다. Primary Main Curve의 B8192/S1024와 SLO는 변경하지 않았다.
