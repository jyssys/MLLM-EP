# LongBench200 Agg DP2와 Native PD의 Trade-off

상태: `MAIN_CONFIRMATION_COMPLETE / P_BOTTLENECK_CONFIRMED / DECODE_SAFE_HEADROOM_NOT_ESTABLISHED`.

## 통계량의 차이

최종 QPS1.65의 Agg Mean TBT는54.79ms였다. Request TPOT p95는298.67ms였다. 두 값은 모순이 아니다. Mean TBT는 모든 Token Interval을 합친 평균이다. TPOT p95는 요청별 평균 Interval의95분위다.

같은 조건에서 Agg GovReport의 TPOT p95는71.45ms였다. HotpotQA는329.86ms, MuSiQue는298.42ms, NarrativeQA는323.80ms였다. 각 값은 독립 Start3회의 평균이다. 짧은 QA는 한 번의 큰 Interval이 요청 평균에 크게 반영될 수 있다. 긴 요약 요청은 전체 Token Interval Pool에서 더 큰 비중을 차지한다.

이 설명은 관측된 길이와 통계 정의에 근거한다. 모든 Spike를 특정 Prefill Kernel에 인과적으로 연결했다는 뜻은 아니다. [Task별 Start-level 결과](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_task_start_level_statistics.json)를 보존했다.

## 비교 계약

Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`을 사용했다. GPU0/1의 H100 SXM80GB 두 대를 사용했다. 두 방법 모두 Original BF16, Full36, Exact Causal Attention, Greedy와 Thinking Disabled를 유지했다.

Runtime은 vLLM0.26.0+cu129, PyTorch2.11.0+cu129와 NIXL1.4.1이었다. GPU0 UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`였다. GPU1 UUID는 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`였다. [Dataset Revision과 Hash](dense_ar_longbench200_dataset_contract.md)는 두 방법에서 같았다.

모든 Worker의 실제 B/S는8192/1024였다. GPU당 KV Budget은42GiB였다. Prefix Cache를 사용하지 않았다. 공식 Batch-invariant 모드를 두 방법에 동일하게 적용했다. 이 결과를 더 빠른 Stock BI0 경로의 Production 성능으로 일반화하지 않는다.

Agg는 Standard vLLM Continuous Batching과 Round-robin Replica2개를 사용했다. QPS0.7의 실제 기록에는 Replica별 Mixed Prefill–Decode Iteration204/203회가 있었다. Custom Cohort-only Scheduler는 사용하지 않았다.

Native는 Standard NIXL Connector와 Continuous Scheduler를 사용했다. 기존 Custom Reservation ACK와 Whole-cohort Guard를 이 경로에 추가하지 않았다. P의 첫 Sample은 버리고 D의 첫 Sample을 Client-visible Output으로 사용했다. D는 원래 Prompt의 마지막 Query1개를 다시 준비했다.

## DuetServe와의 차이

이 연구는 DuetServe-style Methodology를 새 Workload에 적용했다. Figure2의 수치 재현이 아니다. [DuetServe 원문](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf).

| 항목 | DuetServe Figure2 및 논문 Setup | 이번 실험 |
|---|---|---|
| 모델 | Qwen3-8B, Revision 미확인 | 위 Revision 고정 |
| 자원 | H1002개 | 동일 GPU0/1의 H100 SXM80GB2개 |
| Agg | DP2, Round-robin, Budget8192 | 같은 역할, B8192/S1024 실제 검증 |
| S 설정 | General Setup의 Batch Limit1024와 Figure2 CLI를 구분 | 실제 Worker S1024 저장 |
| Runtime | vLLM0.10.1-v1, Dynamo0.5.1 | vLLM0.26.0+cu129, 직접 LLM Worker/Coordinator |
| Workload | Input8000/Output200 고정 | LongBench v1 Mixed200, 실제 가변 길이 |
| 생성 | 고정 길이 조건 | Natural EOS, 공통 Cap2048, Cap Hit0/200 |
| Offered QPS | 2–10 | Pilot 이후 동결한0.4–2.35 |
| 주요 통계 | Mean TTFT/TBT와 논문의 Throughput 정의 | TTFT p95, ITL p99, TPOT p95, 출력 Token/s |

논문의 Figure2가 S1024 CLI를 명시했다고 단정하지 않는다. 또한 논문의 Total Tokens/s와 이번 Output-only Throughput을 같은 분모라고 가정하지 않는다.

## 첫 Arrival Seed의 관측

각 Cell은200개 요청과 독립 Start1회다. Arrival Seed는2026112021이다. 두 방법의 Prompt, 순서, Arrival Offsets와 Output Sequence는 모두 일치했다. 각 실행은25865 Output Token을 반환했다.

| QPS | Method | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Output tokens/s | Joint SLO % |
|---:|---|---:|---:|---:|---:|---:|
| 0.4 | Native | 2458.268 | 49.472 | 32.666 | 49.761 | 63.0 |
| 0.4 | Agg | 1927.698 | 91.827 | 34.443 | 49.794 | 69.5 |
| 0.7 | Native | 2765.036 | 99.739 | 34.205 | 85.451 | 53.5 |
| 0.7 | Agg | 1938.268 | 320.877 | 72.496 | 85.484 | 60.0 |
| 1.0 | Native | 3339.799 | 120.433 | 35.403 | 119.634 | 49.0 |
| 1.0 | Agg | 1975.449 | 329.840 | 119.372 | 119.741 | 47.5 |
| 1.15 | Native | 4006.973 | 136.099 | 36.783 | 136.060 | 43.5 |
| 1.15 | Agg | 1973.140 | 335.913 | 119.096 | 136.314 | 46.5 |
| 1.2 | Native | 4611.834 | 132.134 | 38.256 | 141.497 | 41.5 |
| 1.2 | Agg | 1982.821 | 343.627 | 177.166 | 141.771 | 44.0 |
| 1.65 | Native | 12063.391 | 151.184 | 40.103 | 175.647 | 3.0 |
| 1.65 | Agg | 2114.330 | 414.284 | 255.432 | 189.185 | 30.0 |
| 2.35 | Native | 46554.458 | 138.617 | 38.888 | 176.833 | 3.0 |
| 2.35 | Agg | 5066.959 | 460.781 | 346.506 | 250.994 | 8.5 |

이 표는 첫 Seed의 원시 요약을 보존한다. 최종 반복 평균과 CI를 이 표의 단일 Start 값으로 대체하지 않는다. Main Figure는 후속 Start가 완료되면 반복 평균으로 갱신한다.

첫 QPS0.4 Pair는 읽기 전용 Allocation Observer의 Source가 달랐다. Model Math, Sampling과 Runtime Version은 같았다. Observer 차이의 비용은 별도로 측정하지 않았다. 이 Cell을 Byte-identical Wrapper Source 비교라고 표현하지 않는다. QPS0.7 이후 Pair의 Runtime 파일은 일치했다. Source Hash를 실행별로 보존했다.

## 현재까지의 해석

Native의 TTFT p95는 QPS1.2와1.65 사이에서 크게 증가했다. QPS1.65와2.35의 Output Throughput은175.647와176.833tokens/s였다. 동시에 TTFT p95는12.063초에서46.554초로 증가했다. 이는 이 Finite Cohort의 처리율 제한과 Queue 증가를 보여준다. 장기 Sustainable Capacity의 측정은 아니다.

Agg의 Decode Tail은 크게 증가했다. 첫 Seed에서 ITL p99는91.827ms에서460.781ms로 증가했다. TPOT p95는34.443ms에서346.506ms로 증가했다. 이 두 Tail 통계를 Mean TBT의 증가율로 표현하지 않는다.

Native의 ITL p99도49.472ms에서138.617ms로 증가했다. 따라서 Native Decode가 모든 통계에서 부하와 무관하게 일정했다고 주장하지 않는다. Native의 TPOT Tail은 Agg보다 낮았지만, 기존 절대 SLO를 계속 만족한 것은 아니다.

QPS1.65 Native의 P Waiting은 시간 가중 평균9.405, p9514개였다. D Ready Waiting은 관측한 상태에서0개였다. P Worker Accepted→첫 Prepare 평균은6930.943ms였다. Source-ready→D Prepare 평균은243.503ms였다. 긴 TTFT를 전부 ACK 또는 전송 대기로 설명하지 않는다.

이 Queue 기록은 GPU Kernel Busy가 아니다. 같은 실행의 Kernel Activity와 Observer Gate는 별도 검증 중이다. D Ready Queue0만으로 Decode-safe Headroom을 선언하지 않는다.

## SLO와 통계

Frozen SLO는 TTFT5000ms 이하, 모든 ITL34ms 이하, Request TPOT33ms 이하다. 결과를 보고 이 기준을 변경하지 않았다. 이 값은 연구 계약이며 Production Standard가 아니다.

ITL p99는 Client-visible Interval의 전체 Pool에서 계산한다. TPOT p95는 Request별 `(마지막 Token−첫 Token)/(Output Tokens−1)`의 p95다. 두 통계의 표본 단위와 가중치는 다르다.

Raw Output Throughput은 모든 완료 요청의 실제 출력 Token 수를 전체 측정 Window로 나눈 값이다. Warmup을 제외했다. 모든200개 요청의 Drain은 포함했다. 미완료 요청을 버리지 않았다.

최종95% CI는 독립 Server Start를 재표본화한다. 요청200개를 독립 Start200회로 취급하지 않는다. 같은 Seed의 두 방법은 같은 도착 기록을 사용한다. 반복하지 않은 Cell은 CI를 표시하지 않는다.

## 자료

- [실행별 통계와 Source 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_full_analysis.json)
- [전체 Table CSV](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_table.csv)
- [QPS1.65 Native Host Timeline 분석](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_Native_LongBench_N200_q1.65_s2026112021_BI1_v1_timeline_analysis.json)
- [동결한 Grid와 SLO](dense_ar_native_pd_real_workload_reproduction_20261119/configs/frozen_LongBench200_Main_grid_v1.json)

## 독립 Start3회 확인

QPS1과1.65의 반복을 완료했다. 전체 Main22회에서4400개 요청의 Output과 정상 Retirement를 확인했다. 위 첫 Seed 표는 변경하지 않았다. 최종 평균과 CI는 [Main 결과 보고서](dense_ar_longbench200_qps_sweep_results.md)에 별도로 기록했다.

QPS1.65에서 Native/Agg의 TTFT p95 비율은8.1316[6.1875,9.7377]이었다. ITL p99 비율은0.3575[0.3500,0.3639]였다. TPOT p95 비율은0.1308[0.1162,0.1501]였다. 이 Ratio CI는 동일 Arrival Seed의 독립 Start Pair를 재표본화했다.

같은 QPS의 Raw Output 비율은0.8699[0.8220,0.9187]였다. Qualified Output 비율은0.1021[0.0687,0.1620]였다. Native는 Decode Tail이 낮았지만, 이 Frozen Joint SLO에서 Throughput 또는 Goodput 우위를 보이지 않았다.

이 결과는 단일 지표의 완전한 성능 역전이 아니다. Agg는 TTFT와 Raw Throughput에서 유리했다. Native는 Decode Tail에서 유리했다. Native의 낮은 ITL/TPOT를 절대 SLO 성공으로 확대하지 않는다.
