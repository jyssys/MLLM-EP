# Generic과 Modality-aware 온라인 비교

이 보고서는 동일한 scheduler와 실제 C3 실행의 비교만으로 Serving 판정을 내린다. 먼저 B8 QPS5의 N1 Natural EOS를 스크리닝하고 유망 조건을 같은 request/arrival/GPU/output 계약에서 독립 N3 이상으로 확인한다. B8과 Free Native reference는 모두 비교 범위에 있다.

## 이번 Native QPS5 실측

| Method / observer phase | 완료 | TTFT p50 / p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Decode SLO |
| --- | ---: | ---: | ---: | ---: | ---: |
| NativeStrong-B8 clean | 100/100 | 3.931 / 12.264 | 14.452 | 5.179 | 95% |
| NativeCommon-B8 clean-before | 100/100 | 3.133 / 12.029 | 13.682 | 5.247 | 94% |
| NativeCommon-B8 clean-after | 100/100 | 3.901 / 12.516 | 12.431 | 5.187 | 93% |
| NativeCommon-Free clean-before | 100/100 | 3.396 / 11.897 | 13.642 | 5.618 | 95% |
| NativeCommon-Free clean-after | 100/100 | 3.883 / 12.595 | 13.596 | 5.324 | 92% |
| NativeStrong-Free clean | 100/100 | 4.595 / 13.314 | 13.062 | 5.357 | 98% |
| Native-Matched B8 on-demand clean | 100/100 | 16.823 / 25.027 | 12.635 | 5.167 | 100% |
| Native-Matched B8 inline-redirect clean | 100/100 | 3.878 / 12.436 | 13.639 | 5.276 | 100% |
| 기존 Idle10-B8 clean | 100/100 | 5.603 / 12.570 | 15.220 | 6.793 | 96% |

Clean-before와 clean-after는 각 한 번 시작한 서비스에서 수집한 CPC bracket이다. Native-Matched는 초기 late route와 같은 on-demand EC descriptor를 사용한 진단용 baseline이다. 평균 P HTTP가 11.884 s였고 Natural EOS 출력은 7,248 tokens였다. Idle10은 5,344 tokens, NativeStrong-Free는 5,142 tokens를 출력했다. 이 한 번의 시작은 느려진 P 구간을 보여주며 descriptor 방식의 순수 효과나 Dynamic 방법의 이득은 아직 분리하지 못한다. [원시 영수증에서 재생성한 온라인 표](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_cohorts.csv)

Inline-redirect Native에서는 같은 Main100에서 P HTTP 평균 2.069 s 및 Natural 출력 5,382 tokens를 관측했다. 두 cohort의 P는 원본 이미지 1,251장과 logical BF16 7,855,955,968 bytes를 clone했다. On-demand의 139개 EC worker 호출에서 descriptor 준비 host 평균은 143.752 ms였으며 그중 export RPC 평균은 143.598 ms였다. Inline의 138개 호출에서는 모든 이미지가 inline descriptor를 사용해 descriptor 준비는 평균 0.944 ms, export RPC는 0이었다. 소유권 확인 뒤 client window 안에서 완료한 enclosure를 사용했다. 두 실행은 N1이며 Natural EOS 길이와 batch geometry가 달라 순수 TTFT 기여와 신뢰구간은 이 대조로 추정하지 않았다. [CUDA IPC worker enclosure 원자료](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_ec_enclosures.csv)

이번 Idle10-B8 clean은 33/100건에서 suffix를 실행해 전체 입력의 3.04%를 이전했다. 5/33건만 P-ready 시점에 coordinator의 D active와 pending이 모두 없었다. Suffix 요청의 P-ready→D-submit p95는 3.438 s였다. 같은 시작의 별도 counter 실행에서는 D prepared max가 8이면서 single-request forward 비율은 90.10%였다. Clean 토큰 수와 batch 수치를 같은 실행의 결과로 합치지 않았다. [요청별 C3 경계와 실제 prepared batch](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_requests.csv) · [Observed 배치 원자료](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_actual_d_batch.csv)

기존 pinned scorer로 Natural EOS를 비교했다. NativeStrong-Free와 짝지은 독립 N1에서 Mantis-Eval/DocVQA/MP-DocVQA 순으로 Idle10-B8과 Native-Matched on-demand는 0/0/0 pp, Native-Matched inline-redirect는 −4/0/0 pp였다. 동일 Greedy token ID 전체가 일치한 요청은 각각 74/100, 75/100, 75/100이었다. MMLongBench-Doc에는 로컬 RAW 진단만 적용했다. 이 N1 비교는 새 Method의 비열등성이나 공식 MMLongBench-Doc extraction을 검증하지 않았다. [기존 scorer와 요청별 결과 provenance](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_quality_receipt.json) · [태스크별 진단 표](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_quality_by_task.csv)

## QPS5 Shadow Natural EOS

Natural-hold inline pipeline의 P/D는 실제로 서빙했다. Shadow는 돌려받은 D snapshot의 reservation 가능성만 기록하며 P가 항상 전체 prefill을 했다. 아래 세 실행이 각각 100/100 요청을 완료했고 실제 D suffix는 모두 0이었다. Host 시간은 full-prefill control의 MEASURED_ONLINE, 예약 가능 판정은 DIAGNOSTIC이다.

| Shadow arm | Callback 요청 | Eligible 요청 | TTFT p50 / p95 (s) | ITL p99 / TPOT p95 (ms) | Decode SLO | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Late10 | 13 | 6 | 4.098 / 12.861 | 13.189 / 5.310 | 98% | 7,388 |
| B Generic | 29 | 0 | 5.066 / 13.383 | 13.652 / 5.260 | 98% | 5,393 |
| C Hard | 27 | 0 | 5.186 / 14.143 | 14.144 / 5.311 | 96% | 5,530 |

Method A의 6건이 요구한 suffix는 합계 12,193 tokens였다. 나머지 7 decision은 D admission이 busy였다. B는 20개 budget insufficiency, 8개 D busy, 1개 cost support 밖이었다. Budget 검사 20건의 중앙값은 8.319 ms, 예측 비용은 241.043 ms였다. Hard는 13개 D busy, 11개 image interior, 3개 budget insufficiency였다. Hard의 3개 fit check에서 중앙 budget은 17.025 ms, 예측 비용은 205.184 ms였다.

Shadow의 A가 비용 예측 없이 D를 조회하는 범위와 B/C의 safe-fit을 구분한다. Eligible은 실제 D 사용, 관측된 suffix 시간, 또는 향후 overlap을 보장하지 않는다. B/C의 0건은 이 정책과 관측한 상태에서의 결과다. 그 구간의 실제 D headroom은 평가하지 못했다. [Shadow per-request 결정과 host 시간](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_requests.csv) · [Shadow decision figure](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig12_shadow_decision_eligibility.pdf)

## QPS5 Equal-output Replay

세 arm이 모두 100건을 완료했고 요청별 frozen lengths를 지켜 정확히 5,269 tokens의 출력이 발생했다. 실제 D suffix는 0이다. 세 arm은 각각 독립 시작 N1이다.

| Arm | Callback / Eligible 요청 | TTFT p50 / p95 (s) | ITL p99 / TPOT p95 (ms) | Decode / Joint SLO | SLO-goodput (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native-Matched B8 inline | — | 4.706 / 13.537 | 14.012 / 5.188 | 97% / 51% | 67.246 |
| B Generic shadow | 32 / 0 | 4.929 / 13.108 | 14.557 / 5.259 | 97% / 49% | 80.809 |
| C Hard shadow | 19 / 0 | 4.024 / 12.679 | 14.799 / 5.755 | 97% / 53% | 87.602 |

Generic은 17개 D busy, 14개 predicted budget 부족과 1개 cost support 밖을 기록했다. Fit 검사 14건의 중앙 budget/예측 비용은 20.088/221.353 ms였다. Hard는 8개 D busy와 11개 image interior를 반환했다. Callback 있는 요청의 host 합 p95는 Natural B/C 0.282/0.125 ms, Replay B/C 0.262/0.137 ms였다. 두 출력 계약에서 Generic/Hard 모두 실제 예약 가능 판정이 0건이었다. 동일 요청·arrival의 shadow pairs 200개를 보존했고 실제 suffix cost나 causal TTFT 차이는 이 데이터에서 계산하지 않는다. [Shadow B/C 요청 쌍](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_shadow_matched_BC_requests.csv)

별도 prepared-batch Counter의 Replay max/step mean/single-forward 비율은 Native 6/1.404/68.6%, B shadow 5/1.307/73.9%, C shadow 5/1.412/68.4%였다. 이 Counter는 active suffix 조건과 timed clean run을 대체하지 않는다. [Counter 출처와 분포](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_actual_d_batch.csv)

## Active A/B/C

실제 suffix와 Native follower가 함께 decode한 N8 두 조건 모두 exact Greedy 7/8이었다. Active A/B/C의 timed C3, 강한 Native 대비 paired N3, QPS4/6 및 신규 Heldout은 NOT_RUN이다.

| 실제 offload 지표 | Method B | Method C | paired 차이 |
| --- | --- | --- | --- |
| TTFT p50/p95 | NOT_RUN | NOT_RUN | — |
| ITL p99 / TPOT p95 | NOT_RUN | NOT_RUN | — |
| Absolute Decode / Joint SLO 및 SLO-goodput | NOT_RUN | NOT_RUN | — |
| Actual D Batch / single-request forward 비율 | NOT_RUN | NOT_RUN | — |
| 실제 이전 token 및 D Coordinator Wait | NOT_RUN | NOT_RUN | — |
| D idle 활용, false-safe | NOT_RUN | NOT_RUN | — |
| Actual transfer bytes, peak live buffer | NOT_RUN | NOT_RUN | — |
| 온라인 Task Quality | NOT_RUN | NOT_RUN | — |

입력, 출력, 후보 search budget, attention work, D occupancy budget 및 receipt fingerprint를 비교에 붙인다. Natural EOS 및 Equal-output Replay는 별도 표로 생성한다. 95% CI에서 토큰이나 개별 요청 수를 서버 start 수로 사용하지 않는다. 요청 차이는 개선·유사·악화로 나누고, 불리한 예시도 전체 분포에서 보존한다.
