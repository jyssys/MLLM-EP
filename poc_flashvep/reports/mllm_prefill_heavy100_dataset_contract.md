# MLLM100 Representative Workload Contract

기존 Main400과 Raw Evidence는 보존했다. 이후 주 실험은 Main100을 사용한다.
원본 400개 안에서 실제 요청 100개를 선정했다. 이미지·질문·정답·Prompt는 수정하지 않았다.
성능·SLO·정확도는 선정 입력으로 사용하지 않았다. 이미 측정한 자연 출력 길이는 선정에 사용했다.
따라서 출력 길이를 보기 전의 무작위 표본은 아니다. 전체 Production Traffic의 대표 표본도 아니다.

Manifest: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/data/main100_manifest_template_v2.json`
SHA256: `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`
Parent Main400 SHA256: `7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592`
Selection Seed: `2026112405`

## 요청 구성

| Source | Main400 | Main100 |
|---|---:|---:|
| TIGER-Lab/Mantis-Eval | 100 | 25 |
| lmms-lab/DocVQA | 50 | 13 |
| lmms-lab/MP-DocVQA | 100 | 25 |
| yubo2333/MMLongBench-Doc | 150 | 37 |

Discovery50과 Heldout50은 기존 Split 소속을 유지한다. Calibration32도 그대로 유지한다.
세 집합 사이의 이미지·문서 중복 검사를 통과했다. 요청 ID는 100개 모두 고유하다.

## 통계 비교

| Metric | Statistic | Main400 | Main100 |
|---|---|---:|---:|
| total_input_tokens | mean | 11021.550 | 11029.590 |
| total_input_tokens | median | 9803.500 | 9867.000 |
| total_input_tokens | p95 | 25672.700 | 25726.600 |
| total_input_tokens | min | 105.000 | 556.000 |
| total_input_tokens | max | 30648.000 | 30536.000 |
| visual_tokens | mean | 10951.722 | 10959.760 |
| visual_tokens | median | 9769.500 | 9836.000 |
| visual_tokens | p95 | 25551.250 | 25600.050 |
| visual_tokens | min | 40.000 | 433.000 |
| visual_tokens | max | 30608.000 | 30492.000 |
| text_tokens | mean | 39.807 | 39.810 |
| text_tokens | median | 31.500 | 31.500 |
| text_tokens | p95 | 82.000 | 82.000 |
| text_tokens | min | 19.000 | 20.000 |
| text_tokens | max | 183.000 | 122.000 |
| special_tokens | mean | 30.020 | 30.020 |
| special_tokens | median | 11.000 | 11.000 |
| special_tokens | p95 | 87.000 | 87.300 |
| special_tokens | min | 7.000 | 7.000 |
| special_tokens | max | 157.000 | 125.000 |
| image_count | mean | 12.510 | 12.510 |
| image_count | median | 3.000 | 3.000 |
| image_count | p95 | 41.000 | 41.150 |
| image_count | min | 1.000 | 1.000 |
| image_count | max | 76.000 | 60.000 |
| page_count | mean | 11.750 | 11.740 |
| page_count | median | 2.000 | 2.500 |
| page_count | p95 | 41.000 | 41.150 |
| page_count | min | 0.000 | 0.000 |
| page_count | max | 76.000 | 60.000 |
| natural_output_tokens | mean | 52.590 | 52.690 |
| natural_output_tokens | median | 25.000 | 25.500 |
| natural_output_tokens | p95 | 159.200 | 141.500 |
| natural_output_tokens | min | 2.000 | 2.000 |
| natural_output_tokens | max | 2048.000 | 609.000 |

전체 Percentile·표준편차·CDF 차이는 대응 CSV와 JSON에 보존했다.
400개와 100개의 통계량이 모두 정확히 같지는 않다. 드문 긴 출력과 최댓값은 특히 달라질 수 있다.
선정용 page_count 통계는 페이지 정보가 없는 요청을 0으로 취급했다. 분모는 각각 전체 400개와 100개다.
이 값은 문서 요청만의 평균 페이지 수가 아니다. 원본 요청의 빈 페이지 정보는 수정하지 않았다.
Output Cap Hit는 Main400에서 1/400개, Main100에서 0/100개다.
상한 도달 요청을 잘라내거나 EOS를 수정하지 않았다. 표본에 포함되지 않은 요청도 Main400에 남아 있다.
선택한 요청의 Input Token 합계는 1,102,959개다.
자연 출력 Token 합계는 5,269개다.
Image Reference 합계는 1,251개다.

## 실행 계약과 한계

Model·Processor Revision과 생성 계약은 Main400과 같다. Natural EOS 상한은 2,048 Tokens다.
현재 표는 Processor 측정과 공식 단일 GPU의 기존 출력 길이를 재사용했다.
Main100의 EPD 성능은 아직 미실행이다. Task Quality도 아직 미확정이다.
같은 QPS에서 도착 구간은 대략 짧아진다. 초기화·Compile·Warm-up·Drain 시간은 같은 비율로 줄지 않는다.
모든 방법에 같은 Main100·요청 순서·Arrival Trace를 적용한다. Main400 수치와 섞어 평균을 계산하지 않는다.
대응 표는 `analysis/main100_profile_comparison.csv`다.
요청별 표는 `analysis/main100_request_profile.csv`다.
선정 기준은 `configs/main100_selection_plan.json`에 실험 전 고정했다.
향후 Client 기본 Manifest는 `configs/active_workload.json`에서 읽는다.
