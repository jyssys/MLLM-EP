# Track A1 · Visual KV와 Decode 비용

Discovery200 Native 전체에서 Visual KV는 Prompt 행의 **98.28%**, 실제 생성 이후 남는 논리적 KV 행의 **95.68%**였다. 이 수치는 Main100의 긴 입력이 지배한다. MMReason에서는 같은 비율이 **66.82%에서 29.78%**로 내려간다. Pooled 비율만으로 모든 요청의 Generation 비용을 설명하기 어렵다. [요청별 KV 및 Page](../mllm_specific_disaggregation_discovery/results/a/native_kv.csv) · [Workload Figure](../mllm_specific_disaggregation_discovery/results/a/figures/figure7_visual_kv_memory.png)

| Cohort | 요청 | Prompt Visual 행 비율 | 마지막 출력 직전 논리적 Visual 행 비율 | 생성 시 실제 소비된 출력 행 | P가 Export한 Page 수 |
|---|---:|---:|---:|---:|---:|
| Main100 | 100 | 99.37% | 98.88% | 5,424 | 8,666 |
| MMMU | 40 | 84.66% | 84.55% | 43 | 293 |
| MathVista | 30 | 72.51% | 39.34% | 10,639 | 115 |
| MMReason | 30 | 66.82% | 29.78% | 15,529 | 114 |
| 전체 | 200 | 98.28% | 95.68% | 31,635 | 9,188 |

KV Head 4, Dimension 128, K/V 두 배열, BF16 2 bytes, Full28에서 논리적 KV는 **57,344 B/행**이다. 마지막 출력은 다음 Decode Step에 쓰이지 않으므로 Generated KV는 실제 Output Count에서 1을 뺀다. 이는 관측한 출력과 모델에 근거한 논리적 사용량이다. D 엔진의 실제 최종 Page를 독립적으로 추적한 값은 별도 계측 범위에 남긴다.

Native P의 Prompt Block ID 9,188개가 200/200에서 `ceil(L / 128)`과 일치했다. 총 1,176,064개 예약 행 가운데 Prompt 행은 1,163,122개였고 padding은 12,942개였다. Page 전체가 Visual인 Block은 7,525개, Visual과 다른 행 또는 padding이 섞인 Block은 1,558개다. Visual을 정확한 행 범위로 옮기려면 이 Page의 layout을 재구성하거나 선택한 행을 따로 pack하는 비용이 생긴다. D 엔진 KV Cache는 786,432 Token이었고 42GiB의 Memory Pool을 사용했다.

## 실제 Native transport와 GPU Decode

QPS0.75 Native의 EPD D 로그에서 전체 200개 실행에만 걸친 것으로 식별한 약 10초 구간에는 **199개의 성공 NIXL 전송**이 기록됐다. 해당 구간의 전송당 평균은 **323.06 MiB와 10.03ms**였다. 로그는 집계구간이며 전송별 Raw Timestamp나 GPU-exclusive Time을 제공하지 않는다. 이 시작에서 D KV Pool의 로그상 최대 관측 점유율은 5.0%였다. QPS0.75의 실제 Decode HTTP 중앙값은 252ms다. 정확한 Page 할당·해제 수명은 미계측이다. [Native NIXL Log 관측](../mllm_specific_disaggregation_discovery/results/a/native_nixl_low.json) · [Native Serving](mllm_discovery200_native_baseline.md)

독립 QPS2.5 Native의 D connector Counter 차이는 **200개 전송, 67,440,214,016 bytes**였다. 전송당 평균 **321.58 MiB**로 원본 Block ledger의 합과 일치했다. NIXL xfer time 합 2.535s는 전송당 평균 **12.67ms**, post time 합 1.209s는 평균 **6.05ms**다. 두 시간은 별도 telemetry이며 중복 여부에 따라 단순 합산하지 않는다. 해당 시작의 10초 로그에서 D KV Pool의 최대 관측치는 3.2%였다. 전송별 P→D Queue 및 물리 HBM 사용량은 집계 Counter에서 분리할 수 없다. [Connector Histogram 차이](../mllm_specific_disaggregation_discovery/results/a/native_nixl_mid.json) · [Before/After 원자료](../mllm_specific_disaggregation_discovery/results/a/native_nixl_mid_raw_prometheus.json)

Native는 FLASH_ATTN과 NIXL HND Page Cache를 사용했다. 원본 Qwen의 E Embedding과 Full28을 유지한 별도 H200 실행에서 QKV/Norm/MRoPE, KV Write, FA3, Output Projection, MLP를 CUDA Event로 구분했다. 이 재생은 실제 원본 상태와 연산을 사용한다. Serving Runtime/Queue, Paged KV 및 CUDA Graph의 구체적인 최적화 비용은 포함하지 않는다. HBM Read/Write Hardware Counter는 미계측이다. [GPU Decode Figure](../mllm_specific_disaggregation_discovery/results/a/figures/figure8_decode_cost.png) · [수치 및 Batch 실측](mllm_visual_memory_generation_exact_attention.md)
