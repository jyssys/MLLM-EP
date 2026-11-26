# Track A3 · 원격 Visual Attention 손익분기점

H200 GPU6/5, NV18에서 원본 Qwen Full28 BF16의 5개 실제 Prompt를 측정했다. Native one-shot KV 이동의 회피 가능 비용과 Decode마다 누적되는 원격 비용을 분리했다. 모든 파티션에 실제 GQA Attention, 동일한 Query Projection, KV와 MRoPE를 사용했다. [Full28 원자료](../mllm_specific_disaggregation_discovery/results/a/attention/) · [수치 대조](mllm_visual_memory_generation_exact_attention.md)

## Full28 시간과 초기 KV 복사

| Prompt 행 | Native 출력 토큰 | Local ms/step | Visual ms/step | Visual의 초기 순절감 ms | Native 출력 길이의 예상 순절감 ms |
|---:|---:|---:|---:|---:|---:|
| 424 | 104 | 12.693 | 18.051 | −0.800 | −552.6 |
| 612 | 2 | 13.116 | 18.509 | −6.396 | −11.8 |
| 2,343 | 205 | 12.823 | 18.290 | −0.120 | −1,115.5 |
| 8,007 | 48 | 12.697 | 18.072 | +0.736 | −251.9 |
| 28,845 | 25 | 12.749 | 18.200 | +1.461 | −129.4 |

초기 순절감은 이미 준비한 P HBM의 Full28 KV Copy에서, 선택한 D 행의 P-side Packing과 물리 전송을 뺀 값이다. P-side pack은 단일 CUDA Event 관측이라 작은 Prompt의 cold 비용 변동에 민감하다. Packing을 0으로 둔 낙관 조건에서 가장 긴 Prompt의 절감은 **3.30ms**이고 D→P 실질 지연을 포함한 반복 비용은 **5.45ms/Decode Step** 늘었다. 이 경로에서는 한 Decode Step 이전에 초기 이득이 소진된다. Fixed Prompt 전체를 P에 두는 대조도 최대 초기 4.79ms를 절약했으며 같은 조건에서 Decode Step이 5.44ms 늘었다.

Visual 경로에서 양의 초기 이득을 가진 두 요청의 손익분기점은 **0.14 및 0.27 Decode Step**이다. 그림의 곡선은 실측한 초기 비용과 고정 Teacher Step의 중앙 시간을 실제 Native 출력 길이에 적용한 분석이다. 각 출력 길이의 실제 온라인 실행을 의미하지 않는다. NIXL 등록, KV Admission, Queue, P의 다른 작업 및 vLLM CUDA Graph는 이 표에 포함되지 않는다. [손익분기점 Figure](../mllm_specific_disaggregation_discovery/results/a/figures/figure10_visual_memory_break_even.png) · [재생 CSV](../mllm_specific_disaggregation_discovery/results/a/decode_cost_and_break_even.csv)

## 반복 전송과 Sync

Batch1의 매 Layer에서 D→P Query **7,168 bytes**, P→D 정규화 Partial **7,168 bytes**와 FP32 LSE **112 bytes**를 실제 전송했다. Full28에서 합계 **404,544 bytes/Decode Step**이다. 128회 별도 측정의 NVLink counters는 단일 Layer의 D→P와 P→D에서 각각 **917,504 B**와 **933,888 B**를 관측했다. Transfer payload의 이론값은 각각 917,504 B와 931,840 B다. Counter는 link별 KiB 단위와 장치 범위를 가진다.

이 실행의 CUDA Event에서 Full28 Query Copy 구간은 **0.75–0.76ms**, Result Copy 구간은 **0.48–0.66ms**, Merge는 **0.81–0.83ms**, D의 Local Attention 완료 이후 remote 결과를 기다리는 구간은 **0.31–0.32ms**였다. 이 범위는 각 Layer를 합산한 관측이다. Local 및 Remote Stream이 겹치고 Host 발행 간격도 영향을 주므로 이 구간들을 단순 합산하여 물리 통신 지연을 구하지 않는다.

Local FA3 Attention은 계측한 B1 Full28 Decoder Layer 합에서 **8.5–11.0%**였다. Local MLP/Norm/Residual과 QKV/Norm/MRoPE가 대부분의 시간이었다. 실제 입력의 Visual KV 비중이 높은 경우에도 KV만 옮겨서 전체 Decode 작업을 줄일 수 있는 범위는 제한적이었다. 이 계측은 실제 Model과 CUDA kernels를 실행한 격리 경로다. Native의 운영 TPOT는 별도 보고에 남겼다. [Decode 비용 Figure](../mllm_specific_disaggregation_discovery/results/a/figures/figure8_decode_cost.png) · [Native 기준선](mllm_discovery200_native_baseline.md)

## Host 발행 비용 대조

424행과 28,845행에서 fixed Full28 CUDA Graph 대조를 네 방식으로 시도했다. Local Graph는 캡처됐다. Cross-device 경로는 uncaptured readiness event를 external event로 바꾼 뒤 PeerAsync의 capture 제한을 확인했다. UVA Copy로 모두 캡처할 수 있었지만 반복 replay에서 invalid argument 또는 SIGSEGV가 발생했고, Layout별 Buffers/Events를 독립적으로 유지한 재시도에서도 완료되지 않았다. Graph 대조의 비교 가능한 시간은 **FAILED_EXECUTION**이다. Eager 경로의 메모리와 손익분기 결과는 해당 구현에 한정된다. [실패 로그와 실행 영수증](../mllm_specific_disaggregation_discovery/results/a/graph/) · [Control 소스](../mllm_specific_disaggregation_discovery/attention_graph_control.py)
