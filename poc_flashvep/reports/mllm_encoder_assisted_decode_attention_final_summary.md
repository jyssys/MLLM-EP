# H200 EADA · 종합 결과

**최종 판정 `NO_GO_ONLINE_BENEFIT`**. 동일 3GPU의 실제 open-loop에서 정적 EADA는 Decode 지연을 늘리고 SLO-goodput을 낮췄다. 물리적으로 여유 VRAM을 활용하는 것은 가능했다. 로컬 Attention보다 빠른 격리 형상은 발견됐으나 실제 Serving에서 준비된 배치의 KV 행 수가 그 구간에 도달하지 않았고 Vision Encode가 원격 Attention 지연을 크게 늘렸다.

EADA는 세 H200에서 원본 Qwen2.5-VL의 Vision Encoder와 일부 Prompt KV를 한 GPU에 두고, D가 생성한 Query와 두 GPU의 Partial Attention 결과를 Layer마다 주고받았다. Encoder GPU의 Weight는 **1.260 GiB**로 Vision 모델만 포함한다. 두 LM Worker는 각각 약 **14.185 GiB**를 사용했다. 모든 원본 Prompt KV 행을 보존했으며 Sequence 또는 Weight pruning, quantization, 근사 attention은 사용하지 않았다.

## 독립 검증

| 질문 | 실측과 증거 범위 |
|---|---|
| E에 실제 KV 공간이 있는가 | 별도 Stock 시작에서 실제 **80 GiB**를 E에 할당하고 모두 써서 같은 QPS2.5 Main100 **100/100**을 완료했다. E peak VRAM은 **35.68→116.28 GiB**였다. P는 같은 도착에서 물리 **64 GiB**를 보유한 채 **100/100** 완료했다. Warm P free **77.65 GiB**에서 사전 4 GiB 보호 기준에 따라 P 80 GiB 할당을 거절했다. E80의 ITL p99가 **9.04→10.91 ms**여서 메모리 상주와 latency 보호를 별개로 판정했다. |
| D Attention과 E의 자원 여유가 겹치는가 | Stock QPS2.5 두 trace에서 D Attention과 E 물리 idle의 교집합 **27.90–30.29%**, E kernel과 E HTTP 둘 다 없는 교집합 **11.22–11.63%**였다. P의 같은 교집합은 **4.86–5.24%**였다. D Attention이 실행된 GPM window에서 E DRAM 활동 지표는 평균 **21.21–21.93%**였다. 두 Observer 대조 모두 5% 검증에 실패해 이 값들은 **DIAGNOSTIC**이다. |
| 전체 Decode의 Attention 비중 | Stock 두 trace에서 D의 GPU-active 시간 중 **11.32–11.34%**였다. 첫 trace의 D HTTP union 기준으로는 **7.14%**였다. 같은 물리 범위에 20% Attention 단축을 적용하는 이상적 계산은 D GPU-active 시간의 약 **2.26%**에 해당한다. |
| 실제 GPU 간 Attention이 빨라지는가 | Qwen GQA, BF16, B8×32K의 고립된 CUDA Graph에서 **133.13→103.26 µs**, **22.44%** 개선. 실제 Qwen ragged B4에서는 최선의 범용 25% 분할도 **41.08→44.90 µs**, **9.31%** 악화. Controlled 25개 shape 중 10개에서 격리 20% Gate를 통과했다. |
| Vision Encode와 동시에 실행해도 이득이 남는가 | Controlled B8×32K에서 stock Vision의 522행과 동시 실행하면 EADA attention이 같은 동시 Local보다 **10.7%** 빨랐다. 2,262/7,935행에서는 **8.6/8.1%** 느려졌다. 실제 ragged B4는 그 세 Vision 크기에서 **20.2/39.4/99.0%** 느려졌다. Encoder-first는 단일 D burst에 **45.4–501.8 ms** 대기를 추가했다. |
| 실행 중 발생하는 이동량은 얼마인가 | B8×32K 단일 Layer에서 Q **57,344 B**, E→D partial+LSE **58,240 B**. 128번 반복한 실제 NVLink DATA가 각각 **7,340,032/7,454,720 B**였다. QPS1 Prototype Graph 대조는 100요청에서 P→D **63.25→47.44 GB**, 새 P→E **15.81 GB**, 28-layer 반복 E↔D 논리 payload **2.10 GB**였다. |
| 계산 및 출력이 보존되는가 | Local BF16 FA3 대비 FP32 계산 상대오차 최대 **0.00328**. Full28 Graph에서 Prompt KV shard는 **모든 검사에서 56/56**, 동일 teacher prefix의 Argmax는 **90/90**. Natural B1 **1/2**로, 긴 요청은 생성 token index 34에서 갈렸다. 동일 prefix 재생에서 Local top-2 margin이 0.125, EADA는 BF16의 0.0으로 바뀌었다. |

[Phase 0 리소스와 계측](mllm_encoder_assisted_decode_attention_motivation.md) · [Physical FlashAttention-3 및 NVLink 결과](mllm_encoder_assisted_decode_attention_microbenchmark.md)

## 실제 온라인 Primary 비교

세 H200, 같은 Main100, 같은 예정 도착, 원본 Vision SDPA 및 128행 Paged FA3를 사용했다. Prototype 대조의 token emission은 동일한 Python 내부 경계에서 기록했다. Stock vLLM EPD는 최적화된 별도 서비스의 HTTP SSE client 경계다.

| QPS1, Natural EOS | 완료 | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO Goodput |
|---|---:|---:|---:|---:|---:|
| Prototype Graph Local | 100/100 | 15.312 s | 23.81 ms | 34.59 ms | 0.328 req/s |
| Prototype Graph EADA · 일반 25% | 100/100 | 20.489 s | 103.93 ms | 148.20 ms | 0.132 req/s |

정적 EADA는 같은 엔진의 Local 대비 TTFT p95 **+33.81%**, TPOT p95 **+336.4%**, ITL p99 **+328.4%**, Goodput **−59.7%**였다. 실제 Batch 최대는 **4→7**이었지만 Attention이 분산 실행에서 20% 빨라진 controlled Grid의 가장 작은 교차 geometry인 **262,144 KV rows**에 도달한 Decode Step은 두 arm 모두 0이었다. Encoder GPU service enclosure p95는 **1,865→2,204 ms**였고, 요청당 입장과 초기 KV 전달 p95는 **38→172 ms**였다.

QPS2.5의 세 독립 시작에서 같은 200건을 매번 양쪽 모두 완료했다. EADA는 매 블록에서 TTFT p95 **33.1–39.5%**, TPOT p95 **210.9–257.1%**, ITL p99 **305.8–393.8%** 증가했다. Joint SLO 완료는 Local **4/8/10건**, EADA **0/0/0건**이었다. Local A/A의 exact Natural 출력은 **173–174/200**, Local/EADA는 **155–159/200**이었다.

원본 Vision backend 및 동일 64 output tokens의 실제 이미지 Replay에서도 두 비교군 모두 **6,400 tokens**를 생성했다. 정적 50% EADA에서 TPOT p95는 **19.29→89.43 ms**, ITL p99는 **32.21→154.86 ms**, Goodput은 **0.347→0.100 req/s**였다.

두 최적화된 HTTP 대조의 QPS1 Goodput은 3-replica co-located vLLM **0.924**, NativeStrong EPD **1.015 req/s**였다. 2 GiB D arena를 포화시킨 별도 capacity stress에서는 실제 동시 Prepared Batch가 **3→5**로 늘었고 페이지를 기다린 요청이 **80→75/128**로 줄었다. 두 arm 모두 **128/128** 완료했고 완료율은 **0.739→0.579 req/s**로 낮아졌다. QPS1에서 같은 행 수를 옮긴 Visual-first는 TPOT p95가 일반 EADA보다 낮았으나 ITL p99 **148.20→154.04 ms** 및 Goodput **0.132→0.124 req/s**였다.

동일 Backbone geometry와 고정 64-token의 Qwen2.5 text 대조에서 2P/1D의 P-assisted는 전체 100건 중 47건에 적용했다. 초기 peer transfer를 **58.904→43.855 GiB** 줄였고 TPOT p95 **12.32→52.16 ms**, ITL p99 **18.01→246.12 ms**, Goodput **0.948→0.688 req/s**였다. 동일 Qwen-VL checkpoint의 text-only C3에서는 Vision 실행이 없는 E가 참여해도 Goodput **0.706→0.633 req/s**, ITL p99 **102.14→311.50 ms**였다. 같은 모델의 실제 이미지 역할 대조에서 P-assisted가 EADA보다 높은 Goodput **0.281 vs 0.100 req/s**를 유지했다. [동형 대조, Natural LongBench 및 실제 KV 전송](mllm_encoder_assisted_decode_attention_llm_control.md)

공식 공개 LongBench의 Natural EOS 대조에서도 양쪽이 **128/128** 완료했고 P-assisted의 TPOT p95 **11.90→25.14 ms**, ITL p99 **16.27→27.79 ms**, Goodput **0.798→0.557 req/s**였다. P-assisted와 E-assisted는 서로 다른 역할에서 초기 전송과 용량의 비용을 절약할 수 있었으나 검증된 온라인의 Local 경로를 넘지 못했다.

InternVL3-8B에서도 실제 1/3/6/16장 이미지 입력의 Native 고정 16-token Decode를 12/12 완료했다. 실제 16-image 요청에서 캡처한 post-RoPE KV 28,785행의 B1 물리 연산은 Local BF16 FA3를 기준으로 한 정합 검사 12/12를 통과했고 가장 유리한 원격 분할도 tuned Local보다 **51.0%** 느렸다. 50% 범용 분할은 **25.96→40.98 µs**였다. InternVL의 EPD 온라인 통합은 `NOT_RUN`이다. [두 번째 모델의 Native 실행과 실제 post-RoPE Tensor](../encoder_assisted_decode_attention/internvl_diagnostic.md)

독립 FP32 QK·Softmax·V와 비교한 별도 C0에서 Controlled B8×32K **9/9**가 통과했다. InternVL 실제 입력의 25개 비교는 모두 output relative L2 **≤0.001913**였고, Local 및 원격 경로 모두 같은 head에서 LSE 절대오차 **0.000244**로 고정 기준 **0.0002**를 초과했다. 문제 head에서는 Local과 모든 분할의 LSE가 동일했으며, 이는 독립 FP32 참조와 1 ULP 차이다. 모든 head에서 원격 대비 Local LSE 차이는 **≤0.0000611**였다. 절대 기준의 실패는 유지했다. [독립 FP32 검사](../encoder_assisted_decode_attention/results/operator/operator_fp32_witness.json) · [FP64 교차 진단](../encoder_assisted_decode_attention/results/operator/operator_fp32_lse_diagnostic.json)

[실제 open-loop 경계와 정확성](mllm_encoder_assisted_decode_attention_online_serving.md) · [동일 3GPU의 일반 LLM PD](mllm_encoder_assisted_decode_attention_llm_control.md)

## 연구 판단에 필요한 기준

이 최적화의 가능한 시스템 차별점은 표준 Dense Decoder의 Softmax 분할이 아니라, 기존 1E/1P/1D에서 LM Weight 없이 E의 VRAM과 서비스 여유를 활용하는 Placement다. Infinite-LLM은 이미 Query를 KV 소유 GPU로 보내고 Partial Attention을 결합하는 연산을 구현했다. EAServe는 Vision GPU가 LM Prefill을 맡는 경로를 연구했고 HydraInfer는 인코더와 Decode co-location을 연구했다. EADA의 독립적인 주장에는 같은 GPU 예산의 기존 구성과 P-assisted를 넘는 실측이 필요하다. [선행 시스템의 원문과 코드](../encoder_assisted_decode_attention/prior_work.md)

| Gate | 판정 및 증거 |
|---|---|
| 0 · Resource opportunity | **80 GiB 물리 상주 실측**. 두 profiler의 phase 교집합과 HBM 비율은 `DIAGNOSTIC`. 미래 500 µs fit은 `OFFLINE_ORACLE`. Deadline-safe 보조 실행 비율은 `INCONCLUSIVE`. |
| 1 · Physical execution | **Controlled 조건 통과**. 실제 Peer DMA와 LSE merge를 포함해 B8×32K는 22.44% 빨랐고 독립 FP32 검사는 9/9였다. 실제 Main100 ragged B4는 더 느렸다. InternVL의 추가 strict LSE 기준은 Local과 모든 분할에서 미충족이었다. |
| 2 · Online benefit | **미충족**. 동일 엔진의 QPS1과 QPS2.5 N3, fixed-output Vision 대조와 실제 KV 압력에서 latency와 goodput 기준을 충족하지 못했다. Pinned Scorer의 DocVQA floor와 MMLongBench-Doc 추출 미실행으로 Task Quality 비열등성은 `INCONCLUSIVE`. |
| 3 · Generality | **부분 검증**. InternVL3-8B의 실제 Native와 D/E operator는 `CORRECTNESS_ONLY` 및 격리 실측. 두 번째 모델의 EADA 온라인은 `NOT_RUN`. |
| 4 · MLLM-specificity | **비용·용량 차이 실측**, **온라인 추가 이점 미입증**. 실제 E의 VRAM은 더 많이 보유할 수 있었고 P는 초기 peer 전송을 더 적게 사용했다. 같은 Qwen2.5-VL의 수정 후 E/P 대조에서 E는 P보다 TPOT가 길고 Goodput이 낮았다. |

후속 연구는 원본 D의 KV admission이 실제 운영에 병목인 workload를 먼저 확인하고, 그 조건에서의 용량·Tail SLO를 한 쌍으로 평가하는 편이 타당하다. 이번 tested grid에서는 Remote Attention 가속을 주 연구 방향으로 계속 개발할 근거가 부족하다. Production vLLM에 EADA를 직접 통합한 경로와 장시간 최대 지속 QPS sweep은 `NOT_RUN`이다.

Prototype graph의 분석 테이블에는 **32 cohorts, 3,832번 도착**이 들어 있다. 동일 trace를 여러 arm과 restart에서 재사용했다. 원본 해시까지 재검증한 timing archive는 초기 진단을 포함해 **35 cohorts**다. 연구 checkout의 base는 `d6f1becf93ba843e98d289d247aae77d08b81d23`이다. 물리 GPU 실험을 마친 후 GPU 0–7의 `train_lm.py` 소유권과 현재 실행을 모두 확인했다. [Raw와 lease 영수증](../encoder_assisted_decode_attention/results/raw_online/manifest.json) · [GPU 복원 영수증](../encoder_assisted_decode_attention/results/burner_restore.json)

[고정 실험 계약과 Gate](mllm_encoder_assisted_decode_attention_poc_spec.md) · [재현 경로](../encoder_assisted_decode_attention/README.md)
