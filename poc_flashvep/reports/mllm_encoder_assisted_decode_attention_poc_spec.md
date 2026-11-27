# Encoder-Assisted Decode Attention · 실험 계약

기존 1E/1P/1D와 H200 NV18를 같은 방식으로 사용한다. E는 원본 Vision Encoder와 선택된 Prompt KV를 보유한다. P는 원본 Full LM으로 MRoPE를 적용한 Prefix KV를 생성한다. D는 LM 가중치와 Generated KV를 갖고, Layer마다 현재 Query를 만들어 E/D에서 독립적인 FlashAttention-3를 수행한다. FP32 log-normalizer로 Partition의 값을 결합한다. P가 만든 원본 K/V의 실제 행을 보존한다. E에는 LM Weight를 추가하지 않는다.

## 실험 순서 및 동결한 성공 기준

| Gate | 실측 대상과 기준 |
|---|---|
| 0 | 물리 H200에 실제 안전한 E의 KV capacity, 유효한 동시간 E/D Timer 교집합, Decode attention share. HBM metric의 정의와 Observer A/A를 기록한다. |
| 1 | 실제 query/result peer transfer를 포함한 격리 Attention 20% 이상 감소. FP32 reference에서 같은 행과 GQA, causal visibility를 검증한다. Relative L2 ≤0.005 및 cosine ≥0.9999를 보조 수치로 사용한다. 원본 BF16의 반복 간 변동과 Greedy/KV 정합성은 별개다. |
| 2 | 같은 도착과 GPU 수의 실제 open-loop에서 TPOT ≥10% 개선 또는 SLO Goodput ≥15% 개선. TTFT p95와 ITL p99의 regression은 각각 5% 이하. Encoder service p95 및 queue p95도 같은 arrival의 Local 대조 대비 5% 이하 악화. Zero queue는 milliseconds 차이를 함께 본다. 전체 실패를 분모에 넣는다. |
| 3 | 서로 다른 Dense MLLM의 같은 방법에서 각자의 online strong baseline 대비 실측. InternVL3의 Qwen2 계열 언어 backbone과 다른 encoder geometry를 구분한다. |
| 4 | 별도 Qwen2.5 text model 2P/1D, MLLM Prefill-assisted, 같은 3GPU의 Encoder-assisted와 동형 제어로 실제 자원·초기 전달·온라인 성능을 비교한다. |

실용 SLO는 비교의 기준선을 얻기 전에 정한다. TTFT ≤5,000 ms, 모든 ITL ≤34 ms, TPOT ≤33 ms를 기존 사용자 기준으로 유지한다. Profiler-on의 변동이 clean arms 대비 5%를 초과하는 경우 해당 타임라인은 diagnostic으로 남긴다. 동일 요청의 여러 token 또는 한 GPU 시동 안의 여러 cohort는 독립 반복으로 집계하지 않는다.

## 모델, Workload 및 runtime

- Qwen2.5-VL-7B-Instruct 원본 BF16과 Main100을 사용한다. 과거 Discovery200에서 같은 Main100을 원본 이미지, Prompt, visual rows 및 MRoPE metadata와 함께 찾는다. 최소 128 요청의 비교가 필요할 때 순환 도착을 명시하거나 stock의 읽기 전용 200-request trace를 그대로 재사용한다. Source 이미지가 같은 도착에는 개별 index를 부여한다.
- Independent control은 Apache-2.0 Qwen2.5-7B-Instruct와 원본 Qwen-VL text-only를 지원한다. Controlled text는 원본 MLLM의 각 LM token count를 강제한다. 자연 Text workload는 별도 문서형 공개 데이터로 구분한다. InternVL3-8B의 모델별 Processor로 actual tokens를 기록한다.
- NativeStrong vLLM H200 EPD의 CUDA IPC, NIXL, Paged FA3, Continuous Batching 경로가 프로덕션 비교군이다. 실험적 open-loop arm은 E/P/D가 실제 별도 GPU에서 작동하고, paged KV와 연속 D 입장을 사용하며 Python 내부 token emission을 기록한다. 동일 prototype의 Local FA3 arm을 method effect 비교에 사용한다. HTTP와 Python 경계는 각각 기록한다.
- 정적 E와 P helper는 Physical KV 분할을 바꾼다. Adaptive admission은 이미 관측한 E in-flight와 queue 상태만 보고 새 KV 배치를 선택한다. 기존 state를 매 step 이동하거나 재복제하는 정책 비용을 면제하지 않는다. E가 늦은 상황의 instant fallback은 replica 없이는 제공하지 않는다.

[기존 Track A의 같은 모델 P-assisted 대조](https://github.com/jyssys/MLLM-EP/blob/b4947a556134afc5214bdb1dd0412c7e961f989d/poc_flashvep/reports/mllm_visual_memory_remote_attention_break_even.md) · [선행 시스템과 코드](../encoder_assisted_decode_attention/prior_work.md)
