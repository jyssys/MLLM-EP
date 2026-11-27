# 실제 세 H200의 EADA 온라인 Serving

동일 1E/1P/1D에서 원본 Qwen2.5-VL BF16을 실행했다. E는 실제 원본 이미지를 읽고 Vision Encoder를 호출했다. P는 원본 LM Prefill과 MRoPE를 사용했고, D는 128-token paged KV로 연속 입장과 Decode를 수행했다. E를 보조로 쓰는 경우에도 E가 추가로 보유하는 것은 Vision 모델과 Prompt KV다. D가 매 Layer 전달하는 것은 현재 Query와 반대 방향의 partial output 및 LSE다.

## 대조군과 측정 계약

- **NativeStrong EPD**는 기존 vLLM/IPC/NIXL의 HTTP SSE 고객 경계다. D-only Eager 또는 사용자 코드 때문에 느려진 비교군을 주 Offline 기준으로 사용하지 않았다.
- **Prototype Local**은 실제 A/B에서 같은 E/P와 paged FA3를 사용하고, 모든 KV를 D에 둔다. 정적 EADA, P-assisted, Visual-first, Admission-gated의 직접적인 Method 비교다. Queue, ingest 및 생성 토큰의 방출 시각을 동일한 Python 내부 경계에서 측정했다. 첫 토큰은 P가 계산하고 D에 페이지가 입장할 때 방출했다.
- QPS1의 100개와 QPS2.5의 200개 원본 cohort는 이미 실측한 NativeStrong의 요청 순서와 예정 시각을 그대로 재사용한다. QPS2.5의 200번 도착은 Main100의 두 순환이다. 4–32K 입력과 16–256 출력의 고정 통제는 Natural EOS와 별도다.
- 역사적 joint SLO는 TTFT ≤5,000 ms, 모든 ITL ≤34 ms, TPOT ≤33 ms다. 실패는 분모에 들어간다. Cohort 종료율과 Output throughput은 도착과 drain까지 포함한다. 장시간 지속 arrival에서의 maximum sustainable QPS와는 다른 측정이다.
- 초기 Prototype은 한 Worker에서 전처리와 원본 Transformers Vision SDPA를 실행했다. 개선한 대조는 CPU renderer를 별도 Queue로 분리하고 D/E에 고정 CUDA Graph를 사용한다. Packed FA3 Vision은 별도 대조로 검사했으며 입력 4개 중 1개의 Prefill 첫 토큰을 바꿨다. Primary Graph 실험은 원본 Vision SDPA를 유지했다.

## 정확성 검증

Eager 전환에서는 556, 30,536행의 B1과 556/4,651/9,944/30,536행의 ragged B4를 검증했다. Local 반복의 logits 및 마지막 generated KV가 요소별 같았다. D와 E에서 검사한 실제 Prompt shard는 요청 및 shard별 56/56 Full28 K/V Tensor가 원본과 같았다. 같은 teacher prefix의 EADA Greedy **90/90**, 별도 B1 Natural **2/2**가 일치했다. 다만 BF16 partition rounding이 전체 모델을 거치면서 최대 logit Relative L2는 0.0366, generated KV는 0.00273였다. 전체 State의 bitwise invariance와 공식 비열등성은 별도 기준이다. [실제 Full28 witness](../encoder_assisted_decode_attention/results/model_witness_eager.json)

최종 Graph의 25% 분할은 같은 입력에서 원본 Eager Local 대비 Teacher Greedy **90/90**, Graph Local의 logit bitwise 일치, 모든 실제 Prompt shard의 56/56 K/V 일치를 확인했다. Assisted의 최대 logit Relative L2는 0.02914, generated KV는 0.00316이었다. B1 Natural은 **1/2**가 일치했다. 30,536행 요청의 첫 차이는 0-based 생성 토큰 34였으며 두 결과 모두 EOS를 방출했다. Operator가 계산하는 효율적 Softmax 분할은 수학적으로 같은 KV와 분포를 사용한다. BF16에서 결과가 달라진 범위는 Task 품질 비교에 남겼다. [Canonical Full28 witness](../encoder_assisted_decode_attention/results/model_witness_graph_canonical.json)

독립 Full28 replay에서 첫 차이가 난 생성 토큰 index 34 직전까지 Local과 EADA에 같은 token prefix를 강제했다. Local은 ID 419의 29.000, ID 330의 28.875를 기록해 0.125 차이로 419를 선택했다. EADA는 둘 다 29.125가 된 BF16 tie에서 낮은 ID 330을 선택했다. 같은 prefix에서 logit Relative L2는 0.01473, generated KV는 0.00146이었다. 첫 차이 이후의 Autoregressive 입력 변화와 동일 prefix의 수치 차이를 구분했다. [첫 차이 재생](../encoder_assisted_decode_attention/results/model_witness_profile.json)

## 첫 Eager 대조

| QPS1 Natural, 동일 100개 | 완료 | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| Prototype Local | 100 | 21,733.6 ms | 19.77 ms | 30.78 ms | 40/100 | 0.3252 |
| Eager E helper · 일반 25% | 100 | 28,007.0 ms | 78.15 ms | 141.91 ms | 2/100 | 0.0151 |

같은 두 open-loop의 최대 D prepared batch는 각각 5였다. E GPU의 Vision enclosure p95는 1,818.2→2,367.5 ms였다. E-assisted Decode host step의 75.79%가 E GPU Encode의 host enclosure와 겹쳤다. 여기에는 GPU 발행과 wait가 포함되므로 실제 kernel overlap과 분리한다. 요청별 Natural token sequence는 76/100 일치했다. 첫 prototype Local은 native production TTFT p95 2.78 s보다 느렸다. 원인은 주로 E queue p95 약 20.36 s였으며, E의 Vision 통합을 먼저 개선한 뒤 양쪽 attention graph 대조를 다시 구성했다. [Eager raw 요약](../encoder_assisted_decode_attention/results/eager_online/summary.json)

## Canonical Graph와 통신 비용

같은 100개 도착에서 Local과 Static EADA가 각각 5,467개와 5,289개의 Natural tokens를 생성했다. 둘 다 100건을 EOS까지 완료했고 token sequence는 76/100 일치했다. 동일 생성 계약이라도 수치 차이가 실제 출력량을 바꿀 수 있으므로 Fixed64 Replay를 별도로 수행했다.

| QPS1 Graph, Natural EOS | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO | Goodput req/s | D→보조 왕복을 사용한 요청 |
|---|---:|---:|---:|---:|---:|---:|
| 1E/1P/1D Local | 15.312 s | 23.81 ms | 34.59 ms | 38/100 | 0.328 | 0 |
| E-assisted, 일반 25% | 20.489 s | 103.93 ms | 148.20 ms | 16/100 | 0.132 | 100 |
| P-assisted, 일반 25% · ingress 수리 전 | 14.380 s | 34.85 ms | 143.94 ms | 30/100 | 0.265 | 100 |
| E-assisted, Visual-first 25% | 20.259 s | 95.59 ms | 154.04 ms | 15/100 | 0.124 | 100 |
| E-assisted, Admission-gated 25% | 15.586 s | 128.40 ms | 195.27 ms | 26/100 | 0.226 | 24 |
| Encode+Prefill / 두 Local D | 40.184 s | 40.28 ms | 52.54 ms | 15/100 | 0.104 | 0 |

E-assisted에서 P→D의 논리 Prompt KV는 63.25→47.44 GB, 새 P→E는 15.81 GB였다. 왕복 논리량이 2.10 GB 추가됐다. P-assisted는 15.81 GB를 원래 P 장치에 보유했다. Local과 Static E의 초기 입장/전달 p95는 38.17→172.40 ms였고, full P→D 이후에만 방출하는 SSE stock connector 계측과 경계가 다르다. 보조 GPU를 사용하는 D host-step 전체 시간의 79.38%는 E의 GPU 호출 enqueue–완료 범위와 겹쳤다.

Actual B1에서 E 실행 플래그가 꺼진 상태와 켜진 상태의 D step 중앙값은 Local **10.79→14.02 ms**, EADA **11.73→49.70 ms**였다. EADA Actual B2는 **11.63→64.85 ms**였다. 실제 요청의 Context 및 출력 차이에 영향을 받는 관측 조건으로, 같은 workload에서 실행을 순차와 동시에 전환한 격리 간섭 대조와 함께 판단한다. [격리 간섭과 GPU 분할](mllm_encoder_assisted_decode_attention_microbenchmark.md)

Visual-first는 선택마다 generic arm과 같은 수의 KV 행을 옮겼다. TPOT p95는 낮았고 ITL p99와 Goodput은 악화했다. 현재 E의 Encoding Queue가 비어 있을 때만 보조 KV를 배치한 arm은 Static보다 높은 Goodput을 기록했다. 이후 새 Encode가 들어오면 이미 배치된 KV를 로컬에서 즉시 읽을 수 없다. 이 때문에 TPOT/ITL tail이 더 커졌고 Local을 넘지 못했다. 이번 정책은 request admission 단계에서만 실행된다. Migration, demand fetch 및 KV replica는 이 arm의 비용과 행위에 포함되지 않는다.

| QPS1 mixed trace 내부 구간 | 건수 | Input rows p50 | Local→E helper TPOT p95 | Local→E helper Joint SLO 요청 |
|---|---:|---:|---:|---:|
| 이미지 1장 | 13 | 4,847 | 16.57→104.95 ms | 8→5 |
| 이미지 2–4장 | 45 | 4,651 | 24.89→103.14 ms | 21→8 |
| 이미지 5–19장 | 8 | 17,700 | 20.98→99.63 ms | 2→0 |
| 이미지 20장 이상 | 34 | 15,667 | 21.91→99.79 ms | 7→3 |
| MMLongBench-Doc | 37 | 14,869 | 21.39→99.54 ms | 8→3 |

각 행은 같은 mixed arrival에서 특정 입력을 골라 집계했다. MMLongBench-Doc는 이미지 수 구간과 겹친다. 따라서 1장 구간에도 다른 이미지 요청의 Encode가 동시에 실행될 수 있다.

Normalized 품질의 해석은 작은 Main100의 실제 점수 및 평가 범위를 따른다. QPS1 Local과 Static E는 Mantis 11/25, DocVQA raw ANLS 0/13, MP-DocVQA raw ANLS 0.04287로 각각 같았다. MMLongBench-Doc는 원본 응답을 직접 비교한 진단만 수행했고 공식 추출은 `NOT_RUN`이다. DocVQA의 floor와 MP-DocVQA의 낮은 baseline 때문에 이 표본에서 품질 비열등성 Gate는 미확정이다. [동결 scorer와 요청별 결과](../encoder_assisted_decode_attention/results/online/quality.json)

각 요청의 동일 step, 총 KV rows, 실제 B1 및 이미 생성한 모든 token prefix를 맞춘 사후 대조에서 E/P가 그 step 전체 동안 비활성이었던 Local 대비 E helper의 paired median은 **+0.90 ms**였다(449쌍). E가 step 시작 때 활성이고 P가 비활성인 경우는 **+51.79 ms**(127쌍)였다. 같은 조건에서 P helper는 **+0.94 ms**(466쌍)였다. Stage enqueue는 서로 다른 시행에서 발생하므로 조건에 맞는 요청만 남긴 진단이 런 간 인과 d-separation을 보장하지 않는다. [동일 prefix 및 batch 교차 계측](../encoder_assisted_decode_attention/results/online_interference_diagnostic.md)

별도 고정 64-token 검증에서는 양쪽에 Packed FA3 Vision을 사용해 원래 Prototype의 Encoder 병목과 Natural 출력 길이가 결과를 가리는지 검사했다. 양쪽 모두 **100/100**과 **6,400 tokens**를 완료했다. Local→E helper의 TTFT p95는 **4.597→8.432 s**, TPOT p95 **49.34→86.36 ms**, ITL p99 **165.79→817.31 ms**, Goodput **0.635→0.264 req/s**였다. 양쪽 모두 실제 B8을 사용했고 E helper arm의 live KV 최대는 128,754행이었다. Packed Vision 수치 경로의 차이 때문에 이 결과를 Canonical Natural 및 Task Quality 비교에는 섞지 않았다.

## 200-arrival QPS2.5 대조

| Medium-load Graph, Natural EOS | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO | Goodput req/s | 전체 완료 req/s |
|---|---:|---:|---:|---:|---:|---:|
| 독립 시작 1 · Local | 81.805 s | 31.26 ms | 40.98 ms | 4/200 | 0.0238 | 1.188 |
| 독립 시작 1 · E helper | 112.886 s | 97.20 ms | 166.28 ms | 0/200 | 0 | 0.993 |
| 독립 시작 2 · Local | 84.207 s | 29.53 ms | 38.95 ms | 8/200 | 0.0467 | 1.168 |
| 독립 시작 2 · E helper | 117.509 s | 102.26 ms | 163.71 ms | 0/200 | 0 | 0.972 |
| 독립 시작 3 · Local | 80.725 s | 28.97 ms | 38.18 ms | 10/200 | 0.0598 | 1.196 |
| 독립 시작 3 · E helper | 107.422 s | 103.45 ms | 188.54 ms | 0/200 | 0 | 1.022 |

세 독립 시작의 두 arm이 모두 **200/200** EOS까지 완료했다. 두 번째 블록은 E helper를 먼저 실행했다. 쌍별 악화 범위는 TTFT p95 **33.1–39.5%**, TPOT p95 **210.9–257.1%**, ITL p99 **305.8–393.8%**였다. Prepared B 최대는 **7/8**이고 D Attention step에서 실제 전체 KV rows는 각각 최대 **60,227/87,668행**이었다. N3의 관측 범위를 기록했다. 같은 Torch 기반 엔진에서 E/P Queue가 형성되는 QPS2.5 값으로 prototype의 병목도 포함한다. 같은 도착의 별도 Stock vLLM 실험은 TTFT p95 8.19 s였다. [능동 온라인 원자료와 Stage 지표](../encoder_assisted_decode_attention/results/online/summary.json) · [Stock QPS2.5](mllm_encoder_assisted_decode_attention_motivation.md)

200번 동일한 도착을 비교한 Local A/A 세 조합의 Natural token sequence exact는 **173, 174, 174/200**이었다. 각 독립 블록의 Local/E helper agreement는 **156, 155, 159/200**이었다. EADA의 추가 출력 drift와 Continuous Batching에서 비교 경로 자체의 drift를 따로 관찰했다. 이 exact sequence 통계는 태스크 평가와 별개다.

## 최적화된 동일 3GPU HTTP 대조

| 같은 QPS1, 같은 100 도착, Native vLLM | 완료 | TTFT p95 | TPOT p95 | ITL p99 | E2E p95 | Joint SLO | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong 1E/1P/1D | 100/100 | 2.784 s | 5.07 ms | 9.00 ms | 3.175 s | 99/100 | 1.015 |
| vLLM Colocated DP3 | 100/100 | 2.309 s | 8.08 ms | 7.71 ms | 3.010 s | 90/100 | 0.924 |

DP3는 각각 독립 BF16 vLLM replica를 두고 동일 입력을 round-robin으로 라우팅했다. GPU별 90% VRAM 예산과 128행 KV block, chunked prefill, 2개 renderer, Prefix/processor cache off를 사용했다. 최초 startup은 processor cache와 다중 renderer 조합이 runtime에서 거절됐다. 동일한 unique 이미지 도착을 cache off로 다시 실행했고 세 replica 각각의 warmup과 모든 measured request의 원본 Input Tokens 검증을 통과했다. DP3의 client ITL p99는 낮았고 최대 ITL은 **458.96 ms**였다. 모든 ITL을 확인하는 SLO에서는 NativeStrong이 두 생산 기준선 중 높은 Goodput을 냈다. 설치 엔진에서 같은 Hq28/Hkv4와 3GPU를 유지하는 stock DCP 대조는 구성 불가였다. [HTTP/SSE Raw SHA와 모든 percentile](../encoder_assisted_decode_attention/results/online/http_controls.json)

## 수정 후 E-assisted와 P-assisted 역할 대조

Text-only 재현에서 동일 GPU에 Prompt KV를 남기는 P ingress의 index buffer가 다른 CUDA stream에서 재사용될 수 있는 문제가 드러났다. index 업로드와 gather를 동일 ingress stream에서 실행하도록 수정했다. P가 원래 생성한 Prompt를 실제 P arena에 복사해 모든 검사에서 **56/56** K/V와 Teacher Greedy **90/90**을 확인했다. 같은 소스로 QPS1의 세 역할을 다시 독립 실행했다. 이전 N3 EADA 비교는 그대로 동일 기존 소스에서 쌍을 맞췄다. [수정 후 물리 P witness](../encoder_assisted_decode_attention/results/model_witness_prefill_fixed.json)

| 같은 QPS1 Natural, 수정 소스 | 완료 | TTFT p95 | TPOT p95 | ITL p99 | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Local | 100/100 | 14.715 s | 24.34 ms | 37.84 ms | 0.344 |
| E-assisted, 25% | 100/100 | 19.872 s | 94.81 ms | 145.39 ms | 0.100 |
| P-assisted, 25% | 100/100 | 15.010 s | 30.91 ms | 172.86 ms | 0.281 |

두 helper arm은 각각 14.724 GiB의 논리 Prefix KV를 보유했다. E-assisted는 그만큼의 신규 P→E peer transfer를 수행했다. P-assisted에서 초기 총 peer bytes는 Local **58.904 GiB**에서 **44.180 GiB**로 줄었다. 두 방식 모두 Local보다 Goodput이 낮았고 서로의 tail 비용도 달랐다. [수정 후 동일 역할 Raw](../encoder_assisted_decode_attention/results/llm_control/role_confirmation/)

## KV Capacity 및 Encode-heavy 진입

같은 128개 이미지 다수 요청, QPS6 offered Poisson arrival, 고정 128 token과 같은 D arena 4 GiB를 사용했다. E helper arm에서만 E에 약 8 GiB의 보조 arena를 추가했다. 실제 P→E KV를 할당하고 그 상태로 Full28 Decode를 진행했다. 할당량, D/E peak active pages, waiting retries와 완료 시간을 모두 기록했다.

| D arena 4 GiB | 전체 완료 | 최대 실제 Prepared B | D/E peak pages | KV wait 시도 | 전체 완료 req/s | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| Local | 128/128 | 4 | 502/0 | 0 | 0.721 | 0.0113 |
| E helper · 일반 50% | 128/128 | 8 | 581/574 | 162 | 0.668 | 0 |

이 대조의 Local은 가용 585 page 중 502개만 사용했다. 관측된 B 증가는 E에 실제 KV를 유지할 수 있다는 근거다. EADA가 Decode에 더 오래 머물러 큐가 커져 생긴 부분도 있다. Local에서 KV page wait가 없었고 EADA에서 ITL p99가 31.27→235.51 ms로 늘었다. 같은 trace에서 처리율이나 SLO-goodput 이득은 관측되지 않았다. [실제 capacity cohort](../encoder_assisted_decode_attention/results/online/summary.json)

32개 실제 다중 이미지 요청을 256 output tokens, QPS6, D arena 24 GiB, configured B16으로 주입한 별도 stress는 두 arm 모두 32/32 완료했다. Actual B 최대는 **5→14**, 최대 live KV rows는 **84,441→230,167**, TPOT p95는 **16.30→66.13 ms**, ITL p99는 **29.39→169.34 ms**였다. 처리율은 **0.652→0.600 req/s**였고 양쪽 Goodput은 0이었다. 실제 긴 Decode에서조차 관측된 KV rows는 격리 시험의 20% 물리 단축을 처음 얻었던 262,144행보다 작았다.

같은 고정 128-token workload에서 D의 arena를 **2 GiB**로 줄이면 양쪽 모두 292장의 D page를 실제 점유했다. 이 아레나 제한은 프로덕션의 42 GiB와 별개인 KV 압력 스트레스다. EADA는 E에서 289장을 추가로 썼고 최대 Prepared Batch가 **3→5**로 늘었다. KV 페이지를 한 번이라도 기다린 요청은 **80→75/128**, 시도는 **8,763→4,273회**였다. retry 횟수는 실제 대기 시간과 다른 측정이다. 양쪽이 **128/128**을 완료했지만 처리율은 **0.739→0.579 req/s**, TTFT p95 **144.84→196.13 s**, ITL p99 **30.11→392.82 ms**로 악화했다. 이 부하의 Joint SLO는 양쪽 0건이었다. [동시 입장 및 실제 사용 페이지](../encoder_assisted_decode_attention/figures/online_capacity.png)
