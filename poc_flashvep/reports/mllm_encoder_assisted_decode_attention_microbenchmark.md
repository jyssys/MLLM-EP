# EADA · 물리 GPU 분산 어텐션

## 측정 계약

Qwen2.5-VL-7B의 BF16 Q28/KV4/head128 형상에서 한 query가 유효한 모든 기존 row를 읽는다. 원래 MRoPE를 적용한 뒤의 K를 사용한다. Prompt KV를 Partitions 사이에 정확히 한 번만 배치하고 이번 decode의 신규 K/V 한 row는 항상 D에 둔다. 각 partition은 local softmax로 정규화한 가중 평균과 FP32 LSE를 제공한다. D에서 Global softmax 가중치를 복원해 두 partial 출력을 결합한다. BF16 partial의 결과는 Local BF16 FA3 기준과 비교하며 오차 지표는 FP32에서 계산한다.

물리 GPU4는 E, GPU5는 P이자 초기 KV source, GPU6은 D다. 같은 GPU6에서 128-row Paged KV, vLLM FlashAttention-3, CUDA Graph와 자동 및 수동 split-K를 비교하고 측정상 가장 빠른 Local 구성을 사용한다. 각 partition의 Attention은 FA3로 수행하며 D의 결합은 vLLM `merge_attn_states` CUDA 커널을 사용한다. 외부 stream은 LM weight를 추가로 적재하지 않는다. 실제 요청의 P에는 전체 원래 모델이 있고, isolated benchmark 중 E는 attention만 수행한다. [Operator 소스](../encoder_assisted_decode_attention/peer_attention.py)

## 구현과 대조

- **Eager PeerAsync.** D query event를 받은 E stream이 query와 partial state를 물리적으로 전달한다. D partial과 기기 event로 합류한다. 단계별 CUDA event는 별도의 Eager 계측이며 Host 제출 간격을 포함할 수 있다.
- **CUDA Graph DMA.** D/E 각각의 FA3를 로컬 child graph로 capture하고 Q와 두 FP32/BF16 응답 buffer의 UVA CUDA memcpy node를 dependency로 연결한다. 실제 reuse에서는 D launch 한 번을 기록한다.
- **CUDA Graph peer kernel.** E stream을 capture해 D 포인터에서 Q를 읽고 FA3를 실행한 뒤, BF16 partial과 FP32 LSE를 같은 sender kernel에서 D 포인터에 쓴다. 두 Graph 방식의 논리 payload는 동일하다. [CUDA Graph 런타임](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__GRAPH.html)
- **동일 geometry 대조.** Controlled case는 4K/8K/16K/32K/64K 및 B1/4/8/16/32다. Real ragged case는 동결 데이터의 원래 이미지, 모델, MRoPE, 첫 decode query 및 실제 cache를 사용한다. Visual-first와 동일 행 개수 generic prefix를 비교한다.

모든 candidate는 exact softmax와 같은 키 집합을 사용한다. Isolation 타이밍은 한 decoder layer, 고정 주소, 이미 준비된 batch다. 같은 주소에서 Q를 변경해 graph replay가 새 값을 사용한다는 witness도 확인한다. [실행 파라미터와 원자료 형식](../encoder_assisted_decode_attention/run_operator_sweep.sh) · [CSV 및 Figure 생성](../encoder_assisted_decode_attention/operator_summary.py)

최소 읽기량으로 계산한 KV 바이트는 분석값이다. NVML의 NVLink DATA 카운터는 준비된 런에서 link별 KiB를 합산한다. 측정한 물리 Link 구간과 HBM Read 대역폭은 별개다. 짧은 Context의 고정 반복에서는 KV가 D의 L2에 남을 수 있다. 실제 입력 및 다층 결과는 별도 범위로 보고한다.

## 실제 두 GPU의 전송 포함 결과

두 H200의 독립 FA3를 한 실행 그래프로 합성한 CUDA DMA 경로가 실행됐다. 최초 SM peer-copy 대조는 Triton이 한 분기의 BF16 포인터와 다른 분기의 FP32 포인터에 같은 이름을 받아 컴파일 오류로 중단했다. CUDA DMA 측정은 독립 프로세스에서 완료됐다. D attention과 Q copy → E attention → partial + LSE copy가 독립 그래프 branch이고, D merge는 두 branch가 끝난 뒤 실행된다. Q 변경 후에도 모든 결과를 확인했으며 양방향 NVLink DATA를 직접 관측했다. [GPU 원자료와 정합성 영수증](../encoder_assisted_decode_attention/results/operator/result_receipt.json)

| 실제 Prepared Shape | 가장 빠른 시험 분할 | Local Graph µs | Assisted Graph µs | Attention 지연 변화 |
|---|---:|---:|---:|---:|
| B1 × 32K | 25% | 28.17 | 40.28 | +42.98% |
| B4 × 32K | 25% | 72.42 | 63.12 | −12.84% |
| B32 × 8K | 50% | 128.65 | 100.90 | −21.57% |
| B16 × 16K | 50% | 133.27 | 103.96 | −22.00% |
| B8 × 32K | 50% | 133.13 | 103.26 | −22.44% |
| B4 × 64K | 50% | 130.48 | 102.04 | −21.79% |
| B32 × 32K | 50% | 488.93 | 278.29 | −43.08% |
| B32 × 64K | 50% | 963.46 | 519.56 | −46.07% |

Controlled 25개 shape에서 4개 분할씩 총 100개를 실행했다. 그중 10개 shape는 최소 한 분할에서 20% 물리 attention 단축에 도달했다. 표는 각 실제 prepared shape의 최선 비율 탐색 결과다. 이때 원래 prompt 길이에 이번 D token 한 row를 더했다. [전체 JSONL](../encoder_assisted_decode_attention/results/operator/controlled_paged.jsonl) · [측정 분할별 Figure](../encoder_assisted_decode_attention/figures/operator_partition_attention.png)

Physical B8 × 32K smoke는 Local 132.29 µs, 동시 CUDA DMA 101.53 µs, Eager PeerAsync 200.07 µs였다. B32 × 32K는 각각 489.02/280.27/298.66 µs였다. 편도 CUDA event는 Eager의 제출과 실행 범위다. B8의 정확한 논리 메시지는 Query 57,344 B, partial 57,344 B, LSE 896 B이다. 128 반복 동안 측정한 실물 NVLink DATA는 D→E 7,340,032 B, E→D 7,454,720 B로 모두 해당 논리 payload의 반복량과 일치했다. B32 및 다음 실제 Qwen ragged 대조에서도 일치했다. [전송 검증 JSONL](../encoder_assisted_decode_attention/results/operator/smoke_memcpy.jsonl) · [단계별 Eager 측정](../encoder_assisted_decode_attention/figures/operator_communication_enclosures.png)

## 실제 이미지 입력과 Visual-first 대조

Qwen에서 실제 4개 요청의 영상과 MRoPE를 이용한 Layer0 ragged B4는 [4,852, 8,008, 19,243, 28,846] KV rows였다. 가장 빠른 범용 25% 분할은 Local 41.08→44.90 µs로 9.31% 느려졌다. Visual-first와 같은 개수 prefix를 비교한 분할의 실제 CUDA DMA 결과는 다음과 같다.

| 명목 Visual Offload | 실제 helper rows | Visual-first µs | 동일 개수 generic prefix µs |
|---|---:|---:|---:|
| 25% | 1,204 + 1,983 + 4,800 + 7,200 | 44.83 | 45.39 |
| 50% | 2,409 + 3,967 + 9,600 + 14,400 | 53.07 | 56.39 |
| 75% | 3,614 + 5,951 + 14,400 + 21,600 | 57.60 | 58.40 |
| 100% | 4,819 + 7,935 + 19,200 + 28,800 | 67.25 | 69.10 |

이 시작에서 동일 작업량의 Visual-first는 최대 약 5.9%의 작은 차이를 관측했으나 두 분할 모두 Local보다 느렸다. 구조적 추가 이득을 가르려면 배치와 실제 배치 반복 간의 변동을 구분해야 한다. 별도로 P가 paged layout에 pack하고 실제 GPU로 복사한 setup은 Local 1.35 ms, 25% generic 4.86 ms였다. 이 값은 편성별 한 번의 동기화 범위이며 production NIXL 전송은 별도다. [원본 Qwen Layer0 및 실제 링크 원자료](../encoder_assisted_decode_attention/results/operator/native_qwen_ragged.jsonl)

Graph 시작 2개, controlled 100개, 실제 Qwen 12개 중 record한 changed-query witness는 114개 모두 통과했다. 모든 Graph 결과의 최대 relative L2는 0.00328, Eager normalized LSE의 최대 absolute error는 1.53e−5였다. BF16 partial 출력이 들어가는 수치 경로의 정합성이다. Encode 간섭, Full-model 출력 및 Online TTFT/TPOT/Goodput은 별도 실험이다. [재생 CSV](../encoder_assisted_decode_attention/results/operator/operator_summary.csv)

## 독립 FP32 연산 대조

같은 BF16 post-RoPE Q/K/V를 독립 FP32 GQA 연산으로 계산했다. 각 query head는 해당 KV head의 모든 유효 행을 읽는다. 기준은 FP32 scaled QK, logsumexp 및 softmax 가중 V이며, Local paged FA3와 물리 Eager·CUDA Graph split의 결과를 각각 비교했다. 통과 기준은 출력 relative L2 ≤0.008, cosine ≥0.9999, LSE 절대 오차 ≤2e−4였다.

Controlled B8×32K는 **9/9**를 통과했다. 최대 출력 relative L2는 0.00284, LSE 절대 오차는 1.907×10⁻⁶이었다. InternVL3-8B의 실제 28,785행 B1 fixture는 **0/25**였다. 출력 기준은 모두 충족했고 최대 relative L2는 0.00191, 최소 cosine은 0.99999857이었다. Local 단독과 모든 partition 경로의 최대 LSE 절대 오차는 동일하게 0.000244141이었으며 기존 한도를 넘었다. 전체 **9/34**의 엄격한 결과를 유지한다. [독립 참조 구현](../encoder_assisted_decode_attention/fp32_reference.py) · [엄격 기준 GPU 원자료](../encoder_assisted_decode_attention/results/operator/operator_fp32_witness.json)

같은 InternVL fixture를 독립 시작에서 FP64 LSE와 함께 재검사했다. Q/K의 절댓값 최대는 48/176였고 독립 FP32 LSE 범위는 −1149.764에서 2702.126이었다. 기존 FP32 절대 한도를 넘은 값은 25개 경로 각각의 query head 15 한 개였다. 그 값은 Native FA3와 모든 분산 경로에서 **2702.12548828125**, 독립 FP32에서 **2702.125732421875**로, 그 크기의 FP32 인접 수 한 칸만큼 달랐다. 독립 FP64 값 **2702.1258788496**과의 차이는 **0.000390568**이었다. 전체 head에서 분산 경로와 Native FA3의 LSE 차이는 최대 6.104×10⁻⁵와 1 ULP였고, 독립 FP32 기준과의 ULP 차이는 절대 오차가 작은 다른 head에서 최대 4였다. 절대 오차 한도 실패는 Native에서도 관측됐다. 재시작 결과는 같은 기준의 **0/25**이며, `--diagnose-lse`는 같은 기준을 적용하고 exit status 2로 종료한다. [FP64와 ULP 진단 원자료](../encoder_assisted_decode_attention/results/operator/operator_fp32_lse_diagnostic.json) · [원자료 해시](../encoder_assisted_decode_attention/results/operator/result_receipt.json)

## 온라인을 위한 동적 page-table 재사용

Graph를 시작할 때 D와 E의 고정 Arena 주소를 사용한다. Exact B와 고정 Context ceiling마다 Q, block table 및 valid-length tensor를 보관하고 replay마다 실제 값만 교체한다. E 메타데이터 갱신 후 물리 CUDA event로 D에 의존성을 연결한다. 현재 요청의 K/V 위치는 원래 Page owner가 저장하고 D만 새 토큰의 KV를 추가한다. Helper가 없는 요청이 같은 Batch에 섞이면 일반 Eager 경로를 사용한다. 이 때 빈 partial을 0으로 중화하는 FP32 merge를 실행한다. [재사용 구현](../encoder_assisted_decode_attention/paged_graph.py)

2개 Layer, B4, Context ceiling 256의 Local/Assisted Graph를 valid dummy page로 먼저 준비했다. 같은 실행 주소에서 두 종류의 실제 valid length, block table offset 0→8과 다른 Query를 사용했다. 8/8 출력이 독립으로 발행한 FA3 결과와 bitwise 일치했다. 두 Layer의 두 Bundle 준비 시간은 869 ms였다. Mixed-helper Batch는 명시한 Eager fallback에 들어갔다. 이 값들은 동적 메타데이터 correctness 대조다. 실제 Full28 준비 시간 및 Hot/Clean percentile은 Online 영수증에 분리된다. [Graph reuse GPU witness](../encoder_assisted_decode_attention/results/operator/paged_graph_witness.json)

Full28의 최초 Replay는 긴 B1에서 0/15, ragged B4에서 0/60 Teacher Argmax였다. 같은 BF16 Prompt의 물리 KV Import는 맞았고 작은 동적 page-table 대조는 통과했다. 후속 GPU 진단은 180–30,537개 행의 난수 BF16 KV로, Dummy Warmup 이후 테이블과 유효 길이를 바꾸며 독립 Eager와 비교했다. B1/B4에서 num_splits 0/1/4/8의 2-layer와 num_splits 0/1의 28-layer 비동기 발행이 모두 수치 범위를 통과했다. 후자의 최대 relative L2는 0.00359였다. 이 결과는 고정되지 않은 split 수나 단순한 Layer 공유 버퍼만으로 관측된 Full28 차이를 재현하지 못한 대조다.

독립적인 Local Full28 단계 계측은 실제 556/4,651/9,944/30,536행의 B1과 그 네 요청을 묶은 B4에서 실행했다. B1의 GPU event enclosure 중앙값은 13.74–13.77 ms, Attention 비중 중앙값은 **11.45–13.44%**였다. B4는 **13.96 ms 및 14.82%**였다. 같은 event에서 QKV·RoPE·KV append, output projection, MLP의 경계를 각각 기록했다. Python 제출 구간이 포함될 수 있는 event enclosure이므로 Stock CUPTI kernel share와 별도로 해석한다. [단계별 Figure](../encoder_assisted_decode_attention/figures/decode_event_breakdown.png) · [실제 step 원자료](../encoder_assisted_decode_attention/results/model_witness_profile.json)

CUDA Graph launch와 destroy, Full28 driver에 D device scope를 명시한 뒤 모든 90개 Teacher Argmax가 일치했다. 별도 Eager-local reference와 Graph-local은 BITWISE 일치했다. 이 복구는 device scope와의 연관성을 지지한다. 같은 재시험에서 packed Vision은 첫 P 토큰 1/4을 바꿨고 Assisted Natural 출력은 2개 중 1개에서 달랐다. 따라서 E용 Encoder 최적화의 수치 계약과 D Graph의 실행 오류는 따로 남겼다. [두 Full28 영수증과 난수 KV 대조](../encoder_assisted_decode_attention/results/operator/graph_device_scope_association.json)

## 실제 Vision Encoder와 간섭

같은 세 H200에서 원본 Qwen2.5-VL Vision Encoder의 BF16 weight 390개(1.260 GiB)를 E에 상주시켰다. E stream 우선순위는 −1이고 원본 SDPA 경로를 사용했다. 이미지 1/2/21장, Visual row 522/2,262/7,935인 동결 요청의 Vision forward를 실행했다. D에서는 실제 P가 만든 Q/K/V로 ragged B4 25% 분할과, 단독 실행에서 이득이 있었던 controlled B8×32K 50% 분할을 별도로 검증했다. 각 조합은 Native local·EADA의 단독/동시 실행, Encode 단독 및 Encode 우선으로 나누어 무작위 순서로 4회 실행했다. D 타이밍은 물리 CUDA Graph 160회 발행을 포함한 CUDA event 구간을 호출 수로 나눴다. [실행 소스](../encoder_assisted_decode_attention/encoder_interference.py)

| Attention · Visual rows | 단독 Local → EADA (µs) | Encode 동시 Local → EADA (µs) | E Encode 동시 Local → EADA (ms) | Encode 우선일 때 D 대기 (ms) |
|---|---:|---:|---:|---:|
| Controlled B8 · 522 | 131.64 → 103.93 | 131.97 → 117.79 | 46.62 → 55.21 | 45.36 |
| Controlled B8 · 2,262 | 131.67 → 103.61 | 131.73 → 143.08 | 148.16 → 149.39 | 147.69 |
| Controlled B8 · 7,935 | 131.82 → 103.94 | 131.71 → 142.34 | 491.03 → 493.20 | 492.38 |
| 실제 Qwen ragged B4 · 522 | 41.83 → 45.72 | 42.24 → 50.78 | 45.01 → 46.88 | 45.77 |
| 실제 Qwen ragged B4 · 2,262 | 41.74 → 46.69 | 42.57 → 59.36 | 148.03 → 152.62 | 147.23 |
| 실제 Qwen ragged B4 · 7,935 | 41.92 → 48.42 | 42.23 → 84.01 | 490.07 → 491.45 | 501.82 |

Controlled B8의 단독 이득 21.1–21.3%는 이미지 한 장이 도착했을 때 10.7%로 줄었고 나머지 두 Encode에서는 8.1–8.6% 손해였다. 실제 Qwen ragged B4는 동시 실행에서 Local 대비 20.2–99.0% 느렸다. 같은 스레드 구성의 동시 Local 비교에서 Encoder의 CUDA event 구간은 0.3–18.4% 늘었고 작은 이미지 측정의 실행 간 편차가 컸다. Encode 단독은 다른 Host 제출 경로를 사용하므로 이 비교에는 동시 Local 기준을 사용했다. 해당 요청의 Encode가 끝난 뒤 보조 Attention을 여는 정책은 각 D burst 시작에 45.4–501.8ms의 대기 구간을 만들었다.

별도 계측 Graph에서 D burst 중간에 한 E 보조 Attention과 Encoder를 같은 E clock의 CUDA event로 표시했다. 24회 중 24회에서 0.0197–0.3063ms의 구간 교집합이 있었다. 이는 실제 동시 발행의 증거이며 개별 SM의 동시 실행률, HBM 대역폭 또는 Kernel preemption은 이 대조에서 계측하지 않았다. 모든 비교는 단일 attention layer이며, 실제 Online SLO는 Online 보고서에 기록했다. [Raw JSONL](../encoder_assisted_decode_attention/results/operator/encoder_interference.jsonl) · [CSV](../encoder_assisted_decode_attention/results/operator/encoder_interference.csv) · [Encoder 간섭 Figure](../encoder_assisted_decode_attention/figures/operator_encoder_interference.png)

```bash
python -m poc_flashvep.mllm_specific_disaggregation_discovery.gpu_epoch \
  --run-dir "$eada_results/encoder_interference_gpu_lease" --devices 4,5,6 \
  --wall-seconds 900 -- "$eada_runtime/bin/python" -u \
  -m poc_flashvep.encoder_assisted_decode_attention.encoder_interference \
  --model "$eada_snapshot" --manifest "$tier1_24_manifest" \
  --out "$eada_results/encoder_interference.jsonl" --vision-backend sdpa \
  --encoder-priority -1 --encoder-first --calls 160 --repeats 4
```

### CUDA stream 우선순위 민감도

원본 SDPA, 동일 GPU UUID, 이미지 요청, BF16 KV, 분할 비율, 각 조합 4회 및 실행 순서를 유지하면서 E Encode stream을 보조 Attention과 같은 우선순위 0에서 한 번 더 검증했다. 각 열은 해당 시작의 동시 Local 대비 EADA의 CUDA event 지연 변화다.

| Attention · Visual rows | D 지연, 우선순위 −1 | D 지연, 우선순위 0 | E 지연, 우선순위 −1 | E 지연, 우선순위 0 |
|---|---:|---:|---:|---:|
| Controlled B8 · 522 | −10.7% | −8.6% | +18.4% | +9.2% |
| Controlled B8 · 2,262 | +8.6% | +5.0% | +0.8% | +1.7% |
| Controlled B8 · 7,935 | +8.1% | +6.9% | +0.4% | +0.5% |
| 실제 Qwen ragged B4 · 522 | +20.2% | +18.1% | +4.1% | +0.1% |
| 실제 Qwen ragged B4 · 2,262 | +39.4% | +30.4% | +3.1% | +0.2% |
| 실제 Qwen ragged B4 · 7,935 | +99.0% | +78.0% | +0.3% | +0.5% |

동일 우선순위에서도 24/24 CUDA event 구간이 교차했고, Encode 우선일 때 D는 45.2–489.1ms 대기했다. 실제 ragged B4에서는 모든 이미지 조건에서 느렸으며 큰 이미지의 상대 손해는 99.0%에서 78.0%로 줄었다. 두 우선순위는 각각 독립 시작 한 번으로 평가했으므로 크기 차이에는 시작 간 변동이 포함된다. 이 결론은 단일 layer의 간섭 실험에 한정한다. [우선순위 0 Raw JSONL](../encoder_assisted_decode_attention/results/operator/encoder_equal_priority.jsonl) · [비교 CSV](../encoder_assisted_decode_attention/results/operator/encoder_priority_sensitivity.csv) · [우선순위 Figure](../encoder_assisted_decode_attention/figures/operator_encoder_priority_sensitivity.png)

위 실행의 `--encoder-priority`를 `0`으로 변경하고 별도 출력 경로를 사용한다. 두 원자료를 `operator_summary`의 `--encoder-run`과 `--encoder-equal-priority-run`에 각각 전달하면 비교 Figure와 CSV가 재생성된다.
