# MLLM EPD Runtime 정합성 검사

현재 상태는 `CORRECTNESS_PENDING`이다.
아래 결과는 `CORRECTNESS_ONLY`다. 성능 결과가 아니다.
정합성 Gate를 통과하기 전에는 Main C3 결과를 확정하지 않는다.

## 공통 계약

- 모델은 Original BF16 Qwen2.5-VL-7B-Instruct다.
- Model·Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
- 원본 Vision Encoder·Weight·Attention·Processor를 유지한다.
- 설치 Runtime은 vLLM `0.26.0+cu129`다.
- Encoder Embedding Gate는 Relative L2 ≤0.02와 Cosine ≥0.999다.
- MRoPE Position·Delta와 Image Placeholder Coverage도 검사한다.
- Capture RPC는 원래 연산을 실행한다. 입력·출력을 보정하지 않는다.
- Capture·Weight Fingerprint의 CPU Copy 때문에 해당 실행은 성능 비교에서 제외한다.

## 확인한 문제와 대조 실험

| 항목 | 실제 근거 | 판정 범위 |
|---|---|---|
| Processor Cache0 | Renderer가 전달한 UUID를 Engine-local 순번으로 교체했다. E/P 이미지 식별자가 달라졌다. | 실제 설치 Renderer의 CPU 회귀 검사로 재현했다. |
| Processor Cache1GiB | 요청별 고유 UUID를 E/P/D에서 보존했다. | CPU 회귀 검사 통과. Native32 실행은 32/32 완료했다. 모델 품질 PASS를 뜻하지 않는다. |
| Eager Custom Ops | E의 `custom_ops=none` 대조 실험에서도 Embedding 차이가 남았다. | 이 변경만으로 해결하지 못했다. |
| V1/V2 Runner | E3와 V1 Colocated R3의 N1 입력·Weight·Buffer·Attention Backend가 같았다. | Runner Family를 맞춰도 차이가 남았다. |

### 실제 입력·Weight가 같은 N1

`E3_legacyR3_matched_N1_v0`의 결과다.
Calibration Request는 `lmms-lab__DocVQA__7621`이다.
Image Grid는 `[1,160,120]`이다.
Embedding Shape는 `[4800,3584]`다.

두 Arm의 실제 BF16 Pixel SHA256이 같다.
Vision Parameter별 Shape·Dtype·SHA256이 모두 같다.
등록된 Vision Buffer별 Shape·Dtype·SHA256도 같다.
두 Arm은 V1 GPUModelRunner와 FLASH_ATTN Backend를 사용했다.

| Metric | E3와 R3 비교 |
|---|---:|
| Relative L2 | 0.0827502128839792 |
| Cosine | 0.996576012208262 |
| Max Absolute Difference | 10.09375 |
| Strict Tensor Match | False |
| Frozen Embedding Gate | FAIL |

이는 N1의 근거다. N400 정합성으로 확장하지 않는다.
### Eager Colocated Teacher 대조 실험

`E3_eagerR4_matched_N1_v0`에서 Colocated Teacher를 Eager·Native Custom Ops 경로로 실행했다.
두 Arm의 실제 Pixel·Vision Parameter·Buffer·Attention Backend가 같았다.
Embedding은 Bitwise 일치했다.
Relative L2는 0이고 Cosine은 1이었다.

이는 Colocated Teacher의 실행 설정을 바꾼 `CORRECTNESS_ONLY` 대조 실험이다.
Strong Compiled Agg 기준선을 Eager로 약화한 Main 성능 비교가 아니다.
기존 Compiled Teacher와의 차이가 해결됐다는 뜻도 아니다.

다음 대조 실험은 E의 `--enforce-eager`만 제거한다.
원본 Weight·Processor·Vision Encoder와 `custom_ops=none`은 유지한다.
기존 Strong Compiled Teacher와 다시 비교한다.

## 대형 Tensor Metric의 누적 정밀도

기존 FP32 Reduction에서 Cosine이 1을 넘는 사례가 있었다.
FP64 Reduction으로 검사 도구를 수정했다.
원본 Capture와 이전 분석 JSON은 보존했다.

| Calibration Index | 이전 Cosine | FP64 누적 Cosine | FP64 누적 Relative L2 |
|---|---:|---:|---:|
| 0 | 0.9990567564964294 | 0.996576012208262 | 0.0827502128839792 |
| 1 | 1.0029176473617554 | 0.9968386312892881 | 0.07953911730512854 |

이 수정은 Embedding을 바꾸지 않는다.
Gate Threshold도 바꾸지 않는다.
Index0의 Gate 실패는 수정 후에도 유지됐다.
CPU Protocol·Numerical 회귀 검사는 5/5 통과했다.

## 현재 제한

- Native EPD의 Task Quality Gate는 아직 통과하지 않았다.
- Native32와 Colocated32의 출력 차이를 모두 정상 BI0 차이라고 확정하지 않았다.
- Encoder의 GPU-direct 전달을 검증한 상태가 아니다.
- 현재 E→P는 ECExampleConnector의 CPU/tmpfs 경로다.
- P→D는 설치된 NixlConnector 경로다.
- Generic Exact Token Split 정합성은 `NOT_RUN`이다.
- Main400 성능·Decode-safe Headroom의 검증 완료를 주장하지 않는다.

## 후속 실행 경로·Batch 대조 결과

E4에서 `--enforce-eager`를 제거했다.
같은 N1의 Embedding은 Strong Compiled Teacher와 Bitwise 일치했다.
Resolved RMSNorm Priority도 Eager의 `vllm_c,native`에서 Compiled의 `native`로 바뀌었다.
따라서 이를 컴파일러 연산만의 단독 효과라고 확정하지 않는다.
실행 설정과 Norm Dispatch의 결합 차이가 확인된 상태다.

E4의 이미지별 Fanout N8에는 Relative L2 최대 `0.0813638192300149`가 남았다.
동일 요청의 전체 이미지를 함께 전달하면 85개 중 83개가 Bitwise 일치했다.
나머지 두 이미지는 28-page 요청의 마지막 두 이미지였다.
Compiled Teacher는 이 요청의 이미지를 `14/12/2`로 나누어 실행했다.

`E6_matched_geometry_N8_v1`은 Teacher의 실제 Batch를 재생했다.
8개 요청의 85개 이미지 모두 실제 Pixel·Patch 수·Image Batch를 맞췄다.
85/85 Embedding이 Bitwise 일치했다.
이는 `CORRECTNESS_ONLY`다. 이 Replay를 Online 정책이나 성능 개선으로 표시하지 않는다.

E5의 Encoder B8192 대조 실험은 5개 요청 후 Admission에서 정체됐다.
Running0·Waiting1과 Prompt Throughput0을 확인했다.
대기 중이던 여섯 번째 요청은 Calibration Index17이었다.
이 설정은 `FAILED_EXECUTION`이다. Index20·31은 `NOT_RUN`이다.
부분 Capture와 실패 기록은 보존했다.
검증되지 않은 B8192 Encoder 설정을 Main에 사용하지 않는다.

새 Native EPD N1은 Natural EOS 16개 Output Token을 생성했다.
Compiled Colocated Teacher와 16개 Token ID가 모두 일치했다.
P와 D의 실제 Vision Encoder 호출은 각각 0회였다.
세 Stage가 같은 EC 경로를 사용했다.
Consumer의 Cache Miss에는 오류를 발생시킨다. Local Vision Fallback을 허용하지 않는다.
이 결과를 N8·N32·Main400 전체 정합성으로 확장하지 않는다.

## Raw Evidence

- `mllm_epd_prefill_heavy_token_split_20261124/analysis/E3_legacyR3_matched_N1_v0.json`
- `mllm_epd_prefill_heavy_token_split_20261124/analysis/nativeops_E_FP64_repro_N1_v2.json`
- `mllm_epd_prefill_heavy_token_split_20261124/raw/encoder_reference/E3_legacyR3_matched_N1_v0_E/`
- `mllm_epd_prefill_heavy_token_split_20261124/raw/encoder_reference/E3_legacyR3_matched_N1_v0_R/`

Raw에는 실제 Pixel Hash, Embedding, Weight·Buffer Fingerprint와 Runner Family를 보존했다.

## N32 출력과 동일 Prompt KV 대조 실험

Native Calibration32 Natural 실행은 32/32개를 완료했다.
Compiled Colocated Teacher와 Token ID가 완전히 같은 요청은 26/32개였다.
독립적인 공식 NIXL PD N8은 Colocated Teacher와 7/8개가 일치했다.
이 공식 PD는 2GPU 정합성 대조군이다. 3GPU 성능 기준선으로 사용하지 않는다.

출력이 다른 Mantis `cw_9` 요청에서 LLM과 KV를 직접 Capture했다.
Prompt 길이는 2,961 Tokens다. 아래 결과의 분모는 이 요청 1개다.

| 검사 | 결과 |
|---|---|
| Native와 Official PD의 P 입력 Token ID | Strict 일치 |
| P의 전체 입력 Embedding·MRoPE Position | Strict 일치 |
| P의 마지막 Prompt Hidden State | Strict 일치 |
| 두 Arm 각각의 P→D Prefix KV | 28/28 Layers, 2,960 Positions Strict 일치 |
| 두 Arm 사이의 D 전체 Prompt KV | 28/28 Layers, 2,961 Positions Strict 일치 |
| 네 P/D 경로의 MRoPE Delta | 모두 −2,832 |
| 출력이 처음 달라진 Token Index | 3, 0-based |

첫 두 Decode Hidden State는 Strict 일치했다.
출력 Index2·3의 Hidden Relative L2는 각각 0.00575634·0.00559842였다.
이는 실제 측정값이다. 전체 요청에 대한 새 허용 오차를 정한 것이 아니다.
Native의 Token21·20 Logit은 각각 27.0·26.875였다.
Official PD에서는 두 Logit이 27.0·27.0으로 같았다.
Greedy 결과는 각각 Token21과 Token20이었다.

이 사례에서 Prompt KV 손상 가설은 반증됐다.
이 결과를 전체 Task Quality PASS로 확장하지 않는다.
Probe는 CPU Copy를 수행했다. 해당 실행의 Latency와 Throughput은 성능 결과에서 제외한다.
재계산 결과는 `analysis/Native_OfficialPD_Mantis6_prompt_KV_and_greedy_margin.json`에 있다.

전체 요청 Encoder 경로와 과거 Teacher의 N8 비교에서는 83/85개 Image가 수치 Gate를 통과했다.
나머지 두 Image의 최대 Relative L2는 0.07027769였다.
최소 Cosine은 0.99752788이었다. 이 실패를 통계에서 제외하지 않았다.
기록된 Teacher Batch Geometry를 맞춘 별도 검사는 85/85개 Strict 일치했다.
이 Geometry Replay는 Online 정책이 아니다. 실제 Native의 미해결 Batch 차이는 계속 추적한다.

공식 MMLongBench-Doc 평가의 GPT 답변 추출은 API 키가 없어 아직 미실행이다.
출력 길이·수치 검사·Task Quality는 서로 다른 결과로 유지한다.
Native의 최종 상태는 아직 `CORRECTNESS_PENDING`이다.

## 주 실험 크기 변경

사용자 요청에 따라 향후 주 실험은 MLLM100을 사용한다.
Main400·Raw Output·기존 정합성 결과는 보존했다.
Discovery50과 Heldout50은 기존 Split 소속을 유지한다. Calibration32는 바꾸지 않았다.
Manifest SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`이다.
이 축소는 정합성·Decode 보호 Gate를 완화하지 않는다.

## 같은 요청의 Processor 결과 전달 대조 검사

추가 검사는 같은 GPU4/5/6의 Native EPD에서 수행했다.
기존 경로와 `full`, `grid_only` 전달 경로를 비교했다.
E는 모든 경로에서 원본 이미지와 원본 Processor로 온라인 Vision Encoding을 수행했다.
`grid_only`는 P/D에서 불필요한 CPU Pixel Payload만 제외한다.
이미지·Visual Token·Attention 계산을 제거한 것이 아니다.
P/D는 Strict EC를 통해 원본 E Embedding 전체를 요구한다.
Export가 없으면 실패한다. 로컬 Vision 재실행으로 대체하지 않는다.

N1과 N8의 고정16 출력은 세 경로에서 Strict 일치했다.
N8 Natural EOS 출력도 세 경로에서 Strict 일치했다.
각 경로의 N8 출력은 총526 Tokens다.
이 결과는 같은 실행 Geometry의 정합성 증거다.
앞서 남은 Cross-geometry Encoder Gate와 Task Quality Gate는 완화하지 않았다.

별도 Mantis Calibration 요청의 Prompt는2961 Tokens였다.
D가 전달받은 Prefix는2960 Positions였다.
P의 실제 Prompt IDs·전체 MRoPE·Input Embedding·Position·마지막 Hidden이 일치했다.
28개 Layer 모두에서 P→D Prefix KV가 Strict 일치했다.
두 경로의 전체 Prompt KV도28개 Layer 모두 Strict 일치했다.
Natural EOS 출력도 같았다.
이 Probe는 CPU Copy를 수행했다. 모든 Timing은 성능 결과에서 제외한다.
이는 한 요청의 결과다. 전체 요청의 품질 통과로 확장하지 않는다.

증거는 `analysis/KV_Metadata_B2_Mantis6_v0.json`에 있다.
새 경로는 공통 Runtime 수정이다. Token Split의 성능 개선으로 집계하지 않는다.
