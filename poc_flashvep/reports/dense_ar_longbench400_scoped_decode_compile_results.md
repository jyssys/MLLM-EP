# LongBench400: Prefill Norm과 Compiled Decode를 분리한 검증

성능 실행은0회다. 표의 N/A는 실제 지연 측정값이 아니다.
정합성 실패 이후 첫 성능 실행 전에 계획 전체를 명시적으로 대체했다.
원래21개 조건과 NOT_RUN 사유는 Terminal Outcome에 보존한다.

상태: 독립 Start의 Terminal 결과를 확보했다.

## 변경과 대조군

실제 설치 소스의 CUDA Platform은 Compile 여부에 따라 IR Norm 우선순위를 바꾼다.
Mode0는 `vllm_c,native`를 사용한다. 기본 Inductor Compile은 `native`를 사용한다.
두 설정은 같은 BF16 수학을 구현하지만 모든 Tensor Byte가 같지는 않았다.
`custom_ops=all`만으로 IR Norm 우선순위가 바뀌지는 않았다.
기본 D Compile의 엄격한 불일치와 Global CUDA 우선순위의 N1/N8 통과를 모두 보존한다.
이 개입은 RMSNorm과 Fused-add RMSNorm의 Provider Bundle을 함께 바꾼다.
두 Operator 각각의 수치 기여율을 별도로 분해한 것은 아니다.

Scoped 경로는 Eager Partial-prefill 안에서만 `vllm_c,native`를 선택한다.
Full36 Suffix와 Decode의 원래 Compiled Model은 `native`를 유지한다.
D의 Custom-op 설정은 `all`이다. P의 기존 Mode0 경로는 바꾸지 않는다.
Tensor, Sampling Token이나 Client Timestamp를 덮어쓰지 않는다.
원래 Full36 Single-GPU Teacher도 같은 Stage별 Provider 계약으로 독립 실행했다.
Teacher에는 Partition Adapter, NIXL과 수입 KV가 없다.
Teacher의 Warm State를 평가 Request로 세었던 CPU Assertion은 수정했다.
평가 ID0–7과36개 KV Plane을 검사한다. Warm State와 실패 Controller Log는 삭제하지 않았다.

## 동결한 성능 계약

I3의 원문 Prompt16개, Mean Input28296 Tokens, QPS1.253873을 유지한다.
Arrival Seed는2026112551/2552/2553이다. 각 Arm은 독립 Server Start3회다.
모든 새 Paired 성능은 GPU0/1을 사용한다. 원래 GPU4/5 결과와 CI를 합치지 않는다.
모델은 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.
BI0, Original BF16, Exact Causal Attention과 Full36 Decode를 유지한다.
B8192/S1024, KV42GiB/Block128, Prefix Cache Disabled를 유지한다.
Output16은 `ignore_eos=True`인 Mechanism Control이다. Natural EOS 결과가 아니다.
Layer8, Rows-matched Stock Token8, Hybrid4+정확한 L/8을 사용한다.
각 Offload의 Logical P/D Rows는12676608/3621888이다.
D Cap1과 Source Credit8을 모든 Offload Arm에 유지한다.
NativeStrong은 원래 Scheduler를 유지한다. Common-policy Native의 Cap은1이다.
실제 Batch와 Padding은 JSON에 별도 기록한다. B/S를 실제 Batch로 해석하지 않는다.
Source SHA256은 `93b1b133ebd471be1c75c316bb01bc58dd5d42ecd2f0aa309ae5b35e36193293`다.
Dataset SHA256은 `6429dde7a7dd44156636e7f9c8f041d7b7d0b6f91c80bc2cce1697dc51e8c780`이다.
TTFT5000ms, 모든 ITL34ms, TPOT33ms의 Frozen SLO를 바꾸지 않았다.

## 실측 결과

| Arm | 독립 Starts | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) | Raw Output Tokens/s | 관측 Max ITL(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| NativeCommon | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Layer | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| StockToken | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Hybrid | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| LayerScoped | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| HybridScoped | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

## Paired 판정

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode5% | 매 Start Absolute Decode≥95% |
|---|---|---|---|---|---|---|

## 판정의 제한

NI는 NativeStrong과 정확한 Common-policy Native에 각각 적용한다.
두 Native에 대한 ITL p99와 TPOT p95 CI 상한이 모두1.05 이하여야 한다.
Absolute Decode Attainment≥95%도 모든 Start에서 별도로 요구한다.
실패한 Gate를 반올림하거나 큰 Client Interval을 제거하지 않는다.
CI는 Paired Start-level Empirical 95% CI다. N3의27개 Bootstrap 조합을 열거한다.
다중 비교 보정은 하지 않았다. 48개 Request를48개 독립 Start로 사용하지 않는다.
Within-axis 개입에는 D의 Compiler Dispatch와 Stage별 Kernel 정책이 함께 포함된다.
후속 Admission, Batch와 Queue도 바뀔 수 있다. 이를 순수 GPU 시간의 인과 효과로 표현하지 않는다.
GPU-exclusive Operator 비중과 SM/HBM은 이 Source에서 NOT_MEASURED다.
이 결과를 Natural400 또는 다른 Cut·QPS의 성능으로 확대하지 않는다.
원래54개와 등록 수정54개의 실패를 새 Source로 덮어쓰지 않는다.
Online Dynamic Window나 Axis Selection은 이 검사에 없다.
유한16개 Throughput을 Sustainable Capacity로 표현하지 않는다.

- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/scoped_compile_confirmation_frozen_v1.json)
- [전체 결과와 CI](dense_ar_longbench400_2d_20261121/analysis/scoped_compile_confirmation_results_v1.json)
- [기존 수치 불일치와 Negative Control](dense_ar_longbench400_decode_compile_residual.md)
