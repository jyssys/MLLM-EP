# Track A4/A5 · Visual Memory–Generation PoC

두 H200에서 Visual KV Residency, 균형적인 일반 Sequence Split, 고정 Prompt/Generated 분리, Visual과 동일한 행 수의 일반 Prefix 분리를 Full28로 비교했다. 다섯 실제 요청의 입력과 생성 분포를 사용했다. NativeServing 대조와 격리 GPU Timing은 함께 보존했다. [KV 프로파일](mllm_visual_kv_memory_decode_profile.md) · [Exact Attention](mllm_visual_memory_generation_exact_attention.md) · [순이익 분석](mllm_visual_memory_remote_attention_break_even.md)

| 파티션 | Ragged B4 Full28 ms/step | 28,845 행 요청의 D KV MiB | P KV MiB |
|---|---:|---:|---:|
| Full local | 12.860 | 1,689.461 | 0 |
| 균형 Sequence Split | 18.429 | 900.758 | 788.703 |
| 고정 Prefix/Generation | 18.350 | 112.000 | 1,577.461 |
| Visual Residency | 18.328 | 114.461 | 1,575.000 |
| Visual 동일 행 수 Generic Prefix | 18.305 | 114.461 | 1,575.000 |

Prompt가 28,845행인 요청의 D route KV는 Visual을 P에 유지하면 93.2% 줄었다. D/P 총 KV Tensor 크기는 동일했다. 비교표는 각 실현 Layout의 실제 예약 Tensor 크기에 2,048 output reserve를 포함한다. 공통 correctness reference KV는 계측 프로세스 내부에 확보했고 실제 서비스 메모리 해제량은 별도 시스템 경로로 남긴다.

같은 수의 KV 행을 P로 옮긴 Generic Prefix와 Visual 방식의 B1 Full28 시간 차이는 절댓값 0.24% 미만이었다. B4에서는 Visual이 Generic보다 0.023ms 길었다. 최대 Prompt에서 같은 수를 옮긴 상태의 초기 Copy와 Packing은 Visual 약 3.329ms, Generic 0.904ms였다. Visual 경계가 연속적이지 않아 실제 D 소유 행을 다시 모으는 비용이 생겼다. 고정 Prefix 분리는 D 메모리를 이보다 더 줄였고 반복 비용을 비슷하게 지불했다. [같은 자원 비교 Figure](../mllm_specific_disaggregation_discovery/results/a/figures/figure11_modality_vs_generic.png)

고정 Query에 대한 causal downstream Attention은 볼 수 있는 모든 Prompt KV를 처리한다. 따라서 같은 행 수와 layout이면 Visual이라는 label 자체가 Attention의 Dense Model 계산량을 바꾸지 않는다. Context-length 배치와 KV 행 크기 차이를 활용하는 방법은 일반 LLM의 Prompt Prefix에도 적용된다.

## Strong Baseline의 구현 범위

설치된 vLLM에서 stock GQA DCP는 TP가 전체 KV Head보다 커야 하고 Query heads/KV head 비율이 DCP 크기로 나뉘어야 한다. 이 모델에서는 KV Head 4, Query 28이고 비율은 7이다. 대상 2 GPU의 stock DCP 크기 2는 호환되지 않는다. 같은 두 H200에서 실제 FA3 분산 softmax를 이용한 두-device 균형 KV Split을 일반 대조로 실행했다. Stock DCP Serving 결과는 미측정이다. [고정 Runtime 구현 Hash](../mllm_specific_disaggregation_discovery/results/model_audit.json) · [vLLM DCP](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)

## Online 자원 제약

P는 Prompt 상태를 Generation 수명 동안 유지해야 한다. 실제 Native가 다음 요청을 Prefill하는 동시 조건에서 P의 KV Pressure와 Decode 후보와의 간섭은 미계측이다. Local 경로보다 반복 시간이 늘고, 같은 행 수의 Generic에 비해 독립적인 Visual 이득이 부족해 이 배치의 Active C3 후보는 채택 단계에 도달하지 않았다. GPU 추가 할당, D OOM 완화, 최대 동시성 또는 실제 SLO-goodput 향상은 미검증이다.
