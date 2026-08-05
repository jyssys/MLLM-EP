# FlashVEP Offline Expert-Centered Wavefront Quick PoC

## 0. 한 줄 목표

Qwen3-VL의 vision-heavy batched prefill에서 길어진 expert execution을 중심(anchor window)으로 사용하여,

```text
Dispatch(next microbatch) || Expert(current microbatch)
Combine(previous microbatch) || Expert(current microbatch)
```

를 실제로 겹쳤을 때 MoE 처리량이 증가하는지 4-GPU scheduler-free offline replay로 빠르게 검증합니다.

이 PoC는 최종 시스템 평가가 아닙니다. 성공할 경우에만 이후 vLLM end-to-end 통합 및 8-GPU 검증으로 진행합니다.

---

## 1. 연구 프레이밍

### 문제

Vision-heavy multimodal batch에서는 특정 EP rank의 local expert execution이 길어져 전체 MoE layer makespan을 결정할 수 있습니다.

### 핵심 아이디어

FlashVEP는 expert execution을 제거하거나 근사하지 않습니다. 대신 vision-heavy critical expert execution을 다른 stage를 숨기는 중심 구간으로 사용합니다.

Microbatch `i`의 stage를 다음과 같이 정의합니다.

```text
A_i: Attention
R_i: Norm + Router
D_i: Dispatch
E_i: Local expert execution
C_i: Combine
```

Baseline:

```text
A0 -> R0 -> D0 -> E0 -> C0
A1 -> R1 -> D1 -> E1 -> C1
A2 -> R2 -> D2 -> E2 -> C2
```

핵심 expert-centered wavefront:

```text
Warm-up:
A0 -> R0 -> D0

Steady state:
E0  || D1
E1  || C0 || D2
E2  || C1 || D3
E3  || C2

Drain:
C3
```

여기서 `||`는 실제 GPU 동시 실행을 의미합니다.

Dispatch와 combine이 같은 NCCL resource를 공유하여 동시에 실행할 수 없다면, 각 expert window 안에서 다음처럼 순차 배치합니다.

```text
E_i || [C_{i-1} -> D_{i+1}]
```

또는 deadline에 따라:

```text
E_i || [D_{i+1} -> C_{i-1}]
```

Attention/Router 확장형은 핵심 파이프라인이 성공한 뒤에만 검토합니다.

```text
E_i || A_{i+1} -> R_{i+1} -> D_{i+1}
```

단, Attention과 Expert는 모두 GPU compute를 강하게 사용하므로 실제 concurrency가 없거나 서로 느려질 수 있습니다. 이번 Quick PoC의 필수 범위는 `D/E/C`입니다.

---

## 2. 왜 scheduler-free offline replay인가

현재 repository는 vLLM의 `LLM` 경로를 사용하지만, submitted global batch는 engine scheduler에 의해 여러 model microbatch로 재분할됩니다. 따라서 submitted batch와 실제 expert GEMM shape가 직접 대응하지 않습니다.

이번 PoC의 offline은 단순히 API server를 사용하지 않는다는 뜻이 아닙니다. 다음을 제거한 **scheduler-free operator/block replay**를 의미합니다.

- request admission
- continuous batching
- KV-cache admission
- model-call microbatch 재분할
- idle DP wave

목적은 다음을 정확히 통제하는 것입니다.

- 한 번의 microbatch에 들어가는 token 수
- rank별 routed assignment
- expert GEMM shape
- dispatch/expert/combine 실행 순서
- CUDA stream dependency
- 동일 workload에서 serial 대 overlap 비교

Offline 결과만으로 end-to-end serving speedup을 주장하지 않습니다.

최종 연구에서는 다음 3단계 evidence가 필요합니다.

1. Scheduler-free offline operator/block PoC
2. 4-GPU vLLM integration validation
3. 8-GPU vLLM end-to-end throughput evaluation

논문에서의 답변은 다음과 같이 정리합니다.

> vLLM은 motivation, baseline characterization, 그리고 최종 end-to-end validation에 사용한다. Scheduler-free replay는 vLLM의 request scheduler가 workload shape와 measurement를 바꾸지 않도록 mechanism 자체를 통제된 환경에서 검증하기 위한 중간 단계다.

---

## 3. 고정 환경

- Repository: 최신 `jyssys/MLLM-EP`
- Physical GPUs: `4,5,6,7`만 사용
- GPU 수: 4
- EP size: 4
- Expert ownership: rank당 32 experts, 총 128 experts
- Model: 기존과 동일한 Qwen3-VL-30B-A3B-Instruct local snapshot
- Dtype: BF16
- Expert backend: 기존 Phase 1b와 동일한 `TritonExperts` 우선
- Hidden size, expert intermediate size, top-k: checkpoint/runtime에서 읽음
- Selected layer: 기본 layer 24 하나
- 추가 layer: core result가 통과할 때만 layer 12 또는 36 중 하나
- Model weight, routing, precision 변경 금지
- Token pruning/merging/dropping 금지
- Quantization 금지
- Expert placement/replication 금지

---

## 4. 입력 데이터

### 4.1 우선순위

1. 기존 Batch 16/32 artifact에 실제 post-attention hidden state 및 route capture가 있으면 재사용
2. 없으면 layer 24에서 opt-in compact capture를 한 번 수행
3. capture를 위해 전체 activation이나 model weight를 저장하지 않음

필수 capture:

```text
post_attention_hidden
topk_expert_ids
topk_weights
original_token_count
vision_token_count
destination_rank
local_expert_id
```

### 4.2 Capture 제한

- GPU 4,5,6,7만 사용
- vLLM 실행은 capture 1회만 허용
- 동일 fixed 896x896 image/prompt 사용
- 출력 token consistency 확인
- 기존 artifact 덮어쓰기 금지
- capture가 이미 있으면 vLLM 실행 금지

### 4.3 Workload scaling

Request batch 자체보다 실제 routed workload를 기준으로 보고합니다.

Batch-equivalent workload:

```text
B_eq = [16, 32, 64, 128]
```

각 workload는 captured real request를 concatenate/repeat하여 구성할 수 있습니다.

반드시 함께 기록:

```text
real tokens
vision tokens
total routed assignments
rank별 routed assignments
critical-rank assignments
max local-expert token count
active local experts
```

반복된 input을 사용하면 `synthetic batch scaling from real captured request`라고 명시합니다.

---

## 5. 필수 실험 범위

### Phase O1 — Compute-bound crossover

각 `B_eq`에서 serial full-batch MoE stage를 측정합니다.

```text
D -> E -> C
```

측정값:

```text
T_dispatch
T_expert_max
T_combine
T_moe
expert_fraction
communication_to_expert
tokens_per_second
assignments_per_second
critical_rank
```

목표:

- expert fraction이 25%, 40%, 50%를 넘는 workload 확인
- expert window가 1 ms, 2 ms, 4 ms 이상이 되는 workload 확인

### Phase O2 — Expert-centered D/E/C wavefront

Compute-bound 후보 2개만 선택합니다.

권장:

```text
B_eq = 32와 64
```

메모리와 O1 결과에 따라 64와 128로 바꿀 수 있습니다.

각 full workload를 `K`개 microbatch로 나눕니다.

```text
K = [2, 4]
```

비교:

1. `full_batch_serial`
2. `microbatch_serial`
3. `expert_centered_wavefront`

#### `full_batch_serial`

```text
D(full) -> E(full) -> C(full)
```

#### `microbatch_serial`

```text
D0 -> E0 -> C0 -> D1 -> E1 -> C1 -> ...
```

Fragmentation 및 반복 collective overhead 측정용입니다.

#### `expert_centered_wavefront`

필수 schedule:

```text
D0
E0 || D1
E1 || C0 || D2
E2 || C1 || D3
E3 || C2
C3
```

통신 backend에서 dispatch와 combine이 같은 NCCL resource를 공유해 동시에 진행할 수 없다면 통신끼리는 serialize하고 expert와만 overlap합니다.

Rank마다 collective order가 달라지면 안 됩니다.

---

## 6. 선택적 확장 — Attention/Router next-overlap

다음 조건을 모두 만족할 때만 수행합니다.

- O2 actual speedup `>= 1.10x`
- correctness 통과
- expert-centered overlap이 Nsight 또는 CUDA timestamp로 확인
- 남은 시간/범위가 충분함

선택적 비교:

```text
E_i || A_{i+1} + R_{i+1} + D_{i+1}
```

주의:

- Attention과 Expert가 동시에 실행되었다는 것은 enqueue timestamp가 아니라 GPU kernel overlap으로 확인
- 두 compute kernel의 slowdown을 각각 기록
- throughput이 향상되지 않으면 즉시 중단
- 선택적 확장 실패가 core D/E/C 결과를 무효화하지 않음

---

## 7. 구현 원칙

### 7.1 Runtime

- vLLM serving scheduler는 offline replay에서 사용하지 않음
- 가능하면 기존 vLLM `TritonExperts` 및 checkpoint weight layout 재사용
- 통신은 기존 DPEP primitive를 안전하게 재사용할 수 있으면 우선 사용
- 불가능하면 `torch.distributed` NCCL 기반 명시적 collective 사용
- backend가 바뀌면 반드시 보고
- custom CUDA/Triton kernel 작성 금지

### 7.2 CUDA streams

최소:

```text
dispatch_stream
expert_stream
combine_stream
```

Dependency는 CUDA event로 연결합니다.

```text
dispatch_done[i] -> expert_start[i]
expert_done[i] -> combine_start[i]
```

Default stream implicit synchronization에 의존하지 않습니다.

### 7.3 Correctness

Serial reference와 비교합니다.

```python
torch.testing.assert_close(
    wavefront_output,
    serial_output,
    rtol=...,
    atol=...,
)
```

보고:

```text
max_abs_error
mean_abs_error
cosine_similarity
token/order restoration
route identity
```

---

## 8. 측정 프로토콜

### O1

```text
warmup = 10
iterations = 30
```

### O2

```text
warmup = 10
iterations = 30
```

시간이 길면 최소:

```text
warmup = 5
iterations = 15
```

측정:

- CUDA event wall time
- rank critical makespan
- median, p90, mean, stddev
- throughput
- profiler overhead

Nsight Systems는 best candidate 한 configuration에 대해 한 번만 허용합니다.

---

## 9. 핵심 metric

```text
T_full_batch_serial
T_microbatch_serial
T_wavefront
speedup_vs_full_batch
speedup_vs_microbatch_serial
throughput_tokens_per_sec
throughput_assignments_per_sec
expert_fragmentation_penalty
collective_repetition_penalty
expert_busy_fraction
dispatch_expert_overlap_ms
expert_combine_overlap_ms
actual_overlap_fraction
critical_rank
```

정의:

```text
expert_fragmentation_penalty =
    T_microbatch_expert_sum / T_full_batch_expert - 1
```

```text
actual_overlap_fraction =
    measured_concurrent_duration
    / min(candidate_stage_durations)
```

---

## 10. Gate

### GO — 이후 vLLM 통합 PoC 진행

다음을 모두 만족해야 합니다.

1. 실제 output correctness 통과
2. `expert_centered_wavefront` speedup:
   - 하나 이상의 representative workload에서 `>= 1.15x`
   - 다른 workload에서도 `>= 1.10x`
3. tokens/s 또는 assignments/s 증가가 profiler uncertainty보다 명확히 큼
4. actual D/E 또는 E/C kernel overlap 확인
5. expert fragmentation penalty `< 15%`
6. 특정 rank만의 우연한 결과가 아님
7. memory peak가 full-batch baseline 대비 과도하게 증가하지 않음

### HOLD

- speedup `1.05x~1.15x`
- zero-overhead potential은 있지만 NCCL/fragmentation 때문에 결과가 민감
- Attention/Expert compute contention에 따라 결론이 바뀜
- 다른 backend가 필요하지만 아직 검증하지 못함

### NO-GO

- compute-bound workload에서도 speedup `< 1.05x`
- microbatch fragmentation이 overlap 이득보다 큼
- collective 반복 비용이 지배
- 실제 GPU overlap이 발생하지 않음
- correctness 실패
- custom kernel 없이는 테스트 자체가 불가능

---

## 11. 결과물

```text
docs/flashvep_offline_expert_centered_wavefront_poc_spec.md

poc_flashvep/offline_wavefront/
  capture_schema.py
  workload_builder.py
  offline_moe_runner.py
  expert_centered_pipeline.py

poc_flashvep/scripts/
  capture_offline_wavefront_input.py
  run_offline_wavefront_quick_poc.py
  analyze_offline_wavefront_quick_poc.py

poc_flashvep/tests/
  test_offline_wavefront_correctness.py
  test_offline_wavefront_schedule.py

poc_flashvep/results/offline_wavefront_quick_poc_<timestamp>/
poc_flashvep/reports/offline_wavefront_quick_poc.md
poc_flashvep/results/baseline/gate_offline_wavefront_quick_poc.json
```

기존 결과를 덮어쓰지 않습니다.

---

## 12. 시간/사용량 제한

이번 PoC는 빠른 mechanism gate입니다.

금지:

- vLLM end-to-end integration
- 8-GPU 실행
- 전체 48-layer profiling
- custom kernel
- scheduler 구현
- route predictor
- attention kernel 수정
- 여러 모델/해상도 sweep
- batch 외 추가 축 sweep
- 대규모 Nsight trace
- Phase O2 실패 후 반복적인 튜닝

권장 실행 순서:

```text
1. 기존 capture 확인
2. 없으면 layer 24 capture 1회
3. O1 B_eq 16/32/64/128
4. compute-bound 후보 2개 선택
5. O2 K=2/4
6. GO/HOLD/NO-GO
7. 중단
```

---

## 13. 최종 보고

1. 사용 환경과 backend
2. capture provenance
3. B_eq별 routed workload
4. compute-bound crossover
5. serial/microbatch/wavefront 비교
6. 실제 D/E 및 E/C overlap
7. correctness
8. fragmentation 및 collective penalty
9. GO/HOLD/NO-GO
10. 다음 단 하나의 작업

GO여도 vLLM 통합을 자동 시작하지 않습니다.
