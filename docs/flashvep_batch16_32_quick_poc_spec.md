# FlashVEP Batch-16/32 Quick PoC Specification

## 1. 목적

기존 Phase 1b의 batch-1 TP2/DP2/EP4 결과에서는 다음 현상이 관찰되었습니다.

- `T_expert_max ≈ 0.449 ms`
- dispatch와 combine이 expert보다 훨씬 큼
- 기존 oracle speedup이 약 `1.075x`
- batch-1에서는 DP0만 실제 요청을 처리하고 DP1은 idle wave로 참여

이번 Quick PoC의 목적은 **batch를 16과 32로 증가했을 때 local expert execution window가 실제로 충분히 길어지는지** 빠르게 확인하는 것입니다.

이번 PoC는 FlashVEP tile simulator를 구현하지 않습니다. 다음 질문에만 답합니다.

1. Batch 16/32에서 `T_expert_max`가 batch-1보다 얼마나 증가하는가?
2. Expert fraction이 유의미하게 증가하는가?
3. Dispatch/combine 증가보다 expert 증가가 더 큰가?
4. Batch 16/32에서는 FlashVEP overlap headroom이 `1.15x` 이상으로 커지는가?
5. 이후 Phase 2A simulator를 진행할 가치가 있는가?

---

## 2. 연구 목표의 변경

Batch 16/32 실험은 batch-1 단일 요청 latency 연구와 다릅니다.

이번 PoC의 목표는 다음으로 정의합니다.

> High-concurrency multimodal serving에서 vision-heavy batched prefill의 EP straggler와 overlap 가능성을 평가한다.

따라서 결과는 다음처럼 해석합니다.

- Batch 32에서만 효과가 있으면 high-throughput serving 연구로 의미가 있음
- Batch-1 latency 개선을 증명한 것으로 주장하지 않음
- 이전 motivation 실험이 batch 32였다면 동일 batch 조건에서 비교 가능하도록 기록

---

## 3. 고정 환경

기존 Phase 1b와 동일한 환경을 재사용합니다.

- Physical GPUs: `4,5,6,7`만 사용
- TP = 2
- DP = 2
- EP = 4
- PP = 1
- Model: 기존과 동일한 Qwen3-VL-30B-A3B-Instruct local snapshot
- Dtype: BF16
- Backend: 기존 Phase 1b와 동일한 backend를 우선 사용
- Input image: 기존 deterministic 896×896 gray image
- Prompt: 기존과 동일한 prompt
- Prefix caching: off
- `max_tokens=1`
- Model weight, routing, token 수, precision 변경 금지

Batch는 다음 두 개만 측정합니다.

```text
global_batch_size = [16, 32]
```

Batch 16을 먼저 실행하고, Batch 32는 GPU memory가 허용할 때만 실행합니다.

---

## 4. 반드시 재사용할 기존 코드

새 profiling framework를 만들지 않습니다.

반드시 다음을 우선 재사용합니다.

- Phase 1b TP2/DP2 runner
- Phase 1b selected-layer CUDA-event instrumentation
- Phase 1b stage analyzer
- 기존 straggler/routing summary 코드
- 기존 result/report format

새 코드는 batch sweep에 필요한 최소 wrapper와 analyzer 수정만 허용합니다.

---

## 5. 측정 범위

전체 48개 layer를 상세 profiling하지 않습니다.

기본 selected layer:

```text
layers = [12, 24, 36]
```

Layer 0과 47은 기본적으로 제외합니다. 기존 결과와 특이점 비교가 필요할 때만 추가합니다.

측정 stage:

```text
T_layer
T_attention
T_norm_router
T_dispatch
T_expert_max
T_combine_dpep
T_combine_tp_allgather
T_combine_drain
T_full_moe
```

추가 workload metric:

```text
global_batch_size
real_requests_per_dp_rank
prompt_tokens_per_request
total_real_tokens
total_routed_assignments
rank_local_assignments
max_local_expert_batch
active_local_experts
critical_expert_rank
critical_layer_rank
```

---

## 6. 실행 프로토콜

### Batch 16

```text
warmup = 3
measured_iterations = 8
```

### Batch 32

```text
warmup = 3
measured_iterations = 8
```

실행 시간이 지나치게 길면 최소:

```text
warmup = 2
measured_iterations = 5
```

까지 줄일 수 있으나 보고서에 명시합니다.

Batch 내 모든 요청은 같은 image/prompt를 사용합니다. 이는 batch scaling을 비교하기 위한 의도적인 설정입니다.

DP rank별 실제 request 분배를 반드시 기록합니다.

예상:

```text
Batch 16: DP rank당 실제 요청 약 8개
Batch 32: DP rank당 실제 요청 약 16개
```

실제 분배가 다르면 runtime 결과를 우선합니다.

---

## 7. 실행 전 확인

코드를 수정하기 전에 다음을 확인하고 짧게 출력합니다.

1. Repository root
2. 기존 Phase 1b 실행 명령
3. Batch input list가 실제 global batch로 처리되는 방식
4. DP0/DP1 요청 분배 방식
5. 실제 DPEP path가 유지되는지
6. 사용할 파일과 최소 수정 계획
7. Batch 16/32의 예상 GPU memory
8. OOM 시 중단 방식

---

## 8. 즉시 중단 조건

다음이면 해당 batch를 중단합니다.

- 허용 GPU 4,5,6,7 외 GPU가 필요함
- OOM 또는 반복적인 worker crash
- TP2/DP2/EP4 DPEP path가 유지되지 않음
- Batch가 실제로 하나의 요청씩 순차 처리되고 global batch가 형성되지 않음
- DP rank 중 한쪽에만 모든 실제 요청이 배치됨
- profiler overhead가 20% 이상
- 기존 결과를 덮어써야만 진행 가능함

Batch 32가 OOM이면 Batch 16 결과만으로 판단합니다. OOM을 피하기 위해 image resolution, dtype, model, token 수를 변경하지 않습니다.

---

## 9. 분석

### 9.1 Batch scaling table

다음 표를 작성합니다.

| Metric | Batch 1 Phase 1b | Batch 16 | Batch 32 |
|---|---:|---:|---:|
| requests per DP rank | | | |
| total routed assignments | | | |
| max rank assignments | | | |
| max local expert batch | | | |
| `T_dispatch` | | | |
| `T_expert_max` | | | |
| `T_combine_drain` | | | |
| `T_layer` | | | |
| expert fraction | | | |
| dispatch fraction | | | |
| combine fraction | | | |

기존 Batch-1 값은 Phase 1b artifact에서 읽습니다.

### 9.2 Scaling ratio

다음을 계산합니다.

```text
expert_scaling(B) = T_expert_max(B) / T_expert_max(1)
dispatch_scaling(B) = T_dispatch(B) / T_dispatch(1)
combine_scaling(B) = T_combine(B) / T_combine(1)
layer_scaling(B) = T_layer(B) / T_layer(1)
```

또한:

```text
expert_fraction = T_expert_max / T_layer
communication_to_expert =
    (T_dispatch + T_combine_drain) / T_expert_max
```

### 9.3 간단한 overlap oracle

복잡한 tile simulator를 구현하지 않습니다.

동일 iteration/layer timestamp를 사용하여 기존 Phase 1b oracle을 Batch 16/32에 재적용합니다.

추가로 매우 단순한 sensitivity upper bound 하나만 계산합니다.

```text
best_case_extra_hidden =
    min(T_combine_drain, T_expert_max)
```

```text
extended_oracle =
    max(
        existing_oracle - best_case_extra_hidden,
        unavoidable_prelude_and_drain_lower_bound
    )
```

이 값은 combine tile overlap의 **극단적으로 낙관적인 상한**이며 realistic prediction이라고 부르지 않습니다.

새 discrete-event tile simulator는 구현하지 않습니다.

---

## 10. 판단 기준

### GO — Phase 2A simulator 진행

Batch 16 또는 32에서 다음을 모두 만족할 때만 Phase 2A simulator를 권장합니다.

1. `T_expert_max >= 1.0 ms`
2. expert fraction `>= 25%`
3. communication-to-expert ratio가 batch-1보다 명확히 감소
4. 기존 oracle speedup 또는 extended optimistic oracle이 `>= 1.15x`
5. 최소 2/3 selected layer에서 같은 경향
6. profiler uncertainty보다 예상 이득이 큼
7. TP2/DP2/EP4 DPEP path가 정상 유지됨

### HOLD

다음이면 HOLD입니다.

- `T_expert_max`는 증가하지만 expert fraction이 20~25%
- oracle이 1.10~1.15x
- Batch 32만 유리하고 Batch 16은 불리
- profiler noise와 예상 이득이 비슷함
- untuned backend 때문에 결론이 불확실함

### NO-GO

다음 중 하나면 FlashVEP Phase 2A를 진행하지 않습니다.

- Batch 16/32에서도 `T_expert_max < 1.0 ms`
- expert fraction `< 20%`
- dispatch/combine이 expert보다 더 빠르게 증가
- 모든 representative layer의 oracle `< 1.10x`
- Batch 32가 OOM이고 Batch 16도 개선이 없음
- high batch에서도 communication이 계속 지배

---

## 11. 결과물

기존 artifact를 덮어쓰지 않습니다.

최소 결과물:

```text
poc_flashvep/scripts/run_batch16_32_quick_poc.sh
poc_flashvep/scripts/analyze_batch16_32_quick_poc.py
poc_flashvep/results/batch16_32_quick_poc_<timestamp>/
poc_flashvep/reports/batch16_32_quick_poc.md
poc_flashvep/results/baseline/gate_batch16_32_quick_poc.json
```

필요한 instrumentation 수정은 기존 Phase 1b 코드를 최소 수정합니다.

`poc_flashvep/STATUS.md`에는 짧은 결과 section만 추가합니다.

---

## 12. 시간 및 사용량 제한

이번 작업은 빠른 gate 실험입니다.

- 새 simulator framework 구현 금지
- per-token route trace 금지
- isolated collective microbenchmark 금지
- Nsight Systems full trace 금지
- custom kernel 금지
- 전체 48-layer detailed profiling 금지
- image/token length sweep 금지
- scheduler sweep 금지
- Batch 16/32 외 batch sweep 금지
- 실패한 Batch 32를 여러 설정으로 반복 시도 금지

권장 순서:

```text
1. 기존 코드 확인
2. Batch 16 smoke/profile
3. Batch 32 smoke/profile
4. 간단 분석
5. GO/HOLD/NO-GO
6. 중단
```

---

## 13. 최종 보고 형식

1. 실제 환경과 사용 GPU
2. 실제 Batch 16/32 요청 분배
3. 실제 DPEP path
4. 수정/추가 파일
5. Batch 1/16/32 stage 비교
6. Expert latency와 expert fraction 변화
7. Dispatch/combine scaling
8. 간단 oracle 결과
9. OOM/profiler/backend blocker
10. GO/HOLD/NO-GO
11. 다음으로 권장하는 단 하나의 작업

Phase 2A나 live overlap 구현으로 자동 진행하지 않습니다.
