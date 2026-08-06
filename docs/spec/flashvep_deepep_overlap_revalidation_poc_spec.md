# FlashVEP DeepEP 기반 Overlap 재검증 PoC 스펙

## 0. 문서 목적

이 PoC의 목적은 이전 `torch.distributed` 기반 offline wavefront의 NO-GO를 일반화하지 않고, MoE 전용 통신 backend인 DeepEP를 사용할 때 communication–expert overlap이 실제 성능 향상으로 이어지는지 제한된 범위에서 재검증하는 것이다.

이 PoC는 최종 FlashVEP novelty를 구현하지 않는다. 먼저 다음 세 효과를 분리한다.

1. 일반 AllGather/ReduceScatter 대비 DeepEP 통신 backend 자체의 효과
2. DeepEP serial 대비 stock vLLM DBO overlap의 효과
3. 향후 vision-straggler-aware FlashVEP adaptation을 추가할 여지가 남는지

핵심 판정 대상은 다음이다.

> 이전 실패가 overlap 자체의 한계였는가, 아니면 일반 NCCL collective와 Triton expert가 SM/HBM/L2를 경쟁한 구현 substrate의 한계였는가?

---

## 1. 기존 결과와 재검증 가설

### 1.1 기존 결과

기존 offline PoC는 layer 24의 실제 Qwen3-VL hidden state와 route를 capture한 뒤, 하나의 799-token request를 반복하여 B_eq=16/32/64/128 workload를 만들었다.

- request당 token: 799
- vision token: 784
- hidden size: 2048
- expert intermediate size: 768
- total experts: 128
- top-k: 8
- EP size: 4
- local experts/rank: 32
- expert backend: TritonExperts
- dispatch: `all_gather(hidden)`, `all_gather(topk_weights)`, `all_gather(topk_ids)`
- combine: `reduce_scatter(expert_output)`

최고 결과는 B64/K2에서 full-batch serial 대비 1.034x였다. CUDA event상 overlap은 존재했지만 microbatch serial 대비 다음 slowdown이 발생했다.

- Dispatch: +258.5%
- Expert: +5.0%
- Combine: +31.5%

따라서 기존 판정은 다음으로 한정한다.

> 일반 `torch.distributed` AllGather/ReduceScatter 기반 수동 wavefront는 NO-GO이다.

다음 결론은 아직 검증되지 않았다.

> Qwen3-VL MLLM MoE에서 optimized EP communication과 expert compute의 overlap도 불가능하다.

### 1.2 재검증 가설

DeepEP는 MoE dispatch/combine 전용 layout, metadata handle, 비동기 event interface, communication SM budget 제어를 제공한다. 이를 사용하면 기존 AllGather 기반 구현보다 communication–expert contention이 감소할 수 있다.

검증할 가설은 다음과 같다.

- H1: DeepEP serial은 기존 AllGather/ReduceScatter serial보다 D+C latency를 줄인다.
- H2: DeepEP overlap은 기존 wavefront보다 Dispatch slowdown을 크게 줄인다.
- H3: DeepEP serial 대비 실제 overlap speedup이 B64/B128에서 관측된다.
- H4: stock vLLM DeepEP+DBO가 Qwen3-VL vision-heavy prefill에서 end-to-end 이득을 제공한다.
- H5: stock DBO가 모든 이득을 소진하지 않으며, 향후 vision-straggler-aware adaptive policy가 개선할 여지가 남는다.

---

## 2. 엄격한 범위

### 2.1 이번 PoC에서 수행

1. 별도 격리 환경에서 vLLM 0.20 호환 DeepEP 설치
2. DeepEP intranode correctness/performance smoke test
3. Qwen3-VL TP=2, DP=2, EP=4, PP=1 경로에서 backend 검증
4. stock backend 세 가지 end-to-end 비교
5. 기존 captured route를 재사용한 DeepEP operator replay
6. DeepEP serial과 overlap 비교
7. 지원되는 communication SM 설정 범위 내 SM budget sweep
8. best configuration 1개에 대해 Nsight Systems timeline 수집
9. 결과 분석, GO/HOLD/NO-GO 판정
10. 코드, 명령어, 작은 요약 결과, 보고서를 동일 GitHub 저장소에 commit 및 push

### 2.2 이번 PoC에서 제외

- MACS 구현
- ReaLB 구현
- FlashVEP 최종 vision-aware adaptive policy
- 새로운 CUDA/Triton communication kernel 작성
- DeepEP 소스 내부 kernel 수정
- Qwen3-VL fine-tuning
- token dropping/merging
- expert placement/replication
- FP4/FP8 expert quantization
- 정확도를 바꾸는 최적화
- 8-GPU full evaluation
- 여러 모델·여러 benchmark sweep
- DeepEP V2를 현재 환경에 무리하게 섞는 작업
- 시스템 driver 변경 또는 reboot
- GPU 0~3 사용

---

## 3. 저장소와 Git 정책

### 3.1 대상 저장소

- Repository: `https://github.com/jyssys/MLLM-EP.git`
- Expected remote full name: `jyssys/MLLM-EP`
- Default branch: `main`

### 3.2 작업 branch

현재 작업 tree와 branch를 먼저 확인한다.

- 이미 사용자 작업 branch에 있고 clean하면 그 branch를 계속 사용할 수 있다.
- `main`이거나 적절한 작업 branch가 없다면 다음 branch를 생성한다.

```bash
git switch -c flashvep/deepep-overlap-revalidation
```

금지:

- force push
- 기존 unrelated change 삭제
- 다른 사람의 commit rewrite
- 사용자 파일 임의 reset
- 결과를 만들지 않고 문서만 commit

### 3.3 종료 시 필수 Git 작업

실험과 보고서 작성 후 반드시 다음을 수행한다.

```bash
git add <이번 작업의 파일만>
git commit -m "poc: revalidate FlashVEP overlap with DeepEP"
git push -u origin <현재-branch>
```

최종 보고에 반드시 포함한다.

- remote URL
- branch name
- commit SHA
- pushed 여부
- `git status --short`
- 주요 결과 파일 경로

Push가 인증 또는 원격 오류로 실패하면:

1. local commit은 반드시 남긴다.
2. push 명령과 전체 오류를 보고한다.
3. force push나 credential 변경은 하지 않는다.

---

## 4. 하드웨어 및 실행 제약

### 4.1 허용 GPU

물리 GPU `4,5,6,7`만 사용한다.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

GPU 0~3은 어떤 CUDA 명령에도 노출하지 않는다.

### 4.2 목표 topology

주요 runtime topology:

- TP=2
- DP=2
- EP=4
- PP=1
- BF16
- single node
- NVLink intranode
- Qwen3-VL-30B-A3B-Instruct

이 topology를 선택하는 이유는 DP+EP layout에서 실제 dispatch/combine communication을 발생시키기 위해서다.

### 4.3 현재 환경 참고

기존 기록:

- vLLM: 0.20.0+cu129
- PyTorch: 2.11.0+cu129
- CUDA runtime: 12.9
- NCCL: 2.28.9
- GPU: H100 80GB
- model hidden/intermediate/top-k: 2048/768/8
- experts: 128

환경은 실행 시작 시 다시 기록하여 source of truth로 삼는다.

---

## 5. DeepEP 버전 정책

### 5.1 1차 선택

현재 vLLM 0.20 설치 도구가 pin한 호환 commit을 우선한다.

- DeepEP commit: `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`
- NVSHMEM: vLLM 0.20 설치 스크립트가 기대하는 버전 사용
- 별도 conda/venv 환경 사용
- 기존 working environment를 in-place로 깨뜨리지 않는다

### 5.2 DeepEP V2 제한

최신 DeepEP V2는 NCCL Gin 및 더 높은 NCCL 요구사항을 가질 수 있다. 이번 1차 PoC에서는 다음을 금지한다.

- 기존 NCCL 2.28.9의 in-place 교체
- vLLM 0.20과 검증되지 않은 DeepEP main API 혼합
- 시스템 전역 PyTorch/CUDA 변경

V1 호환 pin에서 가능성이 확인되고, SM contention이 핵심 잔여 문제로 확인될 때만 별도 후속 실험으로 V2를 제안한다.

---

## 6. Phase 0 — 저장소 및 환경 감사

실험 전 다음을 저장한다.

```text
poc_flashvep/deepep_revalidation/results/<run_id>/environment.txt
poc_flashvep/deepep_revalidation/results/<run_id>/git_state_before.txt
```

필수 내용:

- hostname
- date/time/timezone
- `nvidia-smi`
- GPU topology
- Python/PyTorch/vLLM/CUDA/NCCL/Triton versions
- DeepEP import 경로와 commit
- NVSHMEM version
- current Git remote/branch/HEAD/status
- exact model snapshot path
- visible devices
- available disk space
- `ulimit -l`

이 단계에서는 코드 수정이나 GPU benchmark를 시작하기 전에 환경 불일치를 식별한다.

---

## 7. Phase 1 — DeepEP 설치 및 자체 smoke test

### 7.1 설치

vLLM 0.20의 공식 EP-kernel 설치 흐름과 pin을 우선한다. 설치는 별도 환경에서 수행한다.

설치 후 다음을 확인한다.

- `import deep_ep`
- DeepEP shared object load
- intranode NVLink check
- 4-rank distributed initialization
- BF16 dispatch/combine 지원
- hidden size 2048 지원
- top-k 8
- experts 128
- EP4

### 7.2 DeepEP 자체 correctness test

DeepEP reference 또는 repository test를 현재 shape에 맞춰 실행한다.

필수 correctness:

- dispatch received token count
- expert별 token count
- top-k ID/weight preservation
- combine output
- source token order restoration
- BF16 tolerance

### 7.3 실패 시

설치/호환성 문제를 무한히 패치하지 않는다.

최대 범위:

- repository pin과 vLLM pin 정렬
- include/library path 수정
- 사용자 권한 내 build fix
- Python package dependency 정렬

금지:

- 시스템 driver 변경
- 무관한 vLLM 대규모 upgrade
- DeepEP kernel source 재작성

Phase 1 실패 시 `BLOCKED` 보고서를 작성하고 local commit/push까지 수행한다.

---

## 8. Phase 2 — Stock vLLM backend 비교

### 8.1 비교 configuration

동일 모델, 동일 입력, 동일 topology, 동일 scheduler budget으로 다음을 비교한다.

#### A. AG/RS baseline

- `allgather_reducescatter`
- DBO off

#### B. DeepEP serial baseline

- `deepep_high_throughput`
- DBO off

#### C. DeepEP stock overlap

- `deepep_high_throughput`
- DBO on

Prefill 중심 workload이므로 high-throughput backend를 우선한다. Backend가 실제로 선택됐는지 log와 object type으로 검증한다. silent fallback은 실패로 처리한다.

### 8.2 입력

우선 controlled request를 사용한다.

- image: 기존 896×896 controlled image
- prompt: `Describe this image briefly.`
- prompt tokens: processor 결과를 다시 기록
- request 수: 가능한 범위에서 1, 4, 8, 16
- warmup: 최소 5
- measured repetitions: 최소 20

OOM 또는 scheduler constraint가 있으면 가능한 최대 request count를 기록하고 임의로 결과를 숨기지 않는다.

### 8.3 측정

- request wall time
- TTFT 또는 prefill completion latency
- tokens/s
- GPU memory peak
- selected backend
- DBO enabled 여부
- correctness output token/text
- per-rank routed assignment
- critical rank
- layer-level D/E/C가 안전하게 측정 가능한 경우 해당 값

### 8.4 핵심 비교

- A → B: DeepEP backend 자체 효과
- B → C: stock DBO overlap 효과
- A → C: 전체 engineering substrate 효과

A → C를 FlashVEP novelty로 주장하지 않는다.

---

## 9. Phase 3 — DeepEP operator-level contention replay

### 9.1 입력 capture 재사용

가능하면 기존 capture를 재사용한다.

- layer: 24
- real request: 799 tokens
- vision: 784 tokens
- hidden: 2048
- top-k: 8
- experts: 128
- EP: 4

기존 capture가 DeepEP input contract와 맞지 않으면 동일 request를 다시 capture하되, hidden/top-k IDs/top-k weights의 checksum과 shape를 저장한다.

### 9.2 workload

필수:

- B_eq=32
- B_eq=64
- B_eq=128

B_eq는 동일 실제 request의 controlled replication이며 실제 serving batch라고 표현하지 않는다.

### 9.3 비교 variant

1. Existing AG/RS serial
2. Existing AG/RS wavefront K=2
3. DeepEP serial
4. DeepEP overlap K=2
5. DeepEP overlap K=4는 fragmentation이 허용될 때만
6. DeepEP supported communication-SM setting sweep

DeepEP V1 API가 전역 static SM setting만 허용하면 각 process run을 분리한다. 지원되지 않는 임의 SM 값을 강제하지 말고 API와 kernel constraints를 확인한다.

### 9.4 SM sweep

자동/default 값과 지원되는 낮은 SM budget을 비교한다.

예시 후보는 API 검증 후 사용한다.

```text
default
24
16
12
8
4
```

지원되지 않는 값은 `UNSUPPORTED`로 기록한다.

각 설정에서 측정:

- D alone
- E alone
- C alone
- D || E
- E || C
- D || E || C wavefront
- D slowdown ratio
- E slowdown ratio
- C slowdown ratio
- wall speedup
- overlap fraction
- memory peak

### 9.5 정확성

DeepEP serial을 reference로 사용한다.

- `torch.testing.assert_close`
- max absolute error
- mean absolute error
- cosine similarity
- route identity
- top-k weights identity
- output token order restoration
- all ranks pass

BF16 accumulation order 차이가 있을 수 있으므로 tolerance를 명시하고, bitwise exactness를 임의로 요구하지 않는다.

---

## 10. Phase 4 — Nsight Systems 검증

best DeepEP overlap configuration 하나와 대응 serial configuration 하나만 수집한다.

필수 확인:

- DeepEP communication kernel과 Triton expert kernel의 실제 동시 residency
- 통신 kernel의 시간 구간
- Expert kernel slowdown
- communication stream/compute stream dependency
- CPU synchronization 또는 unexpected gap
- repeated launch overhead
- HBM/L2 contention을 직접 확정할 수 없다면 추정이라고 표기

대형 `.nsys-rep`는 원격 repository에 commit하지 않는다. 대신 다음을 commit한다.

- 실행 명령
- kernel summary CSV/TXT
- screenshot 또는 작은 exported timeline summary
- 해석 보고서

---

## 11. 결과 metric 정의

### 11.1 Backend speedup

```text
DeepEP backend speedup
= AG/RS serial wall / DeepEP serial wall
```

### 11.2 Overlap speedup

```text
Stock DBO speedup
= DeepEP DBO-off wall / DeepEP DBO-on wall
```

Operator replay:

```text
DeepEP overlap speedup
= DeepEP serial wall / DeepEP overlap wall
```

### 11.3 Stage slowdown

```text
D slowdown = D_overlap_sum / D_serial_sum
E slowdown = E_overlap_sum / E_serial_sum
C slowdown = C_overlap_sum / C_serial_sum
```

### 11.4 Oracle ceiling

각 workload에서 serial D/E/C를 바탕으로 단순 ideal ceiling을 계산하되, achieved result와 구분한다.

```text
ideal wall lower bound ≈ max(E, D+C의 overlap 가능한 부분)
```

---

## 12. Gate

### 12.1 GO

다음을 모두 만족하면 최종 FlashVEP adaptation 구현으로 진행한다.

1. DeepEP correctness PASS
2. 실제 backend가 DeepEP로 선택됨
3. Nsight에서 communication/Expert kernel의 실제 동시 실행 확인
4. DeepEP overlap의 Expert slowdown ≤ 1.05x
5. DeepEP overlap의 Dispatch slowdown ≤ 1.25x
6. B64 또는 B128 중 하나에서 DeepEP serial 대비 ≥1.15x
7. 다른 하나의 workload에서 ≥1.10x
8. stock end-to-end DBO on이 DBO off 대비 의미 있는 이득을 보임
9. stock DBO 결과에 workload별 편차가 있어 adaptive policy의 여지가 관찰됨

### 12.2 HOLD

다음이면 mechanism 가능성은 남기되 최종 구현 전에 추가 debate를 수행한다.

- 최고 speedup 1.05x 이상 1.15x 미만
- contention은 크게 감소했으나 stock DBO 이득이 작음
- operator에서는 이득이 있으나 end-to-end에서 사라짐
- Qwen3-VL shape에서 DeepEP가 비효율적이지만 원인이 명확함

### 12.3 NO-GO

다음 중 하나면 overlap branch를 최종 중단한다.

- optimized DeepEP에서도 최고 speedup <1.05x
- actual kernel overlap이 없음
- Dispatch 또는 Expert slowdown이 계속 심함
- DeepEP integration overhead가 이득보다 큼
- stock DBO가 모든 이득을 가져가고 FlashVEP가 추가로 최적화할 신호가 없음
- correctness를 유지할 수 없음

### 12.4 BLOCKED

환경/설치 문제로 core experiment가 실행되지 않았을 때 사용한다. BLOCKED를 NO-GO로 바꾸지 않는다.

---

## 13. 향후 FlashVEP adaptation 후보

GO 후에만 구현한다.

### 13.1 역할

FlashVEP는 DeepEP/DBO를 대체하지 않는다. 그 위에서 현재 MLLM layer의 routing 상태를 보고 execution mode를 조절한다.

입력 후보:

- vision assignment/rank histogram
- vision critical-rank load
- vision imbalance ratio
- expert token count distribution
- predicted expert window
- D/C volume
- current batch token count
- modality ratio
- optional MACS/ReaLB 활성 상태

Action 후보:

- serial vs overlap
- DBO on/off
- microbatch degree
- DeepEP communication SM budget
- dispatch/combine priority
- overlap schedule

### 13.2 MACS/ReaLB composition

FlashVEP는 실행 scheduling/resource adaptation layer로 정의할 수 있다.

- MACS가 expert capacity와 effective load를 변경하면 FlashVEP는 변경된 D/E/C shape에 맞춰 overlap을 재선택한다.
- ReaLB가 vision-heavy critical rank를 저정밀로 빠르게 만들면 expert window가 짧아질 수 있으므로 FlashVEP는 communication SM budget이나 overlap mode를 다시 조절한다.
- 조합 실험에서는 각 방법의 단독 효과와 조합 효과를 분리한다.

필수 ablation:

1. DeepEP only
2. DeepEP + DBO
3. DeepEP + DBO + FlashVEP
4. DeepEP + DBO + MACS
5. DeepEP + DBO + ReaLB
6. DeepEP + DBO + MACS/ReaLB + FlashVEP

이번 PoC에서는 3~6을 구현하지 않는다.

---

## 14. 산출물

다음 구조를 권장한다.

```text
poc_flashvep/
  deepep_revalidation/
    README.md
    install_deepep_env.sh
    run_deepep_smoke.sh
    run_vllm_backend_matrix.sh
    run_operator_replay.sh
    run_nsight_best.sh
    analyze_results.py
    configs/
    results/
      <run_id>/
        environment.txt
        git_state_before.txt
        backend_matrix.json
        operator_matrix.json
        correctness.json
        nsight_summary.txt
        gate.json
  reports/
    deepep_overlap_revalidation_report.md

docs/
  spec/
    flashvep_deepep_overlap_revalidation_poc_spec.md
  prompt/
    codex_cli_flashvep_deepep_overlap_revalidation_prompt_ko.txt
```

대형 raw trace, model weight, build artifact는 commit하지 않는다.

### 최종 보고서 필수 목차

1. Executive summary
2. Environment and exact versions
3. DeepEP installation result
4. Backend selection proof
5. Stock vLLM A/B/C results
6. Operator-level D/E/C results
7. SM-budget sweep
8. Nsight overlap evidence
9. Correctness
10. Comparison with previous AG/RS wavefront
11. GO/HOLD/NO-GO/BLOCKED
12. Implication for FlashVEP novelty
13. Git branch/commit/push status

---

## 15. 최종 출력 형식

Codex CLI agent는 종료 시 채팅에 다음을 출력한다.

```text
FINAL STATUS: GO | HOLD | NO-GO | BLOCKED

Repository:
Remote:
Branch:
Commit:
Pushed:

Best stock backend result:
Best DeepEP overlap result:
D slowdown:
E slowdown:
C slowdown:
Nsight actual overlap:
Correctness:

Main report:
Gate JSON:
Key result directory:

What changed:
What was not attempted:
Single recommended next action:
```

모든 수치는 파일에 실제 기록된 결과에서 가져오며 추측하지 않는다.
