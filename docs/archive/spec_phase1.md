# Phase 1 Spec — Modality-Aware Expert Placement & Cross-Attention Merging for MoE-MLLM EP Inference

> **이 문서의 목적**: 코딩 에이전트가 Phase 1(GPU 없이, CPU만)에서 작성·검증할 코드의 범위·모듈별 입출력·테스트 케이스를 정의한다. Phase 1의 목표는 **모든 핵심 로직을 순수 PyTorch dummy로 구현하고 CPU 단위테스트로 검증**하는 것이다. 실제 30B forward(통계 수집·speedup·정확도 측정)는 GPU가 도착하는 Phase 2에서 수행한다.

---

## 0. 연구 한 줄 요약 (맥락)

MoE-MLLM의 Expert Parallelism(EP) 추론에서 vision token이 prefill을 지배해 일부 GPU가 straggler가 되고, 동기화 장벽 때문에 전체 latency가 straggler에 묶인다. 기존 연구는 배치(placement)를 고정한 채 증상을 사후 처리한다 — ReaLB는 정밀도(FP4), MACS는 capacity. 본 연구는 **(Method 1)** modality 특화 expert를 GPU에 고르게 배치해 straggler를 원인부터 제거하고(무손실·정적·통신비 0), **(Method 2)** 그래도 batch dynamics로 남는 잔여 straggler를 cross-attention 기반으로 표적 병합한다.

핵심 노블티: vision token을 *token 공간*이 아니라 *token-expert 결합 공간*에서 판단·처리한다(text-vision-expert 삼자 관계).

---

## 1. 범위 (Phase 1)

### 1.1 In scope (CPU, dummy로 구현·검증)
- M1: per-layer modality-balanced placement (LPT greedy)
- M2: cross-attention 중요도 a_j (**raw 버전만**) → load-targeted selection → routing-보존 expert-aware merge (a+b) → cap
- Calibration 통계 수집 **로직** (dummy router 출력으로 테스트): modality 분류 + key/redundant expert 선호 + expert centroid
- 모든 모듈의 CPU 단위테스트

### 1.2 Out of scope (Phase 2, GPU 필요 — 코드에 TODO로 표기)
- 실제 Qwen3-VL-30B forward (통계 수집 실행, attention/​routing 추출)
- de-RoPE 항, CLS 항 (a_j의 옵션 — Phase 1은 raw만)
- (c) redundant token rerouting (지금 설계에서 제외, centroid는 보험으로 수집만)
- DeepSpeed-MoE 통합 (Phase 1은 순수 PyTorch dummy)
- speedup·throughput·정확도 측정, 하이퍼파라미터 튜닝(ρ, τ, cap, λ)

### 1.3 모델·데이터 (Phase 1은 다운로드·문서화까지만)
- 모델: **Qwen3-VL-30B-A3B-Instruct** (HF). layer=48, experts/layer=128, top-k 확인 필요.
- Calibration: **ShareGPT4V 512장** (MODE와 동일, 이미지 포함 필수).
- Main benchmarks(Phase 2 평가용, 지금은 lmms-eval 설치·데이터 확보만): MMMU, MMBench, ChartQA, TextVQA, MMStar (1차). 평가 도구는 **lmms-eval**로 통일.

---

## 2. 환경 세팅 (CPU, 지금)

1. conda/venv 환경, 의존성 고정(requirements.txt + lockfile).
   - torch (CPU), transformers, deepspeed (import만 — 실행 X), lmms-eval, numpy, scipy, pulp 또는 ortools(LPT 비교용, 선택).
2. Qwen3-VL-30B-A3B-Instruct weight + config 다운로드 (HF). 100T 여유로 가능.
3. **모델 구조 문서화** (`docs/model_arch.md`):
   - layer 수, experts/layer, top-k, hidden dim, router 구조(softmax/​gate linear 위치).
   - MoE forward 경로: router → dispatch → expert → combine 이 코드 어디서 일어나는지.
   - **RoPE 종류 파악**: M-RoPE(multimodal/2D RoPE) 여부, rotary 적용 방식. → de-RoPE 구현 가능성·식 변경 여부를 문서화(구현은 Phase 2).
   - vision token / text token이 시퀀스에서 구분되는 방식(token type id 등).
4. ShareGPT4V 512장 + main benchmark 데이터 다운로드, lmms-eval dry-run(평가 파이프라인이 도는지, 정확도 측정 X).

---

## 3. 모듈 명세 (순수 PyTorch, dummy로 구현)

> 공통 원칙: 모든 모듈은 **프레임워크 비의존**. 입력은 plain tensor/dict, 출력도 plain. DeepSpeed·실모델에 나중에 붙일 수 있게 인터페이스만 깔끔히.

### 3.1 Calibration 통계 수집 (`calib/collect_stats.py`)
실모델 forward는 Phase 2지만, **집계 로직**은 dummy router 출력으로 Phase 1에서 검증.

입력 (per layer ℓ, calibration set 전체에 대해 누적):
- `expert_assignment`: 각 token이 라우팅된 top-k expert 인덱스. shape [num_tokens, k].
- `token_type`: 각 token이 text / vision 인지. shape [num_tokens]. (vision은 추가로 key/redundant 구분 — 아래 3.2의 a_j로 판정.)
- `hidden_states`: 각 token의 hidden. shape [num_tokens, D]. (centroid용.)

출력 (per layer ℓ):
1. **modality 분류용 count**: `N_vis[e]`, `N_txt[e]` — expert e가 vision/text token에 의해 선택된 횟수.
   - → modality specialization score `Δ[e] = f_vis[e] - f_txt[e]` (layer-wise normalized).
   - → 분류: vision-spec(Δ≥δ), text-spec(Δ≤-δ), shared(|Δ|<δ). δ는 설정값(기본 0.1, MODE 따름).
2. **key/redundant expert 선호**: `f_key[e]`, `f_red[e]` — key vision token / redundant vision token이 expert e를 선택한 normalized 빈도. (key/red는 a_j 상위 20%/하위 80%, MODE 따름.)
3. **expert centroid**: `centroid[e] = mean(hidden of tokens routed to e)`. shape [D]. (지금 미사용, 보험으로 수집.)

저장: layer별 numpy/torch 파일. 메모리: centroid가 48×128×D×2bytes ≈ 25MB(MODE 검증), 무시 가능.

**Phase 1 dummy 테스트**: 가짜 expert_assignment·token_type·hidden을 만들어, 위 통계가 수식대로 집계되는지(정규화 합=1, Δ 부호, key/red 분리) 검증.

### 3.2 Cross-attention 중요도 a_j (`method2/importance.py`)
**Phase 1은 raw 버전만** (SparseVLM/MODE 식, 인용 부품):

```
P^ℓ = A^ℓ[text_idx, vision_idx]          # text→vision attention sub-block, [L_t, L_v]
a_j^ℓ = (1/L_t) * sum_{t in text} P^ℓ[t, j]   # vision token j의 중요도
```

- 일반형 인터페이스로 설계하되 Phase 1 기본은 `lambda_cls=0`, `derope=False` (= raw).
  - `a_j = (1/L_t) Σ derope(Attn(t→v_j)) + lambda_cls * Attn_CLS(v_j)`
  - de-RoPE·CLS 항은 **Phase 2 TODO** (인터페이스만 남기고 NotImplemented 또는 passthrough).
- key/redundant 판정: layer마다 a_j 상위 ρ_key(기본 20%)를 key, 나머지를 redundant.

**Phase 1 dummy 테스트**: 가짜 attention matrix(text/vision 인덱스 지정)로 a_j가 평균식대로 나오는지, top-20% 분리가 맞는지 검증.

### 3.3 Method 1 — Placement (`method1/placement.py`)
입력: per-layer expert weight `w[e] = ŝ_vis[e] * L̄[e]` (modality score × 평균 부하), GPU 수 G(기본 8).
출력: per-layer expert→GPU 매핑 `pi[ℓ]: e -> g`.

알고리즘: **LPT greedy** (Longest Processing Time)
```
sort experts by w desc
for each expert: assign to least-loaded GPU
```
- 제약 옵션: GPU당 expert 개수 상한(메모리). 가득 찬 GPU는 후보 제외.
- 목적: minimize max_g Σ_{e in g} w[e].
- (ILP 버전은 선택 — pulp/ortools로 LPT와 결과 비교용. 기본은 LPT.)

**Phase 1 dummy 테스트**:
- vision-heavy expert가 한 GPU에 몰린 입력 → LPT 후 max-load가 vanilla보다 감소하는지.
- LPT 4/3 근사 보장 sanity check(작은 케이스에서 최적해와 비교).
- 같은 modality score 입력 시 균등 분배되는지.

### 3.4 Method 2 — Load-targeted selection (`method2/selection.py`)
입력: per-layer expert별 token count `L[g]`(GPU별 부하), expert_assignment, a_j, ρ^ℓ, straggler threshold τ.
출력: 병합 후보 집합 `C^ℓ`.

```
S^ℓ = { g : L[g] > τ }                              # straggler GPU 집합
C^ℓ = { v_j : g(v_j) ∈ S^ℓ                          # straggler로 가는 + 
              AND a_j < quantile_{ρ^ℓ}(a | S^ℓ) }    # 중요도 하위
```

- ρ^ℓ: layer별 비율. Phase 1은 상수(기본 0.3), sigmoid 스케줄은 인터페이스만(Phase 2 튜닝).

**Phase 1 dummy 테스트**:
- straggler 아닌 GPU로 가는 token은 a_j가 낮아도 C에 안 들어감.
- 중요도 높은 token은 straggler로 가도 C에 안 들어감.
- ρ 비율대로 후보 크기가 정해지는지.

### 3.5 Method 2 — Expert-aware merge (a+b) (`method2/merge.py`)
입력: 후보 `C^ℓ`, expert_assignment, hidden_states, a_j.
출력: 병합된 대표 token 집합 + 갱신된 expert 입력.

```
# (a)(b): 같은 expert로 가는 + 유사한 redundant token만 클러스터
for each straggler expert e:
    cand_e = { v_j in C^ℓ : e in topk(v_j) }        # e로 가는 후보만 (routing 보존)
    clusters = cluster_by_similarity(cand_e)        # routing 분포 또는 hidden 유사도
    for G_m in clusters:
        v_hat = Σ_{j in G_m} (a_j / Σ a) * v_j       # 중요도 가중 평균
        replace G_m's contribution to e with v_hat
```

- **routing 보존 핵심**: token을 통째로 합치는 게 아니라 **expert e로 가는 몫만** 병합. v_j가 다른 expert로도 가면(top-k>1) 그 경로는 건드리지 않음.
- 유사도 기준: Phase 1은 hidden cosine(기본). routing-분포 KL은 옵션 인터페이스.

**Phase 1 dummy 테스트**:
- 다른 expert로 가는 token은 같은 클러스터에 안 묶임(routing 보존 검증).
- 가중 평균이 a_j 비례로 계산되는지.
- 병합 후 그 expert의 입력 token 수가 줄어드는지(부하 감소 확인).

### 3.6 Method 2 — Cap (`method2/cap.py`)
입력: straggler GPU 부하, 2등 GPU 부하 `L_(2)`, merge 함수.
출력: 실제 적용할 병합량(ρ_eff).

```
ρ_eff = min{ ρ : L_straggler(merge_ρ) <= L_(2) }    # 2등 수준에서 정지
actual_merge_ratio = min(ρ^ℓ 천장, ρ_eff)
```

**Phase 1 dummy 테스트**:
- straggler가 2등과 같아지면 병합 중단.
- cap 없이 max 최소화 vs cap 적용 — cap 쪽이 병합량 적은지(over-merge 방지 확인).

### 3.7 End-to-end dummy pipeline (`pipeline/dummy_moe.py`)
순수 PyTorch dummy MoE layer로 전체 흐름 통합 테스트:
```
dummy attention → a_j 계산 → router(dummy) → token count → S^ℓ 식별
→ [placement는 사전 고정] → selection → merge(cap 적용) → expert(dummy FFN) → combine
```
- 목적: 모듈들이 인터페이스 맞물려 end-to-end로 도는지. 수치 정확성보다 **연결성·shape·routing 보존** 검증.
- 정확도 영향 측정: dummy expert(예: identity 또는 작은 linear)로 "merge 전후 출력 차이"가 a_j 낮은 token에 집중되는지 확인.

---

## 4. 디렉토리 구조 (제안)
```
project/
  docs/
    model_arch.md          # 3. 구조·RoPE 문서화
    spec_phase1.md         # 이 문서
  env/
    requirements.txt, lockfile
  calib/
    collect_stats.py       # 3.1 (로직, Phase2 실행)
  method1/
    placement.py           # 3.3
  method2/
    importance.py          # 3.2
    selection.py           # 3.4
    merge.py               # 3.5
    cap.py                 # 3.6
  pipeline/
    dummy_moe.py           # 3.7
  tests/
    test_placement.py
    test_importance.py
    test_selection.py
    test_merge.py
    test_cap.py
    test_pipeline.py
  data/                    # ShareGPT4V, benchmarks (다운로드)
  models/                  # Qwen3-VL weight
```

---

## 5. Phase 1 완료 기준 (Definition of Done)
- [ ] 환경 구축 + 의존성 고정, lmms-eval dry-run 통과
- [ ] Qwen3-VL weight·config 다운로드, `model_arch.md` 작성(특히 RoPE 종류·MoE 경로)
- [ ] ShareGPT4V 512장 + main benchmark 데이터 확보
- [ ] 3.1~3.7 모듈 전부 순수 PyTorch로 구현(실모델 비의존)
- [ ] 모든 `tests/` 통과 (CPU, dummy)
- [ ] de-RoPE/CLS/(c)rerouting/DeepSpeed 통합은 코드에 `# TODO(Phase2)`로 명시

---

## 6. Phase 2 예고 (GPU 도착 후, 스펙 갱신 예정)
- 실 Qwen3-VL forward로 calibration 통계 수집 → placement 산출
- de-RoPE/CLS 구현(RoPE 구조 기반) + a_j ablation(raw vs de-RoPE vs +CLS)
- DeepSpeed-MoE 통합(attention DP + MoE 8-way EP), dispatch 전 merge 삽입
- batch·이미지 수 sweep(straggler 가시화), 잔여 straggler 곡선(vanilla / +M1 / +M1+M2)
- speedup·max-load·정확도 측정, baseline(MACS/CAI-MoE) 비교
- 하이퍼파라미터 튜닝(ρ sigmoid, τ, cap, λ)
