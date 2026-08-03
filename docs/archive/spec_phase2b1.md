# Phase 2-B-1 Spec — Method 1 Placement: As-Is vs To-Be (8×H100, vLLM EP)

> **목적**: calibration profile로 만든 modality-aware expert placement(Method 1)를 적용해, naive linear placement(As-Is) 대비 **straggler·end-to-end latency가 감소하고 accuracy는 손실이 없음**을 측정·증명한다. 이것이 ReaLB(정밀도 손실)·MACS(token drop 손실) 대비 본 연구의 핵심 차별점(무손실)의 첫 실증이다.

---

## 0. 현재까지 확립된 것 (전제)

- vLLM 0.20 native 8-way EP 동작 (`tensor_parallel_size=8`, `enable_expert_parallel=True`, `expert_placement_strategy="linear"`, `all2all_backend="allgather_reducescatter"`). rank당 ~11.8GB, expert 16/128 per rank.
- routing 추출: `enable_return_routed_experts=True` → `routed_experts [seq, layer, topk]`.
- P3 calibration 결과 존재: `dist_vision.npy`, `dist_text.npy` (48,128) — layer별 vision/text expert 선호. modality 분리 확인됨(mean TV 0.607).
- M2: straggler는 batch/layer 단위로 봐야 보임(전체 평균 1.04x, batch·layer 단위 최대 rank 1.95x / expert 6.74x).

---

## 1. 핵심 관문 — vLLM에 layer별 custom placement 주입 (먼저 해결)

vLLM은 기본 `linear`/`round_robin`만 노출하고, 보통 **모든 layer에 동일한** expert→rank 매핑을 쓴다. 그러나 본 연구는 **layer별로 다른 modality-balanced 매핑**이 핵심이다 — P3 데이터가 그 근거다(아래).

**왜 layer별이 필수인가 (P3 근거):**
- P3에서 vision/text expert 선호의 TV distance가 layer마다 크게 다름(layer 9=0.734 최대, 상위 9/11/13/12/8/23/25/7에 집중). 즉 "어느 expert가 vision-특화냐"가 layer마다 다르다.
- M2에서 straggler도 layer마다 다른 expert에서 발생(layer 20=expert 23 hot).
- 따라서 전 layer 공통 매핑은 이 layer별 차이를 평균으로 뭉개 straggler를 못 잡는다. **layer별 매핑이 본 method의 본질.**

**Step 0 (게이트): layer별 custom placement 주입 경로 확보.**
- vLLM 0.20에서 expert→rank 매핑을 **layer별로 다르게** 주입하는 경로 조사:
  - `determine_expert_map()`이 layer별로 호출되는지, layer index를 받는지.
  - `FusedMoE` 인스턴스가 layer마다 생성되므로, 각 layer의 `expert_map`을 그 layer 전용 매핑으로 설정할 수 있는지.
  - 공식 플래그/API가 없으면, 각 `FusedMoE` 생성 시점에 layer_id에 맞는 매핑을 주입하는 **최소 패치** 설계.
- **fused MoE 커널은 절대 수정 금지** — layer별 `expert_map`(expert→rank 테이블)만 교체. all-to-all dispatch는 이 매핑 테이블을 따라가므로, 테이블만 바꾸면 layer별 배치가 적용됨.
- `docs/vllm_placement_inject.md`에 layer별 주입 방법·검증 결과 기록.
- **검증**: layer별 매핑 적용 후 (a) rank당 메모리 여전히 ~12GB(복제 아님), (b) routed_experts 정상, (c) generation 안 깨짐, (d) **실제로 layer마다 다른 매핑이 적용됐는지**(예: layer 9와 layer 20의 expert→rank가 다른지) 확인. 이게 통과해야 진행.

**만약 layer별 주입이 fused 커널 수정 없이는 불가능하다고 판명되면:**
- 멈추고 보고. 대안 논의: (i) 모델 로드 시 layer별로 expert weight 텐서를 재배열, (ii) layer별 매핑을 흉내내는 다른 경로. 공통 매핑으로 후퇴하기 전에 layer별 가능성을 먼저 충분히 탐색할 것 — 이게 본 연구의 핵심이므로.

---

## 2. Method 1 — Per-Layer Modality-Balanced Placement 산출

`method1/placement.py`(Phase 1 구현)를 실제 calibration 통계로 구동. **layer마다 독립적으로** placement를 산출한다.

**Step 1. Layer별 expert modality 분류 (P3 재사용).**
- `dist_vision.npy`, `dist_text.npy` (48,128)로 **각 layer ℓ에서** expert별 modality score `v[ℓ][e]` 계산(P3의 layer별 분포 그대로 사용 — 합산하지 말 것).
- layer별 분류: vision-spec / text-spec / shared (threshold MODE δ=0.1 참고).

**Step 2. Layer별 LPT placement.**
- **각 layer ℓ마다** expert 가중치 = 그 layer의 vision 부하 추정치(`v[ℓ][e]` × 평균 호출량). LPT greedy로 8 rank에 배치, vision-spec expert가 한 rank에 몰리지 않게.
- 출력: layer별 `expert_id -> rank` 매핑 (48 layer × 128 expert). `outputs/placement/modality_balanced_map_perlayer.json` (구조: `{layer: {expert: rank}}`).
- 제약: 각 layer·각 rank 정확히 16 experts.
- 비교 기준선: linear 매핑(모든 layer 동일, 0-15→0...)도 같은 형식으로 저장.
- **검증**: layer 9와 layer 20의 매핑이 실제로 다른지 확인(P3에서 vision expert가 다르므로 매핑도 달라야 함).

---

## 3. As-Is / To-Be 측정 (공정 비교)

**동일 조건**: 같은 모델, 같은 입력, 같은 vLLM EP 설정. **유일한 차이 = expert→rank 매핑** (As-Is: linear / To-Be: modality-balanced).

데이터: 메인 벤치 — ChartQA, MMMU, MMStar (+여유되면 TextVQA, MMBench). accuracy는 정식 점수(smoke 아님), 가능한 충분한 샘플 수.

### 측정 3축 (각각 As-Is, To-Be 둘 다)

**(A) Accuracy — 핵심 셀링포인트**
- 각 벤치 정식 정확도 측정 (lmms-eval 또는 검증된 matcher).
- **기대: To-Be ≈ As-Is (무손실).** placement는 expert 위치만 바꾸고 연산 동일하므로 정확도가 같아야 함(BF16 reduction 순서 차이로 미미한 변동만 허용).
- 출력: `outputs/asis_tobe/accuracy.{json,png}` — As-Is vs To-Be 막대 비교.

**(B) End-to-end latency / throughput**
- prefill latency(또는 throughput tok/s) 측정. 동일 batch·반복 측정으로 분산 포함.
- **기대: To-Be < As-Is (감소).** straggler 완화로 prefill 빨라짐.
- naive 공통 매핑이라 이득이 작을 수 있음 — "이득이 있는가/방향이 맞는가"가 1차 목표.
- 출력: `outputs/asis_tobe/latency.{json,png}` — As-Is vs To-Be (평균±분산).

**(C) Straggler 비율**
- routed_experts로 per-rank load 집계, **batch·layer 단위 max/mean imbalance** 측정(전체 평균은 평탄하니 분포로).
- **기대: To-Be < As-Is (imbalance 감소).** vision expert 분산으로 hot rank 완화.
- 출력: `outputs/asis_tobe/straggler.{json,png}` — As-Is vs To-Be imbalance 분포(여러 batch/layer의 imbalance 분포 비교, 또는 대표 batch).

### 종합 figure
- `outputs/asis_tobe/summary.png`: 3축(accuracy / latency / straggler)을 As-Is vs To-Be로 나란히. advisor 슬라이드용 한 장.

---

## 4. 규칙

- **유일한 변수는 placement 매핑.** 그 외(모델·입력·EP 설정·precision)는 As-Is/To-Be 완전 동일. 공정 비교가 생명.
- **모델 연산 수정 금지** — placement는 expert→rank 매핑 테이블만 교체. fused MoE 커널, precision, token은 절대 건드리지 마(그건 ReaLB/MACS 방식이고 본 연구 아님).
- **prefill 기준** latency·straggler 측정.
- accuracy는 충분한 샘플로 정식 측정(20개 smoke 아님). 시간 제약 시 벤치 1~2개라도 정식으로.
- de-RoPE, CLS, Method 2(merge/cap), P2 gating은 이 단계 아님 — `# TODO`.

## 5. 예상 결과 / 해석 가이드

- **Accuracy 무손실이 확인되면** = 핵심 성공. ReaLB(정확도 -3%)·MACS(token drop -5~7%) 대비 "우리는 0% 손실"이 데이터로 섬.
- **Layer별 placement이므로 공통 매핑보다 straggler·latency 이득이 커야 정상.** P3가 보였듯 vision expert가 layer마다 다르니, layer별 매핑이 각 layer의 straggler를 제대로 잡아야 함. 이득이 의미 있게 나오면 "layer별이 본질"이라는 본 연구 주장이 데이터로 입증됨.
- (참고) 만약 layer별 주입이 끝내 불가능해 공통 매핑으로 후퇴하게 되면, 공통 매핑의 약한 이득 자체가 "layer별이 필요하다"는 근거가 되지만 — 1차 목표는 layer별을 실제로 적용해 이득을 보이는 것.
- **Ablation 가치**: 가능하면 (i) linear(As-Is) (ii) 공통 modality 매핑 (iii) **layer별 modality 매핑(To-Be)** 3단 비교를 하면, "layer별이 공통보다 낫다"를 직접 보여 본 연구의 핵심 주장을 강화. 시간 되면 포함.

## 6. 완료 시
`docs/phase2b1_report.md`: placement 주입 방법, As-Is/To-Be 3축 수치, summary figure 링크, 핵심 결론(특히 accuracy 무손실 여부), 이득이 작다면 원인 분석(공통 vs layer별 매핑).
