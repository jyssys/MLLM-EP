# Phase 2-A Spec — Motivation Evidence & Calibration Profiling (8×H100)

> **목적**: 본 연구의 motivation을 실데이터로 실증하고(intro·advisor 설득용), calibration profiling 통계를 수집한다. GPU 8장이 확보된 상태. 이 단계는 **측정·프로파일링만** 수행한다. Method 1/2 적용·정확도 개선·speedup 측정은 다음 단계(Phase 2-B).

---

## 0. 배경 (한 줄)

MoE-MLLM의 EP 추론에서 prefill 시 vision token이 routing을 지배해 일부 GPU가 straggler가 된다. 이 단계는 그 현상이 실재함을 Qwen3-VL-30B-A3B-Instruct에서 직접 측정해 보인다.

---

## 1. 환경·모델·EP 설정

- 모델: **Qwen3-VL-30B-A3B-Instruct** (이미 `models/`에 다운로드됨, 58G)
- GPU: **8×H100 80GB**
- 병렬: **attention = DP, MoE = 8-way EP** (DeepSpeed). MoE expert 128개를 8 GPU에 16개씩 정적 배치(vanilla, 순차 분배).
- 모드: **prefill 중심**. decode는 측정에서 제외하거나 분리(vision token은 prefill에만 존재).
- 프레임워크: DeepSpeed (MACS와 동일 계열). vLLM 아님.

**먼저 sanity check (필수 선행):**
- 30B를 8 GPU에 로드, EP 배치 확인(GPU당 ~7GB weight). 더미 multimodal 입력으로 forward 1회 통과. OOM·통신 오류 없는지. 이게 통과해야 이후 측정이 의미 있음.

---

## 2. 데이터셋 구성 (확정)

### 2.1 Motivation 실험용 (vision token 비중·straggler 시연)
vision token 비율을 끌어올리고 modality 변동을 보이기 위해 **이미 받아둔 벤치 혼합** (data_assets.md):
- **Vision-heavy**: MMMU(multi-image), ChartQA(고해상도 차트), TextVQA(OCR), InfoVQA가 있으면 추가
- **균형/text-heavy 대비군**: MMBench
- 각 벤치에서 **N=64~128 샘플**씩 추출(빠른 측정용). 고해상도·다중이미지 샘플 우선.
- 목적: rank별 vision 비율이 "31~93%처럼 변동"함을 보이기 → ReaLB Figure 2 재현.

### 2.2 Calibration용 (profiling 통계 수집)
- **ShareGPT4V 512장** (이미 `data/sharegpt4v_512/`). MODE 검증 셋.
- 모든 샘플 이미지 포함 → vision token 보장.
- 목적: trace / gating / (L,E) distribution 수집.

---

## 3. 측정 항목 (이 단계의 핵심 산출물)

### 3.1 Motivation 실증 (data 2.1로)

**(M1) Token modality 비중**
- attention→router로 들어가는 token 중 vision vs text 개수·비율.
- per-input, per-batch 집계. "vision token이 수적으로 압도(예: 90%+)" 확인.
- **Figure**: vision:text 비율 막대 또는 분포. (베스트: 입력별 vision 비율 히스토그램.)

**(M2) EP straggler 실증 (ReaLB Figure 2 재현)**
- 8-way EP vanilla 배치에서, **per-rank token load**(GPU별 받은 token 수)를 vision/text로 분해.
- **per-expert token load**도 vision/text 분해(어느 expert가 hot한가).
- 각 rank·expert의 vision token 비율(%) 표기.
- **Figure**: ReaLB Fig 2(a) 스타일 — 상단 per-rank(vision/text 누적막대 + 비율%), 하단 per-expert(vision/text 누적막대 + hot expert 비율%). device-level / expert-level ideal load 선 표시.
- **핵심 메시지**: hot rank/expert가 vision token 비율이 높다 = straggler의 주원인이 vision.
- (선택) iteration별 hot rank 변동(ReaLB Fig 2b/c) — straggler가 동적임을 보이면 보너스.

### 3.2 Calibration profiling (data 2.2로)

**(P1) Trace — token당 선택 expert 통계**
- 각 token의 top-8 expert 인덱스 기록(layer별).
- 집계: expert별 선택 빈도, layer별 분포.
- **Figure**: expert selection frequency. (matrix 권장: layer × expert 히트맵 — 어느 (layer,expert)가 자주 선택되나.)

**(P2) Gating score — token별 softmax 값**
- router softmax 출력(top-k 정규화 전/후 둘 다 기록 가능, config가 `norm_topk_prob=true`).
- token별 gating 분포, expert별 평균 gating.
- **Figure**: gating score 분포. (matrix: layer × expert 평균 gating 히트맵, 또는 token × expert 샘플 히트맵.)

**(P3) Modality별 (L, E) distribution — 테이블/확률 형태**
- **vision token만**으로 집계한 (layer, expert) 선택 확률 분포: `P_vis[ℓ, e]`
- **text token만**으로 집계한 (layer, expert) 선택 확률 분포: `P_txt[ℓ, e]`
- layer마다 expert 위로 정규화(합=1), MODE 방식.
- **출력**: (L=48, E=128) 행렬 2개(vision/text), 확률값. 테이블(주요 layer 발췌) + 히트맵.
- **핵심 분석**: `P_vis - P_txt` 히트맵 → vision/text가 *다른* expert를 선호함을 보임(MODE Fig 2/7 재현). 이게 Method 1(modality-aware placement)의 직접 근거.

---

## 4. 구현 — Phase 2 스텁 채우기

어제 만든 스텁(`hooks/register_hooks.py`, `method2/derope.py`, `pipeline/ep_integration.py`)의 `NotImplementedError` 중 **이 단계에 필요한 것만** 채운다. derope·merge·cap·placement 적용은 이 단계 불필요(측정만).

### 4.1 채울 것
- **`hooks/register_hooks.py`**: 실제 HF Qwen3-VL에 forward hook 등록.
  - router logits 캡처 (`Qwen3VLMoeTextSparseMoeBlock`의 gate 출력) → top-8 + gating score
  - self-attention weights 캡처 (per-layer) → token modality 분석엔 불필요할 수 있음(M1/M2는 routing만으로 됨). attention은 P3 이후 Method 2용이니 **이 단계는 routing·token_type·hidden만 필수**, attention은 optional.
  - token_type 캡처 (`mm_token_type_ids`: text=0/image=1/video=2) → vision/text 구분
  - hidden_states 캡처 (centroid용, calibration에서만)
  - 출력 형식은 Phase 1 calibration 모듈 계약과 일치.
- **EP load 측정 유틸** (`measure/ep_load.py`, 신규): vanilla 8-way EP 배치 기준으로 각 rank·expert가 받은 token 수를 vision/text 분해해 집계.

### 4.2 채우지 않을 것 (Phase 2-B)
- de-RoPE 수식, CLS 항
- Method 1 placement 적용, Method 2 merge/cap 적용
- speedup·정확도 측정, 하이퍼파라미터

---

## 5. 산출물 (이 단계 완료 시)

`outputs/motivation/`:
- `token_modality_ratio.{json,png}` — M1
- `ep_straggler_rank.{json,png}`, `ep_straggler_expert.{json,png}` — M2 (ReaLB Fig2 스타일)

`outputs/calibration/`:
- `trace_freq.{npy,png}` — P1 (layer×expert 히트맵)
- `gating_score.{npy,png}` — P2
- `dist_vision.npy`, `dist_text.npy` — P3 (L×E 확률 행렬 2개)
- `dist_diff.png` — P3 (P_vis − P_txt 히트맵)
- `dist_table.md` — P3 주요 layer 발췌 테이블

각 figure는 본 연구 intro·슬라이드에 바로 쓸 수 있는 품질로(축 label, 비율% 표기, 범례).

---

## 6. 실행 순서

1. 8 GPU 모델 로드 + EP forward sanity check (더미 입력)
2. hook 등록 구현·검증 (작은 샘플 1~2장으로 routing·token_type이 올바르게 나오는지)
3. **Motivation 측정** (data 2.1): M1 → M2, figure 생성
4. **Calibration profiling** (data 2.2, ShareGPT4V 512): P1 → P2 → P3, figure·table 생성
5. 산출물 정리 + 간단 리포트(`docs/phase2a_report.md`): 측정 방법, 핵심 수치(vision 비율, max/mean load imbalance, P_vis vs P_txt 차이), figure 링크

---

## 7. 주의사항

- **prefill 기준 측정.** decode token이 섞이면 vision 분석이 흐려지니, prefill-only로 forward하거나 prefill token만 집계.
- **top-k 정규화**: `norm_topk_prob=true`이므로 gating score는 정규화 후 값을 기본으로, 필요시 정규화 전도 같이 기록.
- **M-RoPE 주의**: 이 단계는 attention 값을 직접 안 써도 됨(routing 기반 측정). attention 캡처는 optional이고, de-RoPE는 Phase 2-B.
- **메모리**: 30B + 8 GPU면 여유. batch는 straggler가 보이는 크기로(rank당 충분한 token). 작은 batch는 skew가 안 보이니 vision-heavy 입력을 충분한 크기로.
- 측정은 모델을 **수정하지 않고**(vanilla 배치) 통계만 수집. placement·merge는 다음 단계.
