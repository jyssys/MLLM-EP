# Layered Prefill × FastPP × NanoFlow — bounded screening 결과

## 최종 screening 판정 — 2026-09-09

**FINAL STATUS: PARTIAL_ENVIRONMENT_BLOCKED**

이번 재개 작업에서 가능한 native baseline/control 수집은 끝냈다. 그러나
**세 방법 전체의 faithful MLLM successor 검증이 완료된 것은 아니다.**
공식 MoE 경로/최소 호환 포트는 실행했고 상당한 대조 실험을 확보했지만,
native VL/PP×EP 전이 및 NanoFlow의 검색 최적화 MoE baseline은 미완료다.
이 한계를 숨겨 `ALL_THREE_NO_GO`나 `FOUND_SUCCESSOR_STRONG_GO`로 판정하지 않는다.
현재 확인된 범위에서 **승격 가능한 paper-level successor는 없다.**

이 보고서는 **Layered Prefill / FastPP / NanoFlow** 작업물이다.
SERE/Libra/MoDES 보고서가 아니며, 아래의 과거 중간 snapshot은 이력으로만 보존한다.

### 범위·시간·증거

- GPU: 물리 **4/5/6/7**. Native Layered TP2는 4/5만 사용했다. 다른 GPU의
  작업을 연구에 사용하지 않았다. 9월 8일 초기 CPU 의도 import의 GPU0 context
  가능성은 기존 안전 로그에 보존돼 있으며, 이후 모든 실행에 GPU/CPU guard를
  적용했다. 전체 이력에 대해 무조건적인 무위반 주장을 하지 않는다.
- 수집 종료: 9월 9일 약 14:58 KST. 시작부터 약 **22시간 43분** 경과했지만,
  사용자 GPU 반납 구간 약 **14시간 40분**을 포함한다. 이를 연속 GPU 실험으로
  계산하지 않는다.
- 보수적으로 기록된 live experiment interval union: **148.99분**.
  실제 사용 GPU 수별 union: **9.155 GPU-hours**, 4-GPU equivalent **2.289시간**.
  모델 로딩·build/JIT·burn·실패 startup 등은 제외했다. CUDA busy-time은 아니다.
- Request index: **33,788 rows**. 서로 다른 모델/계측/정답 상태의 자료를
  검색하는 index이지, 33,788개의 독립 재시작이나 모두 유효한 성능 표본이 아니다.
- 주요 A/B는 동일 trace/arrival/warmup, 무작위 순서, 3 independent restarts.
  Restart pair를 통계 단위로 쓴다. 단일 빠른 run이나 request 단위 bootstrap으로
  신뢰도를 과장하지 않았다. 강한 후보가 없어 5-run successor/Kimi는 실행하지 않았다.

### 세 후보 요약

아래 headroom은 **측정한 기존 설정의 best-static 대비 추가 request E2E**다.
가능한 모든 scheduler/partition의 이론적 상한이 아니다. Native Layered/FastPP의
긴 자유 생성은 같은 설정에서도 일부 달라져 수치는 descriptive로 제한했다.
출력 불일치 NanoFlow pair, Nsight run, teacher-forced run은 성능 승격 근거에서 제외했다.

| 후보 | Native baseline / mechanism | 추가 E2E 및 대조 결과 | 현재 판정 |
|---|---|---|---|
| Layered Prefill | Qwen3-30B-A3B BF16 TP2; graph 21 engines + eager route/layer 계측 3개 | Full-warmup cap4/12/24·chunk2048에서 **0.657%**; held-out 선택 3/3 모두 0%. 기존 chunk512 대비 cap4 bursty **19.67%**는 원 방법 이득 | 측정한 knob 공간은 작은 여지. Native VL/임의 grouping request oracle은 미확립 |
| FastPP | Dense PP4 18 engines; Qwen3 MoE PP4 broad9 + partition6 + full-warmup15 | Broad 기존-policy **0.587–2.351%**; 충분한 warmup/동일 KV의 5-knob 비교 **0%**. Cost-derived partition은 steady/bursty **14.91%/11.98% 악화** | 기존 option이 강한 대조군. 큰 non-trivial successor 여지 미확인 |
| NanoFlow | 공식 H100 Qwen1.5-MoE EP4 FP16; HF sanity·실제 stream overlap 검증; prefill18/decode36/long-decode18 engines | M4096/8192 prefill의 출력 일치 12 pairs에서 finite-plan **0%**. Decode는 충분한 output-exact 계획이 없어 **null / 판정 불충분** | 수동 계획을 공식 검색 optimum 실패로 부를 수 없음. MoE search/native dynamic VL port 제약 |

원 논문 성능과 비교할 때 모델·정밀도·GPU topology가 다르므로, 이 표의 raw
latency를 세 시스템 간 속도 순위로 사용하지 않는다. Native FastPP는 **PP4/TP1,
EP 아님**. Native Layered도 **EP 아님**. NanoFlow는 expert-sharded + NCCL
AllReduce이며 **DeepEP가 아니다**.

### 1. Layered Prefill: 가정과 기존 knob 공격

코드는 항상 고정 group 수를 쓰는 것이 아니라 token volume에 따라 stage 수를
선택한다. 설정 4/12/16/24는 cap/table 선택이므로 이를 잘못 읽은 '고정 group'
실패를 주장하지 않는다. 공식 코드에 없는 cap8도 만들지 않았다.

Full-trace warmup 12-engine control에서 추가 mean-E2E envelope는 0.657%,
9개 all-token SLO의 추가 attained-goodput 최대는 **0.0301%**였다. Tight TBT와
loose TBT의 최선 설정이 다른 것은 원 논문의 기존 trade-off와 겹친다.
Measured-cost group proxy는 일부 phase에서 25–40%가 나와도 native request
counterfactual이 아니므로 E2E gain으로 인정하지 않는다. Expert-weight 자료는
logical-use proxy이지 HBM traffic 실측이 아니다.

### 2. FastPP: 선정된 screening leader의 심층 반증

**BEST_SUCCESSOR_CANDIDATE=FASTPP, FINALIST=NONE.** 가장 충분한 baseline 및
causal control을 얻은 후보라는 의미이며 연구 승격이 아니다.

ALP는 단순 M 기반이 아니다. 실제 startup non-attention profiling과
context/prefill/prefix 상호작용, 온라인 RLS 갱신을 계측했다. 3,950 invocation의
네 PP-rank identity가 join됐고, 3,885개에서 rank0가 가장 느렸다.

1. Layer-cost minimax는 약 23–26% stage-proxy 개선을 제안했다.
2. 실제 equal12 → 8/12/14/14 partition을 동일 KV 용량·full warmup으로 바꾸자
   **모든 3쌍에서 E2E가 악화**됐다. 중앙 paired 감소율은 -14.91%/-11.98%.
3. 추가 15-engine chunk128/512/2048·greedy·ALP 비교에서 PP-only/chunk2048이
   두 workload 모두 가장 빨랐다. 기존 설정 내 추가 E2E envelope는 0%였다.
4. ALP의 paired E2E 감소율은 steady **-21.54%**, bursty **-19.24%**였다.
   이 관측된 손실은 기존 PP-only 선택으로 제거되므로 non-trivial successor가 아니다.

Broad/native 화면의 9-SLO attained-goodput 추가 여지는 최대 12.31%, 충분한
warmup 화면에서는 6.97%였다. 둘 다 **고정 arrival에서 달성한 goodput**이며
maximum sustainable SLO capacity나 새 method의 이득이 아니다.

이 결과는 'local stage 비용이 위치를 바꿔도 그대로이며 곧 request 절감이다'라는
단순 추론을 반증한다. 특정 CPU/attention/communication 원인을 확정하거나 모든
joint partition을 부정하지는 않는다. Exact measured best-chunk 및 joint
chunk/partition/PP×EP request oracle은 아직 미확립이다.

### 3. NanoFlow: overlap, setup, correctness를 분리

공식 smoke는 independent HF의 82-token reference를 통과했다. 별도의
same-prefix/B4/B16/HF 진단에서 mismatch는 작은 top-two margin에 집중됐지만,
이것만으로 전체 benchmark quality 동등성을 주장하지 않는다.

Nsight에서 compute/NCCL overlap은 0 → 약 1.63 ms로 증가했으나 profiled
step은 9.26 → 12.70 ms로 느려졌다. Kernel 수가 증가하고 resident collective
span도 길어졌다. 이는 clean 성능표가 아니며 overlap 비율 자체를 목표로 삼을
수 없다는 진단이다. NCCL residency를 실제 active traffic으로 부르지도 않는다.

M4096/8192 pure-prefill 18-engine 비교는 두 same-shape warmup을 사용했고,
12개 paired first-token 출력이 모두 일치했다. Plain이 두 크기에서 best라
세 수동 계획의 추가 envelope는 0%다. 이것은 paper-searched optimum이 아니다.

16-output decode 36 engines/3,888 requests에서는 27개 whole-cohort pair 중
2개만 token-exact였다. 96-output 18 engines/612 requests에서도 12개 중 1개만
exact다. 따라서 두 decode envelope는 **판정 불충분(null)**이다. Near-tie에
따른 자유 생성 차이를 곧 model-semantic failure라고 부르지 않는다.

첫 decode ITL은 setup/capture를 포함한다. 16-output의 E2E 내 비중은 조건별
중앙값 58–94%, 96-output에서는 30–71%로 줄었다. 필요한 계산도 들어 있으므로
전부 제거 가능한 waste가 아니다. 어떤 wait/setup도 E2E에서 빼지 않았고,
steady ITL을 warm online E2E로 바꿔 부르지 않았다.

### MLLM 전이와 재현 한계

별도 Qwen3-VL BF16 **TP2/DP2/EP4**, DeepEP HT/TritonExperts, DBO/EPLB/prefix
off 경로를 실제 source/runtime에서 확인하고 이미지 요청을 실행했다. 작은
clean/observer 1,248 requests와 큰 real four-chart control 480 requests를
확보했다. 큰 control의 clean visual/text **120 pairs는 actual prompt token 수가
모두 정확히 일치**한다. 다만 내용/route histogram까지 같지는 않다.

Observer overhead와 restart 공통상태 편차가 커서 이것을 세 native 시스템의
MLLM 성능으로 합성하지 않았다. 큰 입력의 zero-entire-TTFT fixed-timeline bound도
encoder/정상 prefill/queue를 포함하며 실제 제거 가능한 scheduler mass가 아니다.

남은 native-port 제약은 다음과 같다.

- Layered/FastPP native Qwen3-VL 처리 및 FastPP PP×EP 실행/정답을 검증하지 못했다.
- NanoFlow H100 Qwen MoE forward/수동 split은 작동하지만 FusedMoE profiling DB/
  update/run hooks가 갖춰지지 않았다. 빈 hook을 zero-cost expert로 입력하지 않았다.
  Dense Llama3-70B 검색을 MoE 검색으로 둔갑시키지 않았다.
- 별도 Meta-Llama 접근 probe의 HTTP401은 model-access 제약이며 method 실패가 아니다.
- Native 동적 VL 계획/정답 및 진짜 joint request oracle은 추가 baseline 포트 작업이다.

따라서 `PARTIAL_ENVIRONMENT_BLOCKED`는 **모든 엔진이 실패했다**는 뜻이 아니라,
요청한 전체 native-transfer/optimized-baseline milestone이 미완료라는 뜻이다.

### Ranking·후속 결론

| 순위 | 후보 | successor evidence 점수 /50 | 판정 |
|---|---|---:|---|
| 1 | FastPP | 14 | Screening leader; paper finalist 아님 |
| 2 | Layered Prefill | 12 | 기존 knob 공간 추가 여지가 작음 |
| 3 | NanoFlow | 8 | 최적 검색 baseline·decode 정답 gate 불충분 |

점수는 원 논문의 우수성을 평가하지 않는다. 누락된 headroom/causal evidence를
0점으로 둔 것은 universal zero-headroom 측정이 아니다. Kimi, 5 paired successor
restarts, 새 prototype은 **NOT_RUN—promotion gate 미통과**다. Winner의 causal
control, oracle, method/intro gate 실패 이유는 `BEST_CANDIDATE/`에 분리했다.

최신 primary-source prior-art 공격에서도 plain adaptive grouping, context-aware
ALP, dynamic nano plan, 단순 shared-expert overlap은 이미 가까운 해법이 있다.
새로운 orthogonal MoE/MLLM failure를 측정하지 못했으므로 novelty를 주장하지 않는다.
상세 비교: [FastPP](https://www.usenix.org/conference/osdi26/presentation/hwang),
[Layered Prefill](https://arxiv.org/abs/2510.08055),
[NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan),
[DynaFlow](https://arxiv.org/abs/2605.21603),
[VPP](https://arxiv.org/html/2608.26523v1),
[Bullet](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf).

**권고:** 현재 자료로 successor method를 시작하지 않는다. 계속한다면 새 아이디어
구현보다 native VL/검색 baseline과 작은 direct joint request oracle을 먼저
완성해야 한다. 이는 미완료 연구 범위이지, burn 중 백그라운드에서 계속 수행되는
실험이 아니다. 모든 연구 worker 종료 후 사용자 요청의 burn만 4–7에서 별도 실행한다.

### Artifacts

- Branch: `flashvep/scheduling-overlap-successor-mining`
- Working files: `poc_flashvep/scheduling_overlap_successor_mining/`
- Results: `poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/`
- Reproduction/index: `COMMON/REPRODUCE.md`, `COMMON/ARTIFACT_INDEX.md`, `COMMON/EVIDENCE_BUNDLE.json`
- Milestones/score/request/time: `BASELINE_MILESTONES.csv`, `SUCCESSOR_SCOREBOARD.csv`,
  `REQUEST_E2E_RESULTS.csv`, `GPU_TIME_LOG.csv`
- Large raw tensors, weights, native builds, reference checkouts and Nsight binaries
  remain local; selected small evidence and harnesses are committed. Existing
  unrelated work and earlier traces are preserved.

---

## 아래는 보존된 과거 진행 이력

## 현재 상태 — 2026-09-09 13:55 KST

**Layered Prefill / FastPP / NanoFlow 연구를 계속 수행 중이다.** 이전
SERE/Libra/MoDES 작업물이 아니며, 아직 final ranking/STRONG_GO가 아니다.
물리 GPU 4–7에서 실험은 하나씩 실행하고, 측정 중 burn은 돌리지 않는다.

- Layered 추가 12-engine control 완료: cap4/12/24와 chunk2048, full-workload
  warmup, 각 3 restart. Best-static 대비 추가 finite-policy E2E headroom
  **0.657%**, held-out 선택은 3개 block 모두 0%. 전체 partition의 상한은 아니다.
- NanoFlow large-prefill 18-engine control 완료: M4096/8192, plain/split2/split4.
  12개 paired first-token 비교 모두 일치. 두 규모 모두 plain이 median E2E에서
  가장 빨라 추가 finite-plan envelope는 **0%**다. 공식 searched optimum 재현이
  아니라 native splitter/executor를 이용한 bounded portfolio다.
- 큰 Qwen3-VL actual-image 대조 완료: clean/observer 6 engines, 480 requests.
  **120 visual/text pairs가 실제 prompt token 수까지 모두 정확히 일치**한다.
  재시작 공통상태 편차가 커서 modality 효과나 clean stage-cost로 과장하지 않는다.
- 현재 FastPP의 동일 KV 용량/full-trace warmup/기존 chunk128·512·2048·greedy·ALP
  15-engine control 실행 중. 다음은 NanoFlow B4/B64/B256 decode portfolio다.
- 최신 보수적 계측 ledger: **28,040 request rows**, measured interval union
  약 **131.9분 / 8.02 GPU-hours**. 모델 로딩/build/JIT/burn 등은 제외하며,
  이 수치는 CUDA busy-time이나 사용자 GPU 반납 중 wall time이 아니다.

어느 후보도 아직 material하고 non-trivial한 successor gate를 통과하지 않았다.
Native VL/PP×EP port와 NanoFlow MoE auto-search의 미구현 부분은 method failure와
분리한다. 남은 control 완료 뒤 동일 milestone 기준의 평가를 작성한다.

## 보존된 진행 snapshot — 2026-09-09 13:01 KST

연구는 계속 실행 중이며, 아직 최종 ranking이나 STRONG_GO가 아니다.

- FastPP Qwen3의 **full-workload warmup, 동일 KV 용량, 3 paired restarts**에서
  measured-cost 정적 partition(8/12/14/14)은 균등 partition보다 request E2E가
  steady **14.91%**, bursty **11.98%** 악화됐다(각 paired 감소율의 중앙값).
  앞선 23–26% stage-makespan proxy를 E2E gain으로 해석하는 가정을 반증한다.
  Uneven partition의 KV layer count 호환 수정과 CPU 회귀 검사를 별도 보존했다.
- NanoFlow native prefill M=4096/8192 pilot에서 plain과 split2는 첫 출력 4/4가
  일치했으나 E2E 이득이 없었다. M=8192 resource-limited split은 3/4 일치라
  성능 근거에서 제외하고 동일-prefix numerical/HF 대조를 준비했다.
- Qwen3-VL 경량 observer/clean 3쌍은 완료했다. 총 **1,248 measured requests**,
  **14,399 measured logical MoE**가 있으며 unknown request ID는 0이다.
  Observer overhead 중앙값도 family별 **2.5–6.5%**여서 clean E2E와 분리한다.
- 현재 Layered 공식 cap4/12/24 + chunk2048을 full-workload warmup으로 비교
  중이다(12 engines). 공식 TP2라 물리 GPU 4/5만 사용하고, 6/7의 burn은 측정
  간섭을 피하려고 중지한다. 다음 네-GPU 실험은 4–7을 사용한다.
- 추가 real-image 전이 입력은 4개 원본 ChartQA 이미지와 자연 텍스트 대조군이다.
  CPU processor 기준 16 matched pairs의 prompt token 수를 정확히 맞췄다.
  Native VL 포트나 modality-dependent 성능 증거는 아직 아니다.

남은 bounded controls: Layered cap 결과 회수, NanoFlow correctness/plan portfolio,
FastPP 충분한 warmup 및 기존 chunk 옵션, 큰 token-matched VL 관측. 유망 gate를
통과하기 전 Kimi나 successor 구현으로 넘어가지 않는다.

## 보존된 진행 snapshot — 2026-09-09 12:05 KST

**현재 작업은 Layered Prefill / FastPP / NanoFlow이며, SERE/Libra/MoDES가 아니다.**
물리 GPU 4–7에서 순차 실행 중이다. 최종 후보는 아직 선택하지 않았다.

- Layered의 9-restart graph screen에 이어 3개 eager mechanism run도 완료했다.
  Request ID와 두 TP rank의 stack event를 직접 join했다. 계측 run은 별도이며,
  그 긴 CUDA span을 graph-enabled 성능에 대입하지 않는다. 앞선 두 cap의
  제한된 E2E lower envelope는 여전히 best-static 대비 0%다.
- FastPP Qwen3 PP4의 9-restart, **7,200개 요청** 수집 완료. 기존 3 policy의
  best-static 대비 per-workload mean-E2E envelope는 request-mix **0.59%**,
  equal-workload **2.35%**다. 재시작 편차가 크므로 추가 full-workload warmup
  control을 준비했다. 고정 arrival SLO grid의 최대 기존-policy 선택 이득은
  약 12.31%이며, capacity goodput이나 새로운 method gain은 아니다.
- FastPP 새 sparse mechanism trace의 **3,950개 invocation이 4 PP rank에 정확히
  join**됐다. Layer-partition makespan proxy는 약 23–26%지만 request E2E oracle이
  아니다. Rank 0이 3,885개에서 가장 느려, layer cost가 다른 rank로 이동해도
  그대로라는 가정을 별도로 검증해야 한다.
- NanoFlow B4/B16 native logits를 **독립 HF reference**와 같은 prefix에서 비교했다.
  B4는 127/128, B16은 507–508/512 argmax 일치이며, 불일치 모두 reference
  top-two gap ≤0.015625의 near-tie다. 의미가 깨진 split이라고 단정할 증거는
  없지만 benchmark quality 동등성이 증명된 것도 아니다.
- NanoFlow Nsight에서 split2의 NCCL/compute overlap을 실제 확인했다. 그러나
  profiled median step은 unsplit 9.26 ms → split2 12.70 ms로 느려졌다.
  Kernel 수와 resident communication span이 함께 증가했다. **Overlap 비율은
  목적함수가 아니다.** 이 두 profiled run은 clean 성능 수치에서 제외한다.
- 지금은 실제 이미지 Qwen3-VL TP2/DP2/DeepEP4에서 **경량 계측 / 무계측 ×
  3개 독립 restart**를 진행 중이다. 세 base system의 native VL port가 아니라
  공통 transfer diagnostic임을 유지한다.

다음은 마지막 짧은 검증들이다: NanoFlow large-prefill/plan control,
FastPP full-workload warmup 및 안전한 기존 static-partition control,
Layered 추가 공식 cap control. Native-port 제약과 method failure를 분리한다.

## 보존된 진행 snapshot — 2026-09-09 11:00 KST

**GPU 4–7에서 실제 실험 재개, 아직 최종 판정/승자 없음.** 10:27에 점유
문제가 해소됐다. 다른 GPU는 사용하지 않았으며 측정 중 burn은 실행하지 않는다.
공식 Layered 경로는 TP2이므로 실제 계산 장치는 4/5이며, 이를 EP4로 부르지 않는다.

- **Layered Prefill:** BF16 Qwen3-30B-A3B, eager 양쪽 smoke와 graph-enabled
  chunk512/group-cap4/group-cap16 × 3 independent restart를 완료했다. 총 864 measured
  requests. Group4의 chunk512 대비 평균 request E2E 감소율은 restart 중앙값으로
  bursty **19.67%**, steady **3.47%**다. 이는 원 방법/기존 knob 효과이지 successor
  headroom이 아니다. 두 layered group 후보의 best-static 대비 per-regime E2E
  envelope는 **0%**이며, leave-one-restart-out 선택도 동일하다. 미실험 group/
  partition/MLLM 전체를 제한하는 상한은 아니다.
- 여기서 4/16은 **고정 매-request group 수가 아니라 공식 stage 상한/테이블 선택**이다.
  Native `Scheduler.get_num_stages`는 token volume에 따라 실제 1/2/4/8/16개를
  선택한다. 이를 무시하고 "Layered가 항상 고정 group을 사용한다"고 주장하지 않는다.
- Group4는 bursty TTFT를 줄이지만 최대 token 간 지연이 약 50–54 ms로,
  chunk512의 약 28–32 ms보다 커진다. TBT 50 ms와 100 ms에서 결론이 달라지므로
  평균 TPOT만으로 SLO 개선을 주장하지 않는다. Group16의 한 restart는 20% 이상
  느렸고 그대로 포함했다. 단일 유리한 run을 택하지 않았다.
- **NanoFlow:** B4 동일-prefix/동일-history의 4개 새 engine에서 128개 token
  위치 argmax가 모두 일치했다. 같은-plan도 logits는 bitwise 동일하지 않으며,
  cross-plan mean KL은 약 0.00009–0.00011이다. B16 추가 검증 중이다. 이 검사는
  teacher-forced numerical diagnostic으로, 성능/benchmark accuracy 근거가 아니다.
- **FastPP:** 기존 dense 18회 재시작 및 native Qwen3-MoE 계측 자료에 이어,
  native Qwen3 PP-only/greedy/ALP 3-restart 비교를 다음 순서로 수행한다.

모든 짧은 native 정답 sanity는 통과했다. 긴 자유 생성은 동일 설정 재시작에서도
일부 달라져 별도 진단 중이며, correctness가 확정되지 않은 성능 차이는 descriptive로
제한한다. 세 방법의 동일 screening milestone 전에는 ranking하지 않는다.

새 결과: `layered_runs/graph_screen_20260909_v1/`,
`nanoflow_runs/numerical_resume_20260909_b4_retry1/`,
`nanoflow_runs/numerical_resume_20260909_b16/`.

## 과거 기록 — 2026-09-09 10:27 이전 일시 점유 차단 (해결됨)

GPU 4–7 재사용 승인을 받았으나, 재개 초기 확인 직후 `kaist3` 계정의 작업이
같은 GPU를 점유했다. 허용된 장치 범위의 PID만 종료를 시도했지만 OS 권한
오류로 실패했고 비대화형 sudo도 사용할 수 없다. NanoFlow는 점유 검사에서
중단돼 새 GPU 측정은 아직 없다. GPU 소유자/관리자의 반납 조치가 필요하다.
FlashAttention 설치는 완료됐으며 남은 Layered native 빌드는 GPU를 숨긴 CPU
작업으로 시작했다. 상세 상태: `COMMON/RESUME_STATUS_20260909.md`.
아래는 9월 8일 GPU 반납 시점의 연구 결과이며 최종 판정이 아니다.

**상태: 사용자 요청으로 GPU 중단, CPU 분석 진행. 최종 연구 판정 아님.**

2026-09-08 16:15 KST 시작. 마지막 GPU 비교를 정상 완료하고 약 19:47 KST에
물리 GPU 4/5/6/7의 작업을 모두 정리했다. 반환 확인 당시 네 장 모두 compute
process 없음, utilization 0%, 메모리 약 4 MiB였다. Burn도 실행하지 않는다.
새 GPU 사용은 사용자의 재개 요청 이후에만 한다.

현재 명시적 interval로 기록·중복 제거한 live experiment wall-time은 약
**54.4분 / 3.63 GPU-hours**다(네 GPU 기준). 모델 로딩, 빌드/JIT, burn, 일부
초기 미계측 pilot은 제외한 보수적 기록이며 CUDA busy-time 자체는 아니다.

## 9월 8일 반납 시점의 결론 (아래는 보존된 과거 snapshot)

**아직 paper-level successor는 확인되지 않았다.** FastPP에서 workload별 차이는
재현됐지만 기존 greedy 옵션이 강한 대조군이다. NanoFlow는 다양한 입력의 수치
검증이 남았고, Layered Prefill은 native GPU 재현 전이다. 세 방법을 동일 단계까지
검증하기 전에는 ranking/winner를 정하지 않는다.

## 후보별 진행

| 후보 | 확보한 증거 | 현재 해석 / 남은 일 |
|---|---|---|
| Layered Prefill | 논문·코드 감사, 공식 Qwen3 TP2 실행/계측 harness, native dependency CPU 빌드 | GPU baseline 미실행. 단순히 고정 group count인 시스템이 아니라 길이별 group 수 조절이 이미 존재한다. Native correctness와 chunk/group/partition 비교 필요 |
| FastPP | Dense PP4 18회 독립 재시작, 14,400 measured requests. Qwen3-MoE PP4에서 96개 measured requests와 실제 ALP/stage 계측 | 가장 많은 자료가 있지만 아직 MLLM 특이 실패 아님. Native Qwen3 경로는 PP4/TP1이며 EP4가 아님 |
| NanoFlow | 공식 H100 Qwen1.5-MoE EP4, 5개 eager/graph/split smoke 조건에서 각각 HF 기준 82 tokens 일치. 실제 텍스트 cohort 4회 추가 | 긴 자유 생성은 일부 달라 성능 근거에서 보류. 수동 FFN split 계획은 공식 auto-searched optimum이 아니므로 그 차이를 논문 baseline 실패로 부르지 않음 |

각 원 논문과 실제 코드 pin은 후보별 `PAPER_AUDIT.md`, `CODE_AUDIT.md`에 기록했다.
Sarathi는 reference이며 이전 SERE/Libra/MoDES 또는 폐기된 EP 주제를 재개하지 않았다.

## FastPP: 차이는 있지만, 기존 옵션을 이겨야 한다

표는 3개 paired restart block의 **평균 request E2E 감소율 중앙값**이다.
양수는 ALP 개선, 음수는 악화. 긴 출력의 correctness 검증이 끝나지 않아
현재 수치는 descriptive evidence이며 successor 성능 주장에는 사용하지 않는다.

| Workload | 공식 P2P-disabled 조건 | P2P-enabled 대조군 |
|---|---:|---:|
| Heterogeneous steady | +24.25% | +20.63% |
| Heterogeneous bursty | −10.52% | −14.11% |
| Short/high-concurrency | −5.82% | −15.23% |
| Official Azure-arrival control | +7.51% | +11.77% |

- 일부 restart가 크게 느려 single-run/단순 평균은 위험하다. 동일 정책의 긴
  출력도 restart마다 달라, 출력 hash 차이만으로 정책의 numerical failure를
  선언하지 않았다. 짧은 HF 정답 검사는 통과했다.
- Bursty joint SLO(TTFT≤2 s, TPOT≤0.2 s) 충족률 중앙값은 PP-only 87.5%,
  ALP 68.8%였다. P2P-enabled에서는 87.5% vs 66.7%. 이 표본에서 ALP의
  E2E 손실을 SLO 개선으로 설명할 근거도 없다. 이는 최대 SLO capacity 측정은 아니다.
- 그러나 기존 **greedy** 옵션이 강하다. 네 기존 정책의 workload별 restart
  중앙값으로 계산한 best-static(greedy) 대비 per-regime 선택 이득은
  **0.25%**(관측 request mix), **0.64%**(동일 workload 가중치)에 불과하다.
  이는 in-sample·zero switching cost의 제한된 envelope이며, 미실험 partition/
  MoE/MLLM successor 전체의 상한은 아니다. 단순 기존 정책 선택은 유망하지 않다.
- Qwen3-MoE 계측에서 실제 ALP pre-update 예측의 median absolute percentage
  error는 EXTEND **3.06%**(27 matched samples), MIXED **2.69%**(86)였다.
  현재 자료만으로 “MoE에서 ALP predictor가 붕괴한다”고 말할 수 없다.

## NanoFlow: correctness와 초기 비용을 분리

- 2-way split 경로의 sigmoid 복제 오류는 실제 copy method 테스트로 재현,
  한 인자 전달로 수정, 새 GPU worker에서 HF smoke 재검증했다. 연구 finding이
  아니라 trivial compatibility repair다.
- 다양한 FineWeb 텍스트의 32-token 생성은 unsplit/split2 사이 각각 14/16,
  3/4 requests만 완전히 일치했다. 같은 prefix의 logits와 같은-plan restart
  대조 검사는 준비했지만 **GPU 반환으로 아직 실행하지 않았다**.
- 첫 graph capture는 unsplit 약 0.51 s, split2 약 1.93–2.07 s였다. Steady
  decode host step도 B4에서 6.31→8.37 ms, B16에서 8.13→10.87 ms였다.
  단일 수동 계획·단일 restart 결과이며, 이를 NanoFlow 최적 계획의 regret나
  successor headroom으로 계산하지 않는다.

## 공통 Qwen3-VL/DeepEP 전이 자료

Native 세 시스템의 MLLM 포트를 주장하지 않는 **별도 vLLM 진단**이다.

- 실제 경로: BF16, TP2/DP2/EP4, `DeepEPHTPrepareAndFinalize`, `TritonExperts`,
  DBO off. Runtime source/class와 rank 정보를 저장했다.
- Image/text composition 및 DP별 batch 1/4/8, output 32 tokens를 사용했다.
  상세 계측/clean 각각 **208 measured requests**, 총 416개. 추가 smoke 8개.
- 42,672개 logical MoE 기록 모두 4-rank join 가능, unknown request IDs 0.
  Warmup 등을 제외한 measured logical MoE는 **38,160개**다. GPU rank rows를
  request latency에 중복 합산하지 않았다.
- 단일 instrumented/clean restart 대조의 E2E 차이는 **16.37–19.58%**였다.
  이를 최적화 가능한 시스템 waste로 해석하지 않는다. 상세 trace는 진단용,
  request E2E는 clean 자료를 기준으로 하며 추가 overhead replication이 필요하다.
- CPU minimax contiguous partition 계산은 일부 prefill/mixed group-cost 개선을
  보이지만 decode에서는 같은 균등 경계가 최적이었다. **Group makespan 개선은
  request E2E 개선이 아니다.** Native Layered 검증 전에는 후보 승격 금지.
- 현재 fixed-cohort 32-token workload에서 **TTFT 전체를 0으로 만드는 것조차**
  평균 E2E 감소 상한이 text 5.95%, single-image 9.11%, long-text+image 9.77%,
  two-image 11.10%다. 이는 해당 고정 timeline의 prefill-only 상한이며,
  다른 online queue/SLO workload 전체를 배제하는 상한은 아니다.

## CPU로 완료한 정리와 다음 재개

CPU request index, 4-rank/request join 검증, 기존 정책 envelope, ALP 오차 분석,
contiguous partition 진단과 비용 상한을 저장했다. 실제 GPU 시간이 필요한
correctness/causal/oracle/Kimi/prototype은 미실행으로 명시했다.

우선순위: **Layered native baseline → Nano same-prefix correctness → FastPP native
MoE paired runs → 저오버헤드 VL trace → 세 방법 공통 선별**. 그 후에만 winner,
Kimi, 최소 successor prototype으로 진행한다. 현재 winner/STRONG_GO는 없다.

재개 목록: `poc_flashvep/scheduling_overlap_successor_mining/GPU_WORK_DEFERRED.md`.

## 안전·재현 기록

- CPU-only wrapper는 모든 GPU를 숨긴다. GPU launcher wrapper는 pause marker가
  있는 동안 실행을 거부한다. Layered CPU 컴파일은 계속 가능하지만 후속
  CUDA-capable import가 자동 실행되지 않도록 build wrapper를 멈췄다.
- 이전 17:57 CPU 의도 import에서 CUDA가 초기화된 scope incident가 있었다.
  4–7 밖 context 생성 여부를 확정할 수 없어 로그에 남겼고 사용자에게 알렸다.
  따라서 전체 세션의 완벽한 GPU scope 준수를 주장하지 않는다. 반환 시점에는
  4–7에 남은 process가 없으며 다른 사용자의 process를 종료하지 않았다.
- Branch: `flashvep/scheduling-overlap-successor-mining`.
- Results: `poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/`.
- 최종 연구 commit/push 및 최종 판정은 아직 하지 않았다. 이 문서는 중간보고다.
