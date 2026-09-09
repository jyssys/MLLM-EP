# Top-Tier Scheduling/Overlap Successor Mining
## Layered Prefill × FastPP × NanoFlow

## 0. Mission

이번 작업의 목적은 새로운 아이디어를 0부터 발명하는 것이 아니다.

이미 강한 end-to-end 성능을 보고하고 공개 코드가 있는 세 시스템을 대상으로:

1. **Layered Prefill** (`scale-snu/layered-prefill`)
2. **FastPP** (`Sys-KU/FastPP`)
3. **NanoFlow** (`efeslab/Nanoflow`)

다음 순서로 연구한다.

```text
PAPER/CODE AUDIT
→ ORIGINAL BASELINE SANITY
→ MECHANISM REPRODUCTION
→ MoE/MLLM TRANSFER
→ FAILURE-MODE MINING
→ TRIVIAL-FIX ATTACK
→ SUCCESSOR ORACLE
→ ALL-THREE RANKING
→ BEST-CANDIDATE DEEP DIVE
```

최종 목표는 다음 형태의 paper-level successor 후보를 찾는 것이다.

> 기존 top-tier 시스템 X는 assumption A에 의존해 큰 이득을 얻는다. 그러나 A는 MoE MLLM의 workload/runtime에서 체계적으로 깨지며, 이 때문에 기존 이득의 상당 부분이 사라진다. 단순 설정 변경으로는 해결되지 않고, 새로운 scheduling/execution principle B가 필요하다.

세 시스템 모두 충분히 검증한 뒤에만 최종 후보를 고른다.

---

## 1. Research boundaries

Primary research domain:

- LLM / MLLM inference serving
- Mixture-of-Experts
- Expert Parallelism or MoE-distributed execution
- chunked/layered prefill
- pipeline scheduling
- compute–memory–network overlap
- continuous batching

다음은 허용된다.

- Base paper의 limitation을 재현하고 강화하는 successor
- Dense/LLM용 base가 MLLM MoE에서 깨지는 지점을 찾는 연구
- Existing scheduler와 EP runtime 사이의 structural mismatch
- 정확도를 바꾸지 않는 runtime/scheduling method
- Qwen에서 발견 후 Kimi로 generality 검증

다음은 paper-level contribution으로 인정하지 않는다.

- 단일 파라미터를 다른 값으로 바꾸는 것
- 기존 README 권장 설정 사용
- 환경/포팅 버그 수정
- warmup 또는 버퍼 크기 변경만으로 해결
- 특정 한 run에서만 나타나는 speedup
- kernel 또는 wave metric만 좋아지고 request E2E가 안 좋아지는 결과

---

## 2. Hardware and safety

**2026-09-09 user override: GPU experiments are explicitly resumed on physical
4,5,6,7 only.** The pause below is historical. Idle/final burn is reauthorized
on the same subset, stopped before measurements and excluded from research time.
The local burn script defaults to 1–4: always override `VLLM_UTILIZE_GPUS=4,5,6,7`.

**2026-09-08 19:45 KST user override: GPU work is paused for hardware return.**
The last running comparison completed and owned workers were released at ~19:47.
Only CPU work is authorized until the user explicitly resumes GPU experiments.
Idle/final burn is also stopped during this pause. The mapping below applies
only after that later authorization, not as permission to launch now.

모든 GPU 작업은 물리 GPU **4,5,6,7**만 사용한다.

2026-09-08 사용자 정정: 프롬프트의 1–4 지정은 오기이며 4–7로 대체한다.
후속 사용자 승인에 따라 유휴 시간과 작업 종료 후 burn도 4–7에서만 허용한다.
이 정정은 원래의 GPU 번호와 burn 금지 문구보다 우선한다.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

규칙:

- GPU 0,1,2,3 사용 금지
- 다른 사용자의 프로세스 종료 금지
- GPU burn은 유휴·종료 후 4–7에서만 허용. 실험 전에 자신이 실행한 burn을 중지하고 측정 중에는 실행하지 않는다. Burn 시간은 연구 GPU 시간에 합산하지 않는다.
- 실험 종료 후 자신이 띄운 프로세스만 정리
- 각 코드베이스는 별도 conda/env 또는 container 사용
- 기존 working vLLM 환경을 irreversible upgrade하지 않음
- 각 repo/patch/model/runtime commit을 manifest에 기록

공식 baseline이 2GPU를 요구하면 허용된 GPU 4–7 중 2개만 사용할 수 있다. 단, 다른 물리 GPU를 사용해서는 안 된다.

---

## 3. Models and generality

### Baseline reproduction models

각 repo가 공식적으로 지원하는 모델을 우선 사용한다.

- Layered Prefill: official Qwen3-30B-A3B path 우선
- FastPP: official Qwen2.5 dense PP path 우선
- NanoFlow: official supported dense/Mixtral path 우선

### Research target models

Primary discovery:

- `Qwen3-VL-30B-A3B-Instruct`
- 필요한 generic control: `Qwen3-30B-A3B-Instruct`

Cross-model validation:

- `Kimi-VL-A3B-Instruct` 또는 프로젝트에서 사용하는 Kimi MoE MLLM

Kimi는 Qwen에서 다음 조건을 모두 통과한 finalist에만 사용한다.

- material failure
- direct E2E headroom
- non-triviality
- credible novelty gap

---

## 4. Expected time and bounded setup

예상 wall time:

- 캐시/환경이 잘 준비된 경우: **18–28시간**
- NanoFlow build/port 이슈가 큰 경우: **24–40시간**

권장 분배:

- 공통 paper/code audit: 2–4h
- Layered Prefill: 4–7h
- FastPP: 5–9h
- NanoFlow: 6–12h
- winner deep dive: 4–8h

한 repo의 dependency/build 문제에 무한히 매몰되지 않는다.

- setup/debug 약 2시간 초과 시 alternate path 검토
- 3시간 초과 시 `ENVIRONMENT_BLOCKED`와 정확한 원인을 기록하고 trace/oracle 기반 faithful diagnostic으로 전환
- environment failure를 research failure로 부르지 않음

---

## 5. Evidence classes

다음을 명확히 구분한다.

1. `PAPER_CLAIM`
2. `OFFICIAL_BASELINE_REPRODUCTION`
3. `PORT_FUNCTIONAL_SANITY`
4. `TRACE/PLANNER_DIAGNOSTIC`
5. `REQUEST_LEVEL_E2E`
6. `COUNTERFACTUAL_ORACLE`
7. `SUCCESSOR_PROTOTYPE`

한 evidence class를 다른 class로 과장하지 않는다.

예:

- planner score 개선 ≠ request speedup
- overlap timeline 존재 ≠ E2E overlap benefit
- stage bubble 감소 ≠ request critical path 감소
- Qwen text path correctness ≠ Qwen3-VL full path correctness

---

## 6. Common first phase

### 6.1 Project negative map

기존 프로젝트의 NO-GO 결과를 읽고 다음 파일을 만든다.

`PREVIOUS_NEGATIVE_SPACE.md`

최소 포함:

- route/fanout/fragmentation
- modality-aware TP↔EP
- naive encoder/DeepEP overlap
- fixed-shape tail
- speculative partial expert completion
- SERE/Libra/MoDES successor mining

새 successor가 사실상 기존 실패 방향을 이름만 바꾼 것인지 확인한다.

### 6.2 Repository audit

세 repo 모두에서 다음을 기록한다.

- pinned commit
- original model/hardware/runtime
- main performance source
- scheduler/execution mechanism
- configurable knobs
- fast/slow paths
- official limitations
- hard-coded assumptions
- porting boundary
- correctness contract

각 repo별로:

```text
PAPER_AUDIT.md
CODE_AUDIT.md
BASELINE_REPRODUCTION.md
```

을 만든다.

### 6.3 Baseline correctness

Scheduling/runtime method는 원칙적으로 model semantics를 바꾸면 안 된다.

필수 sanity:

- deterministic greedy first-token agreement
- 대표 request의 full-output agreement
- request count / output length / EOS consistency
- invalid run 분리

Output mismatch가 있으면 performance 결과로 사용하지 않는다.

---

## 7. Common workload suite

모든 시스템에 완전히 같은 workload를 강제하지 말고, base mechanism을 공격하는 workload를 사용한다. 단 비교 가능한 공통 suite도 유지한다.

### Text workloads

- long-prefill / short-decode
- ShareGPT-like mixed lengths
- arXiv-like long context
- low/medium/saturation request rate

### MLLM workloads

- text-heavy control
- single-image
- high-resolution image
- multi-image
- ChartQA/GQA real-image prompts
- mixed text-only + image requests
- long/short output
- low/medium/high concurrency

Paired comparison은 동일 request trace와 arrival schedule을 사용한다.

---

## 8. Primary metrics

Primary:

- TTFT p50/p90/p99
- TPOT / ITL p50/p90/p99
- request E2E p50/p90/p99
- throughput
- goodput under the same SLO

Secondary diagnostics:

- stage duration
- pipeline bubble ratio
- expert weight traffic
- HBM bytes
- scheduler choice
- overlap ratio
- GPU utilization
- planner prediction error

Primary conclusion은 request-level metric을 기준으로 한다.

---

## 9. Common failure definition

Base method의 MLLM/MoE transfer failure가 material하려면 다음 중 하나 이상 필요하다.

- base method의 original gain 중 **20% 이상 소실**
- request E2E가 matched baseline 대비 **10% 이상 악화**
- SLO goodput가 **10% 이상 감소**
- one-size-fits-all setting 대비 per-regime oracle가 **10% 이상 개선**
- 동일 correctness에서 successor oracle가 base method 대비 **10% 이상 추가 E2E headroom**

한 run만으로 판정하지 않는다.

- 최소 3 independent engine restarts
- 강한 결과는 5 paired restarts
- run order randomized
- warmup 정책 동일
- confidence interval 또는 bootstrap 보고

---

# PART A — Layered Prefill

## 10. Faithful baseline

Official repo의 Qwen3-30B-A3B path를 우선 재현한다.

최소 비교:

- repo chunked-prefill mode
- layered-prefill mode
- 동일 model / prompt trace / GPU subset / memory limit

방향성 검증:

- TTFT
- TPOT/ITL
- E2E
- expert-weight reload/traffic proxy
- decode stall/bubble

원 논문 숫자의 exact replication은 요구하지 않지만, mechanism 방향은 맞아야 한다.

## 11. Core hidden assumption to test

Layered Prefill은 contiguous layer groups와 정해진 stage count를 사용한다.

핵심 검증 질문:

> 한 번 정한 layer grouping/stage count가 text, Vision-heavy, multi-image, mixed load와 MoE routing 변화에서도 계속 near-optimal한가?

단 이 질문을 정답으로 미리 가정하지 않는다. 코드와 측정이 다른 limitation을 보여주면 그쪽을 따른다.

## 12. Required internal measurements

가능한 범위에서:

- layer/group execution time
- group별 MoE expert time
- group별 attention time
- expert weight load bytes/reload count
- prefill movement across layer groups
- decode work per iteration
- bubble/idle time
- request phase/modality composition

## 13. Failure probes

최소:

1. stage count / group count sweep
2. fixed contiguous partition vs measured-cost balanced partition oracle
3. text-only vs vision-heavy vs multi-image
4. low vs high concurrency
5. pure prefill vs mixed prefill+decode
6. Qwen text MoE vs Qwen3-VL

## 14. Layered Prefill trivial-fix attack

먼저 다음 simple fixes를 시험한다.

- group count sweep
- static per-workload group count
- measured layer-cost balanced contiguous partition
- larger/smaller chunk size
- repo가 제공하는 existing knobs

이러한 단순 설정이 failure의 80% 이상을 회수하면 `INCREMENTAL_ONLY`.

## 15. Layered Prefill successor oracle

계산:

- best static grouping
- per-workload best grouping
- per-request/per-regime oracle grouping
- ideal group duration balancing

중요:

- oracle가 줄이는 것이 실제 request critical path인지 확인
- stage metric 합을 E2E speedup으로 오인하지 않음

Promotion:

- additional request E2E oracle <5%: NO_GO
- 5–10%: weak
- 10–15%: serious
- >=15%: strong

## 16. Potential successor status

강한 후보가 되려면 단순 `vision이면 stage count 8`이 아니라 다음 형태가 필요하다.

> Layer-group scheduling과 MoE/MLLM stage cost를 독립적으로 최적화할 수 없으며, runtime-visible state를 이용한 새로운 grouping/scheduling principle이 필요하다.

---

# PART B — FastPP

## 17. Faithful baseline

Official FastPP artifact의 SGLang PP4 path를 먼저 실행한다.

최소:

- PP only
- dynamic chunk greedy
- dynamic chunk ALP
- batch rebalancing

공식 dense model에서 directionally faithful behavior를 확인한다.

## 18. MoE/EP feasibility audit

다음을 source-level로 확인한다.

- Qwen3 MoE support 여부
- PP와 EP 동시 사용 가능 여부
- 가능한 4GPU topology
- PP2×EP2 또는 다른 faithful topology 가능 여부
- MoE expert placement/communication이 stage model에 포함되는지

지원되지 않는 구조를 억지로 성능 결과로 만들지 않는다.

Native port가 어렵다면 actual Qwen/Qwen3-VL stage traces를 이용한 planner/oracle diagnostic을 먼저 수행한다. 단 trace simulation은 native E2E 결과와 구분한다.

## 19. Core hidden assumption to test

FastPP의 dynamic chunking과 batch rebalancing이 사용하는 load/cost representation이 dense-text serving에는 충분하더라도, MoE MLLM에서는 stage cost를 충분히 표현하지 못할 수 있다.

검증 질문:

> token count와 현재 queue/load를 중심으로 선택한 chunk size와 고정 PP partition이 modality/routing/EP-dependent stage cost에서도 near-optimal한가?

## 20. Required measurements

- per-stage execution duration
- PP bubble ratio
- selected chunk size
- ALP predicted wait/cost vs actual
- stage imbalance
- EP dispatch/expert/combine cost if available
- request queue/running batch composition
- modality/vision token count
- TTFT/TPOT/E2E/goodput

## 21. Failure probes

최소:

1. official dense text baseline
2. Qwen3 text MoE stage-cost transfer
3. Qwen3-VL vision-heavy/mixed stage-cost transfer
4. low/medium/saturation load
5. fixed partition vs measured-cost partition oracle
6. ALP-selected chunk vs exact offline best chunk
7. chunk-only oracle vs joint chunk+partition oracle

## 22. FastPP trivial-fix attack

- static best chunk
- larger ALP training set
- obvious feature addition
- existing rebalancer mode
- one-time PP partition rebalance

단순 feature 1–2개 추가 또는 static partition 변경으로 80% 이상 회수하면 paper core로 약하다.

## 23. FastPP successor oracle

비교:

- current ALP/greedy
- perfect chunk oracle
- perfect stage-cost oracle
- joint chunk + PP partition oracle
- joint PP–EP scheduling oracle if native semantics are valid

Strong candidate:

- base method gain이 MLLM MoE에서 material하게 붕괴
- joint oracle가 current FastPP보다 direct E2E/goodput에서 >=10%
- simple feature/static partition이 대부분 회수하지 못함

## 24. Structural successor requirement

다음 수준이어야 한다.

> Chunk size, PP stage partition, and MoE expert execution are coupled decisions; optimizing them independently creates systematic bubbles or SLO loss.

단순 `ALP에 vision_tokens 추가`는 incremental로 분류한다.

---

# PART C — NanoFlow

## 25. Faithful baseline

Official supported path에서 NanoFlow가 실제로 작동하는지 먼저 검증한다.

가능하면:

- official dense model smoke
- Mixtral/MoE path
- operation schedule generation
- nano-batching
- execution unit scheduling
- online/offline benchmark small reproduction

Gurobi/CUTLASS/MSCCL++ build 등 환경 경계는 정확히 기록한다.

## 26. Core hidden assumption to test

NanoFlow는 model/workload에 맞춰 operation-level pipeline을 탐색한다.

검증 질문:

> 한 번 찾은 execution plan 또는 제한된 plan class가 modality, routing, phase, concurrency가 크게 변하는 MLLM MoE online serving에서 계속 near-optimal한가?

## 27. Required measurements

- chosen pipeline/schedule
- nano-batch sizes
- execution-unit allocation
- compute/memory/network overlap
- critical path
- static-plan regret
- per-regime best plan
- planner/search overhead
- request-level latency/goodput

## 28. Failure probes

최소:

1. official supported model/workload reproduction
2. input/output length regime changes
3. prefill-heavy vs decode-heavy
4. low/high concurrency
5. dense vs MoE if supported
6. text-like vs MLLM-derived operation signatures
7. static plan vs per-regime plan lower envelope
8. mixed-regime trace where workload composition changes over time

Qwen3-VL full port 전에는 current stack의 measured operation traces를 이용해 static-plan regret oracle을 계산할 수 있다. 단 native MLLM result로 부르지 않는다.

## 29. NanoFlow trivial-fix attack

- single static plan retuning
- two-profile manual selection
- rerun offline optimizer per workload
- existing search space enlargement

만약 plan 두 개와 단순 request classification만으로 거의 모두 해결되면 incremental 가능성이 높다. 반대로 state transition, online mixture, co-running interaction 때문에 plan choice 자체가 dynamic coupled optimization이라면 더 강하다.

## 30. NanoFlow successor oracle

계산:

- one global static plan
- best static per workload class
- small plan portfolio oracle
- per-regime/per-interval perfect plan oracle

Promotion:

- static-to-oracle direct E2E/goodput gap <5%: NO_GO
- 5–10%: weak
- 10–15%: serious
- >=15%: strong

## 31. Structural successor requirement

강한 후보는 다음처럼 표현 가능해야 한다.

> A pipeline optimized for one resource mixture becomes systematically suboptimal when modality and MoE routing change the compute–memory–network mixture; the runtime needs a principled plan-adaptation contract rather than one tuned schedule.

단순 schedule table lookup이면 약하다.

---

## 32. All-three equal-screening milestones

최종 ranking 전 세 방법 모두 다음을 완료한다.

| Milestone | Layered Prefill | FastPP | NanoFlow |
|---|---|---|---|
| Paper audit | required | required | required |
| Code audit | required | required | required |
| Official baseline smoke | required | required | required |
| Correctness sanity | required | required | required |
| Mechanism metric reproduced | required | required | required |
| MLLM/MoE transfer probe | required | required | required |
| Trivial-fix attack | required | required | required |
| Successor oracle | required | required | required |
| Prior-art screen | required | required | required |

`ENVIRONMENT_BLOCKED`인 경우 정확한 원인과 대체 diagnostic까지 기록해야 milestone을 조건부 완료로 인정한다.

---

## 33. Candidate scoring

각 base의 successor opportunity를 0–5점 평가한다.

- Baseline fidelity
- Failure severity
- Direct E2E headroom
- Feasible successor headroom
- MoE relevance
- MLLM/modality causality
- Cross-model generality
- Novelty
- Non-triviality
- Method cleanliness
- Code feasibility
- Intro strength
- Evidence

총점 /65.

- >=40: serious
- >=50: finalist
- >=56: strong-go candidate

Hard gates가 점수보다 우선한다.

---

## 34. Prior-art attack

Finalist 전에 반드시 최신 관련 연구를 검색한다.

특히:

- Sarathi-Serve / chunked prefill
- Layered Prefill follow-ups
- PP serving / dynamic chunking
- hybrid PP–EP serving
- NanoFlow-style operation scheduling
- COMET/Flux and fine-grained overlap
- dynamic pipeline portfolios / online plan adaptation
- MLLM serving overlap/scheduling

질문:

> Reviewer가 이 아이디어를 기존 연구의 단순 적용이라고 공격할 때 가장 강한 citation은 무엇인가?

`PRIOR_ART_MATRIX.md`에 problem, assumption, method, metric, exact difference를 기록한다.

---

## 35. Winner selection

세 방법을 모두 screening한 뒤 정확히 하나를 선택한다.

`SUCCESSOR_SCOREBOARD.csv`

선택 기준:

1. Base method가 정상 작동하는가?
2. MLLM/MoE에서 material failure가 재현되는가?
3. failure가 단순 tuning이 아닌가?
4. direct request-level successor oracle가 큰가?
5. Qwen에서 causal evidence가 있는가?
6. Kimi generality를 검증할 가치가 있는가?
7. clean method principle이 최소 2개 이상 가능한가?
8. paper introduction이 자연스러운가?

---

## 36. Best-candidate deep dive

Winner에게 남은 시간의 대부분을 사용한다.

필수:

- 5 independent paired runs
- broader workload/SLO sweep
- exact causal perturbation
- simple-fix adversarial control
- direct request-level oracle
- Kimi phenomenon/oracle validation
- 최소 successor prototype 1개
- correctness and throughput trade-off
- stronger prior-art attack

---

## 37. Strong successor gate

`FOUND_SUCCESSOR_STRONG_GO`는 다음을 모두 만족해야 한다.

1. Original baseline의 mechanism이 directionally reproduced.
2. MoE/MLLM에서 material failure가 재현됨.
3. Qwen과 Kimi에서 동일 causal direction.
4. 단순 config/retuning으로 failure의 80% 이상 회수 불가.
5. Successor perfect oracle >=10% additional request E2E 또는 >=15% SLO goodput.
6. Feasible oracle >=8–10% additional E2E.
7. Minimal prototype이 direct E2E에서 >=5% 개선.
8. 정확도/출력 semantics 보존.
9. Direct prior-art collision 없음.
10. 명확한 paper Introduction과 method principle.

원 방법의 큰 이득을 보존하면서 failure를 제거하는 경우에는 incremental speedup이 작더라도 Pareto 회복을 별도로 평가할 수 있다.

---

## 38. Final statuses

- `FOUND_SUCCESSOR_STRONG_GO`
- `FOUND_INCREMENTAL_ONLY`
- `ALL_THREE_NO_GO`
- `PARTIAL_ENVIRONMENT_BLOCKED`
- `ENVIRONMENT_BLOCKED`

Positive result를 강제하지 않는다.

---

## 39. Research artifacts

Branch:

`flashvep/scheduling-overlap-successor-mining`

Main report:

`poc_flashvep/reports/scheduling_overlap_successor_mining.md`

Directories:

```text
poc_flashvep/scheduling_overlap_successor_mining/
  COMMON/
  LAYERED_PREFILL/
  FASTPP/
  NANOFLOW/
  BEST_CANDIDATE/
```

Common files:

```text
PREVIOUS_NEGATIVE_SPACE.md
BASELINE_MILESTONES.csv
FAILURE_MODE_MATRIX.csv
SUCCESSOR_SCOREBOARD.csv
PRIOR_ART_MATRIX.md
QUALITY_CORRECTNESS.md
REQUEST_E2E_RESULTS.csv
GPU_TIME_LOG.csv
EXPERIMENT_LOG.md
DELIVERY.md
```

Each candidate:

```text
PAPER_AUDIT.md
CODE_AUDIT.md
REPRODUCTION.md
TRANSFER_PROBE.md
FAILURE_ANALYSIS.md
TRIVIAL_FIX_ATTACK.md
SUCCESSOR_ORACLE.md
PRIOR_ART.md
```

Best candidate:

```text
DEEP_DIVE.md
CAUSALITY.md
KIMI_GENERALITY.md
METHOD_OPTIONS.md
MINIMAL_PROTOTYPE.md
PAPER_INTRO.md
```

---

## 40. Checkpoints

약 4시간마다 checkpoint를 작성한다.

```text
CHECKPOINT_H4.md
CHECKPOINT_H8.md
CHECKPOINT_H12.md
CHECKPOINT_H16.md
...
```

포함:

- baseline별 진행 상태
- environment issues
- current strongest failure
- direct E2E headroom
- trivial-fix risk
- next 4h plan
- GPU 4–7 사용 현황 (연구 실험과 burn 별도 기록)

사용자 confirmation을 기다리지 않고, authorized 시간 동안 계속 진행한다.

---

## 41. Final response format

```text
FINAL STATUS:

TOTAL WALL TIME:
LIVE GPU-RESIDENT TIME:
TOTAL GPU-HOURS:
GPU MAPPING:

==================================================
LAYERED PREFILL
==================================================
OFFICIAL BASELINE:
CORRECTNESS:
MECHANISM REPRODUCED:
MLLM/MOE FAILURE:
DIRECT E2E FAILURE SIZE:
TRIVIAL FIX:
SUCCESSOR ORACLE:
PRIOR ART:
SCORE:
STATUS:

==================================================
FASTPP
==================================================
(same fields)

==================================================
NANOFLOW
==================================================
(same fields)

==================================================
RANKING
==================================================
1.
2.
3.

WHY #1 WINS:

==================================================
BEST CANDIDATE DEEP DIVE
==================================================
BASE PAPER:
HIDDEN ASSUMPTION:
MATERIAL FAILURE:
QWEN EVIDENCE:
KIMI EVIDENCE:
CAUSAL EVIDENCE:
DIRECT E2E WASTE:
PERFECT SUCCESSOR ORACLE:
FEASIBLE ORACLE:
TRIVIAL FIX RESULT:
MINIMAL PROTOTYPE:
ACTUAL E2E GAIN:
SLO GOODPUT GAIN:
CORRECTNESS:
CLOSEST PRIOR ART:
EXACT NOVELTY GAP:
METHOD PRINCIPLE:
MINI INTRO:

==================================================
FINAL DECISION
==================================================

NEXT IMPLEMENTATION:
WHAT COULD STILL KILL IT:

Branch:
Commit:
Push:
Report:
Results:
```
