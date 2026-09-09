# Context Parallelism Composition Successor PoC — Working Contract

이번 작업은 Context Parallelism의 세 가지 composition gap을 비교하고, 가장 강한 하나를 paper-level successor 후보로 승격할지 결정하는 focused PoC다.

검증할 세 트랙은 다음이다.

1. DCP × Speculative Decoding
2. PCP ⟂ DCP / zero-transition KV layout
3. MoE-aware CP degree

먼저 아래 spec을 repository에 생성하고 전체 내용을 working contract로 사용하라.

```text
poc_flashvep/reports/cp_composition_successor_poc_spec.md
```

사용자가 제공한 `cp_composition_successor_poc_spec.md`의 전문을 그대로 저장하고 반드시 읽어라.

==================================================
0. GPU SAFETY — ABSOLUTE
==================================================

모든 GPU 실행은 반드시:

CUDA_VISIBLE_DEVICES=4,5,6,7

만 사용한다.

GPU 0,1,2,3은 절대 사용하지 마라.

실험 전 nvidia-smi와 process command를 확인하라.
이전 repository-owned burn이 4–7에서 실행 중이며 PID/command/owner가 명확히 확인되는 경우에만 정상 종료하라.
정체를 모르는 다른 사용자의 process는 절대 종료하지 마라.
non-owned process가 4–7을 점유하면 실험을 시작하지 말고 GPU_CONFLICT_BLOCKED로 보고하라.

실험 중 burn 금지.
작업 종료 후 burn 자동 재시작 금지.

==================================================
1. THIS IS A CURRENT-UPSTREAM STUDY
==================================================

PCP/DCP/speculative support는 빠르게 변하고 있다.
내 기억이나 오래된 문서를 source of truth로 사용하지 마라.

먼저 current vLLM main과 관련 open/merged PR/issues를 조사하라.
최소:

- context_parallel_deployment.md
- #50391
- #46358
- #53673
- #45425
- #51429
- #53573

을 확인하고 UPSTREAM_STATUS.md에 기록하라.

이미 merge된 기능을 새 method로 제안하지 마라.
RFC를 그대로 구현했다는 이유만으로 novelty를 주장하지 마라.

==================================================
2. FIRST PHASE — SUPPORT MATRIX
==================================================

GPU 구현 전에 현재 서버의 model/checkpoint/backend를 감사하라.

SUPPORT_MATRIX.csv를 만들고 최소 다음을 기록:

- model / MoE / MLLM
- attention type: GQA/MLA/hybrid
- q heads / kv heads
- valid TP / PCP / DCP
- EP-off path
- available speculator
- PCP backend
- DCP backend
- CUDA graph support
- current blockers

Qwen3-VL에서 DCP>1이 current config상 불가능하면 fake DCP를 만들지 마라.
Kimi/MLA가 더 적합하면 실제 checkpoint와 memory/backend feasibility를 확인한 뒤 사용하라.
각 트랙이 다른 faithful model을 써도 되지만 범위를 명확히 표시하라.

==================================================
3. COMMON EVIDENCE RULES
==================================================

Correctness before performance.

필수:

- greedy/token agreement
- fixed-prefix logits
- speculative acceptance consistency
- KV slot/ownership validation
- no future-KV visibility
- no missing valid KV

출력 불일치 run은 speedup evidence에서 제외.

Clean timing과 instrumentation을 분리하라.

Primary evidence:

- TTFT
- TPOT/ITL
- request E2E
- throughput / SLO goodput
- peak KV memory / max concurrency

kernel/layer/overlap proxy만으로 GO 금지.

중요 비교는 최소 3 independent restarts, randomized order, matched trace/warmup으로 수행하라.

==================================================
4. TRACK C FIRST — MOE-AWARE CP DEGREE
==================================================

가장 저렴한 screen부터 수행한다.
EP는 기본 OFF.

질문:

Attention/KV만 보고 고른 PCP/DCP degree가 MoE batching과 expert-kernel cost까지 포함하면 잘못된 선택이 되는가?

4GPU fixed-fleet configuration을 구성하라.
모델이 1GPU에 들어가면 예:

- CP1: DP4 × PCP1 × TP1
- CP2: DP2 × PCP2 × TP1
- CP4: DP1 × PCP4 × TP1

TP2가 필요하면 valid configs만 사용.

8K/16K/32K/64K, concurrency 1/4/16을 우선.

측정:

- attention
- CP comm
- router
- expert
- full layer
- active experts
- tokens/expert
- tiny expert group fraction
- TTFT/TPOT/E2E/goodput

비교:

attention-optimal CP degree
vs
full-request-optimal CP degree

그리고:

best static
vs
length-only
vs
MoE-aware per-regime oracle

Best-static 대비 oracle <5%면 Track C NO_GO.
12% 이상 direct request regret가 있고 simple length threshold가 해결하지 못할 때만 승격.

MLLM claim은 matched actual token count에서 Text/Vision/Mixed를 비교하고 modality가 causal할 때만 허용.

==================================================
5. TRACK A — DCP × SPECULATIVE DECODING
==================================================

질문:

DCP-sharded KV에서 multi-token speculative verification을 exact하게 수행하면서 DCP memory/capacity와 speculative latency benefit을 동시에 유지할 수 있는가?

먼저 standalone matrix:

- Vanilla
- DCP-only
- Spec-only
- current combined attempt

을 실행하라.

Historical H와 query j에 대해 rank-local visible KV length가 query마다 다름을 실제 backend에서 검증하라.
Conventional final-length causal metadata가 future KV를 노출하거나 valid KV를 누락하는지 확인하라.

현재 source를 보고 exact strategy를 선택:

1. virtual q_len=1 rows
2. context/current split + LSE merge
3. direct DCP-aware varlen verification

처음부터 세 개를 모두 구현하지 마라.
가장 portable한 exact fallback으로 correctness와 headroom부터 확인하라.

비교 기준:

best(DCP-only, Spec-only)
vs
exact combined path

Strong 후보는:

- exact output/acceptance
- DCP-only 대비 TPOT/E2E >=12%
- Spec-only 대비 KV memory 또는 max concurrency >=25%
- throughput regression <=2%

을 선호.

Exact portable path가 <5%, direct upper bound도 <10%면 NO_GO.

==================================================
6. TRACK B — PCP ⟂ DCP / ZERO-TRANSITION
==================================================

질문:

PCP prefill이 만든 KV를 처음부터 final DCP ownership layout에 기록하여 prefill→decode transition의 복제/redistribution/remap을 제거할 수 있는가?

현재 valid topology만 사용.
Assertion bypass 금지.

먼저:

- PCP-only
- DCP-only
- PCP+DCP current path

를 비교.

다음 경계를 직접 계측:

last prefill completion
→ KV write/copy/redistribution
→ block-table or slot-map update
→ first decode attention
→ first token ready

기록:

- transition bytes
- KV duplication
- collectives
- metadata/remap time
- transition gap
- first ITL
- TTFT/E2E

Oracle:

0. transition copy/metadata zero
1. prefill-time direct final-owner placement
2. direct placement overlapped with remaining prefill

단순히 wait/copy를 앞당긴 것을 saved work로 세지 마라.

Oracle이 유망할 때만 canonical global-position ownership 또는 batched final-owner write를 최소 구현.

Transition direct request share <5%면 NO_GO.
Latency 10–12% 이상 또는 KV duplication/memory 25% 이상 감소로 material capacity gain이 있을 때만 승격.

PCP ⟂ DCP 자체는 active upstream design과 인접하다. zero-transition layout이나 일반 metadata abstraction 등 추가 contribution이 없으면 novelty 없음.

==================================================
7. ALL THREE MUST BE SCREENED
==================================================

한 트랙에서 early positive가 나와도 다른 두 트랙을 audit/cheapest exact diagnostic/headroom 단계까지는 완료하라.

그 후 TRACK_SCOREBOARD.csv로 순위를 정하고 최대 하나만 deep dive하라.

평가:

- support
- correctness
- direct headroom
- capacity value
- causality
- generality
- novelty
- non-triviality
- method cleanliness
- implementation feasibility

==================================================
8. DO NOT REPEAT OLD FAILURE MODES
==================================================

다음을 하지 마라.

- layer/kernel proxy를 request gain으로 곱하기
- observer overhead를 removable waste로 부르기
- weak default와 비교하기
- output이 다른 run으로 speedup 주장하기
- environment failure를 method failure로 부르기
- unsupported DCP/PCP를 fake collective로 흉내 내고 production claim하기
- GPU burn으로 시간을 채우기

==================================================
9. TIME POLICY
==================================================

예상:

- support audit 1–2h
- Track C 2–4h
- Track A 3–7h
- Track B 3–8h
- winner deep dive 4–8h

초기 decisive screen은 8–14h, full deep dive는 12–24h 정도 가능.

하지만 hard gate가 실패하면 즉시 중단하라.
시간을 채우는 것이 목표가 아니다.

==================================================
10. OUTPUTS
==================================================

Branch:

flashvep/cp-composition-successor-poc

Report:

poc_flashvep/reports/cp_composition_successor_poc.md

Workspace:

poc_flashvep/cp_composition_successor/

Results:

poc_flashvep/deepep_revalidation/results/cp_composition_successor_poc_<timestamp>/

필수:

UPSTREAM_STATUS.md
SUPPORT_MATRIX.csv
TOPOLOGY_MAP.md
TRACK_A_DCP_SPEC.md
TRACK_B_PCP_DCP_ZERO_TRANSITION.md
TRACK_C_MOE_AWARE_CP_DEGREE.md
TRACK_SCOREBOARD.csv
GPU_TIME_LOG.csv
EXPERIMENT_LOG.md
PRIOR_ART_MATRIX.md
FINAL_DECISION.md

완료 후 commit/push.

==================================================
11. FINAL RESPONSE
==================================================

반드시 세 트랙 각각에 대해:

- support/topology
- correctness
- clean request metrics
- capacity/memory
- oracle
- prior-art collision
- exact kill/pass gate

를 보고.

최종:

FOUND_STRONG_GO
FOUND_PROMISING_HOLD
ALL_THREE_NO_GO
PARTIAL_ENVIRONMENT_BLOCKED

중 하나를 정직하게 판정하라.

마찬가지로 사용중인 gpu 4,5,6,7은 너가 직접 종료하고 사용해도 괜찮아.
그리고 너가 실험중에 gpu 4,5,6,7을 사용중이지 않을 때는 GPU utilization을 올리기 위해 버닝코드를 돌려야해.
/home/esjung/vllm-ep 위치의 run_utilize.sh을 bash로 돌리면 자동으로 4,5,6,7번 gpu에 버닝이 돌아가니 꼭 gpu가 비어있다면 버닝을 돌려주고, 다시 너가 사용해야한다면 직접 멈추고 gpu를 사용하는 구조로 해줘. 그래야 utilization을 최대로할 수 있어.
마지막으로 결과를 도출해서 나에게 마지막 내용을 전달하고 나서도 run_utilize.sh를 돌려서 gpu 4,5,6,7번이 실행되도록 해줘.
