# Autonomous MoE-EP Research Discovery Sprint

## Headroom-First, Tree-Structured, Paper-Level Search

==================================================

0. Mission

==================================================

이번 sprint의 목표는 특정 method를 구현하는 것이 아니다.

목표는:

4×H100 실제 online MoE serving 환경에서

"충분한 E2E headroom,

새롭거나 blue-ocean인 research gap,

systematic method로 발전 가능한 mechanism"

을 가진 최소 하나의 paper-level research direction을

발견하는 것이다.

Target domains:

- MLLM MoE Expert Parallelism

- LLM MoE Expert Parallelism

- online serving

- prefill / decode / mixed serving

- communication / computation overlap

- scheduling

- dependency / critical path

- runtime state

- continuous batching

- resource interference

- topology

- data-centric workload phenomena

- multimodal interaction

MLLM-specific일 필요는 없다.

Generic MoE-EP phenomenon이어도

novel + general + large headroom이면 충분히 가치 있다.

하지만 MLLM-specific direction을 주장한다면

반드시 modality가 causal variable이어야 한다.

==================================================

1. Hardware

==================================================

반드시:

CUDA_VISIBLE_DEVICES=1,2,3,4

4× H100 only.

Primary model:

Qwen3-VL-30B-A3B-Instruct

Primary serving:

real vLLM online serving.

Primary topology:

TP2 / DP2 / EP4

DeepEP where appropriate.

Generic MoE validation:

Qwen3-30B-A3B optional.

다른 모델/backend는

strong candidate validation 시에만 사용.

==================================================

2. Core research principles

==================================================

모든 candidate는 다음 질문을 통과해야 한다.

Q1. NOVELTY

이 현상 또는 causal framing이

정확히 기존 논문에서 이미 다뤄졌는가?

Q2. BLUE-OCEAN

완전히 최초가 아니더라도

아직 여러 method direction으로 확장 가능한가?

Q3. HEADROOM

perfect E2E oracle이 충분히 큰가?

Q4. FEASIBILITY

실제로 제거 가능한 realistic headroom이 큰가?

Q5. GENERALITY

특정 request / layer / model / backend 하나의 artifact가 아닌가?

Q6. SYSTEMATICITY

단순 hard-coded patch가 아니라

일반적인 runtime principle로 설명 가능한가?

Q7. NON-TRIVIALITY

한 줄짜리 synchronization / parameter tuning /

existing config 변경으로 해결되는 문제가 아닌가?

Q8. INTRO-LEVEL STORY

논문 Introduction에서 다음 흐름이 자연스럽게 가능한가?

Problem

→ existing assumption

→ surprising observation

→ overlooked cause

→ material impact

→ method principle

==================================================

3. Strong-GO definition

==================================================

STRONG_GO는 다음을 모두 만족해야 한다.

### Headroom

Direct request-level perfect E2E oracle:

&gt;= 15%

권장:

&gt;=20%.

그리고 plausible / feasible oracle:

&gt;=12%.

### Controlled signal

대표 controlled experiment에서

원인 변수를 바꾸면:

&gt;=5% direct E2E effect

또는

&gt;=10% target-stage effect

가 반복적으로 나타남.

### Generality

최소 두 개 이상의 축에서 재현:

- workload

- concurrency

- phase

- layer range

- model

- backend/runtime

- modality mix

### Novelty

Exact same phenomenon + same causal mechanism +

same solution principle을 다루는 prior work가 없음.

또는 adjacent prior work는 있으나

중요한 unaddressed dimension이 명확함.

### Non-triviality

다음으로 해결되지 않음:

- single config knob

- torch.cuda.synchronize()

- static threshold tweak

- obvious existing runtime option

- one-line buffer increase

- simple environment upgrade

### Method potential

최소 하나의 clean systematic method family가 존재.

가능하면 2개 이상의 method family가 가능하면

더 strong한 research space로 판단.

==================================================

4. FINAL_GO definition

==================================================

이번 discovery sprint는 반드시

최소 하나의 STRONG_GO를 찾으려고 충분히 탐색한다.

하지만 positive result를 fabricate하지 않는다.

종료 상태:

FOUND_STRONG_GO

or

SEARCH_SPACE_EXHAUSTED_NO_GO

or

HARDWARE_BLOCKED.

단순 HOLD는 최종 상태로 사용하지 않는다.

==================================================

5. Immediate kill gates

==================================================

Candidate를 오래 파기 전에

다음 순서로 빠르게 죽인다.

--------------------------------

KILL GATE 1 — E2E Headroom

--------------------------------

Candidate 발견 후

가능하면 30–45분 내 계산.

Perfect direct E2E oracle:

&lt;10%

→ IMMEDIATE DROP.

10–15%

→ WEAK.

특별히 novelty가 강하지 않으면 DROP.

&gt;=15%

→ PROMOTE.

&gt;=20%

→ HIGH PRIORITY.

--------------------------------

KILL GATE 2 — Feasible Oracle

--------------------------------

실제로 제거 가능한 부분만 계산.

&lt;8%

→ DROP.

8–12%

→ WEAK.

&gt;=12%

→ PROMOTE.

&gt;=15%

→ HIGH PRIORITY.

--------------------------------

KILL GATE 3 — Prior art

--------------------------------

Exact mechanism이 최신 work에 이미 존재:

→ DROP 또는 ADJACENT only.

--------------------------------

KILL GATE 4 — Trivial fix

--------------------------------

existing config / simple synchronization /

one-line parameter adjustment로

E2E 이득 대부분 회수:

→ engineering optimization.

Paper core로 DROP.

--------------------------------

KILL GATE 5 — Generality

--------------------------------

한 request / 한 layer / 한 run / 한 backend에만 존재:

→ DROP 또는 diagnostic-only.

==================================================

6. Research tree architecture

==================================================

이번 sprint는 flat checklist가 아니다.

다음 tree를 유지한다:

ROOT

├── Family A

│   ├── Hypothesis A1

│   │   ├── Observation

│   │   ├── Headroom

│   │   ├── Mechanism hypothesis

│   │   └── Follow-up children

│   └── A2 ...

├── Family B

│   └── ...

└── ...

파일:

RESEARCH_[TREE.md](http://TREE.md)

각 node는 다음 state 중 하나:

UNTESTED

CHEAP_PROBE

PROMISING

DEEP_DIVE

PRIOR_ART_COLLISION

HEADROOM_FAIL

CAUSAL_FAIL

STRONG_GO

CLOSED

==================================================

7. Active frontier rule

==================================================

Final 90분 전까지:

ACTIVE_FRONTIER에

항상 최소 8개 candidate node 유지.

한 candidate가 죽으면

다음 중 하나를 수행:

1. sibling hypothesis 생성

2. failed assumption을 뒤집은 child 생성

3. anomaly에서 new family 생성

4. E2E atlas에서 다른 hotspot으로 이동

Seed list를 다 실행했다고

sprint를 종료하지 않는다.

==================================================

8. Breadth → Depth policy

==================================================

### Breadth stage

한 candidate cheap probe:

15–30분.

목적:

"재밌나?"

가 아니라

"15%+ E2E potential이 있을 가능성이 있나?"

를 판단.

### Promotion

상위 2–3개만

DEEP_DIVE로 올린다.

### Deep dive

각 후보:

60–120분.

반드시 확인:

- direct E2E headroom

- causality

- generality

- prior art

- trivial engineering test

- method sketch

### Finalist

최대 2개 candidate.

각 candidate에 대해

paper-level intro test + mini method PoC.

==================================================

9. First stage:

E2E latency mass atlas

==================================================

가설 전에 먼저

실제 serving E2E latency를 decomposition한다.

대표 workloads:

W1. decode-heavy steady

W2. decode-heavy bursty

W3. long prefill

W4. mixed prefill+decode

W5. multimodal image-heavy

W6. text-heavy control

가능하면 request-level join.

Decompose:

- attention

- router/layout

- dispatch

- expert execution

- combine

- synchronization/waits

- scheduler idle/bubbles

- CPU launch / worker delay

- allocator/memory

- communication overlap

- other

각 component에:

RAW_SHARE

POTENTIALLY_REMOVABLE_SHARE

DIRECT_E2E_CRITICAL_SHARE

계산.

목표:

"어디가 이상한가?"

보다 먼저

"어디에 15% 이상의 돈이 있는가?"

를 찾는다.

==================================================

10. Root Family A:

Dependency / Critical-Path Inefficiency

==================================================

질문:

MoE EP runtime이 throughput을 위해

비동기 작업을 만들면서

불필요한 dependency serialization 또는 bubble을 만드는가?

Explore:

A1.

critical communication completion deadline mismatch

A2.

cross-layer dependency bubble

A3.

cross-request dependency contamination

A4.

producer-ready vs consumer-ready timing gap

A5.

barrier보다 finer-grained dependency로

critical path 단축 가능성

A6.

unnecessary serialization despite independent work

중요:

단순 synchronize는 method로 인정하지 않는다.

찾고 싶은 것:

"실제로 parallelizable한 work가

dependency representation 때문에 serial화됨."

==================================================

11. Root Family B:

Overlap Quality / Hidden Slack

==================================================

단순히 compute와 communication이

동시에 실행된다는 것으로 충분한가?

Measure:

ideal overlap

actual overlap

exposed communication

unused overlap slack

질문:

같은 compute/comm 양인데

왜 어떤 invocation은 잘 overlap되고

어떤 것은 critical path에 노출되는가?

Explore:

B1.

phase-dependent overlap efficiency

B2.

layer-dependent overlap slack

B3.

workload-size dependent overlap failure

B4.

mixed prefill/decode overlap collapse

B5.

sender/receiver timing skew that reduces hideable window

Strong phenomenon:

comm volume 동일해도

schedule/dependency 때문에 exposed comm이 크게 다름.

==================================================

12. Root Family C:

Continuous-Batching Composition

==================================================

현재 scheduler는 token budget을 채우지만

MoE EP execution cost를 충분히 고려하는가?

단 기존 request rebatching /

route-complementary rebatching은 CLOSED.

새 질문:

C1.

같은 scheduled token count라도

prefill/decode composition에 따라

EP critical path가 비선형적으로 달라지는가?

C2.

request age / decode step distribution이

overlap opportunities를 바꾸는가?

C3.

continuous batch packing이

communication-compute concurrency를 파괴하는가?

C4.

scheduler가 만드는 request composition이

downstream EP execution topology와 상호작용하는가?

중요:

simple batching reorder만으로 끝나는 방향은 약함.

Systematic scheduling principle이 필요.

==================================================

13. Root Family D:

Cross-Phase Resource Interference

==================================================

MoE serving에는:

attention

expert compute

EP communication

memory movement

scheduler/runtime

이 같은 GPU resource를 공유한다.

Explore:

D1.

Attention ↔ Expert SM contention

D2.

Expert ↔ EP communication progress contention

D3.

Prefill ↔ Decode co-location interference

D4.

memory bandwidth interference

D5.

communication kernel starvation

D6.

workspace/cache pressure-induced regime changes

중요:

이전 naive Vision encoder × DeepEP overlap은 CLOSED.

새 finding은

generic resource-interference law 또는

systematic scheduling opportunity여야 한다.

==================================================

14. Root Family E:

Temporal / History-Dependent EP State

==================================================

현재 workload W_t만으로

latency를 설명할 수 없는 경우 탐색.

Possible state:

- previous collective

- buffer occupancy

- stream backlog

- request history

- previous expert kernel shape

- cache/workspace state

- allocator state

- communication credit

- transport progress

단 fixed-shape DeepEP giant-tail 방향은

E2E headroom 1.09%로 CLOSED.

새 temporal finding은:

frequency가 높고

direct E2E mass &gt;=15%

이어야 한다.

==================================================

15. Root Family F:

Communication Structure Beyond Load

==================================================

이전 per-token fanout direction은 CLOSED.

따라서 fanout을 다시 하지 않는다.

다른 질문:

F1.

message-size distribution

F2.

small-message fragmentation

F3.

sender timing alignment

F4.

collective launch granularity

F5.

peer-pair concurrency

F6.

topology-local vs cross-link contention

F7.

dispatch/combine asymmetry

단:

same expert histogram/rank load에서

새 feature가 independent signal을 가져야 한다.

그리고 E2E headroom &gt;=15%.

==================================================

16. Root Family G:

Expert Execution × Distributed Runtime Interaction

==================================================

DA-MoE / TEMPO와 collision하지 않도록 주의.

단순:

M_e distribution

active experts

tile padding

grouped-GEMM kernel selection

은 prior-art crowded.

새 질문:

G1.

expert kernel duration이 communication overlap window를

비선형적으로 바꾸는가?

G2.

kernel launch ordering과 EP progress가 상호작용하는가?

G3.

expert execution tail보다

expert timing placement가 더 중요한가?

G4.

same expert work but different temporal ordering이

E2E critical path를 바꾸는가?

즉 workload cost 자체가 아니라

distributed execution interaction을 찾는다.

==================================================

17. Root Family H:

Layer-Level Heterogeneity

==================================================

같은 MoE architecture라도

layer별 runtime role이 동일하다는 가정을 검사.

Explore:

H1.

early/mid/late layer에서

communication criticality가 다른가?

H2.

layer별 available overlap window가 다른가?

H3.

attention-to-MoE ratio가 layer별로 달라

EP optimization target이 바뀌는가?

H4.

layer-specific bottleneck regime이

systematic하게 반복되는가?

단 단순 per-layer tuning table은 약함.

Generalizable structural reason이 있어야 한다.

==================================================

18. Root Family I:

MLLM Modality × Routing × EP

==================================================

MLLM 방향은 강한 novelty 후보지만

다음 gate를 통과해야 한다.

Q1.

Dense MLLM에서도 같은가?

YES

→ MoE specificity 약함.

Q2.

Text-only MoE에서도 같은가?

YES

→ MLLM specificity 약함.

Q3.

modality × routing × EP interaction이

현상 발생에 필수인가?

YES

→ strong MLLM candidate.

Explore:

I1.

Vision/Text transition이 EP overlap window를 바꾸는가?

I2.

vision-token spatial/local routing coherence가

communication scheduling opportunity를 만드는가?

I3.

vision의 global expert diversity가

request composition과 interaction하는가?

I4.

multi-image input이

cross-request EP schedule을 비선형적으로 바꾸는가?

I5.

Vision-heavy / Text-heavy requests가 같이 serving될 때

새 interference phenomenon이 생기는가?

중요:

이미 CLOSED:

- modality bottleneck localization

- modality-aware TP/EP crossover

- spatial chunking

- naive vision encoder overlap

- modality-aware granularity.

이들을 재발견하지 않는다.

==================================================

19. Root Family J:

Data-Centric EP Serving

==================================================

Model architecture를 바꾸지 않고

실제 workload distribution 자체에서

systematic phenomenon을 찾는다.

Explore:

J1.

long-tail expert activation distribution이

time scale별로 다른가?

J2.

request-level routing distribution의

temporal clustering이 runtime cost를 만든가?

J3.

production-like workload mixture가

synthetic benchmark와 다른 bottleneck을 만드는가?

J4.

rare request class가

disproportionate global E2E mass를 유발하는가?

J5.

dataset/modality/content class가

EP execution regime을 predict하는가?

중요:

단 prediction으로 끝나지 않는다.

Predictable state가

systematic runtime method로 이어져야 한다.

==================================================

20. Root Family K:

Scheduler Bubbles / Underutilization

==================================================

직접적으로:

"GPU가 왜 놀고 있는가?"

를 찾는다.

Measure:

- idle SM windows

- idle communication windows

- scheduler gaps

- synchronization bubbles

- load imbalance bubbles

- pipeline bubbles

각 bubble에 대해:

duration

frequency

E2E mass

cause

계산.

Strong candidate:

frequent + systematic + removable bubbles

&gt;=15% direct E2E.

==================================================

21. Root Family L:

Unexpected Regimes / Residual Mining

==================================================

각 serving observation에 대해

best available latency model 구축:

features:

M

phase

layer

expert histogram summary

rank load

concurrency

request composition

backend

modality

etc.

하지만 R² 자체가 목표가 아니다.

Large residual clusters 중:

high-frequency

high-E2E-mass

인 것만 조사.

질문:

"현재 known variables가 같은데

왜 latency가 systematic하게 다르지?"

단 이전 fixed-shape rare tail처럼

E2E mass가 작은 residual은 즉시 버린다.

==================================================

22. Closed-direction list

==================================================

다음을 그대로 다시 제안하지 않는다.

- request route-complementary rebatching

- spatial route-aware chunking

- generic route-aware chunking

- critical-rank expert coalescing

- plain fanout-aware scheduling

- modality-aware TP/EP crossover

- generic dynamic TP↔EP switching

- naive vision-encoder × DeepEP overlap

- modality-aware execution granularity

- plain active-expert fragmentation law

- simple communication-SM adaptation

- simple selective synchronize/wait

- fixed-shape DeepEP giant-tail method

- plain histogram-aware kernel selection

- TEMPO-style token/expert makespan balancing

이전 CLOSED 방향의 일부 변수를 사용할 수는 있으나,

새 causal mechanism과 E2E headroom이 있어야 한다.

==================================================

23. Candidate scoring

==================================================

각 candidate를 0–5점 평가.

N = Novelty

H = Headroom

G = Generality

S = Systematicity

T = Non-triviality

M = Method cleanliness

I = Intro/story strength

E = Evidence strength

TOTAL /40.

Promotion threshold:

&gt;=26:

DEEP_DIVE

&gt;=30:

FINALIST

&gt;=34:

STRONG_GO candidate.

단 H &lt;4이면

TOTAL 점수와 무관하게 STRONG_GO 불가.

==================================================

24. Paper-intro test

==================================================

PROMISING candidate마다

MINI_[INTRO.md](http://INTRO.md) 작성.

반드시 7문장 정도로:

1.

현재 MoE EP serving에서 중요한 문제.

2.

기존 연구가 보는 대표 관점.

3.

그 관점의 hidden assumption.

4.

우리의 surprising observation.

5.

왜 기존 metric/optimizer가 놓치는지.

6.

실제 E2E impact / headroom.

7.

가능한 method principle.

이 intro가 억지스럽거나

"우리가 이것도 최적화했다" 수준이면

candidate 점수 감점.

==================================================

25. Engineering-triviality test

==================================================

각 finalist에 대해 일부러

가장 단순한 fix를 먼저 시도.

예:

- synchronize

- static config

- larger buffer

- smaller batch

- existing scheduler option

- runtime recommended backend

- simple reordering

단순 fix가:

candidate benefit의 &gt;=80%

를 회수한다면:

TRIVIAL_ENGINEERING.

Paper-level core에서 제거.

좋은 research problem은:

simple fix가

중요한 trade-off를 만들거나

현상을 제대로 해결하지 못해야 한다.

==================================================

26. Blue-ocean test

==================================================

완전 최초가 아니더라도

다음을 확인.

### Good blue-ocean

기존 work:

A를 해결.

우리 observation:

A와 다른 orthogonal variable B.

B에서:

scheduler

runtime

placement

communication

co-design

등 여러 method family 가능.

### Bad crowded space

기존 work 여러 개가:

동일 metric

동일 bottleneck

동일 optimization principle

을 이미 수행.

이 경우 DROP.

==================================================

27. Prior-art protocol

==================================================

PROMISING 이상 candidate는

반드시 literature search.

가능하면:

arXiv

OpenReview

conference proceedings

GitHub runtime issues/PRs

사용.

Search terms를 여러 방식으로 생성.

단 title만 보고 판단하지 않는다.

최소:

abstract

problem statement

method section/high-level design

확인.

Record:

Closest prior work

Exact overlap

Exact difference

Novelty risk

Date

PRIOR_ART_[MATRIX.md](http://MATRIX.md) 유지.

==================================================

28. Strong candidate generalization

==================================================

STRONG_GO 후보는 가능하면 다음 중

최소 2개 축에서 재현.

Model:

Qwen3-VL

Qwen3 text-only

Workload:

prefill

decode

mixed

Modality:

text-heavy

vision-heavy

Concurrency:

low

medium

high

Backend:

DeepEP

other viable backend

하나의 환경에서만 크면

system-specific engineering일 가능성.

==================================================

29. Causal standard

==================================================

Correlation-only candidate는

STRONG_GO 불가.

최소:

Observation

+

Controlled perturbation

+

Counterfactual/oracle

필요.

Ideal:

A changes

→ phenomenon changes

A held fixed

→ effect disappears

==================================================

30. Method-stage rule

==================================================

Method 구현은 마지막이다.

순서:

OBSERVATION

→ DIRECT E2E HEADROOM

→ PRIOR ART

→ CAUSALITY

→ GENERALITY

→ FEASIBLE ORACLE

→ SIMPLE METHOD

이 순서를 바꾸지 않는다.

==================================================

31. Time budget

==================================================

Target:

8–12 hours wall time.

Strong unexplored tree가 남아 있으면

최대 14 hours까지 가능.

단:

의미 없는 GPU burning 금지.

각 hour checkpoint에서:

wall time

live GPU minutes

nodes tested

nodes killed

top candidates

next branches

기록.

==================================================

32. Early phase allocation

==================================================

Suggested:

Hour 0–1.5:

E2E latency atlas + residual hotspot map

Hour 1.5–4:

10–20 cheap hypothesis nodes

Hour 4–6:

top 2–3 deep dives

Hour 6–8:

prior art + causality + headroom

Hour 8–10:

finalists generalization

Hour 10–12:

minimal method/oracle if warranted

이것은 guideline이지 strict schedule 아님.

==================================================

33. Autonomous branch generation

==================================================

각 실험 종료 후 반드시 질문:

1.

What did we expect?

2.

What actually happened?

3.

Which assumption failed?

4.

What new variable might explain residual?

5.

Is that variable high-frequency/high-mass?

6.

Can we create a controlled perturbation?

7.

What sibling/child hypothesis follows?

최대 2개의 child를 자동 생성.

단 E2E mass가 작은 anomaly는 child 생성 금지.

==================================================

34. Research value over experiment count

==================================================

많은 experiment 수가 목표가 아니다.

10개 cheap probes 중

2–3개가 PROMISING이면

나머지 breadth를 잠시 중단하고

2–3개를 깊게 검증.

그중:

headroom

causality

novelty

generality

method cleanliness

가 떨어지는 candidate를 제거.

최종 최소 하나의

paper-level candidate를 찾는 데 집중.

==================================================

35. "Would I write the Introduction?" gate

==================================================

각 finalist에 대해 agent 자신에게 질문:

"If this were my paper,

would the introduction sound like

a fundamental overlooked problem,

or just a clever optimization?"

다음이면 weak:

"We notice configuration X is slower,

so we tune X."

다음이면 stronger:

"Current systems optimize X under assumption A.

Real online EP violates A because B.

This creates Y% direct E2E waste.

Existing methods cannot address B because ...

We introduce principle C."

==================================================

36. Required files

==================================================

Branch:

flashvep/autonomous-ep-research-discovery-v3

Main report:

poc_flashvep/reports/

autonomous_ep_research_discovery_[v3.md](http://v3.md)

Maintain:

RESEARCH_[TREE.md](http://TREE.md)

ACTIVE_[FRONTIER.md](http://FRONTIER.md)

E2E_LATENCY_[ATLAS.md](http://ATLAS.md)

CANDIDATE_SCOREBOARD.csv

PRIOR_ART_[MATRIX.md](http://MATRIX.md)

EXPERIMENT_[LOG.md](http://LOG.md)

[ANOMALIES.md](http://ANOMALIES.md)

For finalists:

FINALIST_1_[INTRO.md](http://INTRO.md)

FINALIST_1_[HEADROOM.md](http://HEADROOM.md)

FINALIST_1_[METHOD.md](http://METHOD.md)

FINALIST_2_...

Hourly:

CHECKPOINT_[H1.md](http://H1.md)

CHECKPOINT_[H2.md](http://H2.md)

...

==================================================

37. Final candidate report

==================================================

For each FINALIST:

NAME:

ONE-LINE PROBLEM:

OBSERVATION:

WHY SURPRISING:

HIDDEN ASSUMPTION IN CURRENT SYSTEMS:

DIRECT E2E WASTE:

...%

PERFECT ORACLE:

...%

FEASIBLE ORACLE:

...%

CONTROLLED EFFECT:

...%

GENERALITY:

MLLM SPECIFIC:

YES / NO

IF YES:

why modality × routing × EP is required:

CLOSEST PRIOR ART:

EXACT DIFFERENCE:

TRIVIAL FIX TEST:

WHY NOT JUST ENGINEERING:

METHOD PRINCIPLE:

EXPECTED ACTUAL E2E BENEFIT:

NOVELTY SCORE:

HEADROOM SCORE:

GENERALITY SCORE:

SYSTEMATICITY SCORE:

NONTRIVIALITY SCORE:

METHOD SCORE:

INTRO SCORE:

EVIDENCE SCORE:

TOTAL:

STATUS:

==================================================

38. Final output

==================================================

FINAL STATUS:

FOUND_STRONG_GO

or

SEARCH_SPACE_EXHAUSTED_NO_GO

TOTAL WALL TIME:

TOTAL LIVE GPU TIME:

TOTAL 4-GPU HOURS:

TOTAL TREE NODES:

CHEAP PROBES:

DEEP DIVES:

FINALISTS:

TOP 10 DISCOVERIES:

...

TOP 5 NEGATIVE FINDINGS:

...

FINALIST 1:

...

FINALIST 2:

...

STRONG_GO:

...

IF FOUND:

EXACT PAPER MOTIVATION:

...

PROBLEM:

...

WHY NEW:

...

WHY LARGE:

...

WHY GENERAL:

...

WHY NOT TRIVIAL ENGINEERING:

...

POSSIBLE METHODS:

1.

2.

3.

CHEAPEST NEXT METHOD POC:

...

EXPECTED REAL E2E BENEFIT:

...

WHAT COULD STILL KILL IT:

...

IF NO STRONG GO:

WHY SEARCH SPACE FAILED:

...

WHICH FAMILIES STILL UNEXPLORED:

...

NEXT SEARCH SPACE:

...

Branch:

Commit:

Push:

Report:

Results: