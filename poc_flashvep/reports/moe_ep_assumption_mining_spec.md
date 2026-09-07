# MoE-EP Assumption Mining &amp; Counterfactual Discovery

## Structural Research Search Before GPU Optimization

==================================================

0. Mission

==================================================

이번 작업은 새로운 optimization trick을 찾는 GPU sweep이 아니다.

목표는:

LLM / MLLM MoE Expert-Parallel inference system이

현재 당연하다고 가정하고 있는

execution/runtime/model-system assumptions를 찾아내고,

그 assumption을 깨는 counterfactual execution design 중

- direct E2E headroom이 큼

- 최신 prior art가 정확히 다루지 않음

- 단순 config tuning이 아님

- systematic method로 발전 가능

- 4×H100 single-node에서 검증 가능

한 후보를 찾는 것이다.

핵심 질문:

"현재 시스템은 왜 반드시 이렇게 실행되어야 하는가?"

==================================================

1. Research philosophy

==================================================

이전 discovery는 주로:

trace anomaly

→ hypothesis

→ GPU test

였다.

이번에는:

system design

→ hidden assumption

→ counterfactual execution

→ analytical headroom

→ prior-art attack

→ GPU validation

순서로 진행한다.

GPU experiment는 마지막에만 사용한다.

==================================================

2. Hardware

==================================================

GPU 사용이 필요한 모든 실험에서 반드시:

CUDA_VISIBLE_DEVICES=1,2,3,4

만 사용한다.

GPU 0,5,6,7은 절대 건드리지 않는다.

Primary stack:

Qwen3-VL-30B-A3B-Instruct

vLLM

DeepEP

TP2 / DP2 / EP4

현재 working environment를 우선 사용.

Strong candidate가 나온 뒤

cross-model generality가 필요할 때만

Kimi validation을 고려한다.

==================================================

3. Time budget

==================================================

Expected total wall time:

6–10 hours.

Suggested:

Phase A:

2–4 hours

GPU 거의 사용하지 않음.

Phase B:

1–2 hours

analytical/headroom filtering.

Phase C:

3–6 hours

Top candidates GPU PoC.

Strong candidate가 있으면

최대 약 12 hours까지 허용.

GPU 시간을 채우는 것이 목적이 아니다.

==================================================

4. Read existing project history first

==================================================

먼저 기존 reports / NO-GO / research trees를 읽는다.

다음 파일 생성:

ASSUMPTION_MINING_NEGATIVE_[MAP.md](http://MAP.md)

정리할 것:

- 어떤 research direction을 이미 시도했는가

- 어떤 assumption이 이미 falsified 되었는가

- headroom이 왜 작았는가

- prior-art collision이 무엇이었는가

- 어떤 measurement mistake를 했는가

특히 다음 교훈을 명시:

wave-level improvement

!=

request-level E2E improvement.

앞으로 direct request critical path를 우선한다.

==================================================

5. Do NOT begin with predefined research topics

==================================================

사용자가 제공한 기존 아이디어를

단순히 재배열하지 않는다.

source code,

runtime design,

paper assumptions,

negative results를 읽고

agent 자신이 assumptions를 생성한다.

목표:

최소 30개.

권장:

40–50개.

==================================================

6. What counts as an assumption?

==================================================

좋은 assumption은 예를 들어 다음 형태다.

"Current systems assume X must happen before Y."

"All workload classes use the same execution contract."

"State A and state B are optimized independently."

"Data must move toward compute rather than compute toward data."

"One synchronization scope is assumed to be necessary."

"One routing decision is assumed immutable during execution."

"One scheduling metric is assumed to correlate with request completion."

"All tokens/layers/requests are assumed to require the same execution semantics."

중요:

위 문장들은 예시 형식일 뿐

research topic seed로 사용하지 않는다.

실제 assumption은 agent가 발견한다.

==================================================

7. Assumption sources

==================================================

각 assumption은 최소 하나의 evidence source를 가져야 한다.

Source categories:

A. vLLM source code

B. DeepEP source code

C. existing project measurements

D. prior papers

E. unexplained negative result

F. runtime API semantics

G. scheduler design

H. communication primitive semantics

I. model execution graph

J. request-level serving semantics

==================================================

8. Assumption catalog

==================================================

Create:

ASSUMPTION_CATALOG.csv

Fields:

ID

Assumption

Where observed

Why current systems use it

Potentially unnecessary because

Counterfactual

Potential saved work

Potential added cost

4GPU testability

Novelty risk

Prior-art candidates

Status

Generate &gt;=30 assumptions before selecting finalists.

==================================================

9. Assumption diversity

==================================================

Do not create 30 variations of one idea.

Cluster assumptions yourself.

No one conceptual cluster may exceed 30%

of the catalog.

If it does:

regenerate more diverse assumptions.

==================================================

10. Counterfactual design

==================================================

For each promising assumption ask:

"What would execution look like

if this assumption were false?"

Construct a counterfactual.

Counterfactual must specify:

CURRENT:

...

COUNTERFACTUAL:

...

WHAT WORK DISAPPEARS:

...

WHAT WORK MOVES:

...

WHAT NEW OVERHEAD APPEARS:

...

WHAT DEPENDENCY CHANGES:

...

WHAT REQUEST-CRITICAL PATH COULD SHRINK:

...

==================================================

11. Analytical headroom before GPU

==================================================

Do NOT run GPU PoC immediately.

First derive a rough analytical upper bound.

Use existing traces whenever possible.

Estimate:

direct request-critical fraction

×

fraction affected

×

maximum removable portion.

Call:

ANALYTICAL_DIRECT_E2E_ORACLE.

Rules:

&lt;10%

→ DROP.

10–15%

→ WEAK.

15–20%

→ PROMISING only if novelty strong.

&gt;=20%

→ HIGH PRIORITY.

Preferred GPU candidates should have

&gt;=20% analytical potential.

==================================================

12. Structural gain requirement

==================================================

Theoretical benefit should come from

changing execution structure.

Examples of valid structural gain:

- entire communication/dependency removed

- critical-path serialization eliminated

- duplicated work removed

- expensive execution mode replaced

- phase-specific semantics changed

- unnecessary materialization avoided

- execution ownership moved

- currently hidden parallelism exposed

Weak:

2–3% kernel tuning

small launch optimization

static knob tuning.

==================================================

13. Prior-art attack before GPU

==================================================

Before any candidate gets GPU time:

search literature and implementations.

Act as hostile reviewer:

"What exact work makes this idea non-novel?"

Read enough to compare:

problem

assumption

counterfactual

method

execution semantics

hardware setting

serving objective.

Classify:

NOVEL

BLUE_OCEAN

ADJACENT_HIGH_RISK

CROWDED

DIRECT_COLLISION.

DIRECT_COLLISION:

do not GPU-test as paper candidate.

==================================================

14. Blue-ocean standard

==================================================

Literal first-ever discovery is preferred

but not mandatory.

Strong BLUE_OCEAN candidate:

existing work breaks assumption A

our candidate breaks distinct assumption B

B causes material E2E waste

and B enables multiple systematic solutions.

Weak:

existing work already handles B,

we only apply it to another model.

==================================================

15. Trivial-engineering kill test

==================================================

Before GPU:

ask if counterfactual is equivalent to:

- existing runtime mode

- documented configuration

- static batch-size choice

- one synchronization call

- buffer-size tuning

- warmup

- existing backend switch

- one-line heuristic.

If YES:

DROP unless deeper structural issue remains.

==================================================

16. Research-space test

==================================================

For each top candidate generate

at least 3 distinct solution principles.

METHOD_A:

...

METHOD_B:

...

METHOD_C:

...

If only one ugly special-case fix exists:

candidate is probably engineering.

If multiple clean principles exist:

better research space.

==================================================

17. Intro test

==================================================

For each PROMISING candidate write

a provisional paper Introduction:

1. Why the serving setting matters.

2. What current systems optimize.

3. What hidden assumption they rely on.

4. Why the assumption is unnecessary/violated.

5. What request-level E2E waste it creates.

6. Why prior systems cannot address it.

7. What new execution principle could solve it.

If the story sounds like:

"We tune parameter X."

DROP.

==================================================

18. Candidate scoring

==================================================

Score 0–5:

Novelty

DirectHeadroom

Structurality

Generality

Systematicity

NonTriviality

MethodSpace

IntroStrength

4GPUFeasibility

Evidence

Total /50.

Promotion:

&gt;=32:

PROMISING.

&gt;=38:

GPU_CANDIDATE.

&gt;=43:

STRONG candidate.

Hard headroom gates override score.

==================================================

19. Selection before GPU

==================================================

After 30–50 assumptions:

select at most TOP 5.

Then perform deeper analytical/prior-art review.

Reduce to:

TOP 2–3 GPU candidates.

Do not GPU-test 20 weak assumptions.

==================================================

20. GPU Candidate requirements

==================================================

GPU candidate should preferably satisfy:

Analytical direct E2E oracle &gt;=20%.

No direct prior-art collision.

Not trivial config tuning.

Counterfactual can be approximated on 4GPU.

Clear measurable causal perturbation.

Potential method space &gt;=2 viable designs.

==================================================

21. GPU phase

==================================================

Only now use:

CUDA_VISIBLE_DEVICES=1,2,3,4.

For each candidate first implement

the CHEAPEST COUNTERFACTUAL ORACLE.

Not a production method.

Goal:

"If the assumption is broken,

does request E2E actually move?"

==================================================

22. Oracle PoC

==================================================

Possible oracle types should be generated

from the candidate itself.

Do not default to one technique.

The PoC may use:

controlled replay

execution bypass

synthetic completion

idealized scheduling

selective disabling

offline counterfactual

isolated timing

bounded runtime patch

if semantically valid.

Always report what the oracle exaggerates.

==================================================

23. Direct E2E requirement

==================================================

Primary metric:

request-level critical path.

Report where relevant:

TTFT

ITL / TPOT

request completion latency

throughput

Do not promote based only on:

kernel latency

wave latency

MoE-stage latency

dispatch latency.

==================================================

24. GPU candidate kill gate

==================================================

Fresh direct GPU oracle:

&lt;10% E2E:

CLOSE.

10–15%:

WEAK.

15–20%:

possible HOLD only if novelty excellent.

&gt;=20%:

DEEP_DIVE.

==================================================

25. Causality

==================================================

For DEEP_DIVE candidate:

perform controlled perturbation.

Show:

assumption pressure increases

→ cost increases.

counterfactual removes/relaxes assumption

→ cost decreases.

replicate independently.

==================================================

26. Common-state confound protection

==================================================

Previous work found:

A and B often moved together

due to shared runtime state.

Therefore important comparisons require:

randomized order

repeated runs

fresh baseline interleaving

warmup controls.

Where relevant consider:

clock

allocator

KV/workspace

host gap

scheduler history

worker state.

Do not interpret common regime transition

as treatment effect.

==================================================

27. Generality

==================================================

Do not spend Kimi time early.

Once candidate survives:

direct E2E &gt;=15–20%

+

causality

+

novelty

then validate the phenomenon on Kimi.

Strong paper target:

Qwen + Kimi.

Exact magnitude need not match.

Causal direction and problem must remain.

==================================================

28. MLLM rule

==================================================

If candidate is claimed as MLLM-specific:

modality must be causal.

Test proper controls.

If not:

reclassify generic MoE.

A generic strong MoE-EP candidate

is acceptable.

==================================================

29. Structural novelty examples are NOT seeds

==================================================

Important:

Do not mechanically pursue concepts such as

overlap,

routing,

placement,

prefill/decode,

modality.

Those words are not required.

A strong candidate can involve any

execution assumption found by the agent.

The research question must arise from

assumption mining.

==================================================

30. Rethink checkpoints

==================================================

After:

10 assumptions

20 assumptions

30 assumptions

40 assumptions

write:

ASSUMPTION_RETHINK_&lt;N&gt;.md

Ask:

Which categories are overrepresented?

Which runtime contracts are we treating as immutable?

What assumptions are so fundamental

that we have not even written them down?

What would a completely different systems researcher question?

Generate additional assumptions accordingly.

==================================================

31. Cross-paper assumption mining

==================================================

Read recent strong MoE inference papers

not merely for methods.

For each paper ask:

"What assumption did this paper break?"

Create:

PAPER_ASSUMPTION_[MATRIX.md](http://MATRIX.md)

Fields:

Paper

Old assumption

Counterfactual/new execution

Why headroom existed

What dimension remains unchanged

Potential unexamined assumption

Do not copy paper ideas.

Use them to learn how researchers

identify structural assumptions.

==================================================

32. Existing implementation assumption mining

==================================================

Inspect vLLM / DeepEP implementation.

Look for:

semantic boundaries

ownership rules

lifetime assumptions

collective scope

materialization points

scheduler/runtime separation

state coupling

immutable decisions

fallback paths

global/local decisions.

Do NOT call any of these problems

without measurement.

They only generate assumptions.

==================================================

33. Final STRONG_GO gate

==================================================

STRONG_GO requires:

1. Structural assumption identified.

2. Clear reason it is unnecessarily restrictive

   or mismatched with real workload.

3. Direct request-level E2E headroom &gt;=15%.

Preferred &gt;=20%.

4. Feasible method headroom &gt;=12%.

5. Controlled causal evidence.

6. No trivial fix.

7. No direct literature collision.

8. Generality evidence.

9. At least 2 clean method directions.

10. Convincing paper intro.

11. Minimal method/oracle PoC moves E2E.

==================================================

34. Stop conditions

==================================================

Final status:

FOUND_STRUCTURAL_STRONG_GO

or

ASSUMPTION_SPACE_NO_GO.

ASSUMPTION_SPACE_NO_GO may only be returned after:

&gt;=30 assumptions catalogued

&gt;=10 serious counterfactuals

&gt;=5 analytical headroom analyses

&gt;=2–3 top candidates subjected to

full novelty/structural review

unless zero candidate passes analytical gates.

No need to burn GPU if

no analytically credible candidate exists.

==================================================

35. Required artifacts

==================================================

Branch:

flashvep/moe-ep-assumption-mining

Create:

poc_flashvep/reports/

moe_ep_assumption_[mining.md](http://mining.md)

Maintain:

ASSUMPTION_CATALOG.csv

ASSUMPTION_MINING_NEGATIVE_[MAP.md](http://MAP.md)

PAPER_ASSUMPTION_[MATRIX.md](http://MATRIX.md)

TOP_[COUNTERFACTUALS.md](http://COUNTERFACTUALS.md)

CANDIDATE_SCOREBOARD.csv

PRIOR_ART_[MATRIX.md](http://MATRIX.md)

GPU_EXPERIMENT_[LOG.md](http://LOG.md)

Checkpoint files:

ASSUMPTION_RETHINK_[10.md](http://10.md)

ASSUMPTION_RETHINK_[20.md](http://20.md)

ASSUMPTION_RETHINK_[30.md](http://30.md)

ASSUMPTION_RETHINK_[40.md](http://40.md)

For top candidates:

CANDIDATE_1_[COUNTERFACTUAL.md](http://COUNTERFACTUAL.md)

CANDIDATE_1_[INTRO.md](http://INTRO.md)

CANDIDATE_1_[ORACLE.md](http://ORACLE.md)

etc.

==================================================

36. Final report format

==================================================

FINAL STATUS:

FOUND_STRUCTURAL_STRONG_GO /

ASSUMPTION_SPACE_NO_GO

TOTAL WALL TIME:

TOTAL LIVE GPU TIME:

TOTAL 4-GPU HOURS:

ASSUMPTIONS GENERATED:

ASSUMPTIONS AFTER DEDUP:

COUNTERFACTUALS:

ANALYTICAL HEADROOM TESTS:

GPU CANDIDATES:

DEEP DIVES:

FINALISTS:

--------------------------------

TOP 10 ASSUMPTIONS

--------------------------------

...

--------------------------------

WHAT CURRENT SYSTEMS ASSUME

--------------------------------

Most important hidden assumptions discovered:

...

--------------------------------

TOP COUNTERFACTUALS

--------------------------------

For each:

CURRENT ASSUMPTION:

...

COUNTERFACTUAL:

...

ANALYTICAL E2E ORACLE:

...%

PRIOR ART:

...

WHY STRUCTURAL:

...

STATUS:

...

--------------------------------

GPU VALIDATION

--------------------------------

Candidate:

...

CUDA_VISIBLE_DEVICES:

1,2,3,4

Direct E2E effect:

...%

Controlled effect:

...

Replications:

...

--------------------------------

BEST CANDIDATE

--------------------------------

ONE-LINE PROBLEM:

HIDDEN ASSUMPTION:

WHY CURRENT SYSTEMS MAKE IT:

WHY IT IS UNNECESSARY:

DIRECT E2E WASTE:

PERFECT ORACLE:

FEASIBLE ORACLE:

CAUSAL EVIDENCE:

QWEN:

KIMI:

CLOSEST PRIOR ART:

EXACT NOVELTY GAP:

WHY NOT ENGINEERING:

METHOD 1:

METHOD 2:

METHOD 3:

MINI INTRO:

EXPECTED REAL E2E GAIN:

WHAT COULD KILL IT:

--------------------------------

FINAL DECISION

--------------------------------

...

Branch:

Commit:

Push:

Report:

Results: