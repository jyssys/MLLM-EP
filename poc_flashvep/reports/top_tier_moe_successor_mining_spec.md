# Top-Tier MoE Successor Mining

> Resource amendment, 2026-09-08: latest user authorizes the deferred work on
> physical GPUs4,5,6,7 ONLY. This supersedes original1,2,3,4 mappings below.
> User-requested idle/final utilization may run only on4–7, never concurrently
> with measurements, and is excluded from research evidence/accounting.

## SERE × Libra × MoDES → MLLM Successor Discovery

==================================================

0. Mission

==================================================

이번 작업의 목표는 새로운 research problem을

0부터 발명하는 것이 아니다.

이미 최신 top-tier venue에서

큰 speedup/headroom을 입증한 강한 baseline:

1. SERE

2. Libra

3. MoDES

를 대상으로:

PAPER / CODE UNDERSTANDING

→ FAITHFUL REPRODUCTION

→ MLLM TRANSFER

→ FAILURE MODE DISCOVERY

→ CROSS-MODEL GENERALITY

→ SUCCESSOR OPPORTUNITY

→ BEST CANDIDATE DEEP DIVE

를 수행한다.

최종 목표:

"기존 strong method의 중요한 hidden assumption이

MLLM에서는 깨지며,

이를 해결하는 successor가

명확한 paper-level contribution이 될 수 있다"

는 후보 최소 하나를 찾는 것이다.

==================================================

1. Core philosophy

==================================================

이번에는 다음을 우선한다.

BAD:

"새 phenomenon을 발견했는데

파이가 있는지 모르겠다."

GOOD:

"기존 top-tier method가 이미

큰 performance opportunity를 증명했다.

하지만 assumption A 때문에

MLLM workload에서는 그 benefit을

잃거나 quality가 크게 망가진다.

우리는 A를 해결한다."

즉 이미 검증된 performance pie를

더 안전한 research starting point로 사용한다.

==================================================

2. Target papers

==================================================

반드시 세 개 모두 조사한다.

A. SERE

Recent top-tier expert rerouting/substitution method.

B. Libra

ICLR 2026.

HarMoEny 계열을 발전시킨

lookahead / replication / overlap-oriented method.

C. MoDES

CVPR 2026.

MLLM modality-aware expert reduction/skipping method.

각 논문의 official paper와

official code repository를 우선 사용한다.

Unofficial reimplementation은

official code가 불가한 경우에만 사용한다.

==================================================

3. Hardware

==================================================

모든 GPU 실험:

CUDA_VISIBLE_DEVICES=1,2,3,4

만 사용.

물리 GPU:

1,2,3,4

외 GPU 절대 사용 금지.

GPU 0/5/6/7 금지.

Primary hardware:

4× H100 single node.

==================================================

4. Models

==================================================

최종 generality target은 반드시:

Qwen MoE MLLM

+

Kimi MoE MLLM

이다.

현재 프로젝트에서 검증된 Qwen3-VL 계열을

primary discovery model로 사용.

Kimi는 candidate가 충분히 안정화된 뒤

cross-model validation에 사용.

단 세 baseline 모두에서

가능한 한 Qwen/Kimi transferability를

분석해야 한다.

한 모델에서만 발생하는 failure는

강한 successor motivation으로 보기 어렵다.

==================================================

5. Time budget

==================================================

Expected wall time:

12–20 hours.

Environment / porting 문제가 있으면

최대 24 hours까지 허용.

Suggested:

Paper/code audit:

3–4 h

Baseline reproduction / adaptation:

3–6 h

MLLM stress/failure mining:

4–6 h

Best candidate deep dive:

4–8 h

이것은 guideline이며

강한 candidate가 나오면 시간을 재배분한다.

==================================================

6. Do not confuse environment failure with research failure

==================================================

Official code가 현재 vLLM/runtime와

즉시 호환되지 않는다고:

METHOD_FAIL

로 부르지 않는다.

다음을 구분:

ENVIRONMENT_BLOCKED

PORTING_REQUIRED

BASELINE_REPRODUCED

ALGORITHM_REPRODUCED

RESEARCH_FAILURE.

한 baseline당 environment setup에

무한정 매몰되지 않는다.

대략 60–90분 이상

dependency setup만 반복하면:

isolated env

또는

minimal faithful algorithm extraction

으로 전환.

==================================================

7. Read previous project negatives

==================================================

먼저 기존 project reports를 읽고:

PREVIOUS_NEGATIVE_[SPACE.md](http://SPACE.md)

작성.

목적:

우리 프로젝트에서 이미 죽은 문제를

baseline successor라는 이름으로

되살리지 않기 위함.

특히 이미 closed된:

plain load balancing

fanout

critical-rank coalescing

simple TP↔EP switch

naive overlap

simple synchronization

partial-expert speculation

등과 새 successor가

실질적으로 동일한지 검증.

==================================================

8. Common evaluation principle

==================================================

이번 연구에서 단순 latency만 보면 안 된다.

각 baseline은 quality-efficiency tradeoff를 가진다.

따라서 primary comparison은:

QUALITY-MATCHED EFFICIENCY

and

EFFICIENCY-MATCHED QUALITY

이다.

예:

same accuracy:

which method is faster?

same latency:

which method has better quality?

==================================================

9. Common baseline metrics

==================================================

각 paper마다 가능한 범위에서:

TTFT

TPOT

ITL

request E2E

throughput

MoE time

dispatch

expert

combine

memory overhead

를 기록.

Quality:

benchmark accuracy

task metric

logit/output consistency

paper-specific quality metric

를 사용.

==================================================

10. MLLM stress dimensions

==================================================

각 baseline을 다음 상황에서 평가.

TEXT-HEAVY

VISION-HEAVY

MIXED

SINGLE IMAGE

MULTI IMAGE

LOW RESOLUTION

HIGH RESOLUTION

SHORT OUTPUT

LONG OUTPUT

LOW CONCURRENCY

HIGH CONCURRENCY

모두 수행할 필요는 없지만

method assumption을 공격하기 적합한

stress dimensions를 선택한다.

==================================================

11. Baseline A — SERE audit

==================================================

SERE에 대해 먼저 정확히 조사:

- exact problem

- original speedup

- target models

- routing/substitution mechanism

- expert similarity computation

- calibration process

- runtime overhead

- quality preservation mechanism

- implementation details

- official limitations

- code-level assumptions

특히 질문:

What exactly makes expert A substitutable for B?

Is expert similarity:

global?

static?

dataset-conditioned?

layer-conditioned?

token-conditioned?

modality-conditioned?

Do not assume the answer before reading code/paper.

==================================================

12. SERE MLLM transfer tests

==================================================

SERE를 MLLM으로 transfer할 때:

Text

Vision

에 대해 expert substitution error를 분석.

Potential measurements:

expert-pair output similarity

substitution output error

router-weight-conditioned error

layer-conditioned error

modality-conditioned error

content-conditioned error

그리고 실제:

quality-speed Pareto.

Strong failure example:

LLM:

large speedup + negligible quality loss

MLLM:

similar speedup but substantial quality degradation

or

MLLM quality를 유지하면

SERE speedup이 크게 collapse.

==================================================

13. SERE hidden-assumption mining

==================================================

반드시 다음 질문을 스스로 생성/검증:

Which assumption explains the MLLM gap?

Do not force:

"modality-conditioned similarity"

as the answer.

Potentially it may instead be:

layer

router confidence

vision semantics

batch mixture

runtime load

expert pair stability

calibration distribution

critical path

등일 수 있다.

Evidence가 결정하게 한다.

==================================================

14. Baseline B — Libra audit

==================================================

Libra를 정확히 조사:

- HarMoEny에서 무엇을 계승했는가

- HarMoEny의 어떤 bottleneck을 해결했는가

- lookahead mechanism

- next-layer expert prediction

- expert replication

- token sharding

- overlap mechanism

- predictor accuracy

- planning overhead

- official evaluation

- implementation assumptions

- official limitations

특히:

Which assumption enables Libra's lookahead?

Where does its overlap benefit actually come from?

==================================================

15. Libra MLLM transfer tests

==================================================

MLLM에서 다음을 측정:

expert prediction accuracy by:

modality

layer

token type

vision/text boundary

router entropy

image resolution

multi-image

batch composition

그리고:

wrong prediction cost

wrong replication cost

wasted copy/planning

missed hot expert

actual overlap ratio

request-level benefit.

Strong failure:

Libra prediction/overlap works in text

but degrades materially in Vision/Mixed regime.

==================================================

16. Libra MLLM-specific caution

==================================================

이전 프로젝트에서:

simple cross-layer expert persistence

는 약했다.

하지만 그것만으로

Libra failure라고 결론내리지 않는다.

Libra predictor가 실제로 사용하는

hidden-state/gating mechanism을 그대로 구현하고

정확히 측정해야 한다.

==================================================

17. Baseline C — MoDES audit

==================================================

MoDES는 이미 MLLM method이므로

단순히:

"make it modality-aware"

는 successor가 아니다.

정확히 조사:

- expert skipping/reduction criterion

- modality handling

- layer handling

- threshold design

- calibration

- online/offline decisions

- quality preservation

- prefill vs decode behavior

- speedup source

- official limitations

- code heuristics.

==================================================

18. MoDES successor mining

==================================================

핵심 질문:

"What MLLM variation does MoDES still treat too coarsely?"

But do not force a predefined answer.

Stress:

different visual tasks

different image density

OCR/chart/object/background

single vs multi-image

resolution

cross-dataset calibration

prefill vs decode

mixed request batches

cross-model transfer.

Look for:

same modality

but radically different safe expert budget.

or

calibration policy that fails under distribution shift.

==================================================

19. MoDES candidate must beat existing MLLM literature

==================================================

MoDES successor는 특히 novelty gate가 강하다.

Compare against:

AnyExperts

MACS

other recent modality-aware expert allocation/skipping

and any newer relevant work discovered.

Simple:

token importance

semantic importance

per-layer threshold

만 추가하는 것은

기존 literature collision 위험이 높다.

==================================================

20. Faithful baseline first

==================================================

각 paper에서 먼저:

BASELINE WORKS

를 보여라.

성능이 안 나오면:

method가 나쁜지

port가 틀렸는지

evaluation이 다른지

구분.

Successor failure claim은:

faithful baseline이

reasonable regime에서 정상 작동한 뒤에만 허용.

==================================================

21. Baseline reproduction tolerance

==================================================

원 paper 숫자를 정확히 똑같이

재현할 필요는 없다.

Hardware/model이 다르므로:

directionally faithful

해야 한다.

예:

original:

+30%

ours:

+18%

가능.

하지만:

original:

+30%

ours:

-3%

이면 먼저 implementation을 의심.

==================================================

22. Failure-mode mining

==================================================

각 baseline에 대해 반드시:

FAILURE_MODE_[MATRIX.md](http://MATRIX.md)

작성.

Rows:

workload

modality

layer

quality

latency

throughput

failure severity

possible cause.

Failure를 의도적으로 stress한다.

하지만 unrealistic adversarial input만으로

paper motivation을 만들지 않는다.

==================================================

23. Material failure gate

==================================================

A failure가 research candidate가 되려면

material해야 한다.

Examples:

QUALITY FAILURE:

&gt;=2 percentage points quality loss

or

baseline paper에서는 negligible였던 quality loss가

MLLM에서 크게 증가.

EFFICIENCY FAILURE:

&gt;=10% lost E2E benefit

or

&gt;=20% relative loss of baseline speedup.

PARETO FAILURE:

MLLM에서 baseline의

quality-efficiency Pareto frontier가

materially 악화.

==================================================

24. Cross-model gate

==================================================

Strong failure는:

Qwen

+

Kimi

에서 same causal direction이 보여야 함.

Magnitude는 달라도 됨.

One-model-only:

MODEL_SPECIFIC_HOLD.

Not final paper candidate.

==================================================

25. Successor opportunity estimation

==================================================

Failure를 찾았으면:

"이 failure를 완전히 해결하면

얼마나 얻는가?"

를 계산.

SUCCESSOR_ORACLE.

Important:

Original baseline vs vanilla

만 보지 않는다.

Compare:

vanilla

original baseline

ideal fixed baseline.

Example:

Vanilla:

100 ms

SERE:

80 ms but quality -4%

quality-safe SERE:

95 ms

Ideal MLLM-aware SERE:

80 ms and quality preserved

Successor opportunity:

15 ms at matched quality.

==================================================

26. Strong successor opportunity

==================================================

Preferred:

At matched quality:

&gt;=10% additional request-level E2E improvement

over original baseline

or

At matched efficiency:

&gt;=1.5–2.0 pp benchmark-quality recovery

with &lt;=2% performance loss.

Especially strong:

&gt;=12% additional E2E

or

recover most baseline quality failure

while preserving &gt;=80–90% of baseline speedup.

==================================================

27. Do not demand 12% in every paper

==================================================

Successor paper can be strong even if

incremental speedup is smaller than original paper.

For example:

original method:

1.8×

but MLLM quality collapse.

successor:

1.75×

with quality restored.

This can be stronger than:

1.9×

with same quality problem.

Evaluate Pareto improvement.

==================================================

28. Causal failure analysis

==================================================

Do not stop at:

"SERE performs worse on Vision."

Need:

WHY?

Controlled perturbation.

For example:

if candidate mechanism is calibration mismatch:

recalibrate

→ failure shrinks.

If modality-conditioned similarity:

control expert pair

→ modality changes substitution error.

If prediction uncertainty:

confidence increases

→ Libra success increases.

Causal evidence required.

==================================================

29. Trivial-fix gate

==================================================

Attack each failure with simple fixes.

Examples:

recalibration

different threshold

larger calibration set

static Vision/Text split

existing runtime option

different batch size.

If simple fix solves &gt;=80% of failure:

likely incremental engineering.

Need deeper reason for paper-level candidate.

==================================================

30. Method-space test

==================================================

For each serious failure generate:

METHOD_A

METHOD_B

METHOD_C

conceptually different.

If only:

"change threshold"

exists:

weak.

If failure opens:

prediction

calibration

routing

scheduling

runtime

quality-aware control

multiple directions:

stronger research space.

==================================================

31. Candidate scoring

==================================================

For each of the three baseline families

score 0–5:

OriginalBaselineStrength

FailureSeverity

CrossModelGenerality

MLLMSpecificity

SuccessorHeadroom

Novelty

Causality

NonTriviality

MethodSpace

CodeFeasibility

IntroStrength

Evidence

Total /60.

Suggested:

&gt;=38:

SERIOUS

&gt;=46:

FINALIST

&gt;=52:

STRONG successor candidate.

Hard gates override score.

==================================================

32. Complete all three before final choice

==================================================

Do NOT choose SERE after 2 hours

and skip Libra/MoDES.

All three must receive:

paper audit

code audit

baseline sanity

MLLM failure probe

headroom evaluation

prior-art screen.

Only then rank.

==================================================

33. Early allocation

==================================================

Possible order:

SERE

→ Libra

→ MoDES

or different if setup favors otherwise.

Each gets a bounded first pass.

Do not allow one environment issue

to consume the entire sprint.

==================================================

34. After all three:

select TOP 1

==================================================

Create:

SUCCESSOR_SCOREBOARD.csv

Rank:

#1

#2

#3

with explicit reasons.

Choose exactly one:

BEST_SUCCESSOR_CANDIDATE.

Then begin deep dive.

==================================================

35. Best-candidate deep dive

==================================================

The top candidate receives:

- more workloads

- multiple independent runs

- causal diagnostics

- Qwen + Kimi validation

- quality-efficiency Pareto

- failure distribution

- trivial-fix attack

- stronger prior-art search

- successor oracle

- minimal method prototypes

==================================================

36. Deep-dive novelty attack

==================================================

Search specifically for papers solving:

the exact failure

not merely the base method.

Ask:

"If reviewer says this is already solved,

what is the strongest citation?"

Classify:

NOVEL

BLUE_OCEAN

INCREMENTAL

COLLISION.

==================================================

37. Deep-dive Intro test

==================================================

Write:

BEST_CANDIDATE_[INTRO.md](http://INTRO.md)

Structure:

1.

Recent method X achieves major efficiency gains.

2.

X relies on assumption A.

3.

We show A breaks systematically in MLLMs.

4.

The failure appears in Qwen and Kimi.

5.

This causes measured quality/performance loss Y.

6.

Simple recalibration/tuning cannot fix it because B.

7.

We propose principle C.

If this story is natural:

good.

If it requires many caveats:

weak.

==================================================

38. Minimal successor PoC

==================================================

Do not build full production method.

Implement up to 2 small successor mechanisms.

Purpose:

show failure is fixable using the identified cause.

Compare:

Vanilla

Original baseline

Baseline + trivial fix

Successor prototype.

==================================================

39. Strong-GO standard

==================================================

FOUND_SUCCESSOR_STRONG_GO requires:

1.

Original baseline faithfully works.

2.

Material MLLM failure discovered.

3.

Failure reproduced across Qwen + Kimi.

4.

Causal explanation.

5.

Simple fix insufficient.

6.

Meaningful successor oracle.

7.

No direct prior-art collision.

8.

At least one minimal successor mechanism

moves Pareto frontier positively.

9.

Clear Introduction.

10.

Reasonable implementation path.

==================================================

40. Possible final results

==================================================

FOUND_SUCCESSOR_STRONG_GO

FOUND_INCREMENTAL_ONLY

ALL_THREE_NO_GO

ENVIRONMENT_BLOCKED

Do not force STRONG_GO.

==================================================

41. Required artifacts

==================================================

Branch:

flashvep/top-tier-successor-mining

Main report:

poc_flashvep/reports/

top_tier_successor_[mining.md](http://mining.md)

Create:

PREVIOUS_NEGATIVE_[SPACE.md](http://SPACE.md)

SERE/

  PAPER_[AUDIT.md](http://AUDIT.md)

  CODE_[AUDIT.md](http://AUDIT.md)

  [REPRODUCTION.md](http://REPRODUCTION.md)

  MLLM_[FAILURES.md](http://FAILURES.md)

  [HEADROOM.md](http://HEADROOM.md)

  PRIOR_[ART.md](http://ART.md)

LIBRA/

  PAPER_[AUDIT.md](http://AUDIT.md)

  CODE_[AUDIT.md](http://AUDIT.md)

  [REPRODUCTION.md](http://REPRODUCTION.md)

  MLLM_[FAILURES.md](http://FAILURES.md)

  [HEADROOM.md](http://HEADROOM.md)

  PRIOR_[ART.md](http://ART.md)

MODES/

  PAPER_[AUDIT.md](http://AUDIT.md)

  CODE_[AUDIT.md](http://AUDIT.md)

  [REPRODUCTION.md](http://REPRODUCTION.md)

  MLLM_[FAILURES.md](http://FAILURES.md)

  [HEADROOM.md](http://HEADROOM.md)

  PRIOR_[ART.md](http://ART.md)

Common:

FAILURE_MODE_MATRIX.csv

SUCCESSOR_SCOREBOARD.csv

QUALITY_EFFICIENCY_PARETO.csv

EXPERIMENT_[LOG.md](http://LOG.md)

GPU_TIME_LOG.csv

Best candidate:

BEST_CANDIDATE_DEEP_[DIVE.md](http://DIVE.md)

BEST_CANDIDATE_[CAUSALITY.md](http://CAUSALITY.md)

BEST_CANDIDATE_[ORACLE.md](http://ORACLE.md)

BEST_CANDIDATE_[METHODS.md](http://METHODS.md)

BEST_CANDIDATE_[INTRO.md](http://INTRO.md)

BEST_CANDIDATE_PRIOR_[ART.md](http://ART.md)

==================================================

42. Final response format

==================================================

FINAL STATUS:

...

TOTAL WALL TIME:

LIVE GPU TIME:

TOTAL 4-GPU HOURS:

==================================================

SERE

==================================================

BASELINE STATUS:

ORIGINAL CORE IDEA:

HIDDEN ASSUMPTIONS:

MLLM FAILURE FOUND:

YES / NO

QWEN:

KIMI:

FAILURE SIZE:

QUALITY-EFFICIENCY GAP:

CAUSAL EXPLANATION:

TRIVIAL FIX:

SUCCESSOR ORACLE:

CLOSEST PRIOR ART:

PAPER POTENTIAL:

SCORE:

==================================================

LIBRA

==================================================

same fields.

==================================================

MODES

==================================================

same fields.

==================================================

COMPARISON

==================================================

RANK 1:

...

RANK 2:

...

RANK 3:

...

WHY RANK 1 WINS:

...

==================================================

BEST CANDIDATE DEEP DIVE

==================================================

BASE PAPER:

FAILURE:

WHY MLLM CAUSES IT:

QWEN EVIDENCE:

KIMI EVIDENCE:

CAUSAL EVIDENCE:

FAILURE FREQUENCY:

QUALITY IMPACT:

EFFICIENCY IMPACT:

SUCCESSOR PERFECT ORACLE:

SUCCESSOR FEASIBLE ORACLE:

TRIVIAL FIX RESULT:

METHOD A:

METHOD B:

METHOD C:

MINIMAL PROTOTYPE RESULT:

QUALITY-MATCHED GAIN:

EFFICIENCY-MATCHED QUALITY RECOVERY:

PRIOR ART:

EXACT NOVELTY GAP:

MINI INTRO:

==================================================

FINAL DECISION

==================================================

FOUND_SUCCESSOR_STRONG_GO /

FOUND_INCREMENTAL_ONLY /

ALL_THREE_NO_GO /

ENVIRONMENT_BLOCKED

NEXT IMPLEMENTATION:

...

EXPECTED PAPER CONTRIBUTION:

...

WHAT COULD STILL KILL IT:

...

Branch:

Commit:

Push:

Report:

Results:
