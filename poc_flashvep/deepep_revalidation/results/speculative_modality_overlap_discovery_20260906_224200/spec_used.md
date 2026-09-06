# Speculative Modality-Aware MoE Overlap Discovery

## Tree-Structured Phenomenon Discovery Before Method Design

==================================================

0. Mission

==================================================

이번 sprint의 목표는 특정 method를 처음부터 구현하는 것이 아니다.

목표는:

MLLM MoE Expert Parallel inference에서

"strict expert-completion dependency를

quality-safe speculation으로 완화하여

새로운 compute-communication overlap opportunity를

만들 수 있는가?"

를 체계적으로 탐색하는 것이다.

단 하나의 가설만 검증하지 않는다.

다음 broad research space에서

high-headroom / high-novelty phenomenon을

tree-structured하게 발견한다.

- partial expert completion

- speculative hidden-state execution

- expert-surrogate approximation

- missing-residual prediction

- modality-dependent tolerance

- layer-dependent tolerance

- token-dependent tolerance

- spatial redundancy

- cross-token residual similarity

- cross-layer residual reuse

- late-expert criticality

- verification / fallback

- speculative depth

- newly-created overlap window

- EP critical-path impact

==================================================

1. Core research question

==================================================

Current MoE execution:

Dispatch

→ all routed experts complete

→ Combine

→ exact MoE output

→ next computation

Question:

Must every token wait for every routed expert contribution

before dependent computation can begin?

More specifically:

Are Vision tokens, certain layers, certain routing states,

or certain expert contributions sufficiently tolerant to

partial/approximate MoE outputs that we can:

partial completion

→ provisional hidden state

→ speculative downstream compute

while:

late remote expert computation continues in parallel?

==================================================

2. Important distinction

==================================================

This is NOT primarily an:

- expert skipping project

- expert pruning project

- rerouting project

- expert similarity project

Those can be diagnostic baselines.

The intended systems contribution, if one exists, is:

"Use approximation/speculation to relax a strict distributed

execution dependency and CREATE an overlap window

that did not previously exist."

The method must ultimately target critical-path reduction,

not merely lower FLOPs.

==================================================

3. Hardware

==================================================

Use only:

CUDA_VISIBLE_DEVICES=1,2,3,4

4× H100.

Do not touch GPU 0/5/6/7.

Primary discovery:

Qwen3-VL-30B-A3B-Instruct

Primary topology where feasible:

TP2 / DP2 / EP4

DeepEP HT.

Any strong phenomenon must eventually be checked on:

Kimi MoE MLLM

because final research goal requires cross-model generality.

Do not spend Kimi GPU time on weak hypotheses.

==================================================

4. SpecMoE relationship

==================================================

Read and summarize:

SpecMoE

arXiv:2604.10152

Important reusable concepts:

- partial/draft MoE execution

- unavailable expert approximation

- expert-affinity surrogate

- target-model verification

- self-assisted speculation

But do NOT claim the following as novel:

"replace an unavailable expert with a similar expert."

SpecMoE already uses expert affinity for draft execution.

Our potential novelty must instead come from:

- intermediate hidden-state speculation

- distributed EP dependency relaxation

- compute/communication overlap

- modality-dependent speculation

- partial expert completion

- critical-path-aware acceptance

Explicitly maintain:

SPECMOE_OVERLAP_[MATRIX.md](http://MATRIX.md)

Columns:

SpecMoE concept

Our use

Exact overlap

Important difference

Novelty risk

==================================================

5. Absolute research gates

==================================================

Interesting approximation quality alone is insufficient.

Interesting overlap alone is insufficient.

Candidate must eventually show all three:

QUALITY TOLERANCE

+

REAL OVERLAP WINDOW

+

DIRECT E2E HEADROOM

==================================================

6. Headroom rules

==================================================

For any promoted candidate calculate:

Perfect speculative E2E oracle.

&lt;10%:

KILL.

10–15%:

WEAK/HOLD unless novelty is exceptional.

&gt;=15%:

PROMISING.

&gt;=20%:

HIGH PRIORITY.

Then calculate feasible oracle including:

- surrogate computation

- verification

- fallback probability

- recomputation

- launch overhead

- imperfect overlap

Feasible E2E oracle:

&lt;10%:

KILL.

10–12%:

WEAK.

&gt;=12%:

FINALIST eligible.

&gt;=15%:

STRONG.

==================================================

7. Quality rules

==================================================

Do not use representation cosine alone

to claim quality preservation.

Evaluation hierarchy:

Level 1:

hidden-state metrics

- cosine similarity

- relative L2

- residual norm error

Level 2:

next-layer propagation

- next hidden cosine

- next hidden L2

Level 3:

model-output metrics

- logit KL

- top-1/top-k prediction agreement

- next-token probability change

Level 4:

real benchmark accuracy/quality

Only Level 3/4 can support strong research claims.

Early cheap probes may use Level 1/2.

==================================================

8. Quality budgets

==================================================

Report trade-offs at multiple budgets.

Suggested:

near-lossless:

benchmark loss &lt;=0.1%p if measurable

strict:

&lt;=0.5%p

moderate:

&lt;=1.0%p

exploratory:

&lt;=2.0%p

Do not choose one threshold because it makes the result look best.

Report Pareto curves:

quality loss

vs

E2E oracle

vs

acceptance rate.

==================================================

9. Research tree

==================================================

Maintain:

RESEARCH_[TREE.md](http://TREE.md)

ACTIVE_[FRONTIER.md](http://FRONTIER.md)

Node states:

UNTESTED

CHEAP_PROBE

PROMISING

DEEP_DIVE

FINALIST

STRONG_GO

QUALITY_FAIL

HEADROOM_FAIL

NOVELTY_FAIL

CAUSAL_FAIL

CLOSED

Final 60 minutes 전까지,

unless finalists already exist:

ACTIVE_FRONTIER &gt;= 8 nodes.

Seed hypotheses are only initial roots.

==================================================

10. Root Family A:

Partial Expert Completion

==================================================

Exact output:

y = sum_{i=1..K} g_i E_i(x)

Study provisional outputs using only subsets.

Examples:

top-1

top-2

top-4

top-6

top-7

top-k by router mass threshold.

Test variants:

A1.

plain partial sum

A2.

renormalized partial sum

A3.

router-mass threshold

Example:

proceed once cumulative router mass &gt;=:

0.70

0.80

0.90

0.95

0.99

Questions:

Does quality depend more on expert count

or cumulative router mass?

Does Vision differ from Text?

Does layer matter more than modality?

==================================================

11. Root Family B:

SpecMoE-Inspired Expert Surrogates

==================================================

For missing/late expert E_j:

approximate with another cheap/available expert.

Candidate affinities:

B1.

weight-space nearest expert

B2.

output-space nearest expert from calibration traces

B3.

router-coactivation affinity

B4.

local-rank nearest expert

B5.

modality-conditioned expert affinity

Important:

This is a BASELINE / ingredient,

not automatically a research contribution.

Questions:

Does surrogate improve provisional hidden-state quality

enough to increase speculative acceptance?

Does its extra compute destroy overlap benefit?

==================================================

12. Root Family C:

Missing Residual Prediction

==================================================

Represent:

y_exact =

y_ready + r_missing

Predict:

r_hat_missing

Explore cheap predictors only.

C1.

zero residual

C2.

scaled ready residual

C3.

previous-layer residual

C4.

same-token previous MoE residual

C5.

moving-average residual by layer/expert/modality

C6.

small linear predictor

C7.

tiny MLP only if simpler candidates show strong headroom

The goal is not ML prediction accuracy alone.

Metric:

quality gained per added speculative latency.

==================================================

13. Root Family D:

Vision Spatial / Token Redundancy

==================================================

Prior observations suggest local spatial routing coherence.

Use this only as motivation,

not as proof.

Explore:

D1.

neighboring vision token residual as surrogate

D2.

2D spatial-block mean residual

D3.

same-expert neighboring token output reuse

D4.

representative vision-token residual

D5.

image-region-conditioned surrogate

Compare against:

random vision token

text-token surrogate

global mean

Questions:

Can Vision-specific redundancy make

missing expert contributions more predictable?

This is potentially stronger MLLM specificity.

==================================================

14. Root Family E:

Cross-Layer Reuse

==================================================

Explore whether missing expert residuals

are predictable from nearby layers.

E1.

L-1 residual

E2.

L-2 residual

E3.

early/mid/late layer-specific reuse

E4.

same expert across neighboring layers if meaningful

Do not assume expert identities correspond semantically

between layers.

Measure empirically.

==================================================

15. Root Family F:

Critical vs Non-Critical Expert Contribution

==================================================

Not all routed experts are equally important to:

quality

or

EP critical path.

For each token-expert assignment estimate:

QUALITY VALUE:

impact on hidden/logit quality if approximated.

SYSTEM VALUE:

potential reduction in critical-path completion time.

Investigate mismatch.

Potential important phenomenon:

Two assignments have similar quality importance,

but one is on the critical remote completion path

and creates far more overlap opportunity.

Define:

Speculative Value

≈

recoverable E2E time / quality cost

This may be more novel than simple router-score pruning.

==================================================

16. Root Family G:

Modality-Dependent Tolerance

==================================================

Primary MLLM hypothesis:

Vision tokens may tolerate provisional MoE outputs

better than Text tokens.

But do not force this result.

Compare:

Vision

Text

special/control tokens

at matched:

layer

router mass

expert count

M_e where useful

hidden norm

router entropy

Questions:

G1.

At equal top-k fraction, which modality has lower error?

G2.

At equal quality loss, which modality allows earlier completion?

G3.

At equal quality loss, which modality offers more E2E overlap?

G4.

Does modality effect disappear after controlling router confidence?

If YES:

do not claim modality causality.

==================================================

17. Root Family H:

Layer-Dependent Speculation

==================================================

Potentially only some layers are tolerant.

Test:

early

mid-early

mid

mid-late

late

Questions:

Can a sparse subset of highly tolerant layers

produce most available overlap?

Example:

Only 12/48 layers speculative,

but recover 70% of oracle benefit.

This could yield a clean method.

==================================================

18. Root Family I:

Token-Adaptive Speculation

==================================================

Potential predictors:

router entropy

top1/top2 gap

cumulative top-k mass

hidden-state norm

modality

spatial location

expert locality

remote-rank count

late-rank identity

layer

previous approximation error

Goal:

predict:

"Is it safe to speculate this token?"

But do not train a complex classifier early.

First determine whether a simple signal exists.

==================================================

19. Root Family J:

Speculative Depth

==================================================

Possible scopes:

J0.

No downstream speculation.

Only provisional output analysis.

J1.

Speculate next RMSNorm / QKV projection.

J2.

Speculate next attention.

J3.

Speculate next full transformer block.

J4.

Multiple blocks.

Critical question:

How far can approximate hidden state propagate

before verification becomes too expensive?

Measure:

quality drift

fallback cost

overlap headroom

Likely optimal depth may be shallow.

==================================================

20. Root Family K:

Verification / Fallback

==================================================

This is central.

Speculation is useful only if verification is cheap.

Candidate verification:

K1.

hidden relative error threshold

K2.

cosine threshold

K3.

router-mass confidence

K4.

surrogate-vs-partial disagreement

K5.

sampled exact check

K6.

layer/modality static confidence table

Questions:

Can acceptance be determined

without waiting for all expensive work?

If verification itself requires exact completion

and provides no useful overlap,

record this as a fundamental barrier.

==================================================

21. Root Family L:

Asynchronous Correction

==================================================

Explore whether exact late residual can be

incorporated without full recomputation.

Do not assume it is possible.

Diagnostic questions:

L1.

Is downstream transformation locally linear enough

for residual correction?

L2.

Can correction be applied before attention softmax?

L3.

Can only QKV projection be corrected cheaply?

L4.

Can low-rank Jacobian approximation correct result?

L5.

Is recomputation cheaper than correction?

This is high-risk/high-reward.

Do not spend large time unless earlier gates are strong.

==================================================

22. Root Family M:

Actual Overlap Opportunity

==================================================

This family is mandatory.

Approximation quality without real critical-path slack

must be killed.

For each candidate estimate:

T_exact_ready

T_provisional_ready

Delta_ready =

T_exact_ready - T_provisional_ready

Downstream speculative compute duration:

T_spec_compute

Maximum ideal overlap:

min(Delta_ready, T_spec_compute)

Then subtract:

surrogate cost

verification cost

fallback/recompute expectation

extra synchronization

launch overhead.

Compute:

PERFECT_OVERLAP_ORACLE

and

FEASIBLE_OVERLAP_ORACLE.

==================================================

23. Do not fake overlap

==================================================

Success is NOT:

"combine latency decreased."

Success requires:

request-level critical path decreases.

Waiting moved earlier:

FAIL.

Approximate compute added but fully serial:

FAIL.

Speculation runs concurrently but causes enough contention

to erase E2E gain:

FAIL.

Always measure:

TTFT

TPOT / ITL

request latency

throughput

when progressing to live method PoC.

==================================================

24. Completion-order analysis

==================================================

Try to determine:

Which expert/rank contributions actually arrive late?

Questions:

Are low-router-weight experts disproportionately late?

Are late contributions usually remote?

Does Vision have more removable late contribution mass?

Is completion ordering stable enough to predict?

If actual per-expert completion timestamps are unavailable:

use bounded instrumentation or controlled replay.

Do not subtract absolute timestamps across GPUs.

==================================================

25. Surprise-mining rules

==================================================

The sprint must actively search for unexpected findings.

After every experiment ask:

1.

Did modality matter?

2.

Did layer matter more than modality?

3.

Did router mass matter more than top-k?

4.

Did surrogate identity matter?

5.

Did spatial locality matter?

6.

Was quality good but headroom bad?

7.

Was headroom good but quality bad?

8.

Did a simple statistic strongly predict safety?

9.

Did an unexpected token/layer class dominate?

10.

Did the result contradict SpecMoE-style affinity intuition?

Any reproducible unexpected effect:

&gt;=10% representation/target-stage difference

or

&gt;=5% E2E oracle difference

may create child nodes.

But low-headroom anomalies must not consume hours.

==================================================

26. Autonomous child generation

==================================================

Each completed node produces:

EXPECTED

OBSERVED

FAILED ASSUMPTION

NEW VARIABLE

HEADROOM IMPLICATION

Then create 0–2 children.

Examples:

If Top-4 Vision works only in late layers:

→ child:

late-layer-only speculation

If affinity surrogate fails but spatial residual works:

→ child:

modality-spatial surrogate family

If Text unexpectedly more tolerant:

→ reclassify hypothesis.

Do NOT force MLLM story.

If router-mass dominates modality:

→ generic adaptive speculation candidate.

==================================================

27. Breadth / depth policy

==================================================

First 2–3 hours:

broad cheap probes.

Each candidate:

15–30 min maximum initially.

Promote only candidates with:

quality signal

+

possible &gt;=15% perfect E2E oracle.

Then select top 2–3 for deep dive.

Do not run all seed hypotheses blindly

once strong candidates appear.

==================================================

28. Candidate score

==================================================

Score 0–5:

Novelty

Quality tolerance

E2E headroom

Generality

MLLM specificity

Overlap systematicity

Non-triviality

Method cleanliness

Evidence

Total /45.

Promotion:

&gt;=28:

DEEP_DIVE

&gt;=34:

FINALIST

&gt;=38:

STRONG_GO candidate.

Headroom &lt;4/5:

cannot be STRONG_GO.

==================================================

29. Novelty gate

==================================================

Any PROMISING candidate must be compared against:

SpecMoE

speculative decoding

speculative MoE inference

partial expert execution

expert skipping

dynamic top-k

expert pruning

MoE approximation

asynchronous MoE

ScMoE

FarSkip-Collective

MLLM modality-aware expert reduction

relevant current papers discovered during search

Ask:

"What existing paper would a reviewer cite

to say this is already done?"

Record in:

PRIOR_ART_[MATRIX.md](http://MATRIX.md)

==================================================

30. Strong novelty preference

==================================================

Stronger potential contribution:

"Modality-dependent quality tolerance makes

strict EP completion unnecessarily conservative,

and relaxing this dependency creates a new

overlap dimension."

Weaker contribution:

"Vision tokens can skip more experts."

Very weak:

"Similar experts can substitute for each other."

==================================================

31. Engineering triviality gate

==================================================

Finalists must survive simple baselines:

top-k reduction

expert skipping

renormalized top-k

SpecMoE affinity substitute

static Vision top-k

existing runtime overlap

simple re-routing

Question:

Does speculative dependency relaxation give

additional benefit beyond simply doing less work?

If simple skipping gets essentially the same

quality/speed Pareto:

speculative-overlap contribution is weak.

==================================================

32. Generality

==================================================

Any FINALIST must be validated on:

Qwen3-VL

AND

Kimi MoE MLLM

at least at phenomenon/oracle level.

Need not implement full runtime method twice initially.

Report:

same direction?

same layer pattern?

same modality trend?

same headroom order?

If Qwen only:

downgrade.

==================================================

33. MLLM specificity gate

==================================================

A strong MLLM claim should ideally show:

Vision tolerance

≠

Text tolerance

AND

this difference enables a materially larger EP overlap window.

If both Vision/Text behave the same:

reclassify as generic MoE speculative execution.

That can still be valuable.

Do not kill a strong generic finding.

==================================================

34. Strong-GO gates

==================================================

Research STRONG_GO requires:

1.

Reproducible quality-tolerance phenomenon.

2.

Direct perfect E2E oracle &gt;=15%.

3.

Feasible overlap oracle &gt;=12%.

4.

Cross-model generality.

5.

No exact prior-art collision.

6.

Not equivalent to simple expert skipping.

7.

Clear verification/fallback path.

8.

At least one minimal mechanism PoC

moves real E2E in the predicted direction.

Preferred minimal live signal:

&gt;=5% actual E2E improvement

with plausible path to &gt;=12%.

Final method-level target remains:

&gt;=12% real E2E.

==================================================

35. Immediate kill conditions

==================================================

KILL-A:

Vision/Text partial outputs both highly inaccurate.

KILL-B:

Good approximation but overlap oracle &lt;10%.

KILL-C:

Speculation requires full recomputation so often

that feasible oracle &lt;10%.

KILL-D:

Simple top-k skipping dominates quality/speed Pareto.

KILL-E:

SpecMoE/other prior art already covers

same mechanism + same purpose.

KILL-F:

Phenomenon appears only in one model.

KILL-G:

Actual overlap causes contention and erases headroom.

==================================================

36. Time budget

==================================================

Target:

6–10 hours.

Can extend to 12 hours

if strong unexplored branches remain.

Suggested:

Hour 0–1:

instrumentation + exact output collection sanity

Hour 1–3:

A/B/C/G/H broad probes

Hour 3–5:

spatial/cross-layer/surrogate surprise mining

Hour 5–7:

top 2–3 headroom + propagation deep dives

Hour 7–9:

Kimi generality + prior art

Hour 9–10+:

minimal overlap/verification PoC if warranted

Do not intentionally burn GPU.

==================================================

37. Required artifacts

==================================================

Branch:

flashvep/speculative-modality-overlap-discovery

Main report:

poc_flashvep/reports/

speculative_modality_overlap_[discovery.md](http://discovery.md)

Maintain:

RESEARCH_[TREE.md](http://TREE.md)

ACTIVE_[FRONTIER.md](http://FRONTIER.md)

QUALITY_PARETO.csv

OVERLAP_ORACLE.csv

SPECMOE_OVERLAP_[MATRIX.md](http://MATRIX.md)

PRIOR_ART_[MATRIX.md](http://MATRIX.md)

EXPERIMENT_[LOG.md](http://LOG.md)

[SURPRISES.md](http://SURPRISES.md)

For finalists:

FINALIST_1_[PHENOMENON.md](http://PHENOMENON.md)

FINALIST_1_[HEADROOM.md](http://HEADROOM.md)

FINALIST_1_[METHOD.md](http://METHOD.md)

FINALIST_1_[INTRO.md](http://INTRO.md)

and similarly for finalist 2.

==================================================

38. Final report

==================================================

FINAL STATUS:

FOUND_STRONG_GO

or

SEARCH_SPACE_NO_GO

TOTAL WALL TIME:

TOTAL LIVE GPU TIME:

TOTAL 4-GPU HOURS:

TREE NODES:

CHEAP PROBES:

PROMISING:

DEEP DIVES:

FINALISTS:

--------------------------------

CORE PHENOMENON

--------------------------------

Vision vs Text partial tolerance:

...

Layer effect:

...

Router-mass effect:

...

Spatial effect:

...

Cross-layer effect:

...

--------------------------------

BEST APPROXIMATION

--------------------------------

Plain partial:

...

Renormalized:

...

SpecMoE affinity:

...

Spatial surrogate:

...

Residual predictor:

...

Other surprising winner:

...

--------------------------------

QUALITY

--------------------------------

Hidden-state error:

...

Next-layer error:

...

Logit KL:

...

Benchmark quality:

...

--------------------------------

OVERLAP

--------------------------------

Exact completion:

...

Provisional completion:

...

Created overlap window:

...

Perfect E2E oracle:

...%

Feasible E2E oracle:

...%

--------------------------------

VERIFICATION

--------------------------------

Acceptance criterion:

...

Acceptance rate:

...

Fallback rate:

...

Fallback cost:

...

--------------------------------

GENERALITY

--------------------------------

Qwen:

...

Kimi:

...

Generic vs MLLM:

...

--------------------------------

PRIOR ART

--------------------------------

Closest work:

...

SpecMoE difference:

...

Exact novelty gap:

...

--------------------------------

BEST CANDIDATE

--------------------------------

ONE-LINE IDEA:

...

WHY SURPRISING:

...

WHY MLLM:

...

WHY MOE EP:

...

WHY OVERLAP:

...

WHY NOT EXPERT SKIPPING:

...

WHY NOT SPECMOE:

...

EXPECTED REAL E2E:

...%

METHOD PRINCIPLE:

...

PAPER INTRO STORY:

...

--------------------------------

FINAL DECISION

--------------------------------

FOUND_STRONG_GO / SEARCH_SPACE_NO_GO

Branch:

Commit:

Push:

Report:

Results: