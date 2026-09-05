# Online MoE-EP Runtime Regime Discovery Sprint

## 0. Objective

이번 sprint의 목적은 새로운 method를 구현하는 것이 아니다.

목표는 4×H100 실제 online Qwen3-VL MoE serving 환경에서

"현재 static/default runtime configuration이 놓치는

재현 가능한 execution-regime phenomenon"

을 발견하는 것이다.

특히 다음 축을 탐색한다.

- DeepEP communication resource usage

- dispatch vs combine asymmetry

- all-to-all backend regime

- DP/EP synchronization

- online workload composition

- layer-specific runtime regime

- tail-latency anomalies

중요:

이미 실패한 routing-feature 아이디어를 다시 살리려고 하지 않는다.

이번 sprint의 질문은:

"What does the runtime itself do inefficiently

when the online MoE workload changes?"

이다.

==================================================

1. Environment

==================================================

GPU:

CUDA_VISIBLE_DEVICES=1,2,3,4

4× H100 only.

Primary model:

Qwen3-VL-30B-A3B-Instruct

Primary serving topology:

TP2 / DP2 / EP4

DeepEP active.

Primary:

real vLLM online serving.

Strong finding validation only:

Qwen3-30B-A3B optional.

==================================================

2. Prior closed directions

==================================================

다음을 새로운 research direction으로 다시 제안하지 않는다.

- plain routing imbalance / rank-token straggler

- modality-aware TP/EP

- dynamic TP↔EP load switching

- fanout-aware routing

- request complementary rebatching

- spatial chunking

- route-aware chunking

- critical-rank coalescing

- route-history warmth

- visual encoder × DeepEP naive overlap

- modality-aware expert granularity

- simple active-expert / fragmentation law

- per-expert histogram-aware kernel selection 자체

- token/activated-expert makespan balancing 자체

특히:

DA-MoE:

per-expert routing distribution → kernel selection

TEMPO:

token / activated-replica / actual GPU-cost-aware makespan

Moebius:

load-dependent TP↔EP switching

PROBE:

online compute + communication co-balancing

ELDR:

decode expert-locality-aware request routing

Layered Prefill / ZeRO-Prefill / ExpertPlex:

prefill/decode scheduling/disaggregation/resource specialization

과 동일한 phenomenon을 재발견하면

PRIOR_ART_COLLISION으로 표시한다.

==================================================

3. Mandatory Stage 0 — Timing sanity

==================================================

이전 online experiment에서 simple latency models가

held-out R² ≈ 0이었다.

따라서 새로운 hypothesis 이전에 timing pipeline을 검증한다.

Controlled identical serving workload:

same requests

same arrival pattern

same seed

same runtime

를 최소 20회 반복.

대표적인 동일 layer/invocation shape에서:

- Dispatch CUDA

- Expert CUDA

- Combine CUDA

- T_MoE critical span

CV 측정.

또한 검증:

T_MoE &gt;= each component

event ordering valid

layer/step/rank association valid

DP participant matching valid

prefill/decode labeling valid

CUDA asynchronous measurement contamination 없음

Gate:

median CV &lt;= 10%:

PASS

10~20%:

USABLE_WITH_CAUTION

&gt;20%:

FIX_INSTRUMENTATION_FIRST

단 instrumentation debugging에

45분 이상 쓰지 않는다.

해결되지 않으면 이후 실험은

paired/repeated measurement 중심으로 수행.

==================================================

4. Core hypothesis family A:

Communication resource regimes

==================================================

# H1. DeepEP Communication-SM Sensitivity

Hypothesis:

DeepEP communication에 할당되는 GPU SM resource와

expert GEMM compute 사이에 workload-dependent trade-off가 있다.

작은 workload에서는:

많은 communication SM이 불필요하거나 compute를 방해할 수 있음.

큰 workload에서는:

communication SM 부족이 dispatch/combine 병목을 만들 수 있음.

현재 DeepEP/runtime이 사용하는 SM setting을 source-level로 확인.

API/runtime이 허용하면 controlled sweep:

num_comm_sms:

minimum feasible

8

16

24

32

48

64

default

정확한 값은 현재 DeepEP implementation에 맞춰 조정.

Representative M:

64

128

256

512

1024

2048+

Measure:

Dispatch

Expert

Combine

T_MoE

Important:

같은 routing snapshot 사용.

Strong signal:

workload에 따라 optimal num_sms가 달라지고,

best-per-M vs best-static T_MoE gap &gt;=5%.

Very strong:

&gt;=10%.

If runtime version does not expose safe SM control:

BLOCKED

and move on.

Priority: VERY HIGH.

--------------------------------------------------

# H2. Dispatch vs Combine Resource Asymmetry

Question:

dispatch와 combine의 최적 communication resource configuration이

같은가?

H1에서 strong effect가 있다면:

Dispatch-only optimum

vs

Combine-only optimum

을 따로 측정.

Look for:

same num_sms cannot simultaneously optimize both phases.

Strong:

phase-specific optimum difference creates

&gt;=5% full T_MoE headroom.

Priority: HIGH.

==================================================

5. Core hypothesis family B:

Communication backend lower envelope

==================================================

# H3. All-to-All Backend Crossover Atlas

Current available backends를 source/runtime에서 확인.

Potential candidates:

- deepep_high_throughput

- deepep_low_latency

- allgather_reducescatter

- other actually-supported intranode backend

지원되지 않는 backend를 억지로 사용하지 않는다.

Separate engine runs are acceptable.

Runtime switching implementation은 금지.

Sweep online workloads:

scheduled M:

small → large

phase:

prefill

decode

mixed

concurrency:

1 / 2 / 4 / 8 / 16 where stable

Measure:

Dispatch

Combine

T_MoE

TTFT

TPOT

Question:

Does one static backend dominate?

or

Is there a meaningful lower envelope?

Strong finding:

Backend A wins some regime and B wins another,

with each side &gt;=5% advantage.

Important novelty warning:

"prefill prefers HT, decode prefers LL"

only reproduces existing DeepEP/vLLM guidance

and is NOT novel.

Interesting result requires something beyond that, e.g.:

- crossover within prefill workload range

- mixed-step regime not captured by phase label

- layer-specific crossover

- route/load feature determines optimum beyond M

- substantial oracle gap under colocated online serving

Priority: VERY HIGH.

--------------------------------------------------

# H4. Mixed-Step Backend Mismatch

Hypothesis:

같은 total scheduled M이라도:

pure prefill

pure decode

mixed prefill+decode

step이 communication backend에서 다른 비용을 보일 수 있다.

Create approximately matched M.

Example:

A:

512 prefill tokens

B:

512 decode tokens across requests if feasible

C:

256 prefill + 256 decode

Do NOT force impossible scheduler states.

Question:

Does phase composition matter after controlling M and routing load?

Strong:

same M에서 backend relative advantage or T_MoE differs &gt;=5%.

Prior-art risk:

HIGH.

Treat as diagnostic unless a new mechanism appears.

==================================================

6. Core hypothesis family C:

DP × EP synchronization

==================================================

# H5. Source-DP Asymmetry Amplification

vLLM DP+EP requires expert-layer synchronization across DP ranks.

Construct same global scheduled M but different DP-side source balance:

Balanced:

DP0 ≈ DP1

Asymmetric:

DP0 &gt;&gt; DP1

Keep as much as feasible:

global M

destination-rank workload

expert histogram

matched.

Measure:

per-DP arrival-to-MoE timing

dispatch

expert

combine

critical T_MoE

dummy participation

Question:

Does source-side DP imbalance amplify EP critical path

even if destination expert load is similar?

Strong:

&gt;=5% effect after destination-load controls.

Prior-art warning:

simple DP imbalance / synchronization is known.

Only a new nonlinear/uncaptured mechanism is interesting.

Priority: HIGH.

--------------------------------------------------

# H6. Empty/Dummy DP Participant Tax

vLLM explicitly performs dummy forward passes when one DP rank

has no scheduled requests but another is active.

Measure online:

A:

both DP ranks active

B:

one DP active, one dummy

Control useful workload as closely as possible.

Measure:

T_MoE

Dispatch

Combine

full request latency

Question:

What is the exact dummy-participant tax,

and is it nonlinear with M?

This is primarily characterization,

NOT automatically novel.

Interesting only if:

- unexpectedly large (&gt;10%)

- workload-dependent crossover

- interaction with backend or communication SM resources

Priority: MEDIUM.

==================================================

7. Core hypothesis family D:

Layer-conditioned runtime behavior

==================================================

# H7. Per-Layer Communication Optimum

Use identical controlled routing/workload shape

across representative MoE layers.

Test H1/H3 configurations at:

early

mid-early

mid

mid-late

late

Question:

Do identical shapes prefer different runtime configs by layer?

If YES,

inspect:

weight/layout

kernel selection

buffer state

preceding attention timing

communication state

Strong:

configuration ranking changes across layers

with &gt;=5% effect.

If all layers identical:

REJECT.

Priority: HIGH.

==================================================

8. Core hypothesis family E:

Online tail latency origin

==================================================

# H8. Tail Event Decomposition

Do NOT fit another large predictor first.

Collect real online serving.

Take T_MoE:

fastest 20%

middle 20%

slowest 5%

within matched:

same phase

similar M

same/similar layer

similar expert workload

Compare:

dispatch

expert

combine

DP wait

rank arrival skew

previous-step duration

idle-gap length

backend/kernel state

Goal:

find what phase creates unexplained tail.

Question:

When routing/load is matched,

what actually causes the slow tail?

Generate ANOMALY hypotheses from results.

Priority: VERY HIGH.

--------------------------------------------------

# H9. Cross-Rank Tail Co-occurrence

For every slow T_MoE event:

Which ranks are slow?

Classify:

single-rank slow

multiple-rank correlated slow

global step slow

If tail is mostly globally correlated:

routing imbalance is unlikely cause.

If single-rank:

inspect local compute/communication.

Look for:

rank identity persistence

phase dependence

GPU clock/utilization if cheap

communication queue state

Strong:

stable category explains &gt;=10% tail gap.

Priority: HIGH.

==================================================

9. Core hypothesis family F:

Online arrival-pattern effects

==================================================

# H10. Burst vs Steady at Matched Average Load

Construct:

Steady:

regular request arrivals

Bursty:

short high-concurrency bursts + gaps

Match approximately:

total requests

total tokens

average request rate

Question:

Does EP runtime show hysteresis/tail amplification

under bursty arrivals beyond current M?

Measure:

TTFT

TPOT

T_MoE

first-step after idle

dispatch/combine

Important:

DeepEP first-use-after-idle effects have prior reports.

Do not claim novelty from a simple first-call latency spike.

Interesting only if:

- sustained multi-step effect

- backend/resource-dependent

- not explained by known first-use warmup

Priority: MEDIUM.

==================================================

10. Meta hypothesis:

Static runtime configuration oracle

==================================================

# H11. Runtime-Configuration Oracle Headroom

This integrates H1-H10.

For every sampled serving regime, consider all safely measured configs:

- communication SM allocation

- all-to-all backend

- other non-semantic runtime knobs discovered

Do NOT include TP↔EP topology switching.

Calculate:

BestStatic =

single config minimizing total latency over workload

OraclePerRegime =

best config for each workload regime

OraclePerInvocation =

offline best config per sampled invocation if meaningful

Report:

BestStatic / OraclePerRegime

BestStatic / OraclePerInvocation

Decision:

&lt;3% oracle headroom:

NO_GO for adaptive runtime configuration.

3~5%:

WEAK/HOLD.

5~10%:

PROMISING.

&gt;=10%:

STRONG_GO candidate.

This is one of the most important sprint outputs.

==================================================

11. MLLM specificity check

==================================================

# H12. Generic-vs-MLLM Check

ONLY run if H1-H11 produces a strong finding.

Validate one phenomenon on:

Qwen3-30B-A3B text-only.

Question:

Does phenomenon require multimodal serving?

Do not spend GPU time on this if no strong primary result.

==================================================

12. Autonomous hypothesis generation

==================================================

The seed list H1-H12 is NOT a checklist that ends the sprint.

Maintain:

ACTIVE_[QUEUE.md](http://QUEUE.md)

At all times, unless within final 45 min,

keep &gt;=6 pending hypotheses.

Every completed experiment must produce:

- one explanation if positive

- one "what assumption failed?" question if negative

- up to two follow-up hypotheses

Automatically create ANOMALY_xx when observing:

- sign flip

- &gt;5% non-monotonic curve

- same workload / different runtime state &gt;5%

- backend ranking reversal

- dispatch and combine move opposite directions

- one layer behaves differently

- only one rank repeatedly causes tail

- predicted static winner loses &gt;10%

Each ANOMALY follow-up:

&lt;=25 min.

==================================================

13. Novelty gate

==================================================

Before promoting any result to GO,

search current literature if internet access exists.

At minimum compare against:

- DA-MoE

- TEMPO

- Moebius

- PROBE

- ELDR

- ExpertPlex

- Layered Prefill

- ZeRO-Prefill

- DeepEP / vLLM documented backend behavior

If internet is unavailable:

do NOT claim novelty.

Record search keywords for later review.

Classify:

NEW_ORTHOGONAL

ADJACENT_EXTENSION

LIKELY_PRIOR_ART

PURE_CHARACTERIZATION

==================================================

14. Time budget

==================================================

Target wall time:

4 hours.

Minimum exploration:

3 hours,

unless hardware/runtime failure.

Do not intentionally waste GPU.

But do NOT finish in 60–90 minutes just because

seed hypotheses were easy to reject.

Suggested:

0:00–0:30

timing sanity + environment

0:30–1:30

H1/H2 communication-resource regime

1:30–2:20

H3/H4 backend regime

2:20–3:00

H5-H9 online synchronization/tails

3:00–3:30

strongest anomaly follow-ups

3:30–4:00

H11 oracle + replication + report

If H1/H3 are blocked,

immediately reallocate time to H8/H9 and generated anomalies.

==================================================

15. GPU utilization policy

==================================================

Track:

wall time

live GPU experiment time

Each hour.

We do not require artificial GPU utilization.

But if the last hour contains &lt;15 minutes of live measurements

because the agent only analyzed old artifacts,

generate a new controlled live experiment.

New research claims require new live measurements.

==================================================

16. Method implementation forbidden

==================================================

Do NOT implement:

- dynamic backend switcher

- dynamic SM controller

- new scheduler

- RL

- new routing

- expert placement optimizer

- token dropping/merging

- production CUDA kernel

Oracle / controlled config sweep only.

First prove headroom.

==================================================

17. Outputs

==================================================

Branch:

flashvep/online-runtime-regime-discovery

Report:

poc_flashvep/reports/

online_moe_ep_runtime_discovery_[report.md](http://report.md)

Scoreboard:

poc_flashvep/reports/

online_moe_ep_runtime_discovery_scoreboard.csv

Results:

poc_flashvep/deepep_revalidation/results/

online_moe_ep_runtime_discovery_&lt;timestamp&gt;/

Maintain:

ACTIVE_[QUEUE.md](http://QUEUE.md)

EXPERIMENT_[LOG.md](http://LOG.md)

[ANOMALIES.md](http://ANOMALIES.md)

CHECKPOINT_[1H.md](http://1H.md)

CHECKPOINT_[2H.md](http://2H.md)

CHECKPOINT_[3H.md](http://3H.md)

==================================================

18. Final report

==================================================

TOTAL WALL TIME:

TOTAL LIVE GPU TIME:

TOTAL 4-GPU HOURS:

TIMING_SANITY:

PASS / CAUTION / FAIL

HYPOTHESES TESTED:

NEW HYPOTHESES GENERATED:

ANOMALIES INVESTIGATED:

TOP FINDING 1:

...

TOP FINDING 2:

...

TOP FINDING 3:

...

COMM_SM_REGIME:

...

DISPATCH_COMBINE_ASYMMETRY:

...

BACKEND_CROSSOVER:

...

MIXED_STEP_EFFECT:

...

DP_EP_SYNCHRONIZATION:

...

DUMMY_RANK_TAX:

...

LAYER_CONFIG_VARIATION:

...

TAIL_ROOT_CAUSE:

...

BEST_STATIC_CONFIG:

...

ORACLE_PER_REGIME:

...

ORACLE_HEADROOM:

...%

BEST RESEARCH DIRECTION:

...

SECOND BEST:

...

PRIOR_ART_COLLISIONS:

...

DO_NOT_PURSUE:

...

NEXT_DECISIVE_POC:

...

Branch:

Commit:

Push:

Report:

Scoreboard:

Results: