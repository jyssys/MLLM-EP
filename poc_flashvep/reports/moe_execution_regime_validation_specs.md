# MoE Execution-Regime Validation Sprint

## Objective

이번 sprint는 새로운 optimization method를 구현하는 작업이 아니다.

최근 autonomous discovery에서 발견된 다음 현상을

1~2시간 동안 집중적으로 검증한다.

&gt; 동일 assignment/rank load에서도

&gt; M × active experts × EP fanout × layer에 따라

&gt; MoE CUDA latency의 방향이 뒤집힐 수 있다.

특히 기존 observation:

- M=128:

  F1 → F4 expert latency +104~114%

- M=512:

  F1 → F4 expert latency -12.4~-17.1%

이 sign flip이 실제 GPU execution regime인지,

아니면 warmup/order/kernel-state artifact인지 구분하는 것이

가장 중요한 목표다.

새 method는 구현하지 않는다.

==================================================

Environment

==================================================

GPU:

CUDA_VISIBLE_DEVICES=1,2,3,4

4× H100.

Primary model:

Qwen3-VL-30B-A3B-Instruct

Primary runtime:

TP2 / DP2 / EP4

DeepEP high-throughput

Generic validation이 필요한 strong finding에만:

Qwen3-30B-A3B

사용 가능.

기존 branch:

flashvep/autonomous-moe-discovery-loop

에서 이어가거나 별도 branch:

flashvep/moe-execution-regime-validation

사용.

==================================================

Measurement Protocol

==================================================

이번 sprint에서는 기존 first-use/order confound를 제거하기 위해

모든 controlled comparison에 동일 protocol을 적용한다.

각 shape에 대해:

1. process/runtime 초기화 상태 기록

2. global warmup

3. per-shape warmup

4. comparison order randomization

5. measured repetitions &gt;= 10

6. strong signal이면 &gt;= 30 reps

7. median primary

8. paired comparison

가능하면 다음을 기록:

- expert CUDA

- dispatch CUDA

- combine CUDA

- T_MoE

- wall secondary

- iteration index

- first-use flag

- layer

- M

- active experts

- fanout

- assignments/rank

- routing shape

- CUDA graph state if visible

- workspace/allocator marker if visible

- kernel identity if obtainable

Kernel identity를 직접 얻지 못하면 BLOCKED라고 기록하되

latency characterization 자체는 계속한다.

==================================================

H1. Exact Fanout × M Sign-Flip Boundary

==================================================

## Question

M=128과 M=512 사이에서 실제로

fanout effect의 sign이 바뀌는가?

## Sweep

M:

64

96

128

160

192

256

320

384

448

512

640

768

1024

Fanout:

F1

F2

F4

가능하면 동일:

- total assignments

- top-k

- rank load

- active experts

유지.

각 point &gt;=10 reps.

## Required plot

x:

M

y:

F2/F1 latency ratio

F4/F1 latency ratio

Expert / Dispatch / Combine / T_MoE 각각.

## Strong finding

fanout effect가 특정 M 구간에서

reproducibly 1.0을 crossing하고

양쪽 effect magnitude &gt;=5%.

## Very strong

한쪽 &gt;=10% penalty,

다른 쪽 &gt;=10% benefit.

Priority: CRITICAL.

==================================================

H2. Active Experts vs Fanout Disentanglement

==================================================

## Question

기존 effect가 정말 fanout 때문인가,

아니면 active expert count 때문인가?

둘을 분리한다.

### Experiment A

Active experts 고정,

fanout만 변경.

### Experiment B

Fanout 고정,

active experts만 변경.

Suggested:

A:

2 / 4 / 8 / 16

F:

1 / 2 / 4

대표 M:

128

256

512

1024

## Goal

다음 중 무엇인지 구분:

1. active-expert effect

2. fanout effect

3. interaction effect

## Strong finding

A 또는 F 단독 effect &gt;=5%,

또는 명확한 interaction/crossover.

Priority: CRITICAL.

==================================================

H3. Local Expert Kernel vs DeepEP Communication

==================================================

## Question

sign flip이:

- expert kernel 자체

인지

- DeepEP dispatch/combine

인지

구분한다.

동일 controlled shapes를:

A.

local / expert-only FusedMoE replay

B.

real DeepEP TP2/DP2/EP4

에서 비교.

대표:

M=128

M=256

M=512

M=1024

F1 vs F4.

## Interpretation

If:

local expert에서도 sign flip

→ grouped-GEMM/kernel regime 가능성 높음.

DeepEP에서만 sign flip

→ communication / packing / A2A geometry 가능성.

둘 다 있으나 magnitude 다름

→ phase interaction.

Priority: CRITICAL.

==================================================

H4. Alignment / Tile Boundary Probe

==================================================

## Hypothesis

M=128/512 sign flip이

power-of-two 또는 tile/kernel boundary일 수 있다.

## Sweep

다음처럼 경계 주변을 세밀하게 측정:

112

120

127

128

129

136

144

240

248

255

256

257

264

272

496

504

511

512

513

520

528

필요하면 1024 주변도.

Fanout:

F1

F4

## Look for

latency discontinuity:

127 → 128

128 → 129

255 → 256

etc.

## Strong finding

작은 M 변화(&lt;10%)인데

latency 또는 F4/F1 ratio가 &gt;=5~10% discontinuously 변함.

Priority: HIGH.

==================================================

H5. Layer Persistence under Standardized Warmup

==================================================

## Question

기존 layer-specific response가

runtime order artifact를 제거한 후에도 남는가?

Representative layers:

early

mid-early

mid

mid-late

late

예:

0 / 12 / 24 / 36 / 47

실제 MoE layer indexing에 맞게 조정.

For each:

M=128

M=512

F1

F4

모든 layer에 동일 warmup/order protocol 적용.

## Strong finding

같은 controlled shape인데 특정 layer group이

consistent하게 &gt;=10% 다른 response.

## Interpretation

persistent:

layer-dependent execution regime.

disappears:

previous layer effect was state/order confound.

Priority: HIGH.

==================================================

H6. Sender→Destination Geometry at Fixed Fanout

==================================================

## Question

fanout 수가 같아도

실제 sender→destination matrix shape가 cost를 바꾸는가?

Construct at fixed:

- total assignments

- per-rank load

- average fanout

but different traffic geometry:

A.

pair concentrated

B.

uniform

C.

cyclic

D.

one-source-heavy if feasible

Measure:

Dispatch

Combine

T_MoE

## Strong finding

same volume/fanout/rank-load에서

traffic geometry alone &gt;=5%.

Priority: HIGH.

==================================================

H7. Distribution Shape within Active Experts

==================================================

## Question

같은:

- total assignments

- active expert count

- rank load

- fanout

이어도 expert당 token 분포가 latency를 바꾸는가?

Example:

A:

uniform

[64,64,64,64]

B:

mild skew

[112,64,48,32]

C:

heavy skew

[192,32,16,16]

총량은 동일.

## Measure

Expert CUDA first.

## Strong finding

distribution shape alone &gt;=5%.

## Importance

단순 max-rank load로는 실제 GPU cost를 설명할 수 없음을

직접 증명할 수 있음.

Priority: HIGH.

==================================================

H8. Real-Route Transfer

==================================================

Synthetic controlled finding이 실제 model routing에서도 나타나는지 본다.

기존 real Qwen3-VL trace 또는 새 live routing에서

layer/invocation을:

- low-M / low-F

- low-M / high-F

- high-M / low-F

- high-M / high-F

bin으로 분류.

가능하면 matched-volume pairs를 만든다.

## Question

controlled microbenchmark에서 발견된 cost ordering이

real routing에서도 동일하게 나타나는가?

## Strong finding

synthetic regime prediction이 real route에서

&gt;=5% normalized latency 차이를 설명.

Priority: CRITICAL if H1-H3 positive.

==================================================

H9. TP/EP Crossover Boundary Beyond Token Count

==================================================

이전 TP↔EP load crossover 자체는 이미 known/crowded하므로

다시 증명하지 않는다.

대신 질문:

동일 token volume에서도

- active experts

- fanout

- distribution shape

가 TP vs EP relative advantage를 바꾸는가?

대표 M 2개만 사용.

예:

M=256

M=1024

각각 low/high fanout 또는 low/high active-expert shape.

Compare:

TP4/DP1

vs

TP2/DP2/EP4 DeepEP

## Strong finding

같은 M인데 route geometry만으로

TP/EP relative gain &gt;=5% 변화.

## Interpretation

token volume alone보다 richer crossover control signal이 존재.

Priority: MEDIUM.

==================================================

H10. Real MLLM vs Generic LLM Controlled Check

==================================================

강한 H1-H8 finding 하나만 선택.

Qwen3-VL과 Qwen3-30B에서

동일 controlled execution shape 비교.

## Question

phenomenon이:

generic MoE GPU regime인지

MLLM-specific인지

구분.

Full model serving comparison이 아니라

가능하면 controlled expert workload comparison.

Priority: OPTIONAL.

==================================================

Anomaly Generation

==================================================

실험 중 다음을 발견하면 즉시:

ANOMALY_xx

생성.

- non-monotonic latency

- sudden &gt;5% discontinuity

- sign flip

- specific layer only outlier

- same shape / different order &gt;5%

- dispatch/combine만 direction 반대

- unexpected TP/EP preference change

각 anomaly에 대해 최대 20분짜리 follow-up 하나 허용.

==================================================

Decision Tree

==================================================

Case A:

Sign flip disappears after proper warmup/randomization.

→ runtime-state artifact.

→ execution-regime direction NO_GO.

Case B:

Sign flip survives and local FusedMoE에서도 나타남.

→ GPU expert-kernel regime STRONG_GO candidate.

Next direction:

kernel-cost-aware MoE execution/routing.

Case C:

Sign flip survives only in DeepEP.

→ communication geometry direction.

Next:

DeepEP packing/fanout/traffic-shape analysis.

Case D:

Layer-specific response survives.

→ layer-conditioned MoE execution regime.

Case E:

Synthetic effect survives but real route transfer fails.

→ interesting microbenchmark only.

→ research priority LOW.

Case F:

Synthetic + real route both reproduce &gt;=10%.

→ STRONG_GO.

==================================================

Time Budget

==================================================

Target:

90 minutes.

Minimum meaningful exploration:

75 minutes wall-clock

unless hardware/runtime failure.

Maximum:

~120 minutes.

Suggested:

H1: 20 min

H2: 15 min

H3: 15 min

H4: 10 min

H5/H6/H7: 15 min

H8: 10 min

analysis/follow-up: remainder

H9/H10은 시간이 남을 때만.

중요:

초기 H1-H3가 negative라도

75분 이전에는 종료하지 않는다.

대신 H4-H7 또는 anomaly follow-up으로 이동.

GPU를 일부러 idle하게 두라는 뜻은 아니다.

남은 시간에는 더 많은 repetitions/control을 실행한다.

==================================================

Final Output

==================================================

FINAL STATUS:

STRONG_GO

GO

HOLD

NO_GO

MAIN RESULT:

SIGN_FLIP_REPRODUCED:

YES / NO

SIGN_FLIP_BOUNDARY:

...

FIRST_USE_CONFOUND_REMOVED:

YES / NO

PRIMARY_CAUSE:

EXPERT_KERNEL

DEEPEP_COMMUNICATION

INTERACTION

RUNTIME_STATE

UNKNOWN

ACTIVE_EXPERT_EFFECT:

...

FANOUT_EFFECT:

...

ALIGNMENT_BOUNDARY:

...

LAYER_EFFECT:

...

TRAFFIC_GEOMETRY_EFFECT:

...

REAL_ROUTE_TRANSFER:

YES / NO / NOT_TESTED

TP_EP_GEOMETRY_SIGNAL:

...

GENERIC_OR_MLLM_SPECIFIC:

...

BEST NEW RESEARCH QUESTION:

...

NEXT CHEAPEST DECISIVE EXPERIMENT:

...

DO_NOT_PURSUЕ:

...

Artifacts:

Branch:

Commit:

Push:

Report:

Results: