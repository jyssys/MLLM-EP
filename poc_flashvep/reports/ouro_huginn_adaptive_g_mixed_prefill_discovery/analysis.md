# Adaptive recurrent-group prefill with native Mixed

Final promotion verdict: **NO_GO_INCREMENTAL** for the tested event-based fixed-hybrid path and workload states.
Both pretrained models execute legal grouped prefill and Mixed G1.
The measurements do not establish an incremental adaptive method beyond tuned native Mixed and fixed grouping.
This result does not prove that every possible state-aware policy must fail.

## Completed scope

Ouro uses Base 1.4B, BF16, Static R4. Huginn uses pretrained 3.5B, BF16, Static R16.
Ouro retains native async execution. Huginn retains its supported synchronous backend.
Huginn receives one neutral split/Mixed boundary adapter in every non-atomic arm.
C remains fixed within each comparison.
No dynamic model exit, training, approximation, new attention kernel, or TP/PP engine runs.

The fixed-matrix export contains 619 cohorts and 2025 request instances.
It contains 95823 delivered ITL observations.
These repeated mechanism prompts are not independent task-quality samples.
The main confirmation contains three process/GPU blocks per model and background width.
Each block crosses policy order and repeats its complete warm-up workload.

## Main clean confirmation

The table reports reference TTFT divided by candidate TTFT.
A value below 1.000× means a slower candidate.
Each value averages paired within-block ratios, not an aggregate serving-capacity ratio.

| Model | Background N | Fixed Hybrid G | TTFT speedup vs Mixed | Paired 95% interval |
| --- | --- | --- | --- | --- |
| huginn | 1 | 2 | 0.9837× | [0.9467, 1.0128] |
| huginn | 1 | 4 | 0.9966× | [0.9900, 1.0043] |
| huginn | 4 | 2 | 0.9999× | [0.9935, 1.0056] |
| huginn | 4 | 4 | 1.0055× | [1.0043, 1.0068] |
| ouro | 1 | 2 | 0.9945× | [0.9882, 1.0025] |
| ouro | 1 | 4 | 0.9693× | [0.9299, 0.9922] |
| ouro | 4 | 2 | 0.9967× | [0.9917, 1.0042] |
| ouro | 4 | 4 | 1.0128× | [1.0080, 1.0183] |

The small positive points do not establish the registered 3–5% incremental operating region.
The three-block intervals offer limited tail and population power.
They do not support a new two-model serving method.
See [all raw metrics](benchmark_summary.md) for the accompanying ITL and cohort makespan.

## Why nominal grouping often changes nothing

In the primary P2048/C512 interference screen, every target prompt group uses Mixed G1 while decode remains ready.
Ouro therefore executes 16 mixed prompt submissions. Huginn executes 64.
The target uses zero pure G>1 submissions in those Fixed Hybrid cases.
Changing the configured maximum G does not change that target's execution granularity.
The raw mode counts distinguish this fact from a selector score or a nominal configuration change.

With no ready decode, grouped prefill can reduce scheduler boundaries.
Best fixed grouping or Atomic already exposes that benefit.
Pure grouping during active decode changes priority and can increase delivered-token stalls.
Those TTFT–ITL trade-offs are not an additional adaptive goodput result.
The controlled interventions retain their state fingerprints and observer-pause limitations.

## Correctness, scale-out, and limits

Singleton and batched numerical probes use the original checkpoint weights.
Their sampled hidden/logit/KV comparisons pass without changing the registered tolerance.
The Full-depth ledgers report 0 prompt failures and 0 decode failures.
A heterogeneous prompt-remainder bug failed safely before GPU execution.
The scheduler now chooses a common legal group. The preserved regression and live DP retry check that fix.

DP1/2/4/8 portability uses real independent replicas and a common 32-request burst.
It is generic replication, not new model parallelism or qualified sustainable capacity.
The candidate is a fixed hybrid, not an unimplemented adaptive method.
GSM8K/MATH-500 accuracy and qualified Poisson/Burst capacity remain NOT_RUN.
The study does not claim quality-preserving publication success.

## Research decision

Do not build a larger adaptive optimizer from these measurements.
First find an independently reproduced state that beats both tuned Mixed and a strong fixed hybrid.
Then check task accuracy and natural open-loop attainment before claiming a method.
The paused native-vLLM Ouro Loop PP project remains paused.
The prior PoC verdicts remain unchanged.

## Positive-point follow-up: Huginn short-chat DP8

The library-warmed DP8 burst initially favors the fixed hybrid by 1.1234×.
That point receives additional independent starts and crossed arm order.
The follow-up also warms the exact chat prompts and their complete natural outputs before the offered window.
Under matched chat warm-up, the mean ratio is 0.9927×, with 95% interval [0.9439, 1.0254].
The matched condition uses 5 independent cluster starts per arm.
It still uses one repeated short-chat source family and no sustainable-rate qualification.

The initial positive point remains in the raw table.
The warm-up sensitivity limits the architecture claim.
The experiment does not isolate a single graph, JIT, or host component as the cause.
Neither result establishes adaptive superiority over the best fixed hybrid.

## Answers to the eight contract questions

| Question | Evidence-backed answer |
| --- | --- |
| Legal unchanged-BF16 composition on both models? | Yes within the tested source and probes. Huginn requires the neutral common boundary adapter |
| Which static combinations win? | Empty-decode grouping can reduce prompt boundaries. Active-decode hybrids usually execute the same target Mixed G1 plan. The short-chat DP8 positive point fails matched-warm confirmation |
| What causes the crossover? | Grouping changes prefill priority and yield frequency. Mixed changes packing. The remaining runtime outliers lack a complete graph/JIT/host attribution |
| Does an adaptive method change held-out plans and win? | NOT_RUN. No adaptive superiority claim follows from the fixed controls |
| Poisson/Burst attainment and qualified capacity? | NOT_RUN. No manufactured attainment curve or 90% capacity crossing |
| Correctness, Full R, and quality? | Recorded numerical and Full-depth checks pass. Public benchmark accuracy remains NOT_RUN |
| Does the execution path scale to DP2/4/8? | Yes in actual finite replica bursts. Generic replication is not a new parallelism contribution |
| What remains novel? | The evidence does not establish a new model-general adaptive scheduling contribution beyond the existing execution primitives and fixed controls |
