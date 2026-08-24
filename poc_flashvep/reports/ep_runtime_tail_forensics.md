# Generic EP Runtime Latency-Tail Forensics

## Result

`RUNTIME_CONTEXT: NO-GO`

The fixed experiment did not reproduce a >=10% dynamic expert-latency tail under
serving-like vLLM/DeepEP execution. The exact-static-work tail frequency across all
layer/rank groups was 3.66% in controlled vLLM and 2.67% in serving-like vLLM. The
preselected layer-45/rank-0 workload had no >=15% tail in either live condition.
This run therefore does not establish either a vLLM-specific or a generic EP-runtime
tail mechanism.

## Scope and fixed policy

- Model: Qwen3-VL-30B-A3B-Instruct, BF16
- Runtime: existing vLLM 0.20 / DeepEP high-throughput environment
- Topology: TP2 / DP2 / EP4 / PP1 on physical GPUs 4,5,6,7
- DBO: off; prefix cache: off; eager execution: on
- Controlled: one active prompt on DP0
- Serving-like: two identical prompts on each DP rank (four active prompts)
- Warmup/measured iterations: 5/30 for each live condition
- Isolated replay: 20/100
- Tail definition, fixed before measurement: expert CUDA latency >= 1.15 times
  the median of the same context/rank/layer/exact histogram

The selection policy was fixed from the prior live-prefill trace before these runs:
among exact-histogram groups, select the group with the most >=15% tail events,
without using modality. It selected request `text_18_tui_main`, layer 45, EP rank 0
(prior trace: N=2978, G=30, Q=45, 7/15 tail events). The fresh controlled capture
and isolated replay both used N=2984, G=30, Q=45 and the exact same 32-expert
histogram. No threshold, workload, or pair was changed after observing results.

## Stage A — Repeatability and tail

### Preselected exact workload

| Context | N/G/Q | Histogram variants | Median (ms) | p95 (ms) | CV | >=15% tail |
|---|---:|---:|---:|---:|---:|---:|
| Isolated kernel | 2984/30/45 | 1 | 0.23546 | 0.24261 | 5.81% | 1.00% |
| Controlled vLLM | 2984/30/45 | 1 | 0.48043 | 0.49182 | 1.94% | 0.00% |
| Serving-like vLLM | 5936/30/66 | 1 | 0.55808 | 0.58491 | 2.60% | 0.00% |

Serving-like batching necessarily changes N/Q, so its absolute latency is not an
iso-work comparison with controlled. Repeatability and tail frequency are evaluated
within each condition at an exact, invariant histogram. The earlier 7/15 event did
not reproduce in the fresh 30-iteration controlled run.

### All 48 layers and four EP ranks

| Context | Median expert (ms) | p95 (ms) | Exact-work median CV | Exact-work p95 CV | Exact-work tail frequency | Critical-rank stability |
|---|---:|---:|---:|---:|---:|---:|
| Controlled vLLM | 0.61976 | 0.67168 | 3.20% | 25.52% | 3.66% | 74.03% |
| Serving-like vLLM | 0.55216 | 0.60372 | 2.77% | 21.63% | 2.67% | 69.38% |

High-CV groups exist, but the preregistered >=15% tail is uncommon and becomes less
frequent, not more frequent, in the serving-like condition. Critical-rank identity
is moderately unstable, but this alone is not evidence for a repeated large tail.

## Stage B — Context comparison

The isolated replay removes DeepEP and surrounding model execution and has a 1.00%
tail rate. That rate does not grow to >=10% in either live context. Controlled and
serving-like execution are not iso-N across contexts, but every repeated observation
inside a rank/layer/context has one exact histogram. Under that defensible comparison,
serving concurrency did not increase expert tail frequency or CV.

The absolute isolated/live latency difference must not be interpreted as runtime
contention alone: isolated timing invokes only the captured local expert operation,
whereas the live hook measures the complete TritonExperts expert-compute region in
the model runtime.

## Stage C — Lightweight runtime context

CUDA events were recorded on the compute and DeepEP communication streams without a
per-layer synchronization; elapsed times were resolved after each measured request.
No Nsight profiling was used.

- Dispatch/expert temporal overlap was 0 in both conditions.
- Expert/combine overlap >0.005 ms occurred in 45.87% of controlled observations
  and 99.48% of serving-like observations; median overlap was 0 and 0.10156 ms,
  respectively.
- In serving-like tail observations, dispatch duration was 17.35% higher than its
  exact-work normalized baseline and the previous-layer expert duration was 12.72%
  higher.
- In controlled tail observations, previous-layer expert duration was 14.83% higher,
  while dispatch was 8.34% lower. Thus no context factor had a consistent direction
  across both live conditions.

The strongest serving-only association is elevated dispatch duration, but it is not
a reproduced causal mechanism: expert tail frequency remained just 2.67%, and the
same dispatch relationship reversed in controlled execution.

## Critical-rank prediction

Predictions use request-iteration-grouped five-fold cross-validation. Models are
simple linear regressions. Runtime features are limited to dispatch timing and
already-completed previous-layer expert/combine timing; current expert latency and
post-expert combine timing are excluded from predictors.

| Predictor | Actual critical-rank accuracy |
|---|---:|
| N only | 37.40% |
| N + G | 39.31% |
| N + G + Q | 39.06% |
| N + G + Q + runtime context | 60.69% |

Runtime context adds +21.63 percentage points overall. The improvement is highly
condition-dependent: +43.06 pp in controlled (18.40% to 61.46%) but only +0.21 pp
in serving-like (59.72% to 59.93%). It demonstrates that dynamic timing carries
rank-identity information in this bounded repeated workload, but it does not rescue
the tail gate and should not be read as a general predictor result.

## Gate and interpretation

`RUNTIME_CONTEXT: NO-GO`

The required pattern was not observed:

1. exact-static-work >=15% expert tails were below 4%, not >=10%;
2. the preselected prior tail group produced 0/30 live tails in both fresh contexts;
3. serving-like concurrency reduced aggregate exact-work tail frequency and CV;
4. tail-correlated stage factors were weak and inconsistent across contexts.

The result does not prove dynamic context is irrelevant. It specifically rejects the
bounded hypothesis that this fixed workload and concurrency increase produce a
repeatable large expert-kernel tail. The current evidence supports neither an MLLM
claim nor a generic EP-runtime research direction.

## Limitations

- Serving-like uses four simultaneous requests but only one prompt and one batch
  construction; it is not a long-running arrival-process serving trace.
- Controlled and serving-like N/Q differ, so only within-context exact-histogram tail
  comparisons are causal for repeatability.
- CUDA-event spans expose stream timing but cannot attribute cache, CTA residency, or
  hardware counters.
- The isolated and live hook boundaries are not identical kernel-only scopes.
- Thirty live repetitions bound runtime, but rare sub-percent events need a longer
  trace to estimate precisely.

## Artifacts

- Raw result: `poc_flashvep/deepep_revalidation/results/ep_runtime_tail_forensics_20260824_141855/`
- Per-iteration metrics: `per_iteration_stage_metrics.csv`
- Critical-rank predictions: `critical_rank_predictions.csv`
- Machine-readable summary: `summary.json`
- Figures:
  - `figures/plot1_latency_distribution_by_context.png`
  - `figures/plot2_same_work_tail_events.png`
  - `figures/plot3_tail_vs_runtime_context.png`
  - `figures/plot4_critical_rank_prediction.png`

## Next single recommended action

Stop this runtime-tail direction and prioritize a different mechanism with a stable,
repeatable effect; do not add Nsight or an optimization based on this non-reproduced
tail.
