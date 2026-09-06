# Autonomous EP research discovery v3

## Executive research judgement

**FINAL STATUS: SEARCH_SPACE_EXHAUSTED_NO_GO**

This sprint added fresh, real Qwen3-VL V1 serving measurements on physical
GPUs 1–4 and two short generic-Qwen3 controls. It found a strong but
state-confounded multimodal boundary anomaly: one text→high-resolution-vision
transition doubled layer-local T_MoE and increased wave E2E latency, while
shape-matched warmup, vision-only alternation, and a telemetry-tagged
replication removed it. The reproducible high-mass effects that remain are
static batching/warmup configuration effects, not a non-trivial method gap.
No candidate simultaneously reached the required direct `>=15%` perfect E2E
oracle, causal robustness, and blue-ocean method potential. The prior
fixed-shape tail branch remains closed by its direct request-level cap (1.09%).
No production method was implemented.

## Accounting

- Wall interval: 17:21:49–20:46 KST (about 3 h 25 min including model loads,
  controls, and analysis).
- Fresh live runs: 20 completed Qwen3-VL traces/controls plus two completed
  Qwen3 dense repetitions and six bounded diagnostics/failures. The long
  matched text control alone contains 12,288 logical observations; the
  primary aggregate atlas has 90,240 Qwen3-VL logical rows after excluding
  the aborted text-c2 partial trace.
- Approximate live GPU time: **about 150 wall-minutes × 4 = 10 GPU-hours**.
  This is a conservative sum of model-load and serving intervals; no GPU 0,
  5, 6, or 7 process was touched.
- Model snapshot: Qwen3-VL-30B-A3B-Instruct,
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0 V1, BF16, TP2/DP2/EP4, DeepEP high-throughput,
  Triton unquantized MoE, linear placement, DBO off, prefix cache off, eager.
- All runs used `CUDA_VISIBLE_DEVICES=1,2,3,4`; physical GPU telemetry was
  collected only for those devices. GPU clocks ranged from 345 to 1980 MHz in
  the telemetry-tagged controls, so clock state is treated as a confound rather
  than an inferred utilization metric.

## Fresh E2E latency mass atlas

The compact atlas is in the result directory under `discovery_atlas/`.
Rank-local worker rows were collapsed by DP/local invocation/layer/phase using
the maximum same-DP CUDA span; rank rows were never added, and warmup rows were
excluded when `active_wave` was present. The hook does not provide exact
request-to-layer joins, so stage sums are explicitly labelled MoE-stage mass,
not E2E speedups.

| Regime | Requests | E2E p50 / p90 / p99 (ms) | Measured T_MoE p50 (ms) | Dominant normal stage mass |
|---|---:|---:|---:|---|
| text c4 | 32 | 443 / 807 / 892 | 1.171 | expert 33.8%, dispatch 21.1%, wait 9.0% |
| text c8 | 160 | 656 / 1,030 / 1,944 | 1.168 | expert 31.0%, dispatch 30.4%, wait 3.3% |
| text c8 rep2 | 96 | 970 / 1,005 / 3,364 | 2.056 | expert 24.9%, dispatch 41.1%, wait 16.1% |
| text c16 | 256 | 773 / 1,759 / 2,194 | 1.399 | expert 34.9%, dispatch 28.4%, wait 14.6% |
| vision-hi c8 | 160 | 758 / 1,645 / 7,618 | 1.218 | expert 42.2%, dispatch 15.1%, wait 7.9% |
| mixed c8 | 224 | 901 / 1,116 / 6,235 | 1.164 | expert 38.4%, dispatch 17.8%, wait 6.0% |
| mixed c8 MBT4096 | 84 | 1,278 / 1,865 / 2,298 | 2.055 | expert 22.4%, dispatch 48.9%, wait 12.0% |
| mixed c2 | 48 | 286 / 390 / 1,149 | 1.209 | expert 33.1%, dispatch 29.3%, wait 12.1% |

Fresh maxima are still dispatch-dominant (up to 2.64 s), but the historical
request-joined fixed-shape analysis measured only 1.09% aggregate removable
request mass. Repeating the same anomaly without exact request joins would
violate the headroom-first rule.

## New C1c shape-transition control

The highest apparent headroom came from a deliberate warmup/transition
control, not from a new scheduler. With a text warmup followed by the fixed
high-resolution vision template, T_MoE p50 was 2.395 ms and event-wait p50
0.334 ms; the same vision shape after high-resolution vision warmup was 1.204
ms/0.028 ms. Alternating text and vision stayed at about 2.12 ms/0.49 ms,
whereas alternating two text shapes was 1.21–1.25 ms/0.02–0.04 ms and
alternating two vision shapes was 1.16 ms/0.029 ms. A 32-wave text
shape-matched run (12,288 logical rows) remained stable at 1.175 ms p50.

This is a real operational diagnostic: a multimodal boundary can leave a
large DeepEP-visible state penalty in one runtime state. However, an
independent telemetry-tagged text→vision replication returned to 1.20 ms T_MoE
and 0.04 ms event wait (64 requests), while GPU clocks ramped from 345 to 1980
MHz. The effect therefore fails the sprint's reproducibility requirement and
is best explained by a coupled warmup/cache/DVFS state. Matching the shape is
an obvious static preconditioning fix, so C1c is `CLOSED`, not a paper
finalist. Full per-wave data are in `discovery_atlas/shape_transition_analysis.json`
and `SHAPE_TRANSITION_CONTROL.md`.

## Top observations and killed branches

### 1. Concurrency changes the phase regime, but is a throughput trade-off

The same fixed text template at c4/c8/c16 gave request p50 443/656/773 ms and
request-rate proxies 0.32/0.52/0.77 requests/s. Layer-local c8→c16 p50 rose
1.168→1.399 ms (+19.8%), while event-wait share rose 3.3→14.6%. A second c8
run showed 2.056 ms p50, with dispatch/event-wait dominating, and the partial
c8 rep3 showed p99 47 ms. The exact same-M bins are not stable across runs
(c8 M=114 p50 1.162 ms vs rep2 2.073 ms), so this is a runtime-state anomaly,
not a causal method result. Lowering concurrency is the obvious fix and cuts
throughput; it is not a blue-ocean contribution.

**Status:** `PROMISING_ANOMALY` → `CLOSED` as a paper direction; retain C1a/K1a
for an exact request-join study only if a future scheduler project needs it.

### 2. Visual input changes expert share, not a new MoE cost law

At c8, high-resolution vision raises expert share from 31.0% (text) to 42.2%
and T_MoE p50 from 1.168 to 1.218 ms (+4.3%). E2E p50 rises 15.5%, but that
gap includes vision preprocessing/embedding and different scheduled M. No
matched token/layer intervention isolates an MoE effect. This is not an
MLLM-specific research direction.

**Status:** `HOLD` → `CLOSED` without a direct MoE oracle.

### 3. Fanout and load representations remain null

Across every fresh trace, the time-block model using M, expert distribution,
rank load, fanout mean and F4 fraction has held-out R² approximately zero;
adding fanout changes RMSE by -0.6% to +0.07% in the small traces and at most
about +0.001% in the large clean trace. This re-confirms the prior
DA-MoE/TEMPO-style null and is deliberately not re-framed as novelty.

**Status:** `CLOSED`.

### 4. A multimodal boundary anomaly is high-magnitude but not robust

The fresh transition controls above produce a 2.0× T_MoE difference in one
run and a ~47% wave-level E2E difference relative to shape-matched vision.
Vision-only and text-only alternation are normal, and an independent
telemetry-tagged replication is normal. This is exactly the pattern expected
from a shape/cache/DVFS state transition that static warmup or bucketing can
avoid. It is recorded as `ANOMALY-06` and explicitly rejected as a blue-ocean
method direction.

**Status:** `CLOSED` (`STATE_CONFOUNDED`, `TRIVIAL_ENGINEERING`).

### 5. Max-batched-token budget causes a large but trivial regime change

The same varied request templates at c8 with `max_num_batched_tokens=8192`
versus 4096 produced the same M bins, but M=114 T_MoE p50 increased
1.155→2.133 ms; dispatch increased 0.126→0.834 ms and event wait
0.018→0.273 ms. Request E2E p50 increased 901→1,278 ms while the request-rate
proxy stayed near 0.52→0.50 req/s. This is a useful causal control showing
that scheduler token budget can alter DeepEP phase geometry, but it is an
obvious static configuration knob, exactly the kind rejected by the
non-triviality gate.

**Status:** `TRIVIAL_ENGINEERING`, not a paper core.

### 6. Dispatch-only outliers are real but low-mass for requests

Fresh traces reproduce dispatch maxima while expert/combine remain normal,
consistent with the previous DeepEP asynchronous-state diagnosis. The prior
direct request join capped perfect fixed-tail removal at 1.09%; the new hook's
wave labels cannot improve that bound. This branch fails the direct-headroom
gate and is not revived.

**Status:** `CLOSED`.

### 7. Residual mining finds ordinary high-M shape, not a hidden orthogonal feature

The final 90,240-row residual miner's top 1% is enriched by about +183 tokens
and +10 active experts, with small rank-CV/fanout shifts but no stable
orthogonal variable across trace splits. This is the
expected shape/kernel explanation and fails the orthogonal-variable gate.

**Status:** `CLOSED`.

## Direct headroom gate

For a typical request, 48 MoE layers × 1.16–1.40 ms normal layer-local span
is roughly 56–67 ms. Against fresh request p50 values 443–970 ms, the normal
MoE share is about 6–15% before accounting for overlap and unmeasured
pre/post-MoE work; the realistic text/vision regimes are 6–10%. A method that
only removes the observed fixed-shape tail is strictly below this and was
already capped at 1.09% in the exact request-joined trace. Thus no tested
MoE-only candidate reaches the required 15% direct perfect oracle, and no
candidate reaches a feasible >=12% oracle.

The c4→c16 latency curve does not change this conclusion: its 74% request-p50
increase is accompanied by a 2.4× request-rate increase, so treating
concurrency reduction as removable MoE waste is an invalid counterfactual.

## Causal controls and limitations

- Real model and real image/text requests, not synthetic routing, were used.
- DeepEP activation was verified in every clean run by logs (`DeepEPHTPrepareAndFinalize`,
  EP enabled, 32 local/128 global experts).
- All timing is same-device CUDA-event timing. Cross-GPU absolute timestamps
  were never subtracted.
- `active_wave` separates warmup from measured rows. An aborted c2 text and a
  c8 rep3 partial are retained as bounded diagnostics, not headline runs.
- The hook lacks attention-range and exact vLLM request/layer joins. Therefore
  CPU queue, vision encoder, and attention effects are not attributed to MoE.
- One c8 replication had a different GPU clock state at launch (GPUs 1–4 at
  345 MHz before ramp) and substantially different stage waits. This exposes a
  measurement/hardware-state confound; it is not promoted without a controlled
  clock/telemetry intervention.

## Held-out model check

`discovery_atlas/model_comparison.json` recomputes the descriptive time-block
70/30 split on all 90,240 Qwen3-VL logical rows.  Model 0 (`M`) has RMSE
0.9884 ms; adding active-expert/expert-CV features gives Model 1 RMSE 0.9965
ms; adding rank-load features gives Model 2 RMSE 0.9980 ms; and adding mean/F4
fanout gives Model 3 RMSE 0.9983 ms.  Model 2→3 changes RMSE by **−0.032%**
(slightly worse).  Phase-separated fits are similarly null (fanout changes
RMSE by +0.031% in decode and +0.036% in prefill).  These are held-out
descriptive checks, not a claim that a linear model is a production predictor.

## Top-10 discovery scorecard

| # | Finding | Effect / repetition | Control and disposition |
|---:|---|---|---|
| 1 | Concurrency changes phase geometry | text c8→c16 T_MoE +19.8%, request p50 +17.8%; c2/c4/c8/c16 plus repeats | throughput/load confound; closed as trade-off |
| 2 | MBT budget changes same-M cost | M=114 T_MoE 1.155→2.133 ms; request p50 901→1,278 ms | static `max_num_batched_tokens`; trivial engineering |
| 3 | One text→vision state transition | 2.395 vs 1.204 ms T_MoE; wait .334 vs .028 ms | shape-matched warmup and telemetry replication null; closed |
| 4 | Vision shifts expert share | 42.2% vs text 31.0% at c8; T_MoE only +4.3% | no matched MoE-only oracle; descriptive |
| 5 | Dispatch-only extreme maxima persist | fresh dispatch up to 2.64 s while expert/combine remain normal | prior request-joined removable mass 1.09%; closed |
| 6 | Fanout is not incremental | Model 2→3 RMSE −0.032% aggregate; phase deltas <0.04% | held-out time split; null after load controls |
| 7 | Residual tail is ordinary shape | top 1% +183 M and +11.5 active experts | no orthogonal feature; closed |
| 8 | Layer heterogeneity is shallow | p50 ratio 1.18× prefill / 1.25× decode | low mass, no stable intervention; closed |
| 9 | Generic Qwen3 control is slower but consistent | two fixed-text reps T_MoE ≈2.11–2.15 ms, wait ≈.42–.44 ms | separate protocol/model; no pooled claim |
| 10 | Low-latency backend unavailable here | both attempts fail before serving at NVSHMEM QP-depth assertion | unsupported configuration, not a performance result |

The scorecard is intentionally conservative: the largest effects are either
configuration/throughput trade-offs or fail independent replication. No row
passes the direct-oracle, causal-control, and non-triviality gates together.

## Research tree outcome

The maintained tree contains 16 measured/diagnostic nodes across dependency, overlap,
continuous-batching, interference, temporal state, communication structure,
expert/runtime, layer heterogeneity, modality, data-centric and scheduler
families. Ten active-frontier candidates remain documented, but all nodes with
measured data either fail the direct headroom gate, are a throughput trade-off,
or collide with existing continuous-batching/overlap work. No finalist reached
the `>=15%` perfect oracle + causal + generality promotion rule.

## Prior-art adversarial screen

The concurrency/phase-composition observation is adjacent to asynchronous EP
and batching systems, including [AEP/AMoE](https://arxiv.org/abs/2505.08944),
[EPS-MoE](https://arxiv.org/abs/2410.12247), [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html),
and [Layered Prefill](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html).
Those works already motivate load-dependent scheduling, asynchronous expert
parallelism, model-data co-scheduling, or prefill/decode composition. The
fresh data does not establish an orthogonal causal variable.

## Final research judgement

There is no paper-level STRONG_GO direction in the tested Qwen3-VL EP4
regime. The most useful result is a falsification: normal expert/dispatch
phase mix is workload-sensitive, but the only large request-level curve is
the familiar throughput-versus-latency batching trade-off, while the
previously exciting DeepEP tail has negligible request-level mass. A future
project should first add exact request/layer joins and GPU clock/queue
telemetry; it should not implement a dynamic MoE method based on these traces.
