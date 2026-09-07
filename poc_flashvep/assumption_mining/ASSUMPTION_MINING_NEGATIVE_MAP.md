# Assumption-mining negative map

Date: 2026-09-07
Branch: `flashvep/moe-ep-assumption-mining`
Primary runtime: Qwen3-VL-30B-A3B-Instruct, BF16, vLLM 0.20 V1,
TP2/DP2/EP4, DeepEP high-throughput, eager, DBO off, prefix cache off.

This map is a boundary for the new search.  It is deliberately conservative:
an execution anomaly is not promoted unless it has request-critical mass.

## Already tested or closed directions

| Direction / hidden assumption | Fresh evidence | Direct implication | Prior-art / interpretation |
|---|---|---:|---|
| Wave completion spread represents user latency | H06/H46/H47: wave effects +87--92%, request medians -2.3% to +0.1% | **Closed**; wrong metric | ordinary continuous-batching/API completion semantics |
| Fixed-shape DeepEP tail is removable MoE work | exact route replay has no giant tail; direct request tail mass 1.09% | **Closed** for an optimization paper | real dispatch wait, but economically small |
| Selective wait/drain has useful headroom | wait-aware PoC: coarse drain unstable, throughput cost, no robust E2E gain | **Closed** | synchronizing early can only move the wait |
| Per-token fanout adds latency information | Model2→Model3 held-out RMSE change +0.001% | **Closed** | DA-MoE/TEMPO-adjacent load representations sufficient here |
| Communication-SM configuration is a policy opportunity | SMS lower envelope median 0.11%, max 2.10% | **Closed** | static config gap below gate |
| Modality changes TP/EP optimum or expert granularity | controlled 4-GPU experiments null or token-volume explained | **Closed** | overlaps Moebius/HAP and modality scheduling work |
| Critical-rank coalescing | output-similarity/quality-preserving route did not leave robust headroom | **Closed** | token/expert coalescing is also crowded |
| Visual encoder and DeepEP naive overlap | prior negative: dispatch -12.4%, combine -5.0%, expert -8.9%; real handoff not stable | **Closed** | front-end overlap, not an EP execution contract |
| Fragmentation law / fanout sign flip | ordering/runtime-state confound; no stable causal transition | **Closed** | grouped-GEMM shape effects require isolated kernel study |
| Simple DP imbalance, pinning, dummy participation | request effects about -1.8% to +0.2%; H36 -0.26%, H43 -0.15% | **Closed** | standard DP load/rendezvous concern |
| Async scheduler or NCCL DP toggle is the cause | H31 -1.83%, H32 -0.03%; DeepEP remains active | **Closed** | config switch, not a structural paper problem |
| Host metadata materialization is high-mass | 287,219 calls; median 0.1137 ms, p99 0.145 ms, max 13.9 ms | **Closed as high-mass** | source TODO is interesting but tiny at request scale |
| Attention/MoE co-tail is EP-specific | old enrichment 14.2x, fresh overlap 0.0407% vs 0.25% expected | **Closed as EP method** | generic runtime/state candidate only |
| Output churn, request turnover, burst history | fresh request effects <=2.5%; common-mode transitions | **Closed** | wave/cohort state, not MoE work removal |
| Speculative downstream computation | quality/verification and oracle gates failed | **Closed** | SpecMoE-like space; no overlap mass |

## Measurement lessons carried forward

1. A rank-local CUDA span is not a request critical path until a logical
   invocation is joined once across ranks and then joined to request timing.
2. Same-device CUDA events are valid for duration; absolute event timestamps
   must never be subtracted across GPUs.
3. An observer can perturb an async path: the deferred observer still added
   3.27% request median in H01. New claims need no-hook or randomized paired
   controls.
4. A giant tail can be real and still contribute little mass. The fixed-shape
   tail's projected perfect E2E share is only 1.09% using the direct join,
   despite a 1.9 s rank-local event.
5. “Common state” moving both A/B arms is not a treatment effect. H45/H48
   are retained as state-observability facts, not optimization evidence.
6. Output budgets and wave completion semantics must be equalized before
   interpreting a large wave delta.

## Search boundary for this sprint

The remaining assumptions below are not declared false merely because the
source has an unusual implementation.  They are filtered by (a) whether the
counterfactual removes repeated structural work, (b) analytical request-level
mass, and (c) whether a trivial existing control would solve it.  A candidate
without a measured mass estimate is marked `UNKNOWN`, not promoted.
