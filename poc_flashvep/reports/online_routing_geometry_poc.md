# Online distributed-routing-geometry PoC

**Date:** 2026-09-05  
**Branch:** `flashvep/online-routing-geometry-poc`  
**Decision:** **NO_GO** for a fanout-aware serving method

## Executive result

The premise is observable in the live runtime: the corrected read-only hook
captured 74,112 valid FusedMoE invocations from two real vLLM V1 request-wave
runs (mixed concurrency 8 and a high-burst concurrency 16 run).  Per-token
fanout spans F1--F4.  In prefill (M >= 100), mean fanout has p10/p50/p90 =
3.09/3.63/3.79; decode has a wider 1--4 range but M=1.  Thus natural
variation exists, but the serving data do not show incremental critical-path
information after the M, per-expert, and rank-load controls.  A
histogram-preserving route-transfer replay does show a sizeable but
non-monotonic difference, so it is a diagnostic anomaly rather than evidence
for an online controller.

## Configuration and evidence

| Item | Value |
|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct, local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Runtime | vLLM 0.20.0 V1, BF16, eager, chunked prefill |
| Parallelism | TP2 / DP2 / EP4 / PP1; linear placement; 32 of 128 experts per EP rank |
| Backend | DeepEP high-throughput (`DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`) + Triton Unquantized |
| GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical mapping recorded in each topology JSON) |
| Scheduler | `max_num_batched_tokens=8192`, `max_num_seqs=16`, DBO off, prefix cache off |
| T_MoE | CUDA-event duration of the unmodified FusedMoE apply, covering the stock router/prepare/dispatch/expert/finalize path; rank-local observation |

`online_trace3` is a real mixed online run with five waves at local
concurrency 4.  `online_trace_high2` is a second real run with four waves and
local concurrency 8 (burst regime).  Requests use real Qwen3-VL image inputs
and text controls; the normal vLLM V1 queue and scheduler are used.  The
first `online_trace` directory is retained as provenance but excluded because
its old hook recorded layer=-1.  M=4096 vLLM memory-profile dummy forwards
are also excluded from natural analysis.

The online driver records request waves rather than a private synthetic
operator call.  It does not expose vLLM scheduler iteration IDs or independent
dispatch/expert/combine events, so these are explicit limitations.

## Natural fanout and serving coverage

| Phase / source | Valid rows | M range | mean-fanout p10 / p50 / p90 | F4 range | T_MoE median / p90 / p99 (ms) |
|---|---:|---:|---:|---:|---:|
| Mixed online (trace3) | 37,056 | 1--622 | 1.00 / 3.00 / 4.00 | 0--1 | 0.837 / 0.932 / 1.765 |
| High burst (trace_high2) | 37,056 | 1--749 | 1.00 / 3.00 / 4.00 | 0--1 | 0.861 / 0.998 / 1.970 |
| **Prefill only, pooled** | **4,416** | **8--749** | **3.09 / 3.63 / 3.79** | **0--1** | **0.903 / 1.214 / 3.748** |
| **Decode only, pooled** | **69,696** | **1** | **1.00 / 1.50 / 4.00** | **0--1** | **0.849 / 0.972 / 1.832** |

`NATURAL_FANOUT_VARIATION_SUFFICIENT = YES` for a bounded diagnostic (F1--F4
is present and prefill p10--p90 is nonzero), but the useful prefill variation
is modest and decode's broad variation is at M=1.  It should not be used as
an uncontrolled claim of a large fanout regime.

### Prefill shape by M

Across layers and EP workers, the median mean fanout is stable for normal
prefill shapes: M=107 3.65, 114 3.70, 284 3.65, 321 3.65, 401 3.66,
505 3.66, 622 3.65, and high-burst M=749 3.64.  M=8 is a small warmup-like
shape with fanout 1.00.  This is enough coverage to test the null, but not a
large independent fanout sweep.

## Hierarchical explanatory models

Models are ordinary least-squares fits with a deterministic 70/30 time-block
split.  Rows are kept as worker observations; they are correlated repetitions,
not independent requests.  `Model 1` is the per-expert distribution control;
`Model 2` adds rank-load features; `Model 3` adds per-token fanout and the
conservative sender-destination geometry features.  Full values are in
`models/model_metrics.json`.

| Target / split | Model 0: M RMSE | Model 1: distribution RMSE | Model 2: +rank RMSE | Model 3: +fanout RMSE | Model 2→3 |
|---|---:|---:|---:|---:|---:|
| All valid rows (n=74,112) | 1.1189 | 1.1202 | 1.1202 | 1.1237 | **-0.31%** |
| Prefill (n=4,416) | 0.3532 | 0.5402 | 0.5528 | 0.5529 | **-0.02%** |
| Decode (n=69,696) | 1.1509 | 1.1509 | 1.1509 | 1.1541 | **-0.28%** |

The upper-1% trimmed sensitivity view is still null for decode (-0.11%) and
only +0.80% for prefill; it is not a preregistered positive gate.  The
negative/near-zero incremental value is stable in sign across phase splits.

## Matched natural pairs

The bounded pair search uses same layer, exact M, active experts within 8,
rank max/mean within 5%, and fanout separation >=0.25.  It emits 36
worker-level pairs from 4,032 eligible prefill rows.  Fanout delta p10/p50/p90
is 0.261/0.298/0.401.  Signed T_MoE delta p10/p50/p90 is -17.97/-2.26/+14.19%
and median absolute delta is 6.64%.  The sign is not stable enough for a
causal claim; rows are correlated worker repetitions and the online hook does
not separately timestamp dispatch/expert/combine.

The slow Model-2 residual tail is not enriched for fanout: for prefill rows
with M>=100, the slow residual decile has median fanout 3.620 and F4 0.636,
versus 3.654 and 0.665 for the remaining rows.  This is an **ONLINE_TAIL_ASSOCIATION:
NONE/OPPOSITE**, not evidence of a fanout tail signal.

## Histogram-preserving causal replay

This is a separate, bounded route-transfer diagnostic using a validated BF16
hidden-state capture at layer 24.  It swaps expert IDs between token rows,
preserving exact M=512, top-k=8, total assignments=4,096, every per-expert
histogram, active experts, and four rank assignment totals.  The route variants
are not generated by the online Qwen router and must not be interpreted as a
serving intervention.

| Case | mean fanout | wall/T_MoE (ms) | dispatch (ms) | expert (ms) | combine (ms) |
|---|---:|---:|---:|---:|---:|
| real_original | 3.518 | 2.169 | 0.269 | 0.872 | 0.250 |
| real_low_fanout | 2.537 | 2.424 | 0.264 | 0.891 | 0.260 |
| real_high_fanout | 4.000 | 2.051 | 0.237 | 0.899 | 0.167 |

Relative to low fanout, high fanout is 15.39% faster in full wall, dispatch is
10.37% lower, expert is 0.97% higher, and combine is 35.79% lower.  However,
high is also 5.46% faster than original while low is 11.74% slower than
original; the direction is therefore non-monotonic.  Layout/communication
state and measurement order are plausible explanations.  This replay is
valuable as an anomaly and a future controlled benchmark, but it does not
pass the online novelty gate.

## Causal interpretation and decision

- `DA_MOE_TEMPO_BASELINES_SUFFICIENT = YES`: M, per-expert histogram/active
  experts, and rank-load controls cover the null representation required by
  this PoC.
- `DISTRIBUTED_GEOMETRY_ADDS_NEW_SIGNAL = NO` in the real online trace:
  Model 2→3 held-out RMSE change is below zero (all, prefill, and decode).
- `HISTOGRAM_PRESERVING_REPLAY` is exact for the stated invariants, but its
  non-monotonic result and separate route-transfer process make it a
  **bounded diagnostic**, not a positive online result.
- DeepEP is verified from runtime logs; no synthetic-only evidence is used for
  the online conclusion.

**FINAL STATUS: NO_GO.**  Do not implement a fanout-aware scheduler or
rerouting method from this dataset.  The cheapest decisive follow-up, if the
question is revisited, is a larger controlled serving capture with true
global sender-destination matrices and independent dispatch/combine CUDA
events, plus a remeasured histogram-preserving pair in randomized order.

## Artifacts

- Online trace and route NPZ files: `poc_flashvep/deepep_revalidation/results/online_routing_geometry_poc_20260905_165525/online_trace3/` and `online_trace_high2/`
- Scalar analysis/model outputs: `online_invocations.csv`, `models/model_metrics.json`, `natural_fanout_summary.json`
- Matched pairs: `matched_pairs/summary.json`, `matched_pairs/pairs_detailed.json`
- Causal replay: `causal_replay/cases512/` and `causal_replay/causal_summary.json`
- Plots: `plots/` (M/fanout, expert distribution, residuals, model errors, matched pairs, replay, phase separation)
- Gate and provenance: `gate_summary.json`, `workload_manifest.json`, `source_audit.md`, `dependency_graph.{md,json}`, `resource_compatibility_matrix.{csv,md}`, `overlap_candidate_shortlist.md`

