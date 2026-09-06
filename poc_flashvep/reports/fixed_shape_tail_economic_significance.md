# Fixed-shape DeepEP tail: economic-significance gate

Date: 2026-09-06  
Primary data: `deepep_wait_aware_tail_poc_20260906_124015/stock_run{1,2,3}`  
Secondary sensitivity: `fixed_shape_tail_root_cause_20260906_001917/online_trace`

## Decision

**FINAL DECISION: CONTINUE** under the requested optimistic-oracle rule.

The primary three-run wait-aware STOCK trace contains enough latency mass that
perfect, zero-cost removal of every positive fixed-group excess has a
wave-critical E2E upper-bound projection of **23.83%** using the optimistic but
reasonable matched-p25 baseline.  The p50 and trimmed-mean projections are
**16.11%** and **17.08%**, respectively.  Therefore the optimistic estimate is
above the 15% continuation gate.

This is an upper bound, not an achieved speedup.  The trace has no request ID
or invocation timestamp in the wave summary, so request-level TTFT/request
latency attribution is unavailable.  The E2E values below are explicitly a
critical-path/wave projection; a request-level join must be added before
claiming an end-to-end method result.

## Inputs and accounting

- Model: Qwen3-VL-30B-A3B-Instruct, local snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0 V1, BF16, TP2/DP2/EP4, DeepEP high-throughput,
  DBO off, eager path.
- GPUs: physical 1--4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`).
- Primary trace: three independent STOCK serving runs, 179,328 raw rank rows.
- Each route has two TP rows.  They were collapsed by `route_id` using the
  slower same-DP TP rank, never summed.  This leaves **89,664 valid logical
  MoE invocations**.  DP0 and DP1 remain separate logical serving timelines.
- Total observed logical T_MoE: **150,486.718 ms**.
- Phases present: decode M=1 and prefill; no mixed label.
- No cross-GPU absolute CUDA timestamps were subtracted.

The exact row accounting is in `analysis_manifest.json`.

## Definition

For each `(run, phase, layer, M)` group, the normal value is recomputed from
the same run.  For observation `i`:

`excess_i = max(0, T_MoE_i - matched_group_baseline_i)`.

`TAIL_EXCESS_MASS = sum(excess_i) / sum(T_MoE_i)`.

The primary baseline is matched p50.  Matched p25 and a 10% trimmed mean are
robustness/optimistic sensitivities.  Threshold rows only include observations
above that threshold; they do not treat the entire tail latency as removable.

## Tail frequency and latency mass (primary STOCK, p50 baseline)

| Threshold | Events | Event fraction | Raw latency sum (ms) | Excess (ms) | Raw fraction | Excess mass |
|---:|---:|---:|---:|---:|---:|---:|
| >5 ms | 675 | 0.7528% | 24,120.601 | 21,906.242 | 16.0284% | 14.5569% |
| >10 ms | 196 | 0.2186% | 20,760.960 | 19,563.261 | 13.7959% | 12.99999% |
| >20 ms | 88 | 0.0981% | 19,144.812 | 18,209.981 | 12.7219% | 12.1007% |
| >50 ms | 56 | 0.0625% | 18,128.708 | 17,350.354 | 12.0467% | 11.5295% |
| >100 ms | 40 | 0.0446% | 17,005.152 | 16,635.421 | 11.3001% | 11.0544% |

Only 0.098% of invocations exceed 20 ms, but they account for 12.10% of
logical MoE latency after retaining each group's normal baseline.  The full
positive-excess oracle is larger because it includes all fixed-group
deviations, not only the pre-registered extreme-tail threshold.

## Phase separation and fixed-shape decode anchor

| Phase | Logical invocations | >20 ms events/fraction | >20 ms excess mass | All-positive excess mass |
|---|---:|---:|---:|---:|
| Decode M=1 | 80,352 | 49 / 0.0610% | 12.0028% | 27.9497% |
| Prefill | 9,312 | 39 / 0.4188% | 12.8230% | 29.5784% |
| Mixed | unavailable | unavailable | unavailable | unavailable |

Decode M=1 is the fixed-shape giant-tail class.  Its maximum valid logical
T_MoE is 2,062.456 ms in the primary three-run set.  The old fixed-shape root
trace independently contains a 1,944.237 ms maximum; it is used only as a
sensitivity/reference artifact, not pooled into the primary estimate.

## Baseline robustness

| Normal baseline | All-positive MoE excess mass |
|---|---:|
| Matched p25 (optimistic) | **39.2871%** |
| Matched p50 (primary) | **28.1441%** |
| Trimmed normal mean | **26.9824%** |

The p25/p50/trimmed range is wide because asynchronous-state tails are rare
but very large.  This is why the 20/50/100 ms threshold rows and outlier
sensitivity are reported separately.

## Perfect-oracle counterfactuals

All numbers below use the matched-p50 baseline unless noted otherwise.

| Oracle | Counterfactual | MoE reduction | Wave-critical E2E projection |
|---|---|---:|---:|
| ORACLE-A-20MS | Replace every `T_MoE > 20 ms` with its group normal | 12.1007% | 9.6619% |
| ORACLE-B-TOP1 | Replace global top 1% (897 invocations) | 14.9825% | 10.9972% |
| ORACLE-C-ALL | Replace every positive matched-group excess | 28.1441% | **16.1091%** |

ORACLE-C by baseline sensitivity:

| Baseline | MoE reduction | Wave-critical E2E projection |
|---|---:|---:|
| p25 | 39.2871% | **23.8258%** |
| p50 | 28.1441% | **16.1091%** |
| trimmed mean | 26.9824% | **17.0811%** |

The wave-critical projection collapses DP0/DP1 rows by matching local
invocation/layer/phase/M and taking the critical DP value once.  Its total
observed wave wall is 181,084.828 ms across the three runs (DP wave files are
not double-counted).  The resulting critical MoE share is 50.4153%.

## Outlier sensitivity

| Dataset/baseline | With single maximum | Without single maximum |
|---|---:|---:|
| Primary STOCK, p25 MoE excess mass | 39.2871% | 38.4441% |
| Primary STOCK, p50 MoE excess mass | 28.1441% | 27.1463% |
| Primary STOCK, trimmed mean MoE excess mass | 26.9824% | 25.9685% |
| Primary STOCK, p25 wave-critical E2E projection | 23.8258% | **22.6894%** |

Removing the single 2.062 s sample does not remove the continuation signal in
the optimistic estimate.  In the older fixed-shape reference, p50 all-excess
mass is 15.4173% with its 1.944 s maximum and 12.7210% without it; this is why
the three-run wait-aware stock population is the primary decision dataset.

## Request-level impact and Amdahl limitation

Request-level affected fraction, extra latency per request, request p99 tail
contribution, TTFT, and TPOT are **NOT COMPUTABLE** from these artifacts:
`invocations.jsonl` has no request ID and `waves.dp*.json` has neither
invocation timestamps nor a join key.  This is recorded as
`BLOCKED_NO_REQUEST_ID`, rather than assigning the same rank event multiple
times or inventing request metrics.

Consequently, the 23.8258% number is an optimistic aggregate critical-path
upper-bound projection, not direct request-level E2E speedup.  A wave wall is
used only to provide the requested Amdahl-style scale estimate; the actual
request-level gate remains pending instrumentation.

## Interpretation

> Even with perfect zero-cost elimination of every fixed-shape positive excess,
> the maximum *projected* E2E benefit is 23.83% under the most optimistic
> reasonable baseline (22.69% after removing the single largest outlier).

The explicit 15% rule therefore says **CONTINUE SPILLOVER-PREVENTION**.  The
more restricted extreme-tail oracles alone are below 15% E2E (9.66% for all
`>20 ms`, 11.00% for top 1%), so the continuation is justified only by the
full perfect-oracle upper bound, not by a claim that a threshold policy has
already demonstrated that gain.

The prior wait-aware root-cause result remains consistent: dispatch/DeepEP
dependency spillover is the dominant removable component.  This gate measures
economic headroom only; it implements no synchronization policy or optimizer.

## Reproducibility

Analysis code: `poc_flashvep/fixed_shape_tail_economic_gate/analyze_economic_gate.py`  
Results: `poc_flashvep/deepep_revalidation/results/fixed_shape_tail_economic_significance_20260906_154520/`

Key files:

- `tail_excess_mass.csv`
- `phase_tail_excess.csv`
- `perfect_oracle.json`
- `outlier_sensitivity.csv`
- `analysis_manifest.json`
- `gate_summary.json`

No new GPU experiment or optimization method was run.
