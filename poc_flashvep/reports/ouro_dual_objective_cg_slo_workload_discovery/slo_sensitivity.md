# Frozen SLO sensitivity

The primary SLO is TTFT≤5s and every ITL≤200ms, with a complete response.
The frozen secondary grid crosses TTFT2/5/10s with TBT100/150/200/250/500ms.
All thresholds evaluate the same measured output clocks.
They do not retune the controller, whose exposure reference remains200ms.
Secondary comparisons are exploratory and use pointwise process-block intervals.
They do not replace the registered primary promotion gate.

## Request-level and token-level tails differ

The serial lane contains three starts and36 requests per policy/workload.
No response is capped in these cohorts.
The following table uses TTFT≤5s and the stricter every-gap TBT≤100ms condition.
P99 and maximum entries are means of cohort quantiles, not a single pooled quantile.

| Workload | Policy | Joint attainment % | Pooled-gap ITL P99 ms | Request-maximum ITL P95 ms | Recorded gaps>100ms |
| --- | --- | ---: | ---: | ---: | ---: |
| MIXED | Joint |44.44 |62.81 |125.94 |33 |
| MIXED | G-only C2048 |58.33 |58.86 |107.81 |15 |
| MIXED | G-only C1024 |61.11 |53.99 |93.37 |14 |
|8K stress | Joint |50.00 |71.68 |129.80 |18 |
|8K stress | G-only C2048 |100.00 |71.56 |79.35 |0 |
|8K stress | G-only C1024 |100.00 |63.59 |73.70 |0 |

An average pooled P99 below100ms does not imply that every request satisfies every100ms gap.
Rare gaps can affect many requests while remaining below1% of all token gaps.
The8K Joint result demonstrates this difference: P99≈71.68ms, but every-gap attainment is50%.
The G2048 control has nearly the same pooled P99 and no100ms violations in this sample.
The exposure proxy therefore does not establish complete-token SLO protection.

At TTFT5s/TBT100ms, Joint/G2048 goodput is0.7540× in MIXED and0.4970× in8K stress.
At the primary5s/200ms condition, the ratios are0.9921× and0.9939×.
Across the full serial absolute grid, no Joint/G2048 point ratio exceeds1.000×.
This is a result for the frozen200ms-reference controller, not every possible objective retuning.
The policy also fails the registered primary incremental-performance criterion.

## Calibrated multiplier tiers

The decode reference is the median of24 warmed cohort-median ITLs per context.
It is not a pooled-token median or a complete-next-token deadline predictor.

| Reference context | Reference ms |5× ms |10× ms |25× ms |
| --- | ---: | ---: | ---: | ---: |
|2048 |25.6968 |128.4841 |256.9682 |642.4205 |
|8192 |43.8347 |219.1735 |438.3470 |1095.8675 |

`frozen_multiplier_SLO_results.csv` applies these prespecified tiers without a new GPU run.
Each row retains its workload, lane, arrival parameter, and all-offered denominator.
The absolute and multiplier grids are separate views of the same delivered-token records.

## Capacity limit

The characterization figure contains two registered arrival parameters, not a qualified capacity sweep.
Each arm/cell has36 offered requests across three starts.
The specification requires at least1,000 offered requests per arm/rate and stable queue evidence.
The current finite cohorts cannot establish maximum sustainable rate,90%-attainment capacity, or long-run queue stability.
Those results remain NOT_RUN because the stronger matched baseline already prevents promotion.

Data: `SLO_sensitivity_paired.csv`, `SLO_sensitivity_scope.json`, `slo_lock.json`, and `frozen_multiplier_SLO_results.csv`.
FiguresE/H show finite attainment and correlated token-gap tails, respectively.
