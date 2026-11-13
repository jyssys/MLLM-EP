# Dense AR Dynamic Layer PD: cost model and validation limits

## Model contract

The scheduler uses current Host state and frozen empirical costs.
It does not read future arrivals or future completion times.
It chooses a Cut before submitting the corresponding P prefix.
P remains a full-model replica.
D remains a full-model replica and executes all 36 layers during Decode.

The candidate set is Last0/2/4/8/12.
P executes a true prefix.
D consumes the imported Hidden, Residual, and prefix KV.
D then executes the remaining layers.
After Commit, no worker discards a prefix or silently repeats its layers.

The [frozen cost table](dense_ar_dynamic_layer_pd_crossover_20261113/configs/calibrated_cost_model.json) has SHA256:

`cff7e0f7ab6c33a9a0a0740df86b151d201fe145336fdb8dfd127b826157dae5`

Its calibration uses 80 distinct prompts.
These prompts combine the calibration and discovery pools.
They exclude the held-out pool.
Output2 is a primitive diagnostic, not an Online Serving workload.

## Measured primitive costs

Evidence level: `MEASURED_DIAGNOSTIC_ENCLOSURES`.
The following table uses source batch4 and Input8000.
Each shape has four samples.
This sample count does not establish a worst-case or validated p99 bound.

| D layers | P prefix completion median, ms | Handoff Host median, ms | D suffix event p95, ms |
|---:|---:|---:|---:|
| 0 | 1112.85 | 221.30 | 0.00 |
| 2 | 1049.50 | 244.37 | 58.78 |
| 4 | 989.13 | 248.67 | 118.56 |
| 8 | 856.81 | 208.66 | 239.19 |
| 12 | 731.79 | 187.33 | 358.58 |

P prefix completion combines a CUDA event enclosure with the subsequent Source-ready Host interval.
Handoff includes import, scheduling, metadata, and transport effects.
The Native handoff boundary ends at the released first token.
The partial handoff boundary ends at suffix model submission.
These boundaries support a completion-path model.
They are not exclusive wire-time measurements.

D suffix costs are CUDA event stream enclosures.
They are not exclusive kernel unions.
Commit-to-first-token p95 is approximately 1.94–2.07ms for batch4.
Do not add overlapping telemetry fields to infer exclusive CPU or GPU costs.

## Decode drain prediction

Prepared Decode calibration used the same HND layout and full Decode Graph family.
Measured step p95 values were 26.2574/26.9176/27.7955/30.6027ms for batch1/4/8/16.
These values include Host delivery behavior.
They are not physical Idle Budgets.

The original policy uses the largest calibrated bound for every cohort.
The Batch Lookup variant uses the next measured upper batch shape.
It adds 2ms of prediction margin.
Both variants retain the same actual Decode-cohort execution Guard.
They do not preempt kernels.

Pending D cohorts contribute handoff, suffix, commit, and remaining Decode costs.
Different Cut frontiers remain separate cohorts.
The model does not assume that incompatible frontiers can share a suffix execution.
Batch costs above four use conservative integer-multiple extrapolation.
Exactness qualification does not validate that extrapolation as a tail guarantee.

## Discovery prediction audit

This subsection describes `online_COST_O16_N64_v1` only.
It is not a held-out aggregate.
The run completed 20 Last12 requests and 44 Native requests.
There were five accepted positive-cut decisions.

| Predicted boundary | Samples | MAE, ms | Absolute error p95, ms | Underprediction |
|---|---:|---:|---:|---:|
| Active Decode cohort drain | 4 | 32.82 | 50.92 | 0% |
| Known active and pending work drain | 20 | 96.19 | 158.07 | 0% |
| P Source ready | 5 | 34.16 | 37.17 | 100% |
| Suffix completion to first token | 5 | 18.62 | 20.90 | 100% |

Error equals observed time minus predicted time.
A positive error is an underprediction.
The active and pending drain model was conservative in this run.
The prefix model omitted some submission and preparation delay.
The 5ms Guard did not cover every completion-prediction error.
Actual Decode protection also depends on the execution Guard.
The predictor alone cannot certify safety against unknown arrivals.

The [forecast audit](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/forecast_validation.json) contains each completed run.
Its held-out rows remain separate from this discovery example.
Physical Slack prediction accuracy is `NOT_IDENTIFIED`.
The Clean Kernel Capacity Gate did not pass.

## Held-out prediction audit

This audit pools accepted decisions from the five Output16 COST starts.
The74 decisions represent296 completed Last12 requests.
These decisions are not74 independent server starts.

| Boundary | Decisions | MAE, ms | Absolute error p95, ms | Underprediction |
|---|---:|---:|---:|---:|
| P Source ready | 74 | 37.439 | 50.742 | 100% |
| Suffix completion to first token | 74 | 49.859 | 101.972 | 86.486% |

Known active and pending drain predictions had167 observations.
Their weighted MAE was61.209ms.
Their underprediction fraction was24.551%.
The five start-level absolute-error p95 values ranged from65.346 to134.705ms.
That range is not a pooled p95.

The5ms Guard is not a validated bound for these completion errors.
The actual cohort execution Guard remains necessary.
No result here proves safety against unknown future arrivals.
The model also uses median Host prefix and handoff costs.
Only the suffix table uses an event p95.
Four calibration samples per shape cannot establish a WCET guarantee.

The separate COST event control measured a batch4 Last12 suffix p95 of357.981ms.
The frozen prediction was358.580ms.
This agreement applies to that diagnostic event enclosure.
It does not validate every marker-free online quantum.
The event control failed the overall5% interference gate.

The same diagnostic control measured a full Decode batch4 event median of22.442ms.
The Native control measured21.611ms.
Their Host-step residual medians were4.476 and2.236ms, respectively.
Those residuals include other GPU work and synchronization.
Do not label them exclusive CPU costs.
Do not add independent medians to explain an online p99 regression.

## Transfer accounting

Evidence level: `STRUCTURAL_ACCOUNTING` for this per-request payload table.
The analysis reports observed NIXL bytes separately.

| D layers | Prefix KV, MiB | Hidden plus Residual, MiB | Total, MiB | Change from Native, MiB |
|---:|---:|---:|---:|---:|
| 0 | 1125.0 | 0.0 | 1125.0 | 0.0 |
| 2 | 1062.5 | 125.0 | 1187.5 | +62.5 |
| 4 | 1000.0 | 125.0 | 1125.0 | 0.0 |
| 8 | 875.0 | 125.0 | 1000.0 | −125.0 |
| 12 | 750.0 | 125.0 | 875.0 | −250.0 |

The full Native KV payload is not additional Layer Split overhead.
Native also transfers that KV.
Small suffixes can increase total payload because Hidden and Residual offset the omitted KV planes.

One Native request reported 18000 descriptors and 1179648000 transferred bytes.
Its NIXL transfer field was 68.704ms.
Its post field was 59.363ms.
These fields can overlap.
Their ratio is not an exclusive CPU contribution.
See the [transport audit](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/transport_accounting.json).

## Roofline scope

The scheduler uses a Lookup Table, not a peak-performance roofline prediction.
The [structural roofline analysis](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/roofline_operation_accounting.json) provides an optimistic lower latency bound.
It includes linear-layer FLOPs and causal Attention work.
Its byte floor omits activation traffic and other necessary costs.

The Dense BF16 ceiling is 989.5TFLOP/s.
The bandwidth ceiling is 3.35TB/s.
The vendor's 1979TFLOP/s BF16 figure includes sparsity.
These are vendor limits, not measured sustained rates. [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

Measured HBM traffic, Tensor Core utilization, and spatial compute headroom remain `NOT_MEASURED`.
Theoretical arithmetic intensity does not establish available D capacity.
