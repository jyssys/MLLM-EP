# Native PD Idle Capacity and Loop Offloading Oracle

## Scope

This track measures Native 1P1D execution. It does not implement another scheduler.
The completion-aware scheduler track continues in a separate evidence namespace.
Neither track may replace a failed measurement with a modeled result.

The primary workloads remain frozen.

| Field | Ouro | Huginn |
|---|---|---|
| Model | ByteDance/Ouro-1.4B | tomg-group-umd/huginn-0125 |
| Recurrent depth | R4 | R32 |
| Input/output tokens | 4096/32 | 2048/16 |
| Offered QPS | 6.356 | 1.971 |
| P/D slots | 2/8 | 2/8 |
| Requests per start | 128 | 128 |
| Arrival seeds | 2026110911, 2026110912, 2026110913 | Same |

The model revisions, original BF16 weights, prompt pool, and KV configuration remain unchanged.
Prefix caching remains disabled. Every recurrent computation must occur exactly once.

## Evidence and isolation

Each run uses a unique `oracle_` namespace in the completion-aware evidence directory.
That directory provides one shared resource lock and one approved Burner owner.
Clean timing uses one GPU pair. Other GPU workloads stop during the measurement.
Approved Burners return after the scientific job list ends.

The source snapshot, configuration hash, GPU mapping, and arrival seed identify each run.
The existing staged index and external runtime diff remain unchanged.
The final publication contains only this campaign's artifacts.

## Measurement protocol

First measure Native A/A variation without CUPTI.
Compare a full-cohort activity trace against those controls.
Report TTFT, TPOT, ITL p99, maximum ITL, and elapsed time separately.
Keep the existing 5% interference gate. A failed trace remains diagnostic evidence only.
Maximum ITL changes require a separate tail audit even when p99 passes.

The first collector uses CUPTI concurrent-kernel activity and memory-copy activity.
Its callback retains completed activity buffers without parsing them during serving.
The collector parses and exports records after measured serving ends.
It must report dropped records, invalid timestamps, capture coverage, and its memory limit.
A small CUDA Graph test must pass before a model run.
The collector does not change model kernels or attention math.

CUDA Events provide primitive execution enclosures. They do not prove exclusive kernel activity.
Host submission gaps do not prove physical GPU idle.
The analysis clips activity records to the common measured cohort window.
It computes the time-interval union across streams. It never adds overlapping kernel durations.
Memory copies remain separate. Communication kernels remain separately labeled where kernel metadata permits identification.
Graph metadata identifies kernels that run through graphs. A graph submission is not an additional compute interval.

## Required Native measurements

Report each seed and the aggregate variation.

- Measurement duration and kernel-union busy/idle time.
- Contiguous idle count, mean, p50/p90/p95/p99, and maximum duration.
- Count and total duration above 1/2/5/10/20/30/50 ms.
- P kernel union, P-ready queue, and evidence of simultaneous prefill pressure.
- Decode batch and KV occupancy at each gap, when timestamped observations support attribution.
- ITL p99, maximum ITL, and threshold exceedance counts.

An unavailable attribution remains `NOT_MEASURED`.
A short capture cannot represent the full N128 cohort.

## Primitive calibration

Use qualified recurrent execution with the original full context.
Measure Ouro 4096 tokens with 1/2/3 remaining loops.
Measure Huginn 2048 tokens with 1/2/4/8 remaining loops.
Also measure qualified token tiles of 128/256/512/1024 tokens where feasible.
Retain attention context, prelude/coda costs, and transfer costs in separate fields.
Report whether a cost is a fresh measurement or reused immutable calibration.
Do not equate recurrent token rows with FLOPs.

## Oracle definitions

### Physical Idle Upper Bound

Pack non-preemptive measured quanta into measured contiguous idle windows.
This bound first ignores request availability and transfer costs.
Also report the looser total-idle-time divided by compute-cost bound.
Label both as offline bounds, not implemented serving results.

### Decode-safe Oracle

Preserve the observed Decode schedule and frozen TPOT/ITL contract.
Apply request readiness, causal dependency, KV capacity, handoff, and ACK constraints.
No kernel may cross a protected interval in the strict no-displacement version.
Any oracle that permits Decode displacement must recompute token deadlines and tail metrics.
Report maximum ITL and threshold exceedances as well as p99.
Unknown constraints prevent a verified Decode-safe claim.

### Completion-feasible Oracle

Require D completion before the corresponding Native P completion alternative.
Include transfer, import, scheduling, and continuation delay.
Native Atomic Graphs do not expose every internal source-cut timestamp.
A timestamp derived from serial calibration is a model estimate, not a measured Native boundary.
Report such results as `UNVALIDATED` unless independent GPU execution verifies them.

## Qualified and potential frontiers

Qualified Ouro cuts are R1/R2/R3. Qualified Huginn cuts are R24/R28/R30/R31.
The qualified Huginn set permits at most 384 × 8 = 3072 offloaded loops.
Therefore, 4000 loops cannot occur within that set, regardless of idle capacity.
An earlier cut must leave at least 11 loops on some requests to exceed the per-request average of 10.4167 loops.
Earlier frontiers require exact state-layout inspection and new GPU qualification.
Their costs remain potential compute bounds until qualification passes.

The targets are Ouro 500/1536 loops and Huginn 4000/12288 loops.
Compare the required compute occupancy with measured window capacity.
Do not infer target feasibility from total idle percentage alone.

## Tiled and untiled accounting

Oracle A executes full-prompt loops without token tiling.
Oracle B executes the same loop through dependency-valid token tiles.
Oracle B must include every tile, tile overhead, and required state transitions.
Count one completed loop only after every token tile for that loop completes.
Do not count a partial tile frontier as a completed full loop.

## Gates and deliverables

1. Verify collector coverage and loss counters.
2. Measure Native A/A and profiler interference.
3. Collect three primary starts per model where the collector passes.
4. Compute physical gap statistics and qualified primitive costs.
5. Compute constrained bounds only where the required evidence exists.
6. Report unmeasured constraints and failed traces without replacing them with zero.

The final report answers the five questions in the user request.
It separates measured activity, structural limits, modeled packing, and executed continuation results.
Figures include Native P/D activity and idle histograms.
The report also compares untiled and tiled bounds where calibration supports both.

## Collector recovery ledger

The first full-cohort collector fails with an unspecified CUDA kernel submission error.
Its small CUDA Graph coverage test passes 200/200 kernels with zero dropped records.
That test does not establish compatibility with the complete Native/NIXL workload.

The second collector removes memory-copy activity and starts after graph warm-up.
It also fails in the Ouro Native workload.
The original short-window collector also fails in full-cohort Ouro and Huginn attempts.
These failed executions cannot establish Native idle capacity.

Further isolated controls may collect only the D process.
A periodic non-forced buffer flush may replace a single final flush.
These controls must not synchronize the GPU during measured serving.
Each control requires a separate namespace.

The environment contains CUDA Runtime 12.9 and CUPTI libraries with API versions 26 and 28.
A matching CUDA 12.9 collector is another diagnostic control.
Version selection alone does not establish the root cause.
Record the actual loaded library mappings and selected binary hash in new attempts.

If these controls fail, preserve the short measured windows as diagnostics.
Do not extrapolate their idle fraction to all 384 requests.
Full-cohort idle capacity and constrained Oracle results must remain `NOT_MEASURED` in that case.

The final Oracle report is `ouro_huginn_pd_idle_slack_loop_offloading_oracle_final_summary.md`.

## Optional earlier-frontier qualification

The historical qualified-cut set remains unchanged in Oracle A.
A separate diagnostic may verify Huginn R16 and R21 with exact N8 state comparisons.
It must then verify an actual mixed-cut N32 cohort before serving selection.
These frontiers use the existing one-way state format. They do not assume independent token/depth tiles.
Report a newly qualified set separately from the historical set.
A larger structural ceiling does not establish physical or Decode-safe capacity.
Any extrapolated primitive cost remains `ESTIMATED` until a separate serial GPU calibration measures it.
