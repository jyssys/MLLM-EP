# Huginn R32 Output16/4 idle-capacity diagnosis

## Direct answer

**The increase in clean, usable D idle time remains `NOT_MEASURED`.**
**The additional number of Decode-safe completed loops also remains `NOT_MEASURED`.**
These labels do not mean zero idle capacity.

One Output4 capture records actual kernel inactivity.
However, it fails the 5% interference screen.
The matching Output16 capture fails with a CUDA launch error.
Therefore, the two captures cannot establish a clean physical-idle difference.
The earlier NVML means, 91.96% and 35.15%, do not supply that missing measurement.

The drain-forecast audit did complete.
The configured 93.1062 ms is a Host ITL calibration value, not GPU execution time.
New known-cohort audits show conservative forecasts, with no underpredictions in these two short starts.
This finding is not a future deadline guarantee.

## 1. Scope and execution order

The [Loop28 diagnosis](huginn_loop28_decode_tail_diagnosis_final_summary.md) completed before these GPU controls.
No new scheduler optimization campaign started.

The cap-only controls use Native-path Zero from commit `21a067b8bf2242bb8a16fc8c59ed59193e481f54`.
They use original BF16 Huginn-0125, R32, input 2048, QPS 1.971, and P/D slots 2/8.
GPU pair 4/5, arrival seed 2026110940, KV pools 16384/24832, and graph policy remain identical.
Each control uses the same first 64 controlled prompts and arrival offsets.
Prefix caching remains disabled.
Only the output cap changes between the two workload controls.
These N64 controls do not replace the earlier N128 results.

Evidence: [output_idle_diagnosis.json](huginn_loop28_decode_tail_diagnosis_20261111/analysis/output_idle_diagnosis.json).
Configurations, commands, source hashes, worker exits, and failed logs remain in the same independent evidence directory.

## 2. Marker-free controls

Each cap has two independent starts with the same arrival trace.
Values below are ms. No CUDA kernel profiler runs in these controls.

| Output cap / start | TTFT p95 | TPOT p95 | ITL p99 | Maximum ITL |
|---|---:|---:|---:|---:|
| 16 / first | 6742.066 | 83.284 | 94.149 | 97.720 |
| 16 / repeat | 6643.522 | 81.135 | 90.150 | 92.941 |
| 4 / first | 6538.874 | 74.940 | 78.050 | 79.432 |
| 4 / repeat | 6568.307 | 74.967 | 76.469 | 78.018 |

Both Native A/A comparisons pass the four-metric 5% point screen.
All requested output tokens match their original cap-specific Native references.
Both P and D workers exit normally in all four controls.
Two starts on one arrival trace do not provide a population confidence interval.

The central arrival window also shows lower D workload pressure with Output4.
The following values use coordinator samples and actual Decode submission histograms.
They are not continuous GPU-ready measurements.

| Observation | Output16 | Output4 |
|---|---:|---:|
| Mean coordinator waiting requests | 4.483, n=89 | 4.135, n=89 |
| Mean D reserved blocks / 24832 | 16787.2, n=45 | 9632.0, n=45 |
| Mean D reserved slots | 4.067, n=45 | 2.333, n=45 |
| Actual recurrent Decode submissions | B1=7, B2=233, B3=22 | B1=111 |
| Clean P kernel busy | `NOT_MEASURED` | `NOT_MEASURED` |
| Actual P-ready queue and eligible-cut overlap | `NOT_MEASURED` | `NOT_MEASURED` |

Reserved blocks include requests that have not reached Decode execution.
Coordinator waiting requests do not identify executable P work or completed source cuts.

## 3. Short kernel-capture attempts and interference

The first collector warms before Serving and captures approximately five seconds near measurement second 15.
It records actual kernel intervals across all streams.
Analysis uses their interval union, not the sum of kernel durations.
The common P/D capture denominator is 4.994748 seconds for the completed Output4 attempt.

That Output4 attempt changes ITL p99 by +6.67% and maximum ITL by +15.58% versus its clean control.
The changes versus its repeat are +8.88% and +17.68%.
Thus, it fails the interference screen against both controls.

The profiler-disable call lasts 478.037 ms on P and 22.335 ms on D in that attempt.
These calls occur during Serving and can change later queue behavior.
Their durations are Host calls, not GPU kernel times.

The Output16 attempt ends with `CUDA error: unspecified launch failure`.
A second method reserves storage before measurement and defers profiler disable until after outputs.
Its Output4 run also ends with the same CUDA error class.
The investigation stops that profiling path after this failure.
The prepared deferred Output16 cell does not run.
Fresh marker-free controls complete between these failed attempts.
The precise CUDA-error cause remains `NOT_IDENTIFIED`.

The failed runs do not enter latency, idle, or oracle statistics.
No diagnostic kernel trace qualifies as a clean cap-only idle comparison.

## 4. Preserved instrumented Output4 observation

These are actual intervals from the completed capture, but they are **diagnostic-only**.
They do not establish clean Native idle capacity or Decode-safe slack.

| Observation within 4.994748 seconds | Instrumented Output4 |
|---|---:|
| D kernel union | 1.719620 s, 34.4286% |
| D kernel inactivity | 3.275128 s, 65.5714% |
| P kernel union fraction | 88.5206% |
| D idle-window count | 214110 |
| Idle mean / p50 | 0.015296 / 0.000447 ms |
| Idle p95 / p99 / maximum | 0.002239 / 0.013568 / 525.162339 ms |

Most counted gaps are sub-millisecond gaps between kernels.
They are not useful loop-execution opportunities by themselves.
Kernel inactivity also does not establish inactive copy engines or available KV credit.

| Minimum idle-window length | Window count | Sum of qualifying window lengths |
|---|---:|---:|
| 1 ms | 39 | 3.026509 s |
| 2 ms | 12 | 2.989876 s |
| 5 ms | 10 | 2.984878 s |
| 10 ms | 9 | 2.978271 s |
| 20 ms | 8 | 2.967411 s |
| 30 ms | 8 | 2.967411 s |

Threshold rows overlap and must not be summed.
The matching Output16 values, the whole-cohort idle totals, and their clean differences remain `NOT_MEASURED`.

## 5. Offline oracle limits

A diagnostic packing sensitivity uses the observed Output4 gaps and earlier operation-cost measurements.
It assumes unlimited ready work and omits handoff, source readiness, CODA, Decode reserve, and request availability.
It is not a Decode-safe or completion-feasible oracle.

| Assumed full-loop cost | Windows that fit one unit | Packed loop-sized units |
|---|---:|---:|
| 21.808765 ms, nonterminal calibrated candidate | 8 | 132 |
| 28.894943 ms, earlier serial maximum enclosure | 8 | 99 |
| 34.143634 ms, terminal calibrated candidate | 8 | 83 |

These counts are not offloadable serving loops.
They do not enforce the number or readiness of actual requests.
The costs are operation enclosures from different diagnostics, not measured kernel work inserted into these gaps.

A separate tile256 sensitivity uses 8.206884 ms per tile from the Loop28 instrumented p99 enclosure.
It packs 358 tile units, equivalent to 44 complete eight-tile loops and six unfinished tiles.
The six unfinished tiles do not count as a completed loop.
This sensitivity does not show a tiling benefit over full-loop units for these long diagnostic gaps.
It also does not establish a feasible tiled-loop count.

Qualified full-loop capacity, tiled capacity, Decode-safe slack, and completion-feasible capacity remain `NOT_MEASURED`.
The missing source-cut readiness, exact handoff cost, and safe execution validation prevent stronger claims.

## 6. Provenance and validation of 93.1062 ms

`forecast_cells.py` assigns `preparation_decode_cycle_ms=93.10616207122803` as a literal.
The nearly exact matching datum is Host ITL p99 from `selection_group32_seed2026110940`.
Recalculation from its 1920 output intervals gives 93.10615979135036 milliseconds.
The absolute difference is 0.000002280 milliseconds.
The source does not contain an automatic derivation manifest for the literal.

`predictive_prepare.py` uses this formula:

`forecast = maximum known remaining outputs × max(configured floor, recent observed interval p95)`

The forecast prepares a source checkpoint.
It neither commits ownership nor guarantees a physical idle window.
Actual handoff admission remains a separate current-state decision.

Old preparation logs omit the original advertisement time and exact known-cohort request IDs.
Their exact historical MAE therefore remains `NOT_IDENTIFIED`.
Scoring them against all future arrivals would use the wrong completion target.

Two new N64 Host-only audits record outgoing D advertisements and their exact known request IDs.
The target is the last D output-send receipt among those IDs.
Future arrivals can interfere, but do not enter that target cohort.
The audits reuse the existing forecast-policy configurations without tuning them.
They are separate from the cap-only Native controls.

| Audit | Advertisements | MAE | Absolute error p95 | Underpredictions |
|---|---:|---:|---:|---:|
| Output16 | 1941 | 199.704 ms | 269.948 ms | 0/1941, 0% |
| Output4 | 699 | 74.464 ms | 122.438 ms | 0/699, 0% |

All scored forecasts overpredict known-cohort drain time.
The mean signed error, actual minus prediction, is −199.704 ms and −74.464 ms respectively.
Advertisement samples are correlated and are not independent requests.
Each cap uses one arrival trace and one audit start.
These observations do not prove zero future underprediction risk.
Both audits preserve exact generated output prefixes and normal P/D exits.

Sources: [calibration provenance](huginn_loop28_decode_tail_diagnosis_20261111/analysis/forecast_calibration_provenance.json), [forecast errors](huginn_loop28_decode_tail_diagnosis_20261111/analysis/output_idle_diagnosis.json).

## 7. Final diagnostic decision

Output4 reduces observed Decode batch pressure, KV reservations, and marker-free Decode latency.
However, the clean increase in usable idle seconds, idle percentage, and safe completed loops remains unmeasured.
The correct verdict for that comparison is `INCONCLUSIVE_MEASUREMENT`, not zero headroom or a serving NO-GO.

No new large scheduler campaign follows from these data.
The original staged index and external runtime diff remain unchanged.
The resource guard restored approved burners on GPUs 0–7.

Figures: [kernel capture and gap histogram](huginn_loop28_decode_tail_diagnosis_20261111/figures/output_cap_kernel_idle_diagnostic.png), [known-cohort forecast audit](huginn_loop28_decode_tail_diagnosis_20261111/figures/known_cohort_drain_forecast.png).
The missing Output16 plot explicitly shows `NOT_MEASURED`.
