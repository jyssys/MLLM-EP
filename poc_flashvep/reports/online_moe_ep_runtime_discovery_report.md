# Online MoE-EP runtime discovery (4×H100)

## Executive decision

### Sprint accounting

- Wall time (19:17:11–23:38:53 KST): **4 h 22 min**.
- Live GPU time: **6,591 s** of rank-local event-trace span across 12 traces
  (conservative; excludes model-load wall time and no-row launches).
- Equivalent 4-GPU time: **7.32 GPU-hours**.  Ten completed HT runs plus two
  retained partial traces supplied the primary data; no GPU 0 or 5–7 process
  was touched.

**FINAL STATUS: HOLD (NO_GO for an adaptive runtime-configuration method).**

The clean online evidence does not expose a useful static-configuration
oracle: DeepEP high-throughput with 8, 12, and 20 communication SMs differs by
about **0.11% median** and at most **2.10%** in the exact-M lower envelope.
Adding per-token fanout and sender/destination geometry to expert-distribution
and rank-load features changes held-out error by only **+0.001%** (within
noise).  Thus there is no evidence for
an online fanout/controller direction, and the measured configuration gap is
below the 3% no-go threshold.

The result is deliberately not a claim that every DeepEP regime is optimal:
the low-latency backend could not initialize under the installed DeepEP
queue-depth constraint, and the read-only online hook records one complete
FusedMoE CUDA interval rather than independent dispatch/expert/combine events.
The large fixed-shape prefill tails are a real follow-up anomaly, but their
causal stage is unresolved until rank-synchronised component instrumentation
is available.

## Configuration and provenance

| Item | Value |
|---|---|
| GPUs | physical 1,2,3,4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`) |
| Model | Qwen3-VL-30B-A3B-Instruct (local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`) |
| Runtime | vLLM 0.20.0 V1, BF16, eager, prefix cache off, DBO off |
| Parallelism | TP2 / DP2 / EP4 / PP1, linear expert placement |
| Primary backend | `deepep_high_throughput` (DeepEPHTPrepareAndFinalize) |
| Communication knob | `VLLM_DBO_COMM_SMS` = 8, 12, 20 (DeepEP caps at 20) |
| Scheduler | V1 asynchronous scheduler, chunked prefill enabled; MBT 8,192/16,384/4,096 |
| Observer | read-only `FusedMoE.apply` CUDA events + exact route/histogram capture |

Runtime logs show `world_size=4`, `Using DeepEPHTAll2AllManager`,
`DeepEPHTPrepareAndFinalize`, Triton unquantized MoE, and 32 local / 128
global experts.  The topology proof files are in each clean trace directory.
No GPU 0 or 5–7 process was touched; GPUs 1–4 were free at start and are free
again at completion.

## Hypotheses and decision gates

| ID | Question | Live evidence | Status |
|---|---|---|---|
| H1 | workload-dependent DeepEP communication-SM optimum | route-joined SMS 8/12/20 exact-M envelope: 0.11% median, 2.10% max; no stable winner | NO_GO |
| H2 | dispatch/combine prefer different SM settings | component events are not separately captured in stock online hook | BLOCKED |
| H3 | backend crossover within prefill | HT works; LL fails in both attempts at `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2` | BLOCKED |
| H4 | mixed-step backend/config mismatch | no residual signal after phase/M grouping; component timing unavailable | HOLD |
| H5 | DP-source asymmetry beyond destination load | rank-critical proxy only; no direct synchronized source timeline | BLOCKED |
| H6 | empty/dummy participant tax | not isolated by current workload | NOT_RUN |
| H7 | layer-specific runtime ranking | descriptive layer variation, no repeatable ≥5% SMS ranking | HOLD |
| H8 | fixed-regime tail has a non-route cause | fixed-M/layer tails repeat, but route-joined rank/timing correlation is ≈0.14; mechanism unresolved | PROMISING ANOMALY |
| H9 | cross-rank tail co-occurrence | aggregate rank span exists, but event clocks are rank-local | BLOCKED |
| H10 | burst vs steady tails differ at matched load | burst/steady p90s differ descriptively; route/state confounded | HOLD |
| H11 | safe runtime configurations leave oracle headroom | clean HT lower envelope: 0.11% median, 2.10% max | NO_GO |
| H12 | generic Qwen3 text reproduces strongest anomaly | gated because H1–H11 produced no strong causal signal | NOT_RUN |

Each negative result generated follow-ups in `ACTIVE_QUEUE.md` (H1R–H12R and
tail/state anomalies); the queue remains above the six-item minimum.

## Live online collection

The result directory is
`poc_flashvep/deepep_revalidation/results/online_moe_ep_runtime_discovery_20260905_191711/`.
Ten hook-enabled HT serving runs are primary: SMS20 medium, SMS8, SMS12,
SMS20 burst16, SMS20 long-steady (partial after a stalled worker), SMS20
MBT4096/concurrency8, SMS20 long-c8, and four recovery runs (SMS20/12/8 at
concurrency 8 plus SMS20 at concurrency 2).  A separate MBT4096/concurrency16
trace, the hook-missing SMS4/SMS16 launch attempts, and both LL attempts are
retained as diagnostics but excluded from the primary causal aggregate.

A route-id join (the common `route_XXXXXXXX_lY` identifier) gives
**6,105,264 valid rank/layer observations** and **1,524,288 complete four-rank
invocation aggregates** after the registered `M≤2048` filter.  The latter is
the critical-path analysis unit:

| Phase | Rank/layer rows | Complete invocations | Critical CUDA p50 / p90 / p99 (ms) | Rank critical-span p50 / p90 |
|---|---:|---:|---:|---:|
| Prefill | 86,304 | 21,576 | 0.986 / 3.071 / 4.686 | rank-imbalance 1.063 / 1.397 |
| Decode | 1,437,984 | 359,496 | 0.918 / 2.060 / 3.494 | rank-imbalance 1.062 / 1.226 |

Natural prefill M in the clean aggregate spans 8–2,048 (MBT4096/c8 includes
the 2,048-token bin); decode M is 1.  The route join avoids treating
rank-local timestamps as a global clock.  The partial MBT4096/concurrency16
trace is kept as a scheduler-stall diagnostic and is not mixed into these
numbers.

## Timing sanity and model hierarchy

The sanity grouping uses repeated `(source, phase, layer, M, EP-rank)` rows.
There are 9,191 groups in the stride-5 compact sanity sample; median CV is
**41.9%** and p90 CV **72.4%**.  This is
`INSUFFICIENT` under the pre-registered ≤10% pass / 10–20% caution gate.  The
large CV is why the report uses paired descriptive medians and does not claim
stage-specific causality.

The time-block held-out hierarchy is:

| Model | RMSE (ms) | MAE (ms) | p90 abs. error (ms) | R² |
|---|---:|---:|---:|---:|
| M only | 9.77646 | 0.43564 | 0.63331 | -0.00005 |
| M + expert distribution | 9.77645 | 0.43548 | 0.64429 | -0.00005 |
| + rank-load features (Model 2) | 9.77676 | 0.43562 | 0.65352 | -0.00011 |
| + fanout/traffic geometry (Model 3) | 9.77667 | 0.43571 | 0.65916 | -0.00009 |

Model 2→3 RMSE change is **+0.001%** (effectively zero; compact fit sampled
deterministically from 1,219,430 rank rows),
so natural fanout is not an incremental online latency signal in this trace.
This is a null result, not evidence that fanout can never matter under a
different instrumentation or backend.

## Communication-SM and backend results

Clean rank-local FusedMoE intervals by SMS are:

| SMS | Phase | n rank rows | M median | p50 / p90 (ms) |
|---:|---|---:|---:|---:|
| 8 | prefill | 18,000 complete | 284.0 | 0.973 / 2.845 |
| 12 | prefill | 19,920 complete | 284.0 | 1.044 / 3.338 |
| 20 | prefill | 48,384 complete | 284.0 | 0.976 / 2.979 |
| 8 | decode | 270,864 complete | 1 | 0.911 / 1.800 |
| 12 | decode | 299,664 complete | 1 | 0.948 / 2.077 |
| 20 | decode | 867,456 complete | 1 | 0.912 / 2.034 |

For exact-M bins where multiple SMS values are present, the route-joined lower
envelope has median headroom **0.11%**, p90 **1.71%**, and maximum **2.10%** in
prefill; decode headroom is **0.25%**.  Every value is below the
pre-registered 3% no-go boundary.  The descriptive best setting moves among
8/12/20 by M; it is not a robust workload-dependent crossover.

### Static versus oracle envelope

`BestStatic` is the validated SMS20 HT configuration used for the long traces.
`OraclePerRegime` is the lower envelope over SMS8/12/20 for each exact
`(phase,M)` bin: 0.11% median, 1.71% p90, and 2.10% maximum prefill headroom
(0.25% for decode).  A true `OraclePerInvocation` is not identifiable from
restart-separated natural waves because route identities are not repeated
across SMS settings; no per-invocation gain is fabricated.  The per-regime
bound is already below the 3% gate.

The low-latency backend was attempted at MBT=8192 and 1024.  Both fail during
vLLM dummy profiling before natural requests, with the same DeepEP assertion
in `deep_ep/buffer.py:601`.  It is therefore a runtime feasibility blocker,
not a negative performance comparison.  No unsupported backend was silently
substituted.

## Tail anomaly

Tail mining compares samples within the same source/phase/layer/M group.  The
route-joined aggregate shows broad fixed-shape tails: prefill p50/p90/p95/p99
are 0.986/3.071/3.753/4.686 ms and decode values are
0.918/2.060/2.248/3.494 ms.  Some fixed-M/layer groups contain 10–100×
scheduler-state outliers while assignment count, active experts, and fanout
remain unchanged.  After exact route joining, correlation of critical span
with rank imbalance is only **0.14** in both prefill and decode; the previous
larger correlations came from the timestamp-window proxy.  This is a real
tail anomaly, but its causal stage remains unresolved and it is recorded as
`ANOMALY-03`, not promoted to a method claim.

This remains the strongest unexplained observation, but it is compatible with
a rank-local queue/stream or scheduler-state tail and not yet attributable to
one execution stage.  It is therefore recorded as `ANOMALY-03`, not promoted
to a new method claim.

## What is and is not measured

The stock hook records one CUDA interval around `FusedMoE.apply` on each EP
rank, plus exact top-k routes, expert histograms, rank loads, and request
phase/M.  It does **not** add independent dispatch, expert, or combine CUDA
events in the online worker.  Source audit confirms that DeepEP HT uses
`previous_event`, `EventOverlap`, and communication-stream dependencies for
layout/dispatch and combine; this is an implicit stream/event dependency, not
a new global barrier.  Consequently:

- `T_MoE` below is a conservative complete FusedMoE interval.
- SMS/backend results are descriptive across engine restarts, not matched-route
  causal comparisons.
- rank timestamps are never subtracted as a global clock; route IDs are used
  for the complete-invocation join.
- TensorCore, HBM, and link-utilisation values are not inferred or invented.

## Prior-art boundary and next work

The null fanout result avoids re-framing DA-MoE/TEMPO-style expert-histogram
or makespan work as a new contribution.  The communication-SM control is an
existing DeepEP/vLLM resource knob; a 0–3% lower-envelope gap is not enough to
justify an adaptive controller.  The only worthwhile next experiment is a
short, instrumentation-corrected tail study: rank-synchronised dispatch,
expert, combine events plus scheduler iteration IDs, repeated at fixed M=284
and one decode shape.  If the tail collapses after component association, do
not pursue it; if it remains at fixed route/load and one stage is causal, test
that stage against the relevant prior art before implementing anything.

No dynamic backend switcher, SM controller, scheduler, RL policy, or kernel was
implemented in this sprint.

## Reproducibility artifacts

The compact primary analysis is under
`.../analysis_final/analysis_summary.json`; the exact route-joined aggregate
is `.../analysis_full/aggregate_invocations_route.csv` with its companion
`aggregate_summary_route.json`, tail analysis, and runtime envelope.  All
hook-enabled raw traces are preserved as `invocations.jsonl.gz`, with per-run
status/topology/wave manifests and the LL/SMS4/SMS16 failure logs.  The
analysis scripts (`analyze_runtime.py`, `aggregate_by_route.py`,
`tail_mine.py`, and `regime_stats.py`) are reusable and make no runtime
changes.
