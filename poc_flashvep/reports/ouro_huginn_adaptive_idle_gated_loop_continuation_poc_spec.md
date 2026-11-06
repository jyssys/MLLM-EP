# Adaptive idle-gated recurrent continuation: bounded PoC

## 1. Hypothesis and scope

The candidate moves the remaining recurrent suffix from P to D once.
Its handoff boundary depends on current conservative D execution slack.
A larger budget can admit more remaining loops. A smaller budget admits fewer loops or rejects migration.
Static cuts, fixed placement, and D-local independent probes do not establish this mechanism.
The campaign does not implement D-to-P state return or kernel preemption.

The publication parent is `3b3633dbc64bcd916c2ba1ff513deb4226cc647d`.
The new branch is `flashvep/adaptive-idle-gated-loop-continuation-poc`.
The existing checkout, staged index, private runtime, checkpoints, and prior evidence remain unchanged.
Each run has an immutable namespace, configuration, source hashes, and raw logs.

## 2. Fixed clock and priorities

Start: 2026-11-06 13:05:11 KST. Final deadline: 2026-11-06 21:05:11 KST.
Stop GPU experiments by 20:15:11 KST. Do not start long jobs after 19:55:11 KST.
The clock includes audit, implementation, GPU tests, analysis, push, and approved burner restoration.
A pause or capacity interruption does not restart this clock.
Use isolated two-GPU serving pairs. Other GPUs remain quiet during clean timing.
Stop only verified owned experiment or approved burner sessions. Restore burners on all eight GPUs after GPU closeout.

Priority is qualified slack, full-prompt continuation cost, variable-cut correctness, real admission, decode protection, and Token comparison.
Do not construct a large online campaign when an earlier gate fails.

## 3. Models and controlled workload

| Model | Weights / depth | Prompt / output | Low / medium / heavy QPS | Physical pair | P/D slots |
| --- | --- | --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | Original BF16 / R4 | 4096 / 32 | 1.23460875 / 3.4569045 / 6.356 | 0/1 | 2/8 |
| tomg-group-umd/huginn-0125 | Original BF16 / R32 | 2048 / 16 | 0.37903355455 / 1.06129395274 / 1.971 | 4/5 | 2/8 |

Keep pinned checkpoint revisions, qualified numerical fixes, exact attention, and full recurrent work.
Keep Huginn Prelude, injection, Coda, and its 4096-token context limit.
No early exit, approximation, quantization, weight change, or repeated/missing loop is allowed.
Reuse the 128-record ShareGPT/LongBench controlled-prefix pool. Each primary cohort has 128 distinct prompt IDs.
Keep fixed caps with ignored EOS and disabled prefix caching. Do not call these natural output distributions.
Generate fresh Poisson offsets for each rate. Pair methods by inputs, caps, seed, and offsets.
Discovery and calibration do not use their confirmation outcomes to choose cuts or margins.
The reused text pool is not a claim of unseen-text generalization.

## 4. Frozen SLO and instrumentation contract

| Model | TTFT ms | Request TPOT ms | Additional-work TPOT p95 ms | ITL p99 ms |
| --- | --- | --- | --- | --- |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

Also target at most 5% ITL p99 degradation from a paired Native run.
Report start-level confidence intervals, maximum ITL, and individual deadline misses.
A p99 screen is not a hard per-token deadline guarantee.
Count every offered failure as an SLO failure. Finite-cohort goodput is not sustainable capacity.

Replay the previous failed interference gate before changing instrumentation.
Test bounded host records, preallocated sparse CUDA events, and activity-only CUPTI.
Consider graph-level activity to reduce per-node trace overhead.
Graph envelopes are not exclusive kernel union. A conservative envelope complement must account for non-graph work too.
Report physical kernel idle, scheduler-visible idle, and predicted safe execution budget as different quantities.
Do not fill missing kernel activity with event enclosures or NVML averages.

For each collector, compare matched marker-free and instrumented cohorts.
Retain the 5% local-tail screen for TPOT p95, ITL p95/p99, and maximum ITL.
Check full-cohort duration and fixed arrival-time windows too.
Qualify coverage, dropped records, timestamp validity, and output parity separately.
No in-serving export or global synchronization is allowed merely to flush trace data.
Use capture calibration first. Freeze the selected collector before independent confirmation starts.
An unqualified trace remains diagnostic. Label missing safe-window distributions NOT_MEASURED.

## 5. Safe budget and online decision

Define B_safe as a conservative predicted execution allowance, not known future idle time.
Inputs can include current ready decode, active batch, last token times, observed service bounds, and uncertainty.
Also use KV credit, workspace, queued GPU work, graph readiness, and host submission cost.
An empty host ready queue does not prove an idle GPU.
A measured kernel gap does not guarantee that a future decode cannot arrive inside it.

At a legal P recurrent boundary, compare each remaining suffix against the same frozen safety policy.
The admission cost includes state preparation, exposed transfer, scheduling, remaining recurrent work, and required Coda work.
Reject if credit is insufficient, the state is stale, or predicted relief is nonpositive.
Log every candidate cut, predicted budget, component bound, decision, recheck, and rejection reason.
Measure policy overhead separately from GPU work.
Never use future arrivals or realized future output length in an online decision.

## 6. Boundary, ownership, and execution protocol

P owns the request until D accepts its exact remaining suffix.
P must not re-execute completed loops. D must execute every accepted remaining loop exactly once.
Native ACK protects source state and KV lifetime. Do not release or overwrite in-flight buffers.
After ownership commits to D, rejection cannot silently resume work on P.
D can yield at a completed loop boundary and prioritize ready decode.
An already submitted GPU kernel remains non-preemptible.
Admission of an entire suffix and scheduling of each next loop are separate decisions.
Record transfer-time arrivals, overrun, graph misses, ACK delay, aging, and continuation deadlines.

Keep the Native Atomic fast path when no offload is selected.
Measure any boundary polling or feature-memory cost even when an eventual handoff is rejected.
Do not call a staged fallback Native. Compare zero-offload latency and throughput with Native directly.
If actual mid-loop observation requires runner restructuring, qualify that restructuring before a performance claim.

## 7. Primitive cost and oracle

Measure full-prompt one-, two-, and several-loop suffix costs for each model.
Measure token tiles 16/32/64/128 and legal multi-loop units separately.
Include required Prelude/Coda, hidden and valid KV bytes, descriptors, ACK, and source retention.
CUDA events measure enclosures. Do not label them exclusive compute.
Apply the same graph budget, warm-up, safety margin, and transport policy to Token and Loop candidates.

Compare Native, adaptive Loop-only, adaptive Token-only, and feasible Token-by-Loop candidates.
Label any oracle with future arrival knowledge Clairvoyant Upper Bound.
Never make invalid causal tiles free. Keep dependency and ownership checks.
Validate calibrated costs on independent static GPU executions before claiming feasible online headroom.
Retain the previous model-error limits: median absolute relative error at most 15%, p95 at most 25%.
Reject false Decode-safe predictions. A failed predictor is UNVALIDATED, not measured speedup.
Do not add overlapping compute and transfer intervals or component percentiles.

## 8. Matrix and exactness gates

| Stage | Required cells | Expansion rule |
| --- | --- | --- |
| A0 | Native low/medium/heavy, both models | Marker-free N128 plus bounded matched observer trials |
| A1 | Full-prompt and tiled continuation primitives | Exact legal states and deterministic warm-up |
| B | Offline budgets and fair action families | Separate validated estimates from upper bounds |
| C | Mixed-cut N8 then N32 | At least two different nonzero remaining-loop counts plus fallback |
| D | Native, Static Token/Loop, Dynamic Token/Loop | N128 only after gates pass, paired seeds |
| E | Three independent starts and optional N512 | Only for qualified promising methods |

N32 means simultaneous submission, not 32 GPU residents.
Check generated tokens, valid prompt KV, boundary hidden state, transfer identity, full depth, ownership, credits, and release.
Test acceptance, rejection, stale-admission fallback, delayed ACK, and mixed cuts.
Forced diagnostics do not establish an online adaptive policy or latency success.
Original HF equivalence remains a separate unmeasured item unless explicitly tested.

Gate 1: valid same-run P pressure and D safe opportunity. Failure is NO_GO_IDLE_SLACK. Missing evidence is INCONCLUSIVE_SLACK.
Gate 2: actual variable accepted Loop counts. Failure is NO_GO_DYNAMIC_GRANULARITY.
Gate 3: absolute Decode protection and Native-relative tail targets. Failure is NO_GO_DECODE_INTERFERENCE.
Gate 4: net Prefill/TTFT or goodput benefit after transfer and scheduling. Failure is NO_GO_NET_BENEFIT.
Gate 5: fair Dynamic Token comparison. Report LOOP_ADVANTAGE, GENERIC_DYNAMIC_ONLY, TOKEN_ADVANTAGE, or INCONCLUSIVE_COMPARISON.
If full-prompt granularity demonstrably does not fit, test only a minimal feasible token-tiled variable-loop primitive.
Do not mix its verdict with full-prompt continuation.
Generic logical-layer execution with identical boundaries and tied weights is EQUIVALENT_MECHANISM.

## 9. Figures and closure

Create seven figure families: Native activity/slack, budget versus selected loops, actual continuation timeline, method performance,
Token-versus-Loop work efficiency, net benefit, and budget/loop/tile feasibility.
Label each panel MEASURED, TRACE_ANALYSIS, MODELED, UPPER_BOUND, or NOT_MEASURED.
Do not invent idle intervals, selected cuts, or speedups.
Store per-request outputs, policy records, queue/credit samples, HBM GiB and percentages, source archives, hashes, and failures.
The final report separates the five gates for each model and answers all four user questions.
Commit only this campaign's source, spec, reports, figures, and evidence through a separate Git index.
Use an ordinary push. Check remote SHA, preserved index/runtime hashes, and eight approved burner GPU sessions.
