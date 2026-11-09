# Online idle-time intervention and adaptive recurrent continuation

Implementation note, 2026-11-08: Initial N8 online interventions may use the prior qualified full-prompt cost records plus the frozen guard rule.
These are calibration inputs, not newly measured safe windows.
The new serial calibration and Native A/A controls must check those inputs before final confirmation.
Record admission-window and immediate execution budgets separately.
An empirical multi-opportunity admission estimate is not a guarantee of future slack.

## 1. Research question

Measure actual online recurrent work redistribution, not merely exact static cuts.
Select the P export boundary from current D state after a completed P boundary.
Accept different remaining-loop counts when observed budgets permit them.
Protect Decode latency and measure useful Prefill relief after all handoff costs.
Compare Native PD, strong Token baselines, and Loop methods without assuming a Loop advantage.

The publication parent is `ddbf49ae3ef610fefd78248d765a27641b21b98f`.
The publication branch is `flashvep/adaptive-idle-intervention-dynamic-loop-poc`.
Use the new `idle_intervention_dynamic_loop` source namespace and a separate evidence directory.
Preserve the active checkout, staged index, private runtime, checkpoints, and previous evidence.
Publication uses a temporary Git index and an ordinary push.

## 2. Fixed clock

Start: 2026-11-08 18:07:15 Asia/Seoul.
Final deadline: 2026-11-09 08:07:15 Asia/Seoul.
GPU experiment deadline: 2026-11-09 06:37:15 Asia/Seoul.
Do not start long jobs after 06:07:15.
The fourteen-hour wall clock includes inspection, implementation, GPU work, analysis, publication, and resource restoration.
A capacity interruption does not reset this clock.
Record dispatch time separately from wall time and GPU execution time.

### Explicit completion resumption, 2026-11-09

The user requested completion after the original campaign expired.
This is a separate resumption, not a restart of the fourteen-hour clock.
The resumption starts at 13:00:54 KST on 2026-11-09.
Stop GPU experiments by 16:00:54 KST. Finish publication and restoration by 17:00:54 KST.
Do not start long jobs after 15:40:54 KST.
Record this interval separately in `state/resumption_20261109.json`.
Preserve the original budget, worker receipts, and failed tests.
Recheck pending full-byte observer tests before larger mixed-cut qualification.
Use the same frozen SLO and correctness gates.
Do not interpret an unavailable confirmation arm as measured zero benefit.

Prioritize a working boundary protocol and actual intervention during the first part of the campaign.
Do not spend the campaign attempting a perfect CUPTI trace.
Maintain a bounded execution chain with retained receipts, progress checks, and deadline enforcement.
A watchdog alone is not an owner that selects the next experiment.

## 3. Models and workload

| Model | Revision | Precision / depth | Input / output | Physical P/D pair | P/D slots |
| --- | --- | --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | Original BF16 / R4 | 4096 / 32 | 0/1 | 2/8 |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | Original BF16 / R32 | 2048 / 16 | 4/5 | 2/8 |

Ouro low/medium/heavy offered QPS: 1.23460875 / 3.4569045 / 6.356.
Huginn low/medium/heavy offered QPS: 0.37903355455 / 1.06129395274 / 1.971.
Retain the existing ShareGPT/LongBench controlled model-tokenized prefix pool.
Use 128 distinct prompt IDs, fixed output caps, ignored EOS, and disabled prefix caching.
Verify actual lengths and context limits. Do not silently truncate.
Use open-loop Poisson arrivals without completion-dependent throttling.
Pair inputs, sampling seeds, arrival offsets, GPU count, and precision within every comparison.
Separate calibration and policy selection from held-out arrival seeds.
The reused text pool does not establish unseen-text generalization.
N512 may repeat the 128 prompts only with explicit accounting and disabled reuse.

## 4. Frozen SLO and interpretation

| Model | Request TTFT ms | Request TPOT ms | Additional-work TPOT p95 ms | ITL p99 ms |
| --- | --- | --- | --- | --- |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

Target ITL p99 at most 1.05 times the paired Native value.
This is an empirical non-inferiority target, not a guarantee for future deadlines.
Use independent start-level confidence intervals. Do not bootstrap correlated tokens as independent runs.
Report maximum ITL and counts above the frozen ITL threshold.
A p99 threshold is not a hard per-token deadline guarantee.
Retain all offered failures in attainment denominators.
Finite-cohort drain throughput is not sustainable capacity.

## 5. Evidence and instrumentation

Primary serving uses marker-free output delivery timestamps and bounded host control records.
Use separate instrumented runs for CUDA Event cost calibration and detailed activity traces.
Retain the prior five-percent observer equivalence screen when interpreting such traces as clean activity evidence.
Do not require a trace to pass that screen before a bounded real intervention or protocol correctness test.
If observer equivalence fails, retain the trace as diagnostic and evaluate intervention latency directly.
Measure Native A/A variation near the intervention comparisons.
Do not attribute every paired difference to the profiler when Native restarts also vary.

Physical kernel idle, scheduler-visible readiness, and predicted execution budget are different quantities.
Do not infer safe computation from average utilization.
CUDA Events measure enclosures, not exclusive kernel union or WCET.
Record loaded software versions and resolved profiler libraries when a profiler is used.
Do not infer a version-related cause from package versions alone.

## 6. Phase A: controlled intervention

Use the real qualified recurrent core and graph path.
Initial diagnostic probes may use independent exact states. They do not reduce P work.
Calibrate 16/32/64/128/256-token one-loop slices and full-prompt one-loop work.
Add legal multiple-loop candidates when one-loop execution is correct.
Retain Prelude, injection, Coda, Attention prefix state, and KV dependencies.
Synthetic GEMM, sleep, and busy-wait cannot replace actual recurrent probes.

Start with N24 pilot cohorts. Expand safe candidates to paired N128 starts.
Use at least three independent arrival seeds for final claims and five near a boundary when time permits.
Measure accepted/rejected work, actual executed rows/loops, execution occupancy, TPOT, ITL tails, and host overhead.
Include a zero-probe feature-enabled control and Native A/A starts.
Do not increase work after repeated Decode violations merely to obtain nonzero offloads.
Suspend additional work when three of the last 32 output intervals exceed the frozen ITL threshold.
Also suspend it after any interval exceeds twice that threshold.
Continue accepted requests exactly. Do not skip work, truncate output, or release live state during suspension.
Retain every violation and the resulting throttle state.

## 7. Admission budget and execution budget

Admission estimates the remaining request's resource cost and completion delay after one-way transfer.
Execution estimates the next indivisible scheduler quantum that D may submit now.
Do not combine these budgets into one scalar without recording both meanings.
Use current ready Decode, active batch, last token times, recent service distributions, and uncertainty.
Also use graph readiness, KV leases, workspace, accepted continuation backlog, and observed overrun feedback.
Only current and past observations may influence an online decision.
Future arrivals and realized future output length are forbidden inputs.

Calibrate cost bounds on independent diagnostic states.
Begin with a measured p99 or largest observed cost plus a guard of at least 1 ms or 20%, whichever is larger.
Observed maxima are not certified worst-case bounds.
Do not reduce the frozen guard on confirmation data.
Allow feedback to reduce permitted work or increase the guard after violations.
Separate any later policy tuning from final held-out evaluation.
Log predicted cost, observed cost, budget, overruns, rejection reasons, and resource state.

### Intervention estimator, declared before confirmation

The initial active reserve uses the Native median delivery interval from a separate calibration run.
This initial value is a prediction, not a conservative upper bound for Decode service.
After sixteen observations for an active batch size, the estimator uses that batch's observed p95 plus a 1 ms prediction guard.
The execution candidate retains its separate measured p99 cost guard.
The controller also stops new work after repeated intervals above the calibrated Native-relative target.
These rules require empirical intervention checks. They do not prove future deadline safety.

When no Decode request has an observed outstanding token deadline, the initial policy cap is 200 ms.
This cap is not a measured physical idle gap or a prediction of future arrivals.
A newly ready first Decode step makes the active budget zero until its first output.
An accepted continuation can yield between complete scheduler quanta.
The admission horizon can cover several predicted future opportunities. Record its multiplier separately from the immediate execution budget.

### Controls and attribution

Retain both the original Native wire and a feature-enabled full-state-wire zero-offload control.
The latter controls the transport layout used by Token and Loop continuation.
A difference between these controls is a runtime confound, not a Loop scheduling benefit.
Keep the original Native-relative ITL target in every final comparison.
Also report the matched-wire comparison. Do not replace one reference with the other after inspecting favorable results.

An unarmed request retains the grouped Native execution path.
An armed request that later rejects every offer pays checkpoint and negotiation overhead.
Measure this branch with a separate forced-rejection control.
Do not claim that the zero-method control alone measures the cost of rejected dynamic offers.

The common confirmation cache budget is selected before held-out starts.
Each method warms its required legal geometries under the same policy and cache capacity.
Record startup work, memory, and duration separately.
Exact startup requests never count as online adaptive decisions or measured migrations.

## 8. Dynamic boundary protocol

Use process-local hooks around the qualified runtime. Do not rewrite the private engine on disk.
P retains ownership until an explicit accepted cut commits.
A host submission boundary is not evidence that the GPU completed that boundary.
Wait for the actual native completion event before offering that state.
Use bounded checkpoint candidates or asynchronous D advertisements to avoid polling every loop unnecessarily.
Checkpoint selection is not a precommitted export cut.
Actual export requires a fresh D decision after the completed checkpoint.

Required states are P_RUNNING, P_BOUNDARY_READY, OFFERED, RESERVED, COMMITTED, IMPORTED, D_RUNNING, and RETIRED.
Every offer has a transfer ID, request ID, completed frontier, epoch, and expiry.
An approval for an old epoch cannot stop or duplicate newer P work.
Before commit, timeout or rejection permits P to continue at its next legal frontier.
After commit, D owns the accepted remainder. A late rejection cannot silently resume P.
Preserve native ACK before source retirement and final KV-credit release.
Test stale approval, duplicate message, rejection, timeout, delayed ACK, and cancellation behavior on CPU.
Do not disable native progress guards or ignore GPU errors.

Initial completed-depth candidates are Ouro 1/2/3 and Huginn 24/28/30/31.
These candidates may change only from calibration evidence before confirmation.
Native fallback executes every remaining loop.
Keep untouched Native Atomic execution when a request has no plausible migration opportunity.
Measure a zero-offload control for feature-enabled checkpoint overhead.
Do not label a slower staged path Native.

## 9. D execution and token tiling

After admission, Decode-ready work has priority at each legal scheduling boundary.
D may yield between completed loop or tile quanta. It cannot preempt a running kernel instantly.
Bound admitted continuation backlog and record queue aging and Prefill completion deadlines.
If a full-prompt loop violates empirical Decode protection, investigate exact token-tiled execution.
Define a tile by token width and recurrent-loop count.
Select only dependency-closed ready tiles whose measured cost fits the current execution budget.
Retain per-depth KV prefixes and required hidden/injection state.
Do not treat an arbitrary token-by-loop rectangle as independent.

Prefer one-way ownership of the remaining request. D can tile that owned remainder without returning state to P.
Do not introduce D-to-P state migration silently.
The protocol must distinguish the committed P/D frontier from D's evolving execution tile.
Record both dimensions in every actual decision.
Whole-prompt and tiled results receive separate verdicts.

## 10. Strong Token comparison

| Arm | Candidate family |
| --- | --- |
| A0 | Native Atomic PD |
| A1 | Exact Static Token prefix/suffix |
| A2 | Exact Static whole-prompt Loop cut |
| A3 | Online idle-gated Token prefix/suffix |
| A4 | Online idle-gated whole-prompt Loop cut |
| A5 | Online idle-gated token-tiled variable-loop continuation |
| A6 | Clearly labeled optimistic clairvoyant diagnostic |

Share the estimator, safety contract, credit policy, Decode priority, graph budget, and fallback implementation where applicable.
Token candidates can use efficient large prefixes/suffixes. Do not restrict Token to C256 or one-loop submissions.
Tune legal execution quanta for both families with equal selection budgets.
Same recurrent rows do not imply equal FLOPs, Attention pairs, boundary work, or transfer bytes.
Report Huginn Prelude/Coda work separately.
Distinguish a method's total D occupancy from its largest non-yielding submission.
Do not claim superiority over an unmeasured or deliberately constrained Token baseline.
Generic logical-layer execution at identical boundaries is EQUIVALENT_MECHANISM.

## 11. Correctness ladder

Retain original BF16 weights, KV, qualified numerical fixes, and full fixed R4/R32.
No early exit, approximation, quantization, duplicate loop, missing loop, or invalid KV access is allowed.
Compare exact output tokens, valid KV planes, hidden state, full work, and native lifetime.
Original-HF fidelity remains a separate claim unless directly measured.

1. CPU protocol and dependency regression.
2. N8 Native and static state qualification.
3. N32 mixed-cut protocol qualification.
4. N8 actual online boundary offers and accepts.
5. N32 actual online mixed-cut state qualification.
6. N128 marker-free serving with accepted-work accounting.
7. Optional N512 persistence after repeated success.

Forced mixed cuts establish protocol coverage, not online adaptation.
An adaptive demonstration requires Native fallback and at least two distinct nonzero accepted remaining-loop counts in one online run.
If measured safety permits only one cut, report ADAPTIVE_NOT_DEMONSTRATED instead of forcing another cut.
Do not stop all research merely because detailed profiling fails.
Do stop a state-invalid path and preserve its evidence.

## 12. Metrics and repeat design

Report TTFT, TPOT, ITL, and E2E p50/p95/p99, maximum ITL, and all offered/completed/rejected counts.
Report actual-send QPS separately from offered nominal QPS and completed finite throughput.
Report input/output tokens per second, joint TTFT+TPOT goodput, and separate Decode attainment.
Compute percentiles from request or token records. Do not add component percentiles.
Service TTFT requires actual first model execution, handoff, D continuation, and first output.
If clean actual model entry is unavailable, report NOT_MEASURED instead of substituting admission time.
Record P pressure, D readiness, KV credit, work moved, transfer/ACK, and scheduler overhead.
Keep independent diagnostic probes separate from actual migrated serving requests.
Retain all failures and fixed-seed repetitions. Do not replace unfavorable seeds.

## 13. Decisions

Use GO_ADAPTIVE_LOOP only for actual variable online migration, Decode protection, and repeated serving benefit over Native.
Use GO_LOOP_OVER_TOKEN only after the fair Token comparison also passes.
Use CONDITIONAL_GO for correct adaptive work whose net serving benefit remains uncertain.
Use NO_GO_SLACK only after sufficient real interventions show no useful safe opportunity.
Use NO_GO_GRANULARITY only after both whole-prompt and legal tiled candidates fail to provide useful units.
Use NO_GO_NET_BENEFIT only after actual dynamic migration loses its gain to measured costs.
Use TOKEN_ADVANTAGE when matched Decode-safe Token scheduling performs better.
Use INCONCLUSIVE when actual adaptation or fair clean comparisons remain incomplete.
Record Ouro and Huginn independently. A common GO requires both models.

## 14. Preservation, figures, and closure

Retain immutable configurations, source archives, hashes, GPU mappings, arrival seeds, outputs, failures, and decision logs.
Generate the seven requested figure families from actual evidence.
Actual dynamic-decision plots must contain only real online decisions.
Do not fill unrun methods with zero. Label modeled, diagnostic, instrumented, and marker-free records separately.
Review FluidPD, Prefill Deflection, CDB, and ExactLoop from primary sources.
Do not claim Loop-specific novelty from generic idle borrowing or tied-weight logical-layer cuts.

Use an exclusive quiet lane for clean timing. Stop only verified approved burner sessions.
Adopt the current operational burner supervisor through UID, executable, SID, GPU, and log-path checks.
While no experiment owns resources, restore approved burners on all eight GPUs.
Retry only missing burners after startup failure. Do not stop successful burners or unknown jobs.
Check actual GPU engine ownership and utilization, not only worker creation.
Close all scientific controllers before final restoration.
Complete report, scoped commit/push, and remote-SHA validation before the final deadline.

## 15. Resumption clarifications before held-out confirmation

The final estimator calibration uses N128 Native and matched-wire zero controls.
The calibration arrival seed is 2026110910.
The confirmation arrival seeds are 2026110911, 2026110912, and 2026110913.
All four controls use the same final memory pool and graph policy as their comparison arms.
This separates cohort-size calibration from held-out arrivals. It does not establish unseen-text generalization.

The execution policy protects a predicted deadline margin, not absolute ready-first ordering.
It can submit one bounded Prefill quantum while Decode is ready if the current model admits that quantum.
The next scheduling boundary reevaluates the budget. A running kernel cannot yield early.
Record ready-Decode counts for each submitted quantum.
Evaluate the resulting interference against both the absolute SLO and the Native-relative target.
Do not call this policy strict Decode-ready-first scheduling.

The full-byte state observer changes CPU, pinned-memory, and copy activity.
Its measurements qualify state correctness only.
The Ouro mixed-state stress uses four clusters of eight simultaneous submissions, six seconds apart.
Each request still requires an actual completed source boundary and a fresh destination decision.
No stress request receives a forced Cut.
This stress does not replace primary Poisson serving.

The tiled arm transfers ownership of the entire remaining Loop suffix once.
It adapts legal execution tiles within that owned suffix.
It does not implement arbitrary online Token-by-Loop ownership frontiers or a return to P.
The final comparison must name this scope explicitly.

An unarmed fallback retains Native grouped execution.
An armed request can still incur boundary and negotiation costs before every offer is rejected.
Measure that cost with the rejection control. Do not promise zero added cost for every rejected request.
Keep full-prompt and tiled verdicts separate when full-prompt selection accepts no work.

### Bounded completion scheduling amendment

The GPU deadline remains 16:00:54 KST. The final deadline remains 17:00:54 KST.
The original controller stops new starts at 15:40:54 KST.
Measured startup durations can leave declared confirmation starts unfinished at that boundary.
A separate completion owner can finish already declared confirmation and context controls.
For each start, startup timeout plus run timeout plus 60 seconds must fit before the unchanged GPU deadline.
The inherited dispatcher retains its deadline, exclusive-lane, ownership, progress, and validity checks.
No new scientific cell, configuration, seed, Cut, SLO, or inclusion rule enters this window.
The amendment changes job-start scheduling only. It does not extend either hard budget.
Record the original cutoff and each completion-owner start in the evidence directory.
