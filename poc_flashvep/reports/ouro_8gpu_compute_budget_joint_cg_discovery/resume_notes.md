# Resume state and required next work

## Final completion note

The GPU experiment campaign is complete. Do not restart its queues automatically.
The final incremental decision is NO_GO_INCREMENTAL.
All 12 quiet confirmation starts completed, with 168 cohorts and 1,008 request executions.
G1/J1 per-request plans, output token IDs, and lengths matched on all 144 paired requests.
Joint/G-only quiet output-throughput ratios ranged from 0.996× to 1.003×.
The extra C search added no executed action advantage in these cells.

All 798 quality generations and frozen scoring completed.
J1 scored GSM8K 47/64 and MATH 32/50 extracted, with 30/50 completed-correct and six caps.
G1 had the same correctness labels on all 114 questions. Token sequences matched 113/114.
Native F0A/M0/M1 scored 47/64 and 28/50 extracted, with 26/50 completed-correct and eight caps.
C1A/C1S had GSM8K 45/64 and 43/64. The latter's paired loss was -6.25 pp, with CI [-12.5,-1.5625].
Do not claim uniform quality equivalence. See quality.md and preserved wrong-case files.

Online feedback, H2/H3, full 1,819-question accuracy, and sustainable-rate capacity remain NOT_RUN after the negative gate.
All eight GPUs have owned burners restored.
Final packaging uses the archive/hash manifest and a scoped final commit/push.
Read the current Git HEAD, remote branch, and archive manifest to check that step.
Do not infer its status from an earlier tool-session ID.

## Superseded running checkpoints below

This is a running campaign, not a completed result.
The root branch is `flashvep/ouro-8gpu-budget-joint-cg-optimization-poc`.
The authoritative specification has 466 lines and was read completely before task actions.
The reporting skill and required references were also read.

The new package and report root use `ouro_budget_joint_cg` and `ouro_8gpu_compute_budget_joint_cg_discovery`.
The external checkout is `/home/esjung/external/ouro-budget-joint-rlt`.
It is a detached worktree at `4a350344027960a48384626aa646f1765fda6d4f`.
Only the prior G2/G3 config/scheduler/executor files were copied through apply_patch.
No native-vLLM or PP engine changed.

The original staged binary-diff hash remains `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Preserve all unrelated staged additions and dirty deletions.
Commit only this new package and report root.
The user explicitly authorizes the final push.

## Current checkpoint after Phase 0 and initial Phase 1

### Latest checkpoint: TRAIN finished and quiet confirmation running

The new branch checkpoint `08f2fe77c9669ab3c503baca774ea10ed9f5dfef` was committed and pushed successfully.
All original staged changes still have the preserved binary-diff hash.
There are additional uncommitted report/analysis/quality-coordinator changes after that checkpoint.

All 64 fresh cost jobs and all eight controlled endpoint jobs completed.
The fresh service dataset has 298,842 quantum records across 1,056 observer cohorts.
The frozen model has 334 lookup cells and remains immutable.
Blind Atomic MAE=9.613 ms, upper exceedance=5.786%, block CI [2.343,8.646]%.
Blind Staged Core MAE=2.769 ms, exceedance=3.862%, CI [2.237,6.578]%.
Blind Prelude MAE=0.797 ms, exceedance=4.481%, CI [3.097,8.960]%.
Only three source families supply single-prompt blind prediction targets.
All model-risk labels remain EXPERIMENTAL_BUDGET_RISK. No deadline safety claim.

The additional cycled-frontier queue completed on all eight GPUs.
All 16 non-reference comparisons matched tokens, hidden, logits, and KV exactly.
It covers G1→G3 and G1→G2→G1 continuation patterns.
The local software test suite now has 17 passing tests.

The complete TRAIN budget pilot finished: 32 jobs, 448 natural observer cohorts, all eight budgets and both shapes.
Joint beats severely constrained C-only at some budgets.
G-only is often very close to Joint. Native Mixed usually has much lower P99 at P8177/N8.
No Joint GO is established.
Do not interpret large gains against an infeasible/smallest-progress C-only fallback as sufficient practical evidence.

The selected-credit diagnostic was corrected during TRAIN before reading pilot performance outcomes.
Earlier workers remain immutable and carry their own source snapshots.
Measured R4 completion always comes from actual observer spans.
The quiet confirmation uses the corrected implementation only.
Core runtime files must remain frozen during this confirmation.
Analysis-only edits do not change the running GPU executor.

The quiet plan is `p5_quiet_confirmation_plan.json`: 12 one-at-a-time jobs over B40/B120, P2033/N4 and P8177/N8.
There are three independent model-process starts and two FINAL_ONLINE cohorts per cell.
Warm inputs use CALIBRATION, not FINAL_ONLINE. Cap512, natural EOS allowed.
All eight devices are isolation-reserved. Burners are intentionally OFF during this quiet queue.
This is a focused finite-cohort confirmation, not Poisson capacity.
The original dispatcher tool session was 98603. Check live state if that session is lost.

A detached `confirmation_controller` waits for the quiet queue, analyzes it, then registers/runs the B40 quality pilot.
Its process record is `state/confirmation_controller_process.json` and its log is recorded there.
At creation PID was 45671. Do not trust an old PID without checking its current cmdline.
The planned quality pilot reuses 64 GSM8K + 50 MATH full answers, cap4096, greedy, preserved stops/evaluator.
It now includes seven arms: C1A/C1S/G1/J1/M1/M0/F0A.
F0A is stock native Atomic C1024. M0 is stock native Mixed.
The quality reference is F0A, not C1A.
The evaluator asserts that its actual imported extractor matches the frozen hash.
The controller stops for manual review after quality. It does not make a final verdict or push automatically.

Still required after queues finish:
- Review quiet comparisons without mixing TRAIN and FINAL lanes. `live_pairs.py` now keys these separately.
- Run `export_results.py` for main_table, raw request events, queues, and experiment matrix.
- Run final budget analysis/figures after clean windows. Do not add heavy CPU/I/O work during quiet timing.
- Create final analysis.md, phase_gates.md/decision_gates.json, budget_stall_relationship.md, quality.md, and final summaries.
- Fill matched_pairs.csv from QUIET_FINAL effects. Do not invent capacity/rate measurements.
- Record H2/H3, online feedback, and production rate sweeps as NOT_RUN if incremental evidence remains absent.
- Hash/archive retained raw data locally, preserve failed evidence, and provide exact paths/hashes.
- Review scoped changes, commit only the new package/report root, and push the final branch.
- Restore all eight owned burners before final handoff. Preserve unrelated processes and original staged hash.

The user asked for a progress-only answer during this running turn.
That answer was delivered. The research remains active and must continue.

All 64 Phase 0 jobs completed. Final 8-worker drained-throughput drift ranged from about -4.20% to +0.896%, depending on shape/device.
Use `gpu_contention.md` and the paired CSV, not the earlier partial estimates.
All eight numerical jobs completed: 64 execution records, 48 reference comparisons.
All 48 comparisons had exact hidden/logit/KV/token agreement on the isolated probes.
There were 3,056 ready-stream event records. Maximum alignment bound was 0.066094 ms.
All eight fairness jobs completed: 160 observer cohorts and 480 request pairs.
All 480 pairs matched output length and actual executed decode-loop work.
Token IDs matched in 426/480 pairs. Accuracy remains untested for the new policy.
The quiet observer controls completed: 143/144 token-sequence matches. One native Mixed case remains unresolved.

Phase 2 data collection is running under `data_controller.py`.
Check `/proc` and immutable completion markers before starting another dispatcher.
The original tool session after server recovery was 82146. It may be lost on another server restart.
The standalone dispatch child can survive its parent, so do not assume no GPU work remains.

A detached `pilot_controller.py` is registered in `state/pilot_controller_process.json`.
It waits for all new cost data and the frozen v1 model.
Then it runs the extra dynamic-frontier probes, builds the TRAIN-only budget pilot, and executes it.
It stops for manual review before final held-out, feedback, quality, or capacity expansion.
Its log path is recorded in that process-state file.
`detach.py pilot_controller` detects a live matching coordinator before creating another.

The old `p2_paired_service_plan.json` is registered but NOT_RUN.
The controlled endpoint queue supersedes it for causal C/G component interaction.
That queue uses P2048/P8192 and C128/512/2048, which share the exact final KV endpoint.
It is a component benchmark, not a full-answer serving workload.
Only four TRAIN sources provide P16384. The component P8192 excerpts cycle three distinct families from that pool.

## Earlier calibration registration

`p0_contention_plan.json` contains 64 jobs over all GPUs and 1/2/4/8 worker groups.
Every device uses fixed disjoint CPU affinity `16*GPU .. 16*GPU+15`.
The plan uses legacy prompts for continuity, not fresh held-out inputs.
Quiet calibration reserves all eight devices, including otherwise idle GPUs.
Owned burners return after the queue.
The new resource wrapper recognizes old and new owned burner logs and supervisor records.
It writes only to the new report root.
Never start a second dispatcher while `state/dispatcher.lock` is held.

The Phase 0 dispatcher session was started from the root tool session 86024.
Inspect the current tool/session state and raw completion markers before assuming it ended.
`controller.py` can wait for that queue, analyze contention, and execute only registered numerical jobs.
It stops for review after numerical checks.
It does not automatically dispatch all cost or serving experiments.

## Frozen new sources

`source_family_manifest.json` has 1,099 prompts.
Alpaca rows 1000–1599 and arXiv test rows 2000–2199 are new.
The code asserts family disjointness from the preceding study.
Partitions are TRAIN, VALIDATION, CALIBRATION, NEW_HELDOUT, FINAL_ONLINE.
Variants from one article share a partition.
Natural text supplies P512/P2033/P8177/P16384 where available.
No repeated padding creates unsupported long prompts.

## Implemented and CPU-tested

`CompletionFairness` wraps a native policy.
It cannot clear debt from merely changing `loops_done`.
It can clear debt after native coda delivery changes generated-token count.
Request cancellation/ID reuse and one-of-many completion have tests.
`ChunkActions` wraps only `_make_scheduled_request`.
It stores per-request staged spans and never globally mutates C/G config.
New C is selected only at a fresh chunk boundary.
The CPU integration test runs exact R4 with variable C/G and confirms cleanup.
All five local tests passed.
Four additional optimizer definition tests subsequently passed, for nine tests at that checkpoint.
`optimizer.py` implements single-action enumeration with full remaining R4 costs.
U0 remains a diagnostic partial-work objective.
U2 includes all residual groups and an optional measured decode-yield penalty.
No measured cost model or online optimizer is fitted yet.
`BudgetSelector` exposes an explicit overbudget progress fallback.
Before live use, add phase-policy admission that defers unaffordable prefills for decode when possible.
Limit that fallback by an idle/starvation rule and record every overrun.
Do not silently treat it as budget compliant.

`budget_controller.py` is a causal past-emission definition tool.
Fixed, feedback, and pressure-style modes are not live serving results.
Feedback requires at least 32 past gaps and a 16-output dwell.
It uses a 15% hysteresis band.
These provisional defaults require TRAIN-only review before held-out use.

`ServiceObserver` wraps `prepare_on_gpu` and `_execute` after native waits.
It separately records host dispatch, queue delay, native event readiness, service intervals, and completed prompt credit.
It must be validated on GPU before interpreting timing.
Its close method releases bound runner references to avoid the previous profiler-lifetime OOM.

## Registered next jobs

`p1_numerical_plan.json`: eight GPU jobs, isolated P128/512 or P1536/2048.
Each compares Atomic, fixed G2, fixed G3, and scripted C512/G2 then fresh C1024/G4.
The preregistered hidden/logit/KV relative-L2 tolerance is 0.03.
Token parity remains separate.
Full-R4 and legal-frontier violations must stop the affected mechanism.

`p1_fairness_plan.json`: eight fixed-policy stock/completion-fairness jobs.
All five modes cross on each GPU with two source cohorts.
`p2_service_plan.json`: 64 fresh service jobs at P2033/N4 and P8177/N8.
Each job crosses all six paths and two cohorts at one C.
This is targeted new readiness-separated service, not a repeat of the old opaque duration target.
Do not dispatch the broad queue before reviewing the new observer and integrity probes.

## Still required

1. Finish/review P0 and numerical GPU jobs. Do not stop the whole study for nonneutral contention.
2. Run fairness-only ablations and fresh per-path cost collection.
3. Implement a path-conditioned service lookup with separate new blind calibration. Do not use old global E1 or D1.
4. Implement discrete legal candidate enumeration and a full-R4 remaining-cost objective.
5. Compare fixed budgets 10/20/40/60/80/120/160/240 ms.
6. Run safe live C-only/G-only/Joint pilots even if overrun risk requires an EXPERIMENTAL label.
7. Only extend to feedback budget, H2/H3 planning, or qualified capacity if physical pilot evidence warrants it.

A new chunk may have a separate prelude quantum and recurrent quantum.
Budget adherence is per issued native quantum.
Complete-chunk utility must still include prelude, all residual loops, staging, and scheduling costs.
Do not count partial R1/R2 work as completed prompt tokens.
Native Mixed stays G1 and remains a strong separate baseline.
