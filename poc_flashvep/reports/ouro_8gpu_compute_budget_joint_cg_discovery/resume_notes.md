# Resume state and required next work

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
