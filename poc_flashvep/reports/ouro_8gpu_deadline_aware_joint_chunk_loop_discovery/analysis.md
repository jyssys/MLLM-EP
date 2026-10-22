# Eight-GPU deadline-aware chunk/loop campaign

Verdict: **BLOCKED for online deadline control**.

The experiment completed numerical diagnosis, fixed-policy characterization, and the first blind predictor test.
It did not establish a safe candidate-specific deadline model.
Phases 5–6 remain NOT_RUN under the specification gates.
This does not prove that Joint scheduling has no useful region.

## Current results

- All 64 identity/contention jobs completed. Concurrent-worker drift is not uniformly neutral.
- All 160 isolated numerical comparisons are exact.
- All 1,728 main event-lane cohorts pass Full-R4 work and token-clock checks.
- The matrix contains 530,826 quantum observations and 176,787 observed token gaps.
- The quality pilot contains 342 natural generations: 114 questions for each of Atomic, G2, and Mixed.

One historical mismatch now has an exact operator-level cause reproduction.
Identical BF16 input and pretrained MLP weights reproduce Atomic M1024 and Grouped M48 outputs exactly.
This applies to that first divergence, not every historical mismatch.
The new G3 boundary guard fixes a separate reproduced scheduler descriptor error.

At P8177/N8/C1024, median prefill quanta are 103.424 ms for Atomic and 51.970 ms for G2.
However, staged G4 already changes the observed tail without reducing G.
Its paired mean cohort-p99 change versus Atomic is −28.87%, with 95% CI [−45.27%, −7.87%].
G2 versus staged G4 is +5.25%, with 95% CI [−24.17%, 34.67%].
These are diagnostic event-lane comparisons across four process-start pairs, with three cohorts per start.
They are not clean serving speedups or pooled-p99 confidence intervals.
Tuned Mixed remains a necessary strong control.

The selected quantum lookup yields held-out MAE 1.935 ms and aggregate upper-bound coverage 98.861%.
Its 95% coverage interval is [97.484%, 99.729%], using process-start bootstrap resampling.
That aggregate hides Atomic-prefill coverage of only 79.141% and staged-prefill coverage of 90.901%.
The full-token lookup has MAE 17.311 ms and a 230.204-ms calibration margin.
Its retrospective stock-policy labels include future interruptions.
They cannot directly estimate a changed candidate schedule.

The offline budget envelope finds partial-work operating points unavailable to G4 under tight initial budgets.
It does not establish completed-prompt, deadline, or E2E gains.
Read `joint_headroom.md` before interpreting that plot.

All three fixed policies score GSM8K 47/64 (73.4375%) and MATH 28/50 (56.0%).
MATH complete-correct accuracy is 26/50 (52.0%), with 8/50 capped answers.
All 114/114 output sequences match across these pilot policies.
The paired accuracy interval [0, 0] percentage points is degenerate and does not prove population equivalence.
Adaptive Joint accuracy remains NOT_RUN.

Do not build A3 from the current global margins.
The smallest next experiment is a newly registered path-stratified predictor and event-readiness test on fresh source families.
It must check a candidate-specific complete-token DAG before any online SLO claim.
Existing CONDITIONAL verdicts and raw evidence remain unchanged.

## Historical registration snapshot

The following section preserves the initial progress snapshot.
Its NOT_RUN statements describe registration time, not the current result.

The campaign uses Base Ouro-1.4B with BF16 and Static R4.
Eight physical GPUs run independent single-GPU experiments.
The paused native-vLLM PP research remains excluded.

## Current evidence

The original staged changes remain intact.
A separate source checkout preserves the previous RLT runtime.
The new checkout retains its source pin and the removable grouped-prefill patch.
G3 has an additional configuration allowance and needs GPU qualification.

The saved-output check reproduces the batched parity failure.
It finds 86 exact sequences among 96 historical comparisons from twelve cohorts.
Those counts do not replace the prior 72/80 table.
The first-cause category remains UNKNOWN.

The upstream CPU split-prefill suite produced 33 passes and 28 skips.
Three new frontier/leakage tests passed in a separate command.
Six legal-action/leakage definition tests also passed.
These tests do not establish pretrained GPU quality.

The registered contention campaign has 64 device/start jobs.
It compares one, two, four, and eight simultaneous workers on every GPU.
Each job measures two representative context/concurrency conditions.
The calibration results are not adaptive-policy speedups.

New input families and quality-pilot inputs are frozen.
The quality pilot contains 64 GSM8K and 50 MATH cases per policy.
The proposed cost-surface plan has 96 source-split/shape/chunk jobs.
Each job crosses the six runtime paths within one GPU context.
Those jobs require numerical qualification before dispatch.

## Claims not yet supported

No live Adaptive C-only, G-only, or Joint result exists.
No new quality or sustainable-capacity result exists.
The study cannot yet determine whether Joint exceeds adaptive C-only or Mixed.
Historical CONDITIONAL verdicts remain unchanged.

## Next decision

Finish and review the interference controls.
Then run targeted first-divergence and isolated G1/G2/G3/G4 tests.
Resolve any real state or stream bug before affected adaptive work.
Use exploratory cost analysis only when its correctness limits remain explicit.
