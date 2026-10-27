# Ablations

Status: `VALIDATION_ABLATIONS_COMPLETE_DEPLOYMENT_GATED`.
Model and runtime identities appear in `00_runtime_audit.md`.

| Ablation | Required control | Status |
|---|---|---|
| Identity hook | Same full expert work | PASS on 32 HumanEval requests and two smoke requests per EP4 group |
| Uniform versus router | Same semantic removal budget | 16 validation requests per policy at two budgets. Actual kernels remain dense |
| Refinement versus EP-only versus joint | Same actual executed edges | Matched fixed-trajectory structural replay only |
| Refinement signal | Same router and semantic edge budget | P2/P3 full free-running comparison complete |
| EP signal | Same refinement state and edge budget | NOT_RUN |
| Weight normalization | Original retained weights versus explicit renormalization | NOT_RUN |
| Edge budget | 0%,5%,10%,15%,20% | 0%,5%,20% diagnostics complete. 10%/15% not run |
| NFE | Natural generation and additional matched-NFE control | NOT_RUN |
| Hardware | Group A/B crossover | Baseline repeats and physical operator comparison. No deployed candidate crossover |
| Isolation | One group quiet versus EP4×DP2 | Quiet P0 and operator runs complete. No matched DP2 throughput comparison |
| Batch | 1,2,4,8,16,32 | B1 pilot and operator tests only |
| Block length | 32,64 | Block32 pilot and operator tests only |
| Layer and routing skew | Separate strata | NOT_RUN |

The structural traffic counterfactual is not a quality ablation.
Its removed edges did not execute through a sparse runtime.
It cannot establish actual kernel work reduction.
The campaign must not fill missing policy rows with projected speedups.

## Additional completed controls

The zero-removal semantic control matched all 16 retained-all validation outputs and NFE counts.
An independent baseline restart matched eight additional repeated requests.
The first failed zero-removal attempt remains in the evidence archive.
The revised diagnostic retains communication input buffers until request-end synchronization.

The final uniform control uses an independent random generator.
The original structured hash remains a superseded diagnostic, not the final P1 baseline.
Non-contiguous mask regression tests failed before explicit row/slot indexing.
They pass after that correction.
CPU test evidence appears in `cpu_validation.json` under the result directory.

## NFE and answer transitions

At the higher budget, uniform omission matches P0's aggregate 15/16 correctness.
It still changes one correct answer to wrong and one wrong answer to correct.
Its paired accuracy interval is [−18.75, +18.75] pp.
Mean per-question NFE increases by 56.71%.
One request reaches the generation cap and remains wrong.
Aggregate accuracy alone therefore hides both trajectory changes and workload inflation.

P2/P3 at the higher budget match P0's observed correctness vector.
Their mean per-question NFE ratios are 1.03036× and 1.01751×, respectively.
This small NFE difference does not establish an E2E gain.
P3 adds confidence observation and selection overhead in the diagnostic.
None of these runs removes physical expert work.

## Remaining conditional tests

Live matched-edge P3/P4/P5 accuracy and request latency remain unmeasured.
Weight renormalization remains untested.
The batch/block expansion remains untested after the incremental-benefit gate.
No missing condition is labeled OOM or passed by inference.

## Failure accounting

| Attempt | Observed status | Resolution | Evidence |
|---|---|---|---|
| Initial A zero-removal control | FAILED exact parity | Retain modified buffers through request-end synchronization. Revised 16-request control passes | `logs/A_semantic_zero_control.log` |
| Initial P1 structured-hash control | SUPERSEDED, not uniform enough | Independent-generator P1 replaces it. Old outputs excluded from final P1 counts | `ep4/A_semantic_P1_05/`, `ep4/B_semantic_P1_20/` |
| Non-contiguous deletion mask | CPU regression test failed | Explicit row/slot indexing. Torch and NumPy tests pass | `test_semantic.py`, `test_replay.py` |
| Controller process-snapshot race | FAILED on a vanished `/proc` entry | Retry only vanished snapshots. Retain ownership assertions | `logs/sensitivity_supervisor.log` |
| Initial event-instrumented operator lane | OBSERVER_ONLY | Add a separate clean wall-time pass. Preserve initial observer data | `ep4/*_isolated_operator_cost/` |
| Group B isolated P0 outliers | MEASURED, cause unresolved | Retain all observations and wide uncertainty. Do not assign a hardware cause | `latency_samples.csv` |

Paths in the table refer to the campaign result directory, except source test files.
Exact per-attempt GPU-hours are `NOT_MEASURED`.
The command and job ledgers retain available start, stop, and exit timestamps.
Allocated wall duration does not equal measured GPU busy time.
