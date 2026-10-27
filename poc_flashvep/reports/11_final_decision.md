# Decision checkpoint — not a final verdict

Status: `IN_PROGRESS_REFINEMENT_DISCOVERY`.
No GO, CONDITIONAL, NO_GO, or BLOCKED verdict is justified yet.
Model and runtime identities appear in `00_runtime_audit.md`.

| Phase | Current status | Evidence |
|---|---|---|
| 0 Runtime audit | PASS for Groups A and B | Model revision, ownership, nonzero remote assignments |
| 1 Quality reproduction | PILOT_COMPLETE_ONLY | User-approved GSM8K 128 and HumanEval 32 per backend |
| 2 Refinement sensitivity | BASELINE_TRACE_ANALYSIS_ONLY | 16 retained-all trajectories. Expert omission remains untested |
| 3 EP cost model | PARTIAL_OBSERVER_ONLY | Two fresh sanity traces, no calibrated realistic estimator |
| 4 Sparse runtime | NOT_RUN | Quality and headroom gates pending |
| 5 Full candidate evaluation | NOT_RUN | No candidate selected |
| 6 Ablation and final decision | NOT_RUN | No speedup or quality-preservation claim |

## What is established

The exact non-preview Mini checkpoint loads on the available hardware.
Group A performs physical expert-sharded EP4 execution.
All 32 native HumanEval pilot requests preserve output tokens and NFE through the identity hook.
The original historical quality counts match their saved sample records.
The HF and native temperature-zero paths implement different token selection.

## What is not established

Fresh full benchmark accuracy is not complete.
Group B smoke correctness passed, but clean timing crossover is not complete.
No approximate policy passed a quality test.
No sparse dispatch or expert-kernel reduction is demonstrated.
No isolated E2E speedup is measured.
No incremental EP-aware gain over refinement-only allocation is demonstrated.

## Next acceptance gate

Test controlled expert-contribution omission on the frozen non-test validation requests.
Investigate protocol differences without tuning test accuracy.
Then test causal refinement sensitivity on the frozen non-test partitions.
Do not expand sparse runtime engineering before the conservative headroom gate passes.

The baseline pilot and retained-all trace collection are complete.
Their state files reside in `../results/refinement_aware_ep_expert_allocation/state/`.
No allocation policy or sparse expert execution is validated yet.
The reproduction entry point is `../refinement_ep_allocation/README.md`.
The pilot supervisor restores owned burners when its evaluation queue finishes.
The earlier automatic full-evaluation and isolated-baseline queues are canceled.
Their design remains available but does not execute automatically.
See `refinement_ep_pilot_scope_amendment.md` for the current scope.
Sparse policy development still requires a later scientific gate review.
The previous Looped MoE research remains stopped.
