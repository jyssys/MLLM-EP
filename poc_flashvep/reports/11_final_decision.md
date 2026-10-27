# Decision checkpoint — not a final verdict

Status: `IN_PROGRESS_BASELINE`.
No GO, CONDITIONAL, NO_GO, or BLOCKED verdict is justified yet.
Model and runtime identities appear in `00_runtime_audit.md`.

| Phase | Current status | Evidence |
|---|---|---|
| 0 Runtime audit | PASS for Group A | Model revision, ownership, nonzero remote assignments |
| 1 Quality reproduction | IN_PROGRESS | Full fresh HF and EP4 evaluations run in the durable queue |
| 2 Refinement sensitivity | NOT_RUN | Fresh baseline gate pending |
| 3 EP cost model | PARTIAL_OBSERVER_ONLY | Two fresh sanity traces, no calibrated realistic estimator |
| 4 Sparse runtime | NOT_RUN | Quality and headroom gates pending |
| 5 Full candidate evaluation | NOT_RUN | No candidate selected |
| 6 Ablation and final decision | NOT_RUN | No speedup or quality-preservation claim |

## What is established

The exact non-preview Mini checkpoint loads on the available hardware.
Group A performs physical expert-sharded EP4 execution.
Two native requests preserve output tokens and NFE through the identity hook.
The original historical quality counts match their saved sample records.
The HF and native temperature-zero paths implement different token selection.

## What is not established

Fresh full benchmark accuracy is not complete.
Group B EP4 crossover is not complete.
No approximate policy passed a quality test.
No sparse dispatch or expert-kernel reduction is demonstrated.
No isolated E2E speedup is measured.
No incremental EP-aware gain over refinement-only allocation is demonstrated.

## Next acceptance gate

Complete fresh GSM8K and HumanEval scoring.
Investigate protocol differences without tuning test accuracy.
Verify physical EP4 and no-op parity on Group B.
Then test causal refinement sensitivity on the frozen non-test partitions.
Do not expand sparse runtime engineering before the conservative headroom gate passes.

The durable state resides in `../results/refinement_aware_ep_expert_allocation/state/campaign.json`.
The reproduction entry point is `../refinement_ep_allocation/README.md`.
The supervisor restores owned burners when its evaluation queue finishes.
The previous Looped MoE research remains stopped.
