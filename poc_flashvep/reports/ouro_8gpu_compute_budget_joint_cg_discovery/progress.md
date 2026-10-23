# Compute-budget campaign progress

Phase: 6. Status: COMPLETE_PHASED_NEGATIVE_DECISION.
These counts come from immutable attempt completion markers.

| Queue | Planned | Started | Complete | Failed |
| --- | --- | --- | --- | --- |
| p0_contention_plan.json | 64 | 64 | 64 | 0 |
| p1_numerical_plan.json | 8 | 8 | 8 | 0 |
| p1_observer_control_plan.json | 2 | 2 | 2 | 0 |
| p1_fairness_plan.json | 8 | 8 | 8 | 0 |
| p2_service_plan.json | 64 | 64 | 64 | 0 |
| p2_paired_service_plan.json | 8 | 0 | 0 | 0 |
| p2_controlled_service_plan.json | 8 | 8 | 8 | 0 |
| p1_dynamic_frontier_plan.json | 8 | 8 | 8 | 0 |
| p3_budget_pilot_plan.json | 32 | 32 | 32 | 0 |
| p5_quiet_confirmation_plan.json | 12 | 12 | 12 | 0 |
| p6_quality_plan.json | 8 | 8 | 8 | 0 |

Source HEAD when this snapshot was prepared: `08f2fe77c9669ab3c503baca774ea10ed9f5dfef`.
This is not a self-referential final published commit identifier.
Recognized owned burner GPUs: [0, 1, 2, 3, 4, 5, 6, 7].
During quiet contention controls, otherwise idle GPUs are reserved measurement resources.
Other idle periods use owned burners. Unrelated GPU jobs remain untouched.

Online Joint: MEASURED_TRAIN_AND_QUIET. Qualified capacity: NOT_RUN. Quality: MEASURED_PUBLIC_PILOT_NOT_EQUIVALENCE_PROOF.
Sixteen CPU tests passed before GPU qualification.
These tests cover native software fixtures and policy definitions, not pretrained quality.
The paused native-vLLM PP program remains excluded.
