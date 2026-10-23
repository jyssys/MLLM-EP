# Compute-budget campaign progress

Phase: 5. Status: RUNNING.
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
| p3_budget_pilot_plan.json | 32 | 30 | 22 | 0 |

Last commit: `d3a7d714059c455c64c1fcb23f39a6e35d055e01`.
Recognized owned burner GPUs: [].
During quiet contention controls, otherwise idle GPUs are reserved measurement resources.
Other idle periods use owned burners. Unrelated GPU jobs remain untouched.

Online Joint: RUNNING. Qualified capacity and new adaptive quality: NOT_RUN.
Sixteen CPU tests passed before GPU qualification.
These tests cover native software fixtures and policy definitions, not pretrained quality.
The paused native-vLLM PP program remains excluded.
