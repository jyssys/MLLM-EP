# Compute-budget campaign progress

Phase: 2. Status: RUNNING.
These counts come from immutable attempt completion markers.

| Queue | Planned | Started | Complete | Failed |
| --- | --- | --- | --- | --- |
| p0_contention_plan.json | 64 | 64 | 64 | 0 |
| p1_numerical_plan.json | 8 | 8 | 8 | 0 |
| p1_observer_control_plan.json | 2 | 2 | 2 | 0 |
| p1_fairness_plan.json | 8 | 8 | 8 | 0 |
| p2_service_plan.json | 64 | 59 | 51 | 0 |
| p2_paired_service_plan.json | 8 | 0 | 0 | 0 |
| p2_controlled_service_plan.json | 8 | 0 | 0 | 0 |
| p1_dynamic_frontier_plan.json | 8 | 0 | 0 | 0 |

Last commit: `ff303f04de6dd298d95b3230198372f55dbce72d`.
Recognized owned burner GPUs: [].
During quiet contention controls, otherwise idle GPUs are reserved measurement resources.
Other idle periods use owned burners. Unrelated GPU jobs remain untouched.

Online Joint: NOT_RUN. Qualified capacity and new adaptive quality: NOT_RUN.
Sixteen CPU tests passed before GPU qualification.
These tests cover native software fixtures and policy definitions, not pretrained quality.
The paused native-vLLM PP program remains excluded.
