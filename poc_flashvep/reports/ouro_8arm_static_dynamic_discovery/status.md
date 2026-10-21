# Study status

Updated UTC: 2026-10-21T08:36:29.235050+00:00.

| Arm | Parallelism | Mode | GPUs | Completed target | Status | Fidelity |
|---|---|---|---|---|---|---|
| S4 | single | static | 4 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| T4 | tp4 | static | 4,5,6,7 | 64/300 + 39/100 | IN_PROGRESS | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| L4 | layer_pp4 | static | 4,5,6,7 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| O4 | loop_pp4 | static | 4,5,6,7 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| SD | single | dynamic | 4 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |
| TD | tp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| LD | layer_pp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |
| OD | loop_pp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |

The controller can continue independently of the chat turn. Inspect `state/continuation_controller.json` and `state/continuation_stage.json`.
