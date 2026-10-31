# Host E2E Component Percentages

Evidence: MEASURED_HOST_EXCLUSIVE_INTERVALS. Each setting has 24 requests at G2.
Each column divides summed component time by summed request E2E. It totals 100% before display rounding.
The columns are not GPU-kernel or DMA decompositions.

| Component | Ouro C256 % | Ouro C2048 % | Huginn C256 % | Huginn C2048 % |
| --- | --- | --- | --- | --- |
| harness_delivery | 0.067 | 0.208 | 0.024 | 0.039 |
| KV_or_slot_admission_wait | 13.348 | 0.015 | 47.142 | 43.818 |
| reserve_and_P_receive | 0.179 | 0.500 | 0.050 | 0.054 |
| actual_P_queue_to_first_submission | 3.268 | 0.010 | 0.000 | 0.000 |
| P_prelude_and_pre_core_queue_observed | 0.193 | 0.071 | 0.547 | 0.558 |
| P_service_observed_not_kernel | 57.034 | 13.222 | 8.325 | 4.871 |
| exposed_handoff_observed | 0.940 | 1.205 | 0.057 | 0.275 |
| D_activation_observed | 0.220 | 1.070 | 0.012 | 0.010 |
| first_D_sample_or_continuation | 0.013 | 0.060 | 0.000 | 0.000 |
| generation_observed | 24.737 | 83.639 | 43.842 | 50.374 |

A Prefill service interval can contain queueing and host control work. It is not pure computation.
Cumulative transfer completion time can overlap computation. It cannot form another additive percentage.
