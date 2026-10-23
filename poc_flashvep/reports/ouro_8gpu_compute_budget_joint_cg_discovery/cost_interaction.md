# Conditional C/G cost interaction

The controlled component queue completed eight independent GPU/model-process blocks.
It crossed C128/C512/C2048 with G1/G2/G3/G4 on each device.
Three input families repeat within each block.
Every final chunk ends at the same actual KV endpoint, P2048 or P8192.
This avoids comparing different final context lengths as a G-only effect.

The queue contains 648 endpoint quantum records and 288 complete-chunk component records.
Every prompt completed exactly four recurrent passes.
The one-output-token component runs are not full-answer quality evaluations.
Their timings use the concurrent observer lane, not quiet clean serving.

Define the interaction as:

`I = T(C2,G2) - T(C1,G2) - T(C2,G1) + T(C1,G1)`.

At P2048 and depth zero, C512→C2048 and G1→G4 gave I=31.725 ms.
Its paired process-block 95% interval was [28.319, 38.280] ms.
This contradicts a separable additive cost model for those conditions.
It does not establish a serving speedup.

| Exact endpoint | Additive categorical MAE, ms | Product-only MAE, ms | Interactive categorical MAE, ms |
| --- | ---: | ---: | ---: |
| P2048 | 4.937 | 16.000 | 1.594 |
| P8192 | 14.142 | 23.505 | 13.570 |

These are mean leave-one-GPU/process-out errors on component cell medians.
They are not blind natural-serving prediction errors.
The same three input families appear across process blocks.
The P8192 interaction estimates have substantially more variation.
Do not generalize the P2048 result to every context or backend.

The full-chunk table sums prelude and all required endpoint recurrent groups.
It also records the first-to-last endpoint interval and dispatch count.
No partial R1/R2 span receives completed-token credit.
The component sum is not additive attribution of a whole request's critical path.

Observed cost can reflect matrix shape, repeated setup, GPU staging, and CPU launch gaps.
This study does not isolate their individual causal contributions with hardware bandwidth counters.
It does not treat parameter size as measured HBM traffic.

Data: `controlled_endpoint_service.csv`, `interaction_effects.csv`, `cost_separability_models.csv`, and `full_chunk_completion_cost.csv`.
