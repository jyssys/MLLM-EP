# Discovery matrix

The score is the conservative minimum of a modality-effect lower bound and a diagnostic K6 skip-gain lower bound. Values from different mechanisms are only a gate/ranking aid, not a shared physical unit.

| Hypothesis | conservative score | both lower bounds positive |
|---|---:|---|
| H1_REDUNDANCY | -0.166195 | no |
| H2_PERSISTENCE | -0.103243 | no |
| H3_CIRCUIT | -0.144661 | no |
| H4_RANK_OCCUPANCY | -0.347222 | no |

The frozen candidate is `H1_REDUNDANCY` / `redundancy`. Rationale: H1 structural modality effect passed; exact skip gate failed and is frozen for negative confirmation.

H1–H5 were all executed even though H1 did not pass the functional gate. No derived H6/H7 was added because the trace exposed no strong, direct-to-intervention signal beyond the predeclared set.
