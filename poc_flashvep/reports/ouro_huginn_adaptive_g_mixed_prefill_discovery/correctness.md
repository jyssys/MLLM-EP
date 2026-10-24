# Correctness and fidelity

The registered limits are relative L2 ≤0.03 and maximum absolute error ≤0.125.
The study does not relax either limit after measurement.
The table compares each singleton arm with native Atomic on the same P128/C64 request.
It includes the full valid prompt KV planes and the first sampled coda state.

| Model | Executions | Token-exact comparisons | Comparisons | Maximum relative L2 |
| --- | --- | --- | --- | --- |
| huginn | 7 | 6 | 6 | 0.0 |
| ouro | 6 | 5 | 5 | 0.0 |

The separate batched audit inserts P512/C256 after four background requests begin decoding.
It compares Mixed with Hybrid G2/G4 and grouped-only G2.
The target's valid sampled KV tails, hidden state, logits, and bounded output match in the recorded comparisons.
This is not every KV element of every generated token.

The plan ledger contains 1470 prompt-request records and 1470 decode-request records.
Their failure counts are 0 and 0.
The decoded-position ledger checks each recorded position's actual recurrent-depth submissions.
Recorded exit-depth labels alone do not constitute that check.

Matched bounded outputs agree in 2025/2025 recorded reference/candidate pairs.
These pairs include repeated source content and reference self-pairs.
They do not prove population quality equivalence.
The standalone CPU fixtures use small random weights only for scheduler regression tests.
No research performance or quality claim uses those weights.

The heterogeneous-remainder regression and live DP retry remain separate from the earlier fixed-grid evidence.
See [failure diagnosis](failures.md) for the exact error and repair.
GSM8K and MATH-500 accuracy remain NOT_RUN.
