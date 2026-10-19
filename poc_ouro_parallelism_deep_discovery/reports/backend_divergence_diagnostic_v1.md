# Native/reference output divergence

## Scope

This CPU analysis reopens 128 admitted native/reference pairs from the completed baseline extension.
Each pair has identical prompt token IDs.
The [v2 summary](../analysis/backend_divergence_diagnostic_v2/summary.json) pins all raw inputs and adds explicit EOS/terminal flags.
The [request CSV](../analysis/backend_divergence_diagnostic_v2/requests.csv) retains zero-based first differences and finish reasons.
The original v1 output remains unchanged.
No model or GPU job runs for this diagnostic.

## Results

| Base | Domain | Exact IDs/N | Internal difference/N | First token agrees/N | Median common prefix among divergent pairs |
|---|---|---:|---:|---:|---:|
| 1.4B | GSM8K | 20/32 | 12/32 | 32/32 | 28.5 tokens |
| 1.4B | MATH-500 | 4/32 | 28/32 | 32/32 | 131.5 tokens |
| 2.6B | GSM8K | 25/32 | 7/32 | 32/32 | 53 tokens |
| 2.6B | MATH-500 | 0/32 | 32/32 | 32/32 | 170 tokens |

All79 non-exact pairs contain unequal tokens within both stored arrays.
For78 pairs, the first difference precedes both terminal positions and touches no EOS token.
The remaining pair, Base1.4B `math500/test/58`, first differs at native EOS0 versus staged token198, at index186.
There are zero prefix-only length differences.
Thus, the non-prefix classification must not imply that every difference precedes termination.
The independent review identified this required distinction before publication.
An internal difference appears before token32 for 7/32, 2/32, 0/32, and 4/32 pairs, respectively.

Request-bootstrap 95% intervals for the internal-difference percentages are:

- Base1.4B GSM8K: 37.500%, CI [21.875, 53.125]%.
- Base1.4B MATH-500: 87.500%, CI [75.000, 96.875]%.
- Base2.6B GSM8K: 21.875%, CI [9.375, 37.500]%.
- Base2.6B MATH-500: 100.000%, CI [100.000, 100.000]%.

The last interval describes these sampled bits.
It does not prove population-wide divergence.

## Limits

Different BF16 backends, TP layouts, cache implementations, and stopping mechanisms remain potential factors.
This diagnostic identifies no first unequal operator or numerical cause.
Canonical decoded text and native API text remain separate fields.
Their text equality is not a numerical contract or a stopping-cause test.

First-token agreement does not establish whole-generation parity.
Likewise, equal final accuracy does not establish identical output trajectories.
No candidate quality-preservation or speedup result follows.
The frozen parser, length-cap penalty, and scores remain unchanged.
