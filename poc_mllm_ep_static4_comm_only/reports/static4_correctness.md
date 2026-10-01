# Static4 correctness

Correctness passed three deterministic repeats covering 576 physical
rank-layer comparisons. Each Static4 call was compared against a native shadow
call on the identical BF16 hidden states, route IDs, and route weights.

| Check | Result |
|---|---:|
| Router ID/weight hashes recorded | PASS |
| Same routed paths and ownership | PASS |
| Original source-row reconstruction | PASS |
| Maximum absolute error | 0.0 |
| Maximum relative error | 0.0 |
| Minimum cosine | 0.9999998808 |
| All outputs finite | PASS |
| Deterministic injected next token | identical (`1986`) |

No token, route, weight, expert, precision, or model output approximation was
introduced. Verdict: `PASS_EXACT_BF16_OUTPUT`.
