# Active frontier after analytical filtering

The frontier is intentionally small because the headroom gate precedes GPU.

| Rank | Candidate | Why it survived initial source reading | Kill test | Current result |
|---:|---|---|---|---|
| 1 | token-scoped combine release (A02--A04) | would change a fundamental dependency boundary | exact hidden/output correctness plus request-level ready-time oracle | no measured mass; UNKNOWN, not GPU-promoted |
| 2 | worker/engine ownership split (A11--A12) | could expose cross-request slack without changing model math | direct request E2E with state-aware scheduler oracle | no trace mass; UNKNOWN, prior vLLM architecture risk |
| 3 | device-side metadata lifecycle (A15--A19) | repeated source-level host materialization | no-copy/reuse diagnostic with equal output budget | <1% upper bound; DROP |
| 4 | mid-step admission (A28) | could alter the step atomicity contract | identical request stream, admission at safe boundaries | no mass; crowded chunked-prefill prior |
| 5 | scoped notification (A36) | DeepEP barrier scope is structural | preserve collective correctness, compare critical path | 1.09% direct cap; DROP |
| 6 | state-aware resource contract (A32/A40/A41) | common regimes are unexplained | clock/allocator telemetry and causal perturbation | no measured request mass; UNKNOWN |

No row is `PROMISING`: the required analytical threshold is >=20% for a GPU
counterfactual in this branch.
