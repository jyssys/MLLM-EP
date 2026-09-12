# Methods generated from the signal

Methods were generated only after the shape-predictor signal passed its gate. None passes the implementation threshold.

| candidate | data-derived rationale | perfect E2E ceiling | added costs/risks | decision |
|---|---|---:|---|---|
| F1 cross-wave expert-major coalescing | combine same-expert live rows across independent waves | <=3.31% GSM8K, <=2.85% HumanEval | queueing, larger dispatch bookkeeping; crowded prior art | KILL |
| F2 tiny-expert specialized execution | post-compaction late work has 68.5--79.4% <=4-row groups | 2.12% / 1.13%; 50%-capture 1.06% / 0.57% | new kernel path and numerical validation | KILL |
| F3 fragmentation-aware wave composition | pair count misses shape costs | <=3.31% / 2.85%; prior ready-set RAWS oracle 0.650% | queue delay and schedule constraints | KILL |
| P1 shape-aware cost model | M1 robust RMSE is 45.2% lower than M0 | no independent action oracle | predictor is not a systems contribution | CHARACTERIZATION ONLY |
| R1 route-plan delta reuse | rank geometry stable, exact route volatile | router-only upper bound 6.43% / 7.22% | exact metadata still changes; hidden/output not reusable | KILL |
| C1 confidence-conditioned expert budget | would follow from high-confidence/diffuse-routing paradox | not computed | premise false; approximate quality; REFLEX/TEAM/DES overlap | KILL |

The separate `METHOD_CANDIDATES.csv` records the evidence and prior-art risk in machine-readable form.

## Why no bounded-delay oracle sweep

The zero-wait, zero-overhead perfect fragmentation ceiling is already below 5%. Any 0.1--2 ms holding window can only reduce request latency headroom, so a queue sweep cannot rescue the branch. Throughput might improve in another objective, but the requested direct E2E gate would still fail.

