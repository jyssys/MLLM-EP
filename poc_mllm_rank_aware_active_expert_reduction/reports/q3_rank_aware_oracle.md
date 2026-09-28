
# Q3 rank-aware structural oracle

| Mapping | Policy | Path reduction | Max-rank active/weight reduction | Total active reduction | Remote-path reduction |
|---|---|---:|---:|---:|---:|
| EP2 | O1_RANDOM | 20.312% | 9.457% | 10.191% | 20.275% |
| EP2 | O2_NATIVE_TAIL | 20.312% | 15.337% | 16.475% | 20.695% |
| EP2 | O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER | 20.312% | 24.049% | 33.882% | 19.298% |
| EP4 | O1_RANDOM | 20.312% | 8.589% | 10.255% | 20.362% |
| EP4 | O2_NATIVE_TAIL | 20.312% | 13.816% | 16.475% | 20.245% |
| EP4 | O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER | 20.312% | 25.383% | 30.047% | 20.550% |
| EP8 | O1_RANDOM | 20.312% | 6.896% | 10.116% | 20.321% |
| EP8 | O2_NATIVE_TAIL | 20.312% | 11.156% | 16.475% | 20.306% |
| EP8 | O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER | 20.312% | 28.172% | 27.660% | 20.179% |

At held-out B8 and matched 20.312% path reduction, EP4 native-tail reduces the maximum rank working set by 13.816%; O4 reduces it by 25.383% (**+11.567pp**).

O4 is a nondeployable future-route structural upper bound. The caption trace lacks dense safe-candidate labels, so this is matched path budget—not a demonstrated matched functional-damage budget. Q4 remains required for safety.
