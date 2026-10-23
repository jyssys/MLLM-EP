# Objective catalog

All fourteen families use the same legal native action set.
C changes only at a new chunk. G changes only at a legal recurrent frontier.
An incomplete R1/R2 span receives no immediate completed-token credit.
Every efficiency denominator pays the estimated remaining Full-R4 service.

E is eventual chunk tokens divided by estimated Full-R4 service, normalized by a frozen TRAIN scale.
R combines planned nonpreemptible exposure and already observed decode-gap age.
R is zero when no decode request exists.
R is not a calibrated probability of a future ITL violation.
It omits unresolved remaining decode/coda work and supplies no deadline guarantee.
The service-tail variant uses historical TRAIN tail factors, not counterfactual ITL labels.

| Family | Implemented rule | Main limitation |
| --- | --- | --- |
| O0 | Original U2 eventual Full-R4 efficiency | No explicit decode-risk cost |
| O1 | C-projected E minus G-projected R | Separable diagnostic, not a physical factorization |
| O2 | E minus lambda times coupled R | Utility weights require validation |
| O3 | Maximize E under a normalized risk cap | Estimated cap, not an SLO guarantee |
| O4 | Minimize the risk class, then maximize E | Risk-class boundaries can change choices abruptly |
| O5 | Require minimum normalized E, then minimize R | Efficiency-floor choice affects results |
| O6 | Increase E with observed prompt age and R with decode pressure | Current age is not future TTFT |
| O7 | Penalize empirical service-tail exposure | Service tail is not complete-token ITL CVaR |
| O8 | Pareto actions nearest the state-weighted ideal | Larger Pareto set is not a measured gain |
| O9 | Log E minus lambda times log one-plus-R | A geometric trade-off, not a new primitive |
| O10 | Pay residual yields and retained-state cost | The 0.1-ms yield prior is a sensitivity assumption |
| O11 | Add shape-support and uncertainty penalties | Unsupported estimates remain experimental |
| O12 | Add C/G switching penalties | Fewer switches do not prove better serving |
| O13 | Adjust lambda from at least 32 already observed gaps | Fixed compute budget isolates this feedback from budget adaptation |

The initial grid contains 112 family/weight settings.
Some settings coincide because a family ignores lambda. They are not 112 independent mechanisms.
E-only and R-only endpoints provide independent driver ablations.
O1 projections come from the full common universe before restricting a control's action surface.
This avoids changing objective features merely because G-only fixes C.

The selected four development families are O2, O4, O6, and O10.
Their frozen initial weights are 8, 1, 2, and 8, respectively.
Selection uses historical TRAIN geometry and different mechanisms, not fresh FINAL serving outcomes.
Fresh VALIDATION selected O6, weight 2, before FINAL.
The completed FINAL comparison does not show incremental serving benefit beyond matched G-only.
See `analysis.md` for the NO_GO_INCREMENTAL verdict and its scope.
The catalog does not claim novelty from scalarization, enumeration, or a larger candidate menu.

Source: `objective_families.py`, `objective_features.py`, `objective_normalization_v1.json`, and `objective_grid.csv`.
