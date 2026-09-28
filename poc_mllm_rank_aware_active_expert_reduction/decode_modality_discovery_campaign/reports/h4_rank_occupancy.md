# H4 — modality-conditioned EP-rank occupancy

- B8 singleton-expert effect: 0.131944 (request-bootstrap 95% CI -0.347222, 0.611111)
- rank max/mean effect: 0.011921 (request-bootstrap 95% CI 0.001091, 0.026796)
- working-set HHI effect: 0.000869 (request-bootstrap 95% CI -0.000016, 0.002166)
- mean max-rank active-expert reduction at K6: rank-closure 3.639, router-tail 1.917, random 1.368 experts.
- exact K6 rank-closure probe: candidate damage 0.027185; router-tail 0.017092; random 0.020181; gain vs tail -0.010093 (request-bootstrap 95% CI -0.029179, 0.001061); gain vs random -0.007004 (request-bootstrap 95% CI -0.028126, 0.006175)

Rank-aware choice creates more structural closure, reproducing the generic Q1/Q3 premise, but its modality effect is not robust and K6 exact damage is not better than generic controls.
