# H1 — visual-conditioned concurrent expert redundancy

Exploratory original-minus-neutral effects (positive means more redundancy in the original condition):

- lower effective rank: 0.056238 (request-bootstrap 95% CI 0.030255, 0.082820)
- mean pairwise output cosine: 0.002646 (request-bootstrap 95% CI 0.000316, 0.004746)
- larger 0.95-cosine cluster: 0.000000 (request-bootstrap 95% CI -0.002604, 0.002604)
- lower leave-one-out reconstruction residual: -0.004946 (request-bootstrap 95% CI -0.006281, -0.003735)

The metrics are internally mixed: effective rank/cosine move slightly toward redundancy, but reconstruction residual moves in the opposite direction and no near-duplicate cluster grows. Exact four-step K6 functional probes show:

- redundancy: candidate damage 0.028444; router-tail 0.017092; random 0.020181; gain vs tail -0.011352 (request-bootstrap 95% CI -0.034930, 0.002167); gain vs random -0.008262 (request-bootstrap 95% CI -0.040849, 0.021820)
- redundancy + rank closure: candidate damage 0.039459; router-tail 0.017092; random 0.020181; gain vs tail -0.022366 (request-bootstrap 95% CI -0.051967, -0.001134); gain vs random -0.019277 (request-bootstrap 95% CI -0.048392, 0.001790)

Thus H1 produced a small discovery structural effect but failed the skip-utility gate. It was frozen to distinguish reproducible-but-useless structure from a discovery fluctuation on untouched data.
