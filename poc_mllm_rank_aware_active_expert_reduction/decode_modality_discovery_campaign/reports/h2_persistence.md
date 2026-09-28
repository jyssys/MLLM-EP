# H2 — decode-step routing persistence

- top-k persistence original-minus-neutral: 0.002994 (request-bootstrap 95% CI -0.001733, 0.007823)
- router-distribution persistence: 0.000257 (request-bootstrap 95% CI -0.003365, 0.004019)
- EP-rank-set persistence: -0.006684 (request-bootstrap 95% CI -0.019013, 0.005384)
- persistence-selected K6 gain vs router-tail: -0.094322 (request-bootstrap 95% CI -0.103243, -0.085458)
- persistence-selected K6 gain vs random: 0.126712 (request-bootstrap 95% CI 0.112521, 0.140675)

No request-bootstrap CI establishes a positive modality-conditioned persistence effect, and the diagnostic local skip choice is worse than router-tail.
