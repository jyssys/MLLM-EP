# Frozen cohort and intervention contract

- Calibration: 8 requests; fresh held-out: 24 requests.
- Domain counts: `{'calibration': {'dense_relation': 2, 'docci': 2, 'nocaps': 2, 'scene_layout': 1, 'textcaps': 1}, 'heldout': {'dense_relation': 4, 'docci': 5, 'nocaps': 5, 'scene_layout': 5, 'textcaps': 5}}`.
- The universal prompt is byte-identical (`62ac73f0155eda59725ce06d35e271d198218bd21b3f2ea11ba92376d6c95fc3`).
- Q2/Q2-2 request and image identities were excluded before capture.
- Same-domain counterfactuals were matched within 10% visual-token count.
- One predeclared layer/step site per request; selection used no routing, grounding, or damage outcome.
- Diagnostic reduction is native K8→K6/K4 tail removal, retained-weight renormalization, and no substitution.
