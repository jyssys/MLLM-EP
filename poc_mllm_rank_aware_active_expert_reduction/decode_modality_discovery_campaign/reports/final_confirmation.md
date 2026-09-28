# Untouched fresh confirmation

- Frozen primary: `H1_REDUNDANCY` / `redundancy`; confirmation requests: 24.
- G1 structural modality effect: **True**, 0.061379 (request-bootstrap 95% CI 0.033303, 0.091110).
- G2 exact K6 skip utility: **False**; gain vs router-tail -0.013605 (request-bootstrap 95% CI -0.044405, 0.003632); gain vs random 0.016842 (request-bootstrap 95% CI -0.034751, 0.075181).
- G3 domain robustness: **True**; positive domains 5/5; `{'dense_relation': 0.07228895207589965, 'docci': 0.03847999456810314, 'nocaps': 0.04399508972717281, 'scene_layout': 0.021500115715973322, 'textcaps': 0.13662458729681592}`.
- K6 predicted closure events/request: primary 0.625, rank-aware variant 1.042, router-tail 0.833, random 0.792.
- Verdict: **MODALITY_EFFECT_NOT_USEFUL_FOR_SKIPPING**.

The functional damage contains target-logprob drop, Jensen–Shannon divergence, and top-1 flip over a four-token teacher-forced horizon. It is not a correlation-only result.
