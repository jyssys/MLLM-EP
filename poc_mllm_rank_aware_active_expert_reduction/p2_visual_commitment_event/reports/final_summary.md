# PoC-2 — visual commitment event existence gate

Evidence: **MEASURED_FULL_VOCAB_TEACHER_FORCED_TRUE_EP4**. Frozen rule: `{'threshold': 1.6388980748160977e-05, 'width': 3, 'minimum_position': 12}`.

Calibration rule validity: **False**; calibration event rate: **0.000%**.

Held-out event rate: **0.000%** (0/24). No held-out trajectory satisfied the calibration-frozen persistent-low rule, so event-aligned pre/post JSD and its CI are **N/A**. Post persistence: **0.000%**.

| Held-out decode window | Mean JSD | Median JSD |
|---|---:|---:|
| 1–32 | 0.20512 | 0.108715 |
| 36–64 | 0.119598 | 0.0561675 |
| 68–96 | 0.0885648 | 0.0222463 |
| 100–128 | 0.109589 | 0.0404853 |

Same-image repeat-noise JSD: **0**. Generic fixed-position delta: **0.0904062**; request-bootstrap 95% CI **[0.0698906, 0.11105]**; pseudo-event delta: **0.0904062**. Event-minus-generic CI: **N/A**.

Existence gate: **False**. MLLM-specific beyond-position gate: **False**. Final verdict: **NO_VISUAL_COMMITMENT_EVENT**.

Because the existence gate failed, K8→K6 pre/post damage, post-commit rank-aware closure, and a cheap online proxy are **NOT_RUN_BY_GATE**.
