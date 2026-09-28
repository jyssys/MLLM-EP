
# Rare-but-critical paths

Frozen definition: primary-critical AND router-weight<=calibration-median AND F1-prediction<=calibration-median.

- Held-out count/rate: 1 / **0.78%**
- Domain counts: `{'structured_diagram_layout': 1}`
- Layer counts: `{'47': 1}`
- Decode-step counts: `{'16': 1}`
- Mean vision-mass affinity: 0.01824182265164596
- Protection at 20% skip: `{'F0_GENERIC_LOCAL_ROUTER': 0.0, 'F1_MODES_STYLE_GENERIC': 0.0, 'F2_TEXT_PREFILL': 0.0, 'F3_FULL_PREFILL': 0.0, 'F4_VISION_PRIOR': 0.0, 'F5_VISION_SEPARATED': 0.0}`

This test asks whether F5 protects low-router-weight / low-generic-score but
high-measured-damage paths better than the generic control. Its result is folded
into the Stage-B decision; rarity alone is not evidence of vision causality.
