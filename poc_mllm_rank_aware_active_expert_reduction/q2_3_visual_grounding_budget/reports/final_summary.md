# Final summary — visual-grounding budget-risk Phase 1

The bounded result is **NO_STRUCTURED_DAMAGE_SIGNAL**. The primary score was frozen before damage measurement, evaluated on 24 fresh image-hash-disjoint held-out requests, and required both observational and same-token image-swap causal-delta evidence plus incremental value over a calibration-frozen generic control. MoDES was not compared, and no expert-selection or latency runtime was implemented.

```text
BASELINE_STATUS: READY
COHORT_STATUS: FROZEN_8_CALIBRATION_24_FRESH_HELDOUT
HELDOUT_REQUESTS: 24
PRIMARY_VISUAL_SCORE: LOG_VISUAL_TO_TEXT_PREFILL_SUBSPACE_ENERGY_RATIO
K6_DAMAGE_MEAN: 0.0111
K4_DAMAGE_MEAN: 0.0793
VISUAL_SCORE_DAMAGE_SPEARMAN_CI: -0.0357 [-0.4171, 0.3619]
CAUSAL_DELTA_SPEARMAN_CI: -0.2861 [-0.5739, 0.1088]
BEST_FROZEN_CONTROL: full_energy
VISUAL_INCREMENT_CI: -0.2504 [-0.8166, 0.3702]
PHASE1_GATE: FAIL
PHASE2_PROMOTION: NO
MODES_COMPARISON: NOT_RUN_BY_USER_SCOPE
EXPLORATORY_NONLINEAR_RESULT: NO_VISUAL_INCREMENTAL_RESCUE
TPOT_E2E_CLAIM: NOT_RUN
VERDICT: NO_STRUCTURED_DAMAGE_SIGNAL
NEXT_ACTION: STOP_PHASE2
```
