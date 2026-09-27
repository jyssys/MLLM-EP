# Vision-conditioned decode reduction — final summary

Both independent tracks stop at Stage A. This is a measured negative result,
not a runtime failure: the bounded TRUE-EP4 capture returned exact frozen token
sequences and 1,530 complete real decode token/layer groups after removing
audited dummy/prefill rows.

## Track A: consolidation

The non-deployable oracle protected the two highest-weight paths and chose the
least-error replacement among those already-computed primary paths for each
secondary path. At the required 20% path reduction it achieved 19.14% active
union reduction, but the substituted branch vectors were not remotely close:

- mean raw expert-output cosine: **0.344683**;
- P90 router-weighted contribution relative-L2: **0.170652**;
- P90 token mixture relative-L2: **0.492585**.

The frozen gate required strong branch similarity and P90 weighted/mixture
relative-L2 at most 0.025. It fails decisively. Consequently A1–A4 propagation,
held-out quality, EP2/EP4 timing, and a modified-route EP8 replay were not run.
Vision-conditioned criticality cannot rescue a frontier whose future-truth
oracle is already unsafe; therefore no A4-vs-A2/A3 claim is made.

## Track B: temporal output reuse

Same-expert route recurrence is abundant but does not imply output stability.
Within the captured contiguous windows, the W16 recurrent-path rate is 66.94%.
Yet the best recurrent cached weighted-contribution error is 11.23%, its median
is 101.99%, and the reuse rate is **0%** at every registered 1%, 5%, and 10%
error threshold for W=1/2/4/8/16. Even a 50% error budget accepts only 4.59%.

The 15–20% low-error reuse gate therefore fails. No B1–B4 cached output was
propagated through the model, and no quality or physical timing was run. The
safe EP8 structural effect is zero. The all-pool sparse-checkpoint recurrence
maximum (42.637%) is explicitly unsafe and is not reported as a candidate gain.

## Decision

The result rejects both tested exact bounded families on this model/workload.
It does not reject vision-conditioned decode foresight itself; it shows that
using that signal to force these particular branch substitutions or stale
output reuse does not provide safe headroom. No production optimization was
implemented.

BASELINE_STATUS:
READY_REUSED_AND_TRUE_EP4_VECTOR_CAPTURE_VERIFIED

BATCH_REGIME_STATUS:
STRUCTURAL_B1_B2_B4_B8_B16_COMPLETE_B32_INSUFFICIENT_POOL

TRACK_A_ORACLE_HEADROOM:
20.00% path reduction / 19.14% active-union reduction, but mean branch cosine 0.344683 and P90 mixture relative-L2 0.492585; FAIL

TRACK_A_QUALITY_STATUS:
NOT_RUN_STAGE_A_ORACLE_FAILED

TRACK_A_VISION_INCREMENT_VS_GENERIC:
N/A_STAGE_A_STOP_NO_A2_A3_A4_PROPAGATION

TRACK_A_EP2_TPOT_GAIN:
N/A_NOT_RUN_BY_SPEC_GATE

TRACK_A_EP4_TPOT_GAIN:
N/A_NOT_RUN_BY_SPEC_GATE

TRACK_A_EP8_STRUCTURAL_EFFECT:
STRUCTURAL_EP8_PROJECTION_ONLY_NOT_RUN_NO_PROMOTED_POLICY

TRACK_A_VERDICT:
A_INSUFFICIENT_CONSOLIDATION_HEADROOM

TRACK_B_ORACLE_REUSE_HEADROOM:
0.00% reuse at <=1%, <=5%, and <=10% weighted-contribution relative-L2 for W1/2/4/8/16; FAIL

TRACK_B_QUALITY_STATUS:
NOT_RUN_STAGE_A_FAILED

TRACK_B_VISION_INCREMENT_VS_GENERIC:
N/A_STAGE_A_STOP_NO_B2_B3_B4_PROPAGATION

TRACK_B_EP2_TPOT_GAIN:
N/A_NOT_RUN_BY_SPEC_GATE

TRACK_B_EP4_TPOT_GAIN:
N/A_NOT_RUN_BY_SPEC_GATE

TRACK_B_EP8_STRUCTURAL_EFFECT:
STRUCTURAL_EP8_PROJECTION_ONLY_SAFE_EFFECT_0; recurrence-only maximum 42.637% is unsafe

TRACK_B_VERDICT:
B_INSUFFICIENT_REUSE_HEADROOM

COMBINED_RECOMMENDATION:
PURSUE_NEITHER

NEXT_ACTION:
Stop these two method families; preserve the evidence infrastructure and require a new mechanism with a positive future-truth local-error oracle before any quality or runtime work.
