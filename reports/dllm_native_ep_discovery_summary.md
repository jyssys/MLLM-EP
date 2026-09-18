# dLLM-native Expert Parallelism discovery summary

## Outcome

Both tracks are **NO-GO** for implementation. Track A found a strong and held-out reproducible dLLM-native structural fact—live-MASK adjacent edges persist—but the systems/semantic chain breaks twice. Only 3.349% of measured STAY branches have <=5% raw output relative-L2, and the impossible all-STAY source-cache oracle saves only 0.0399% of the whole simulated EP8 routed stage because current live-MASK work is diluted by prompt/prior/current-decoded full rows. The practical E3 EP8 stage gain is 0.0000036% (rounded to 0.0000% below).

Track B is exact-semantics but weak. EP8 mean most-affine-rank share is 16.131%, early-ref1 predicts the future home only 28.446% of the time, and O2 removes 1.244% of current-block but only 0.0205% of whole remote bytes. After even the minimum 4,096-byte M1 migration, the repeated dLLM EP8 net stage change is -0.0362%, so neither dLLM nor AR one-shot breaks even.

## Evidence boundaries

- Route persistence: measured threshold-.95 GSM8K-128 model trace; discovery 0--63, held-out 64--127.
- Branch stability: measured duplicate-compute diagnostic on 16 fixed requests and layers (1, 5, 10, 14, 19); model outputs were unmodified.
- EP4/EP8 timing: **SIMULATED-EP4/EP8-EP2-CALIBRATED**, not physical EP4/EP8.
- Quality: no reuse policy rollout was run because the systems upper bound failed first.
- No production cache, migration runtime, kernel, threshold policy, freeze policy or pruning method was implemented.

## Required final summary

TRACK_A_LIVE_MASK_STAY_ROUTE_FRACTION:
69.305% (held-out)

TRACK_A_STAY_ROUTER_MASS_FRACTION:
73.319% (held-out)

TRACK_A_STAY_EP_COST_FRACTION:
0.731% calibrated live-MASK stage removable; 68.821% pair-count persistence

TRACK_A_STABILITY_ORACLE_REUSABLE_FRACTION:
3.349% at raw branch relative-L2 <=5%

TRACK_A_PRACTICAL_REUSABLE_FRACTION:
0.051% of held-out STAY branches selected by E3

TRACK_A_EP4_STAGE_GAIN:
0.0000% (practical E3, simulated)

TRACK_A_EP8_STAGE_GAIN:
0.0000% (practical E3, simulated)

TRACK_A_QUALITY_RUN:
NO

TRACK_A_QUALITY_DELTA:
N/A

TRACK_A_DLLM_SPECIFIC_CONTROL:
PASS

TRACK_A_VERDICT:
NO-GO

TRACK_B_TOKEN_HOME_AFFINITY:
16.131% mean most-affine EP8-rank share

TRACK_B_EARLY1_HOME_MATCH_TO_FUTURE:
28.446%

TRACK_B_EP4_REMOTE_BYTE_REDUCTION:
1.464% current-block / 0.0242% whole

TRACK_B_EP8_REMOTE_BYTE_REDUCTION:
1.244% current-block / 0.0205% whole

TRACK_B_EP4_STAGE_GAIN_AFTER_MIGRATION:
-0.0358% (M1)

TRACK_B_EP8_STAGE_GAIN_AFTER_MIGRATION:
-0.0362% (M1)

TRACK_B_AR_ONE_SHOT_BREAK_EVEN:
NO

TRACK_B_DLLM_REPEATED_BREAK_EVEN:
NO

TRACK_B_DLLM_SPECIFIC_CONTROL:
FAIL

TRACK_B_VERDICT:
NO-GO

STRONGER_TRACK:
NEITHER

PRIMARY_DISCOVERY:
Adjacent live-MASK expert edges are genuinely persistent, but their branch outputs are usually not stable enough to reuse and they are a tiny share of full physical EP work.

PRIMARY_BLOCKER:
Full-row prompt/prior dilution kills Track A, while weak EP8 destination affinity and migration cost kill Track B.

NEXT_ACTION:
Do not implement either runtime; seek a dLLM-native mechanism that removes dominant full-row EP work without approximate branch reuse.

DO_NOT_IMPLEMENT_PRODUCTION_RUNTIME_AUTOMATICALLY:
true
