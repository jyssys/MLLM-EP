# Stage 1 + residual balance summary

B1 uses the existing train-0..63 global-static replica map with budget 8 and conservatively leaves communication unchanged. B2 lowers expert critical time to rank mean and is an impossible perfect-balance upper bound.

The prior H1 EP8 perfect-balance-only reference was 12.37%. Crossing 15% here is meaningful only when the selective S1 substrate and B2 oracle are both stated; it is not a live runtime result.

P2 quality safety is promoted on n=128. P6 and its EP-aware incremental comparison are paired on n=32 only; P6 was not promoted to 128 because its measured EP-specific increment did not pass the 3-point discovery gate.

| Configuration | n | quality delta (pp) | EP4 S1 | EP4 S1+B1 | EP4 S1+B2 | EP8 S1 | EP8 S1+B1 | EP8 S1+B2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p1_random_f0_r75_age1_sanity8 | 8 | +0.00 | 9.64 | 13.18 | 15.78 | 9.47 | 15.45 | 21.58 |
| p2_confhigh_f0_r50_age1_sanity8 | 8 | -12.50 | 1.45 | 5.28 | 8.14 | 1.28 | 7.75 | 14.46 |
| p2_confhigh_f0_r75_age1_sanity8 | 128 | +3.12 | 5.98 | 9.75 | 12.38 | 5.90 | 11.93 | 18.43 |
| p2_confhigh_f0_r75_age2_sanity8 | 8 | -12.50 | 15.14 | 18.47 | 20.92 | 14.99 | 20.63 | 26.36 |
| p2_confhigh_f1_r75_age1_sanity8 | 8 | +0.00 | 18.53 | 21.72 | 24.07 | 18.38 | 23.78 | 29.31 |
| p3_conflow_f0_r75_age1_sanity8 | 8 | +0.00 | -15.29 | -10.56 | -7.39 | -15.21 | -7.93 | 0.04 |
| p4_ageconf_f0_r75_age1_sanity8 | 8 | +0.00 | 8.64 | 12.28 | 14.85 | 8.54 | 14.44 | 20.73 |
| p5_epload_f0_r75_age1_sanity8 | 8 | +0.00 | 3.63 | 7.47 | 10.20 | 3.49 | 9.84 | 16.39 |
| p6_confep_l01_f0_r75_age1_sanity8 | 8 | +0.00 | 8.64 | 12.28 | 14.85 | 8.54 | 14.44 | 20.73 |
| p6_confep_l05_f0_r75_age1_sanity8 | 32 | +0.00 | 6.66 | 10.40 | 13.01 | 6.58 | 12.58 | 19.02 |
| p6_confep_l10_f0_r75_age1_sanity8 | 8 | +0.00 | 7.88 | 11.55 | 14.14 | 7.78 | 13.72 | 20.07 |

The required `Stage1+Balance` fields below use B2 perfect balance (oracle); B1 global-static values remain in the table and do not cross the same gate.

Total stage delta includes P6-induced NFE/trajectory changes. S1 max/mean does not improve; therefore the delta is not evidence of material EP load shaping.


PREVIOUS_STEP_PREDICTABILITY:
PASS

ONE_STEP_MASK_FREEZE_SAFETY:
PASS

MAX_SAFE_FREEZE_AGE:
1

BEST_GENERIC_SELECTIVE_POLICY:
P2 previous-confidence-high, F0, 75% active, max_freeze_age=1

BEST_EP_AWARE_SELECTIVE_POLICY:
P6 confidence+EP8 load, lambda_max=0.5, F0, 75% active, max_freeze_age=1

GENERIC_ACTIVE_WORK_REDUCTION:
26.33%

EP4_EP_AWARE_INCREMENTAL_STAGE_GAIN:
0.68 percentage points

EP8_EP_AWARE_INCREMENTAL_STAGE_GAIN:
0.68 percentage points

EP4_STAGE1_ONLY_GAIN:
6.66% (S1)

EP8_STAGE1_ONLY_GAIN:
6.58% (S1)

EP4_STAGE1_PLUS_BALANCE_GAIN:
13.01% (S1+B2 perfect-balance oracle)

EP8_STAGE1_PLUS_BALANCE_GAIN:
19.02% (S1+B2 perfect-balance oracle)

EP8_JOINT_HEADROOM_GE_15_PERCENT:
YES

QUALITY_DELTA_AT_BEST_JOINT_POINT:
+0.00 pp on n=32

PRIMARY_BLOCKER:
EP-aware selection adds less than the required 3 percentage points over confidence-only; the >=15% joint result requires the perfect-balance oracle.

VERDICT:
HOLD

DO_NOT_IMPLEMENT_PRODUCTION_SPARSE_RUNTIME_AUTOMATICALLY:
true
