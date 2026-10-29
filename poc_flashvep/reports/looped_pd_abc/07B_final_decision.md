# Track B: decision

Reviewed decision file present.

Timing correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.

Active model scope: ouro, huginn. Dory is deferred by explicit user instruction. Existing Dory evidence remains historical, not a new verdict.

Campaign status: `FINALIZED_WITH_LIMITATIONS`. Evidence snapshot: 2026-10-29T11:01:16.077115+00:00.

Overall status: `CONDITIONAL`. Immutable-frontier transfer is internally correct. Narrow two-GPU signals remain. Generic layerwise streaming, reference fidelity, three-start uncertainty, and incomplete capacity tests prevent GO.

| Model | Status | Reason |
| --- | --- | --- |
| ouro | CONDITIONAL | P512/C4 gives 1.090x p95 E2E versus DP Mixed, with a 1.012-1.153x conditional CI. Moderate-SLO goodput is only 1.034x. Other repeated regimes lose. No general loop-specific GO. |
| huginn | CONDITIONAL | At P512/C4 and absolute ITL 150ms, bounded goodput improves versus the best measured control. E2E remains worse than DP/Mixed. Official-reference mismatch and generic streaming controls remain unresolved. |

| Model / regime | Baseline | Candidate | Starts / frozen pairs | Exact outputs / N | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup | Paired 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro / serving / P128 C1 | DP | B | 3 / 3 | 72/72 | 635.783 | 711.718 | 0.893× | 0.876–0.939× |
| ouro / serving / P128 C1 | DP_Mixed | B | 3 / 3 | 72/72 | 644.216 | 711.718 | 0.905× | 0.892–0.930× |
| ouro / serving / P128 C1 | PD | B | 3 / 3 | 72/72 | 703.471 | 711.718 | 0.988× | 0.969–1.029× |
| ouro / serving / P128 C1 | PD_G1 | B | 3 / 3 | 72/72 | 703.790 | 711.718 | 0.989× | 0.974–1.041× |
| ouro / serving / P128 C1 | PD_A_control | B | 3 / 3 | 72/72 | 700.595 | 711.718 | 0.984× | 0.969–1.014× |
| ouro / serving / P128 C1 | PD_A_coalesced | B | 3 / 3 | 72/72 | 697.890 | 711.718 | 0.981× | 0.966–1.020× |
| ouro / serving / P512 C4 | DP | B | 3 / 3 | 72/72 | 1102.345 | 981.602 | 1.123× | 1.106–1.173× |
| ouro / serving / P512 C4 | DP_Mixed | B | 3 / 3 | 72/72 | 1070.138 | 981.602 | 1.090× | 1.012–1.153× |
| ouro / serving / P512 C4 | PD | B | 3 / 3 | 72/72 | 1508.632 | 981.602 | 1.537× | 0.957–1.719× |
| ouro / serving / P512 C4 | PD_G1 | B | 3 / 3 | 72/72 | 1172.164 | 981.602 | 1.194× | 0.912–1.473× |
| ouro / serving / P512 C4 | PD_A_control | B | 3 / 3 | 72/72 | 1444.279 | 981.602 | 1.471× | 1.197–1.501× |
| ouro / serving / P512 C4 | PD_A_coalesced | B | 3 / 3 | 72/72 | 1359.870 | 981.602 | 1.385× | 1.250–1.420× |
| ouro / serving / P2048 C8 | DP | B | 3 / 3 | 72/72 | 2180.926 | 4678.616 | 0.466× | 0.413–0.730× |
| ouro / serving / P2048 C8 | DP_Mixed | B | 3 / 3 | 72/72 | 2810.066 | 4678.616 | 0.601× | 0.524–0.926× |
| ouro / serving / P2048 C8 | PD | B | 3 / 3 | 72/72 | 4777.884 | 4678.616 | 1.021× | 0.908–1.466× |
| ouro / serving / P2048 C8 | PD_G1 | B | 3 / 3 | 72/72 | 4305.518 | 4678.616 | 0.920× | 0.745–1.072× |
| ouro / serving / P2048 C8 | PD_A_control | B | 3 / 3 | 72/72 | 4769.008 | 4678.616 | 1.019× | 0.904–1.089× |
| ouro / serving / P2048 C8 | PD_A_coalesced | B | 3 / 3 | 72/72 | 4097.656 | 4678.616 | 0.876× | 0.651–0.991× |
| ouro / serving_long128 / P512 C4 | L128_DP | L128_B | 3 / 3 | 48/48 | 3449.890 | 2898.505 | 1.190× | 1.180–1.197× |
| ouro / serving_long128 / P512 C4 | L128_DP_Mixed | L128_B | 3 / 3 | 48/48 | 2842.743 | 2898.505 | 0.981× | 0.970–1.002× |
| ouro / serving_long128 / P512 C4 | L128_PD | L128_B | 3 / 3 | 48/48 | 3712.556 | 2898.505 | 1.281× | 0.982–1.286× |
| ouro / serving_long128 / P512 C4 | L128_PD_G1 | L128_B | 3 / 3 | 48/48 | 3815.594 | 2898.505 | 1.316× | 0.978–1.321× |
| ouro / serving_long128 / P512 C4 | L128_PD_A_control | L128_B | 3 / 3 | 48/48 | 3784.293 | 2898.505 | 1.306× | 0.967–1.311× |
| ouro / serving_long128 / P512 C4 | L128_PD_A_coalesced | L128_B | 3 / 3 | 48/48 | 3944.288 | 2898.505 | 1.361× | 1.293–1.364× |
| huginn / serving / P128 C1 | DP | B | 3 / 3 | 72/72 | 2626.455 | 2889.741 | 0.909× | 0.862–0.972× |
| huginn / serving / P128 C1 | DP_Mixed | B | 3 / 3 | 72/72 | 2641.859 | 2889.741 | 0.914× | 0.870–0.980× |
| huginn / serving / P128 C1 | PD | B | 3 / 3 | 72/72 | 2736.798 | 2889.741 | 0.947× | 0.910–1.017× |
| huginn / serving / P128 C1 | PD_G1 | B | 3 / 3 | 72/72 | 2876.210 | 2889.741 | 0.995× | 0.924–1.084× |
| huginn / serving / P128 C1 | PD_A_control | B | 3 / 3 | 72/72 | 2879.762 | 2889.741 | 0.997× | 0.992–1.091× |
| huginn / serving / P512 C4 | DP | B | 3 / 3 | 72/72 | 3295.142 | 4065.456 | 0.811× | 0.761–0.896× |
| huginn / serving / P512 C4 | DP_Mixed | B | 3 / 3 | 72/72 | 3537.789 | 4065.456 | 0.870× | 0.813–0.956× |
| huginn / serving / P512 C4 | PD | B | 3 / 3 | 72/72 | 4784.864 | 4065.456 | 1.177× | 1.112–1.266× |
| huginn / serving / P512 C4 | PD_G1 | B | 3 / 3 | 72/72 | 4309.330 | 4065.456 | 1.060× | 0.993–1.170× |
| huginn / serving / P512 C4 | PD_A_control | B | 3 / 3 | 72/72 | 4424.257 | 4065.456 | 1.088× | 0.973–1.167× |
| huginn / serving / P2048 C8 | DP | B | 3 / 3 | 72/72 | 8860.635 | 18358.209 | 0.483× | 0.475–0.490× |
| huginn / serving / P2048 C8 | DP_Mixed | B | 3 / 3 | 72/72 | 11429.320 | 18358.209 | 0.623× | 0.587–0.631× |
| huginn / serving / P2048 C8 | PD | B | 3 / 3 | 72/72 | 27593.847 | 18358.209 | 1.503× | 0.965–1.526× |
| huginn / serving / P2048 C8 | PD_G1 | B | 3 / 3 | 72/72 | 18921.083 | 18358.209 | 1.031× | 0.983–1.047× |
| huginn / serving / P2048 C8 | PD_A_control | B | 3 / 3 | 72/72 | 18408.542 | 18358.209 | 1.003× | 0.948–1.016× |
| huginn / serving_long128 / P512 C4 | L128_DP | L128_B | 3 / 3 | 48/48 | 11907.071 | 13319.751 | 0.894× | 0.893–0.997× |
| huginn / serving_long128 / P512 C4 | L128_DP_Mixed | L128_B | 3 / 3 | 48/48 | 11883.670 | 13319.751 | 0.892× | 0.881–0.987× |
| huginn / serving_long128 / P512 C4 | L128_PD | L128_B | 3 / 3 | 48/48 | 13660.943 | 13319.751 | 1.026× | 1.004–1.109× |
| huginn / serving_long128 / P512 C4 | L128_PD_G1 | L128_B | 3 / 3 | 48/48 | 14062.355 | 13319.751 | 1.056× | 1.025–1.169× |
| huginn / serving_long128 / P512 C4 | L128_PD_A_control | L128_B | 3 / 3 | 48/48 | 13581.226 | 13319.751 | 1.020× | 0.983–1.119× |

## Limits

Official-reference sequence agreement is incomplete for both active models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.

## Final review: strongest controls and scope

| Model | Candidate | Input / concurrency / cap | Best measured E2E baseline | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup [95% CI] | Exact outputs / evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | B | 128 / 1 / 32 | DP | 2626.455 | 2889.741 | 0.909× [0.862–0.972×] | 72/72 |
| huginn | B | 2048 / 8 / 32 | DP | 8860.635 | 18358.209 | 0.483× [0.475–0.490×] | 72/72 |
| huginn | B | 512 / 4 / 32 | DP | 3295.142 | 4065.456 | 0.811× [0.761–0.896×] | 72/72 |
| huginn | L128_B | 512 / 4 / 128 | L128_DP_Mixed | 11883.670 | 13319.751 | 0.892× [0.881–0.987×] | 48/48 |
| ouro | B | 128 / 1 / 32 | DP | 635.783 | 711.718 | 0.893× [0.876–0.939×] | 72/72 |
| ouro | B | 2048 / 8 / 32 | DP | 2180.926 | 4678.616 | 0.466× [0.413–0.730×] | 72/72 |
| ouro | B | 512 / 4 / 32 | DP_Mixed | 1070.138 | 981.602 | 1.090× [1.012–1.153×] | 72/72 |
| ouro | L128_B | 512 / 4 / 128 | L128_DP_Mixed | 2842.743 | 2898.505 | 0.981× [0.970–1.002×] | 48/48 |

## Frozen-grid goodput trade-off

| Model | Candidate | Input / cap | ITL / TTFT limits ms | Best goodput baseline | Baseline / candidate qualified | Baseline / candidate requests/s | Ratio [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | B | 512 / 32 | 150.000 / 5000 | PD_G1 | 56 / 70 of 72 | 0.800 / 1.048 | 1.310× [1.125–1.580×] |
| huginn | L128_B | 512 / 128 | 150.000 / 5000 | L128_PD_A_control | 36 / 47 of 48 | 0.237 / 0.315 | 1.325× [1.070–1.786×] |
| ouro | B | 512 / 32 | 150.000 / 5000 | DP_Mixed | 71 / 72 of 72 | 4.044 / 4.181 | 1.034× [0.975–1.143×] |
| ouro | L128_B | 512 / 128 | 150.000 / 5000 | L128_DP_Mixed | 47 / 48 of 48 | 1.400 / 1.390 | 0.992× [0.969–1.057×] |

No original-model-equivalent serving GO is accepted. Read `09_final_summary.md` for the full limits.
One-start scale points do not supply an accepted confidence interval.
The full frozen SLO grid remains available. Favorable cells are not a new acceptance rule.
