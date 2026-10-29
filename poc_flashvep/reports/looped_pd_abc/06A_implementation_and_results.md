# Track A: measured implementation results

Reviewed decision file present.

Timing correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.

Active model scope: ouro, huginn. Dory is deferred by explicit user instruction. Existing Dory evidence remains historical, not a new verdict.

Campaign status: `FINALIZED_WITH_LIMITATIONS`. Evidence snapshot: 2026-10-29T11:01:16.075472+00:00.

These are actual GPU measurements from completed runs.
Unfinished and failed runs do not contribute latency values.
Natural generation permits EOS and preserves the declared output cap.
Cap termination is not proof of answer completeness.
A lower latency with different output IDs is not an accepted lossless speedup.

| Model / regime | Baseline | Candidate | Starts / frozen pairs | Exact outputs / N | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup | Paired 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro / serving / P128 C1 | DP | A | 3 / 3 | 72/72 | 635.783 | 699.128 | 0.909× | 0.899–0.957× |
| ouro / serving / P128 C1 | DP_Mixed | A | 3 / 3 | 72/72 | 644.216 | 699.128 | 0.921× | 0.909–0.964× |
| ouro / serving / P128 C1 | PD | A | 3 / 3 | 72/72 | 703.471 | 699.128 | 1.006× | 1.002–1.020× |
| ouro / serving / P128 C1 | PD_G1 | A | 3 / 3 | 72/72 | 703.790 | 699.128 | 1.007× | 1.002–1.030× |
| ouro / serving / P128 C1 | PD_A_control | A | 3 / 3 | 72/72 | 700.595 | 699.128 | 1.002× | 0.989–1.027× |
| ouro / serving / P128 C1 | PD_A_coalesced | A | 3 / 3 | 72/72 | 697.890 | 699.128 | 0.998× | 0.995–1.055× |
| ouro / serving / P512 C4 | DP | A | 3 / 3 | 72/72 | 1102.345 | 1215.311 | 0.907× | 0.889–1.320× |
| ouro / serving / P512 C4 | DP_Mixed | A | 3 / 3 | 72/72 | 1070.138 | 1215.311 | 0.881× | 0.812–1.277× |
| ouro / serving / P512 C4 | PD | A | 3 / 3 | 72/72 | 1508.632 | 1215.311 | 1.241× | 0.982–1.688× |
| ouro / serving / P512 C4 | PD_G1 | A | 3 / 3 | 72/72 | 1172.164 | 1215.311 | 0.964× | 0.718–1.720× |
| ouro / serving / P512 C4 | PD_A_control | A | 3 / 3 | 72/72 | 1444.279 | 1215.311 | 1.188× | 0.972–1.755× |
| ouro / serving / P512 C4 | PD_A_coalesced | A | 3 / 3 | 72/72 | 1359.870 | 1215.311 | 1.119× | 1.006–1.616× |
| ouro / serving / P2048 C8 | DP | A | 3 / 3 | 72/72 | 2180.926 | 3687.181 | 0.591× | 0.488–0.891× |
| ouro / serving / P2048 C8 | DP_Mixed | A | 3 / 3 | 72/72 | 2810.066 | 3687.181 | 0.762× | 0.619–1.133× |
| ouro / serving / P2048 C8 | PD | A | 3 / 3 | 72/72 | 4777.884 | 3687.181 | 1.296× | 1.220–1.587× |
| ouro / serving / P2048 C8 | PD_G1 | A | 3 / 3 | 72/72 | 4305.518 | 3687.181 | 1.168× | 0.928–1.432× |
| ouro / serving / P2048 C8 | PD_A_control | A | 3 / 3 | 72/72 | 4769.008 | 3687.181 | 1.293× | 0.965–1.466× |
| ouro / serving / P2048 C8 | PD_A_coalesced | A | 3 / 3 | 72/72 | 4097.656 | 3687.181 | 1.111× | 0.705–1.294× |
| ouro / serving_long128 / P512 C4 | L128_DP | L128_A | 3 / 3 | 48/48 | 3449.890 | 3092.371 | 1.116× | 1.107–1.261× |
| ouro / serving_long128 / P512 C4 | L128_DP_Mixed | L128_A | 3 / 3 | 48/48 | 2842.743 | 3092.371 | 0.919× | 0.912–1.052× |
| ouro / serving_long128 / P512 C4 | L128_PD | L128_A | 3 / 3 | 48/48 | 3712.556 | 3092.371 | 1.201× | 0.917–1.356× |
| ouro / serving_long128 / P512 C4 | L128_PD_G1 | L128_A | 3 / 3 | 48/48 | 3815.594 | 3092.371 | 1.234× | 0.917–1.391× |
| ouro / serving_long128 / P512 C4 | L128_PD_A_control | L128_A | 3 / 3 | 48/48 | 3784.293 | 3092.371 | 1.224× | 1.020–1.360× |
| ouro / serving_long128 / P512 C4 | L128_PD_A_coalesced | L128_A | 3 / 3 | 48/48 | 3944.288 | 3092.371 | 1.275× | 1.212–1.439× |
| huginn / serving / P128 C1 | DP | A | 3 / 3 | 72/72 | 2626.455 | 2742.788 | 0.958× | 0.953–0.988× |
| huginn / serving / P128 C1 | DP_Mixed | A | 3 / 3 | 72/72 | 2641.859 | 2742.788 | 0.963× | 0.939–1.017× |
| huginn / serving / P128 C1 | PD | A | 3 / 3 | 72/72 | 2736.798 | 2742.788 | 0.998× | 0.948–1.065× |
| huginn / serving / P128 C1 | PD_G1 | A | 3 / 3 | 72/72 | 2876.210 | 2742.788 | 1.049× | 1.013–1.124× |
| huginn / serving / P128 C1 | PD_A_control | A | 3 / 3 | 72/72 | 2879.762 | 2742.788 | 1.050× | 0.996–1.148× |
| huginn / serving / P512 C4 | DP | A | 3 / 3 | 72/72 | 3295.142 | 3849.766 | 0.856× | 0.817–0.886× |
| huginn / serving / P512 C4 | DP_Mixed | A | 3 / 3 | 72/72 | 3537.789 | 3849.766 | 0.919× | 0.869–0.958× |
| huginn / serving / P512 C4 | PD | A | 3 / 3 | 72/72 | 4784.864 | 3849.766 | 1.243× | 1.116–1.348× |
| huginn / serving / P512 C4 | PD_G1 | A | 3 / 3 | 72/72 | 4309.330 | 3849.766 | 1.119× | 1.040–1.127× |
| huginn / serving / P512 C4 | PD_A_control | A | 3 / 3 | 72/72 | 4424.257 | 3849.766 | 1.149× | 1.024–1.257× |
| huginn / serving / P2048 C8 | DP | A | 3 / 3 | 72/72 | 8860.635 | 16836.025 | 0.526× | 0.524–0.538× |
| huginn / serving / P2048 C8 | DP_Mixed | A | 3 / 3 | 72/72 | 11429.320 | 16836.025 | 0.679× | 0.643–0.692× |
| huginn / serving / P2048 C8 | PD | A | 3 / 3 | 72/72 | 27593.847 | 16836.025 | 1.639× | 1.063–1.670× |
| huginn / serving / P2048 C8 | PD_G1 | A | 3 / 3 | 72/72 | 18921.083 | 16836.025 | 1.124× | 1.081–1.137× |
| huginn / serving / P2048 C8 | PD_A_control | A | 3 / 3 | 72/72 | 18408.542 | 16836.025 | 1.093× | 1.045–1.111× |
| huginn / serving_long128 / P512 C4 | L128_DP | L128_A | 3 / 3 | 48/48 | 11907.071 | 13703.750 | 0.869× | 0.856–0.957× |
| huginn / serving_long128 / P512 C4 | L128_DP_Mixed | L128_A | 3 / 3 | 48/48 | 11883.670 | 13703.750 | 0.867× | 0.857–0.955× |
| huginn / serving_long128 / P512 C4 | L128_PD | L128_A | 3 / 3 | 48/48 | 13660.943 | 13703.750 | 0.997× | 0.985–1.103× |
| huginn / serving_long128 / P512 C4 | L128_PD_G1 | L128_A | 3 / 3 | 48/48 | 14062.355 | 13703.750 | 1.026× | 0.942–1.109× |
| huginn / serving_long128 / P512 C4 | L128_PD_A_control | L128_A | 3 / 3 | 48/48 | 13581.226 | 13703.750 | 0.991× | 0.941–1.064× |

## Mechanism and controls

A transfers full-prompt recurrent state at an intermediate cut.
The destination completes prefill and then executes full-depth decode.
The matched A control uses the same whole-prompt chunk and staged execution.
Generic logical-depth migration is a novelty counterexample.
The cut sweep is separate tuning evidence, not an accepted single-start gain.

## Limits

Official-reference sequence agreement is incomplete for both active models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.
