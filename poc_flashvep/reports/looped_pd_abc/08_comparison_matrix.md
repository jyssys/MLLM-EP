# Independent A/B/C decision matrix

Reviewed decision file present.

Timing correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.

Active model scope: ouro, huginn. Dory is deferred by explicit user instruction. Existing Dory evidence remains historical, not a new verdict.

Campaign status: `FINALIZED_WITH_LIMITATIONS`. Evidence snapshot: 2026-10-29T11:01:16.078430+00:00.

| Track | Model | Status | Reason |
| --- | --- | --- | --- |
| A | ouro | NO_GO_PERFORMANCE | A loses to the strongest measured equal-GPU DP/Mixed controls in all four repeated two-GPU regimes. This finding does not exclude untested long-context or other PD implementations. |
| A | huginn | CONDITIONAL | At P512/C4, a frozen absolute-150ms ITL cell improves bounded goodput. The strongest E2E baseline is still faster. Multiple SLO cells, three starts, unresolved reference fidelity, and generic migration limit the claim. |
| B | ouro | CONDITIONAL | P512/C4 gives 1.090x p95 E2E versus DP Mixed, with a 1.012-1.153x conditional CI. Moderate-SLO goodput is only 1.034x. Other repeated regimes lose. No general loop-specific GO. |
| B | huginn | CONDITIONAL | At P512/C4 and absolute ITL 150ms, bounded goodput improves versus the best measured control. E2E remains worse than DP/Mixed. Official-reference mismatch and generic streaming controls remain unresolved. |
| C | ouro | NO_GO_CORRECTNESS | Loop versus service-time placement matches 125/144 outputs and gives 0.981x p95 E2E. A one-start synthetic four-GPU signal does not repair the natural-output gate or prove loop-specific novelty. |
| C | huginn | NO_GO_CORRECTNESS | Loop versus service-time placement matches 139/144 outputs and gives 1.016x p95 E2E with a 0.977-1.093x conditional CI. No lossless incremental gain is established. |

| Track | Model | GPU budget | Status | Scope |
| --- | --- | --- | --- | --- |
| A | ouro | 2 | NO_GO_PERFORMANCE | Three frozen starts per regime, exact bounded outputs. No accepted equal-budget benefit. |
| A | ouro | 4 | NO_GO_PERFORMANCE | One-start candidate discovery. No accepted gain. Natural pressure also changes outputs. |
| A | ouro | 8 | NO_GO_PERFORMANCE | Best synthetic candidate is 1.066x in one start, below the target. No gain CI. |
| A | huginn | 2 | CONDITIONAL | Strict-ITL goodput signal only. No deployment or general E2E GO. |
| A | huginn | 4 | NO_GO_PERFORMANCE | One-start candidate discovery does not beat the strongest baseline. No gain CI. |
| A | huginn | 8 | NO_GO_PERFORMANCE | One-start synthetic gains are below the target. Pressure comparisons change outputs. |
| B | ouro | 2 | CONDITIONAL | A narrow 8.3% latency reduction is below the 10% target and does not generalize. |
| B | ouro | 4 | NO_GO_PERFORMANCE | One-start synthetic and pressure discovery do not establish a benefit. |
| B | ouro | 8 | NO_GO_PERFORMANCE | The apparent pressure gain is one-start and output-confounded. Synthetic B loses. |
| B | huginn | 2 | CONDITIONAL | Frozen-grid goodput signal, not sustainable serving capacity or an original-model fidelity GO. |
| B | huginn | 4 | NO_GO_PERFORMANCE | One-start discovery lacks a gain against the strongest baseline. |
| B | huginn | 8 | NO_GO_PERFORMANCE | One-start discovery lacks a gain. Pressure output parity is incomplete. |
| C | ouro | 2 | NOT_MEASURED | A fixed 1P:1D pool has no initial placement choice. No independent C gain is measured. |
| C | ouro | 4 | NO_GO_CORRECTNESS | Repeated natural output parity fails. A single synthetic signal remains unconfirmed. |
| C | ouro | 8 | NO_GO_CORRECTNESS | Synthetic loop versus service is 1.001x in one start. Natural-pressure parity fails. |
| C | huginn | 2 | NOT_MEASURED | A fixed 1P:1D pool has no initial placement choice. No independent C gain is measured. |
| C | huginn | 4 | NO_GO_CORRECTNESS | Repeated natural output parity fails. Strong static controls remain faster. |
| C | huginn | 8 | NO_GO_CORRECTNESS | One-start synthetic loop placement loses to service-time placement. Natural-pressure parity fails. |

## Limits

Official-reference sequence agreement is incomplete for both active models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.

## Measured scale discovery, not accepted speedup

| Model | Regime | Candidate | Best E2E baseline | Baseline / candidate p95 E2E ms | Point ratio | Exact / N | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | pressure4 | A1 | PD1_G1 | 9239.341 / 12812.355 | 0.721× | 111/128 | NOT_MEASURED: one start |
| huginn | pressure4 | A2 | PD1_G1 | 9239.341 / 12802.027 | 0.722× | 112/128 | NOT_MEASURED: one start |
| huginn | pressure4 | B1 | PD1_G1 | 9239.341 / 9908.653 | 0.932× | 110/128 | NOT_MEASURED: one start |
| huginn | pressure4 | C_loop | PD1_G1 | 9239.341 / 12005.007 | 0.770× | 118/128 | NOT_MEASURED: one start |
| huginn | pressure8 | A1 | DP_Mixed | 7756.938 / 9355.433 | 0.829× | 120/128 | NOT_MEASURED: one start |
| huginn | pressure8 | A2 | DP_Mixed | 7756.938 / 10115.449 | 0.767× | 120/128 | NOT_MEASURED: one start |
| huginn | pressure8 | B1 | DP_Mixed | 7756.938 / 8353.819 | 0.929× | 110/128 | NOT_MEASURED: one start |
| huginn | pressure8 | C_loop | DP_Mixed | 7756.938 / 8509.666 | 0.912× | 116/128 | NOT_MEASURED: one start |
| huginn | scale4 | A1 | DP | 3318.935 / 3377.823 | 0.983× | 32/32 | NOT_MEASURED: one start |
| huginn | scale4 | A2 | DP | 3318.935 / 4135.311 | 0.803× | 32/32 | NOT_MEASURED: one start |
| huginn | scale4 | B1 | DP | 3318.935 / 4234.442 | 0.784× | 32/32 | NOT_MEASURED: one start |
| huginn | scale4 | C_loop | DP | 3318.935 / 4673.249 | 0.710× | 32/32 | NOT_MEASURED: one start |
| huginn | scale8 | A1 | DP_Mixed | 3399.207 / 3463.670 | 0.981× | 32/32 | NOT_MEASURED: one start |
| huginn | scale8 | A2 | DP_Mixed | 3399.207 / 3887.011 | 0.875× | 32/32 | NOT_MEASURED: one start |
| huginn | scale8 | A4 | DP_Mixed | 3399.207 / 3358.702 | 1.012× | 32/32 | NOT_MEASURED: one start |
| huginn | scale8 | B1 | DP_Mixed | 3399.207 / 4304.061 | 0.790× | 32/32 | NOT_MEASURED: one start |
| huginn | scale8 | C_loop | DP_Mixed | 3399.207 / 4067.756 | 0.836× | 32/32 | NOT_MEASURED: one start |
| ouro | pressure4 | A1 | PD1 | 2857.962 / 4842.029 | 0.590× | 122/128 | NOT_MEASURED: one start |
| ouro | pressure4 | A2 | PD1 | 2857.962 / 7157.911 | 0.399× | 110/128 | NOT_MEASURED: one start |
| ouro | pressure4 | B1 | PD1 | 2857.962 / 2954.185 | 0.967× | 121/128 | NOT_MEASURED: one start |
| ouro | pressure4 | C_loop | PD1 | 2857.962 / 9051.171 | 0.316× | 110/128 | NOT_MEASURED: one start |
| ouro | pressure8 | A1 | PD1 | 2572.737 / 6244.680 | 0.412× | 122/128 | NOT_MEASURED: one start |
| ouro | pressure8 | A2 | PD1 | 2572.737 / 4789.579 | 0.537× | 123/128 | NOT_MEASURED: one start |
| ouro | pressure8 | B1 | PD1 | 2572.737 / 2286.504 | 1.125× | 122/128 | NOT_MEASURED: one start |
| ouro | pressure8 | C_loop | PD1 | 2572.737 / 5113.081 | 0.503× | 117/128 | NOT_MEASURED: one start |
| ouro | scale4 | A1 | DP | 1090.877 / 1284.988 | 0.849× | 32/32 | NOT_MEASURED: one start |
| ouro | scale4 | A2 | DP | 1090.877 / 1249.086 | 0.873× | 32/32 | NOT_MEASURED: one start |
| ouro | scale4 | B1 | DP | 1090.877 / 1721.722 | 0.634× | 32/32 | NOT_MEASURED: one start |
| ouro | scale4 | C_loop | DP | 1090.877 / 882.644 | 1.236× | 32/32 | NOT_MEASURED: one start |
| ouro | scale8 | A1 | DP | 907.031 / 995.294 | 0.911× | 32/32 | NOT_MEASURED: one start |
| ouro | scale8 | A2 | DP | 907.031 / 850.551 | 1.066× | 32/32 | NOT_MEASURED: one start |
| ouro | scale8 | A4 | DP | 907.031 / 1069.564 | 0.848× | 32/32 | NOT_MEASURED: one start |
| ouro | scale8 | B1 | DP | 907.031 / 1251.026 | 0.725× | 32/32 | NOT_MEASURED: one start |
| ouro | scale8 | C_loop | DP | 907.031 / 1338.983 | 0.677× | 32/32 | NOT_MEASURED: one start |
