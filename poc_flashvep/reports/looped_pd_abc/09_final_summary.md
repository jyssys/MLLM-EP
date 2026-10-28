# Dense Looped PD: evidence and decisions

DRAFT: final scientific review is pending. No GO claim is accepted.

Campaign status: `RUNNING`. Evidence snapshot: 2026-10-28T21:47:48.668731+00:00.

These are actual GPU measurements from completed runs.
Unfinished and failed runs do not contribute latency values.
Natural generation permits EOS and preserves the declared output cap.
Cap termination is not proof of answer completeness.
A lower latency with different output IDs is not an accepted lossless speedup.

| Model | Exact output / N | Wilson 95% CI, % | GSM8K test | HumanEval |
| --- | --- | --- | --- | --- |
| ouro | 88/100 | [80.18790057328857, 93.00059356298051] | NOT_MEASURED | NOT_MEASURED |
| huginn | 89/100 | [81.3687034969197, 93.74580364293543] | NOT_MEASURED | NOT_MEASURED |
| dory | 80/100 | [71.11708344068411, 86.66330666689676] | NOT_MEASURED | NOT_MEASURED |

| Track | Model | Status |
| --- | --- | --- |
| A | ouro | NOT_MEASURED |
| A | huginn | NOT_MEASURED |
| A | dory | NOT_MEASURED |
| B | ouro | NOT_MEASURED |
| B | huginn | NOT_MEASURED |
| B | dory | NOT_MEASURED |
| C | ouro | NOT_MEASURED |
| C | huginn | NOT_MEASURED |
| C | dory | NOT_MEASURED |

## Evidence locations

The master measurement table is `results_raw.csv`.
The repeated comparisons are `paired_comparisons.csv`.
The frozen-grid request SLO results are `slo_results.csv`.
The exact-transfer results are `transfer_identity.csv`.
The host phase partition is `e2e_phase_breakdown.csv`.
The E2E component percentages are `e2e_phase_percent_summary.csv`.
The frozen-timeline handoff oracle is `handoff_frozen_timeline_oracle.csv`. It is not measured speedup.
The request outputs and source snapshots remain under `raw/`.

## Limits

Official-reference sequence agreement is incomplete for all three models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.
