# Independent A/B/C decision matrix

DRAFT: final scientific review is pending. No GO claim is accepted.

User scope update: only Ouro and Huginn remain active. Dory entries below are historical. Read `user_scope_override.md`.

Timing correction: legacy foreground-I/O timings are invalid for clean method comparisons. Read `timing_correction.md`. Canonical nonblocking retiming is pending.

Campaign status: `RUNNING`. Evidence snapshot: 2026-10-28T22:34:18.766027+00:00.

| Track | Model | Status | Reason |
| --- | --- | --- | --- |
| A | ouro | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| A | huginn | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| A | dory | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| B | ouro | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| B | huginn | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| B | dory | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| C | ouro | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| C | huginn | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |
| C | dory | NOT_MEASURED | Final review pending. Do not interpret partial comparisons as a final verdict. |

| Track | Model | GPU budget | Status | Scope |
| --- | --- | --- | --- | --- |
| A | ouro | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | ouro | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | ouro | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | huginn | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | huginn | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | huginn | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | dory | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | dory | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| A | dory | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | ouro | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | ouro | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | ouro | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | huginn | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | huginn | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | huginn | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | dory | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | dory | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| B | dory | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | ouro | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | ouro | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | ouro | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | huginn | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | huginn | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | huginn | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | dory | 2 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | dory | 4 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |
| C | dory | 8 | NOT_MEASURED | Review pending. Consult the measurement tables for actual completed runs. |

## Limits

Official-reference sequence agreement is incomplete for all three models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.
