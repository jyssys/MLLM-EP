# Original-gate adaptive profile

## Scope and validation

These traces describe the existing adaptive baseline.
They do not describe a new scheduler or architecture.
Both Base checkpoints use their unchanged original gate, q=0.5, minimum R2, maximum R4, and `last_exited`.
Prefill still executes full R4.
Generation is greedy and free-running with the frozen calibration prompts and common cap.

Every recorded decode event has an executed depth of 2, 3, or 4.
The analyzer checks that event count equals generated token count minus one.
The first output token comes from full-depth prefill.
The sum of event depths equals the runtime's recorded recurrent-step count.
The runtime histogram uses zero-based depth labels, so the analyzer does not label its key `1` as executed R1.

## Observed depth mix

Each row contains eight requests.
The pooled mean weights each observed decode event equally.
The work comparison uses four steps for that same observed adaptive output length.
It is not a quality-equivalent fixed-R4 generation comparison.

| Model | Domain | Decode events | Pooled mean depth | R2 / R3 / R4 event fraction | Same-length loop-work reduction | Request-bootstrap 95% CI |
|---|---|---:|---:|---|---:|---|
| 1.4B | GSM8K | 867 | 3.0923 | 3.345% / 84.083% / 12.572% | 22.693% | [21.137%, 24.419%] |
| 1.4B | MATH-500 | 8744 | 2.9300 | 14.844% / 77.310% / 7.845% | 26.750% | [23.928%, 28.706%] |
| 2.6B | GSM8K | 852 | 3.0211 | 1.056% / 95.775% / 3.169% | 24.472% | [23.762%, 25.000%] |
| 2.6B | MATH-500 | 34971 | 2.8242 | 21.403% / 74.771% / 3.826% | 29.394% | [27.367%, 31.059%] |

The bootstrap resamples whole requests 4000 times within each model/domain row.
It does not treat individual tokens as independent requests.
Long responses affect the pooled metric more than short responses.
The JSON also reports the unweighted mean across request-level means.

## Interpretation

The actual original-gate path skips recurrent work for both Base sizes in this cohort.
That fact does not establish a new distributed scheduling opportunity.
It also does not establish lower allocated KV memory.
The current runtime retains the full depth-slot allocation.
Its terminal-copy policy and an alias-aware reference remain distinct implementation choices.

The measured quality context remains in [quality_and_exactness.md](quality_and_exactness.md).
Adaptive and fixed natural responses can differ in length, content, and accuracy.
The work percentages above are not E2E speedups.
They cannot become candidate headroom or a predicted DP scheduling gain.

## Evidence

- `analysis/adaptive_profile_01/request_profiles.csv`.
- `analysis/adaptive_profile_01/summary.json`.
- Editable request-level depth figures in that directory.
- `scripts/summarize_adaptive_profile.py`.
- Original acceptance/depth events and runtime statistics in each completed adaptive quality output.

No new model run or learned component supplies this analysis.
The held-out cohort remains untouched.
