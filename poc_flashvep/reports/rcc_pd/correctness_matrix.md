# RCC-PD correctness evidence

## Reused prerequisite

Both models entered this study as `CORRECTNESS_READY` for the matched corrected runtime.
The release commit is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The predecessor investigation completed before RCC GPU execution started.
Its experiment processes no longer owned the selected devices.

| Model | Qualified correction | Prior independent evidence | Scope |
|---|---|---|---|
| Ouro R4 | Down fixed 256-row tiles and small-row RMSNorm | PD_G1 repeat and B parity at N200 and N400, including old failure seeds | Matched-runtime internal qualification |
| Huginn R32 | Same correction plus adapter fixed 256-row tiles | Two N200 seeds and two closed-loop N400 seeds, plus state/order closure | Matched-runtime internal qualification |

The predecessor's original-HF triage matched 8/8 Ouro outputs and 5/8 Huginn outputs.
Those eight-prompt probes are not benchmark accuracy evaluations.
Huginn original-model equivalence remains unresolved.
No new HF or benchmark accuracy result is claimed here.

## New fixed-cut output tests

The three arms are PD_G1, A_sep, and A_native_mixed.
Every comparison uses the same input IDs, seed, checkpoint, precision, depth, and output policy.
The candidate arm uses the existing Native Mixed path.

| Model | Diagnostic 128-token prefixes, cap 32 | Diagnostic 256-token prefixes, cap 16 | Quiet 128-token prefixes, cap 32 | State-probe 128-token prefixes, cap 16 |
|---|---:|---:|---:|---:|
| Ouro | 8/8 token-exact per arm | 8/8 token-exact per arm | 24/24 per arm across three starts | 8/8 token-exact per arm |
| Huginn | 8/8 token-exact per arm | 8/8 token-exact per arm | 24/24 per arm across three starts | 8/8 token-exact per arm |

Each non-reference arm matches PD_G1.
Mixed also matches A_sep directly.
Exit-depth sequences and stop reasons match.
The quiet PD_G1 control repeats exactly across its three starts.
These are repeated instances of the same eight prompts, not 48 independent prompts per arm.
The 128-token and 256-token prefixes share the same source-prompt identities.

All 36 measured cohorts pass exact recurrent row-work accounting.
The identity is `R × Σ(prompt_tokens + output_tokens − 1)`.
The subtraction accounts for the first output token, which uses completed prompt state.
No prescribed recurrent work is skipped.

## Handoff byte identity

Every heavy diagnostic arm verifies source and destination KV plus hidden fingerprints.
Each such run records 16/16 passing checks, including eight warmup and eight measured requests.
The check covers only semantically valid KV and the registered hidden carrier.
The split arms compare the valid prefix below the cut.
The PD_G1 arm compares its full valid prompt transfer.

This verifies each transfer against its own producer.
It does not compare different arms' intermediate states by itself.
The additional prompt-completion test provides that comparison.

## Full prompt-completion state comparison

Evidence class: `MEASURED_HEAVY_CORRECTNESS`.
The probe reads state before the first CODA output is sampled.
The hidden comparison covers the last prompt row, not every final prompt hidden row.
It tests seeds 2026103017 and 2026103018.
Each seed uses a 128-token prompt.
The probe excludes invalid depth planes and unwritten coda positions.

| Model | Pair | Last prompt hidden row | Expected-valid prompt K/V planes per request | K/V identity |
|---|---|---:|---:|---:|
| Ouro | PD_G1 vs A_sep | 2/2 exact | 192 | 2/2 exact |
| Ouro | PD_G1 vs A_native_mixed | 2/2 exact | 192 | 2/2 exact |
| Ouro | A_sep vs A_native_mixed | 2/2 exact | 192 | 2/2 exact |
| Huginn | PD_G1 vs A_sep | 2/2 exact | 264 | 2/2 exact |
| Huginn | PD_G1 vs A_native_mixed | 2/2 exact | 264 | 2/2 exact |
| Huginn | A_sep vs A_native_mixed | 2/2 exact | 264 | 2/2 exact |

For Huginn, prelude and coda KV use depth zero.
Recurrent KV uses all 32 depth planes.
Coda KV contains 127 valid prompt positions before its final prompt-row call.
The probe excludes the not-yet-written final coda position.
This guards the boundary-state comparison against invalid-byte mismatches.

`prompt_state_comparisons.json` contains the per-seed comparisons.
Each raw state probe records a SHA-256 for every valid plane and the final hidden row.
No divergent valid plane or hidden row was observed in these selected cases.

## Limits

These finite probes do not establish correctness for every batch order or arrival seed.
The RCC study does not repeat the predecessor's full 200/400-request qualification on a new mixed path.
It does not test long multi-chunk migration.
It does not establish GSM8K or MATH accuracy.
It does not establish original-HF lossless equivalence.

The new mechanism status is `FINITE_PILOT_PASS`, not universal correctness certification.
Any future admission or long-context implementation needs its own expanded gate.
The present novelty stop does not require a larger serving campaign.
