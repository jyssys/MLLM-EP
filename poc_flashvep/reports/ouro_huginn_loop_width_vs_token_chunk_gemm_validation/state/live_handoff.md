# Live research checkpoint

Recorded during the current campaign on 2026-11-11 UTC.
This file is not the final report.

## Current work

Package: `poc_flashvep/loop_width_gemm_validation`.
Evidence: this file's grandparent directory.
Spec: `poc_flashvep/reports/ouro_huginn_loop_width_vs_token_chunk_gemm_validation_spec.md`.
Final report remains unwritten.
The current work must finish before the queued follow-up starts.

Start: 07:38:55 UTC.
Review: 11:38:55 UTC.
GPU stop: 12:38:55 UTC.
Operational closure bound: 13:38:55 UTC.
These are current-campaign bounds, not the follow-up's ten-hour budget.

Active controller: PID 2588107.
Command: `python -m poc_flashvep.loop_width_gemm_validation.final_controls`.
Log: `logs/final_controls_controller.log`.
It ran five v2 serial controls. It then started 27 N128 confirmation jobs.
Read `state/confirmation_progress.json` for completion and unsafe-arm stops.
The controller generates final serving analysis and paired statistics after those jobs.

## Completed evidence

The initial projection search used three independent starts per model.
The additional batch-shape search used three more starts per model.
There are 441 projection cells across 12 processes.
Workspace cap: 64 MiB. Candidate cap: 32 configurations per geometry.
The policy uses start 1 only. Later starts do not reselect kernels.

The fixed-policy Huginn dense-time ratio is approximately 4.08–4.24 across starts.
This is Token64 × R32 divided by full-prompt Loop1.
It sums isolated projection timings with structural multiplicities.
It is not serving speedup.
Projection-only compatible Token batch 8 nearly removes that dense gap.
Ready work, attention, and KV admission are absent from that batch model.

Actual legal Huginn core graph enclosures also retain a gap.
Tuned Loop1: 15.996 ms p50.
Tuned Token64 × R32: 69.440 ms p50.
Prelude and Coda are outside those core enclosures.
Output clones and host launch gaps can remain inside them.
Every recorded Token core covers depths 0–31 exactly once.

Both policies passed N8 and N32 state qualification for every planned arm.
There are 720 qualification test requests across repeated prompts and controls.
The v2 map includes actual batched P geometries.
V2 policy SHA256: `cbfed632621f5631056d6aff875eecec1500efc74d85d8182e945cdaee3ce1d6`.
All 15 N64 marker-free serving arms passed output, work, and lifetime checks.
All measured packed-prefill graph selections hit existing entries.
There were no measured graph captures or eager fallbacks in those N64 arms.

## N64 findings

Huginn Output4 Loop28 versus tuned Zero:
TTFT p95: 6425.7 versus 10615.2 ms.
ITL p99: 77.39 versus 76.46 ms.
These are preliminary, one-start results.

Huginn Output16 Loop28 improves TTFT but worsens ITL substantially.
Ouro Loop3 and Token3072 fail the frozen Decode screen.
Those unsafe Ouro split arms do not receive N128 confirmation.
Three Ouro Native controls remain in the confirmation plan.

Huginn Output4 confirmation includes Zero, Loop31, Loop28, Token64, and Token256.
Loop31 versus Token64 is the requested equal-recurrent-row comparison.
Loop28 versus Token256 is a larger equal-recurrent-row comparison.
The latter Token arm remains a stronger throughput challenger despite its N64 relative-tail failure.
Huginn Output16 confirmation includes Zero, Loop28, and Token256.
Arrival seeds: 2026111111, 2026111112, 2026111113.
The plan uses balanced run order.
An absolute Decode failure stops later repeats of that arm.
Never omit that failed result.

## Retained diagnostic problems

The first core observer missed BulkWrittenGraphs.execute.
Its absent recurrent samples are NOT_MEASURED, not zero time.
Fresh `core_cost_*` controls observe the actual graph instance.
The original whole-submission enclosures remain valid.

One Ouro serial startup failed because Ouro lacks prepared Prelude/Coda methods.
The observer now checks model-specific methods.
The failed raw namespace remains intact.
Fresh Ouro `*_serial8_s2` controls passed.

The initial Token cap-1 configuration excluded Prelude.
It actually executed all 32 Loops in one submission.
Its total throughput cost is valid, but it is not a small-quantum result.
V2 includes Prelude in the cap and requires an actual quantum audit.
V2 Token64 N32 recorded 32 Prelude spans and 992 recurrent spans, each with one Loop.

## Required closure

Inspect final controls and every correctness result.
Run `cost_analysis`, `serving_analysis`, `confirmation_stats`, and `inventory` as needed.
The direct Loop/Token ratios are in `analysis/paired_method_comparisons.json`.
Generate figures only after clean GPU timing ends.
Write the final English report with measured/modelled/untested distinctions.
The current campaign does not implement a new adaptive scheduler.
It does not prove exclusive Loop novelty. Equivalent logical-layer cuts remain equivalent.
Physical kernel idle, HBM counters, and clean online Service TTFT are NOT_MEASURED.

Run `closure` to check source and preserve user state.
Create `state/CAMPAIGN_CLOSED.json` only after scientific GPU work ends.
Reuse the approved restoration helper with this namespace.
Verify all eight burners, resident GPU processes, and completed generation rounds.
Create and verify the lossless evidence bundle.
Run this package's `publish` module.
It uses a private Git index and preserves the active checkout.
Verify the final remote SHA.

Publication branch: `flashvep/loop-width-gemm-validation-20261111`.
Publication parent: `4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd`.
Active checkout remains the unrelated recovery branch at `f4ecea59f36f620c69c4fef09c590bf790802cd9`.
Index-entry SHA256 must remain `d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3`.
External runtime diff SHA256 must remain `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

## Queued follow-up

The user supplied a separate Phase 1/2 workload-discovery task.
See `state/queued_followup.json` and the full user prompt.
Start its ten-hour budget only after current review, reporting, push, and restoration.
It requires both models, minimum Output16, and physical kernel-union evidence.
P busy must reach 80% and service proximity must reach 85%.
P compute-ready pressure and usable D slack must occur in the same window.
Profiler interference must remain within 5%.
Current Output4 results do not satisfy that follow-up gate.
Do not start Phase 2 before qualified Phase 1.
Do not build its excluded Phase 3/4 systems.
