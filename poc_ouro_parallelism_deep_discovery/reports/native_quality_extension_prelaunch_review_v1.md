# Native TP4 quality extension: bounded prelaunch review

## Result

No source, ownership, finish-rule, or receipt blocker remains in current v2.
This is a CPU/source review.
It does not establish actual model outputs or native quality.

The [v2 manifest](../artifacts/native_quality_extension_contract_v2.json) freezes 14 named inputs.
All 14 hashes match.
The [structured review](../artifacts/native_quality_extension_prelaunch_review_v1.json) preserves earlier findings, tested versions, and limits.

## Earlier findings and remedies

| Finding | Remedy in current v2 |
|---|---|
| An inherited FourModelProbe flag could replace OuroModel.forward at import. | Reject inherited POC4_* and storage-control variables. Use read-only KVWorkerProbe directly. |
| Unsupported finish reasons with an unset stop reason became EOS. | Admit native stop/length reasons only. |
| A synthetic length-capped result ending in EOS passed the first validator. | Require the final length-capped token to differ from EOS. |
| A generate exception could leave the request alarm armed. | Cancel the alarm in finally. |
| The first validator expected external source keys in a campaign-only guard receipt. | Check the complete runner receipt separately from the guard's in-root receipt. |
| Initial v1 lacked established job-local compile-cache provenance. | Current v2 requires fresh VLLM, Inductor, and Triton roots before native imports. |

Main reports that native v1 stopped during compilation at 18:43:54 UTC.
It remains NOT_ADMITTED.
Its frozen manifest, runner, and validator archives remain preserved.
This review does not infer an incorrect compiled function or numerical discrepancy from that precautionary stop.

The source-receipt finding concerns the earlier inspected validator revision.
Current v2 contains the in-root exclusion and the complete runner receipt.
The review tests that current implementation, not the earlier condition.

## Existing native path and ownership

The 19 engine-config keywords match native_quality_pilot.py with TP4 and async scheduling enabled.
Only the worker-extension class changes.
The new class exposes read-only ownership inspection without the earlier import-time controls.

The runner retains BF16, fixed R4, native attention, native collectives, and greedy sampling.
It retains full prefill, disabled prefix/chunked prefill, and the 8192-token cap.
It submits one request at a time.

Runtime ownership gates require four physical GPUs and four depth-indexed banks per physical layer.
They require 24 or 48 physical layers, TP world size four, and four local Q/KV heads.
The physical UUID set must match ordered GPUs 4/5/6/7.
These are admission assertions, not observations from this review.

The frozen extension contains exactly 32 requests per domain.
No new outcome selects its IDs.
This is not a fresh held-out cohort or candidate confirmation.

## Completion and comparison

Installed native scheduling checks EOS before the length cap.
The revised validator rejects EOS-at-cap metadata labeled as length.
It also binds prompt IDs, canonical text, raw IDs, model pins, and finish metadata.

The canonical evaluation text is the pinned tokenizer's complete decode of raw IDs.
The native API text remains separate.
This avoids silently equating API text trimming with token parity.

Native stopping matches decoded-text substrings.
Staged stopping matches exact encoded-token suffixes.
The native API can trim text inside the final decoded token.
Thus, equal stop strings do not establish identical terminal token sequences.

Native-versus-staged differences describe whole operating points.
They do not isolate a numerical-core cause, candidate quality loss, or architecture gain.
The unchanged parser and completed-answer override remain frozen.
Transitive scorer package review belongs to the separate literature audit.

## Receipts, clocks, and admission

The runner checks all named source hashes before native imports.
It records their full manifest receipt.
The legacy guard separately records in-root Python files.

The validator now respects those distinct namespaces.
It checks the complete runner receipt and exact job-local cache paths.
These receipts are not signed attestations or a complete binary build receipt.

The per-request watchdog is 1200 seconds.
The frozen outer timeout is 3600 seconds per model.
Alarm cancellation occurs on both normal and exception paths.

Only complete 64-request arms with native status and successful outer guards enter analysis.
An operational timeout does not justify a smaller quality denominator.
The recorded wall is a quality-run diagnostic, not architecture timing.

## CPU checks and limits

Nine unit tests pass.
Five synthetic analysis cases check complete admission, unavailable-arm exclusion, partial-arm rejection, and changed-raw rejection.
Two fake request cases check alarm cancellation.

Seven tests execute the current validator's receipt AST.
The valid campaign-only guard passes with the full runner receipt.
Missing entries, wrong hashes, late setup, and wrong cache ownership reject.

The review hashes the staged reference summary without parsing its outcomes.
It reads only source and selected guard metadata.
No actual quality result, model execution, GPU call, or protected result enters this review.

No candidate E2E, free-running candidate quality, noninferiority, or held-out PASS follows.

## Actual work

The actual interval is 2026-10-19 18:36:18 UTC through 2026-10-19 18:48:33 UTC.
The active interval through this checkpoint is 735 seconds.
No prior pause interval enters this work.
Concurrent worker time does not add campaign elapsed time.

