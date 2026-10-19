# Post-launch baseline analyzer review

## Findings

The original analyzer needs additional integrity checks before a complete arm can receive strong admission claims.
Four synthetic cases expose missing bindings.
They do not establish a discrepancy in any actual record.
I opened no partial accuracy output or protected result.
I changed no scorer, sample, threshold, cap, gate, runner, or analyzer.

| Synthetic case | Original analyzer result | Missing binding |
|---|---|---|
| Replace prompt IDs with other IDs of the same length | Accepts | Frozen prompt to encoded input |
| Change text while keeping output IDs unchanged | Accepts, and the synthetic grade changes | Output IDs to decoded text |
| Declare `stop` with no configured frozen stop | Accepts as completed | Stop label to a frozen token suffix |
| Give the synthetic launch contract a mismatched protocol hash | Admits the otherwise complete synthetic arm | Protocol to its prelaunch anchor |

[The retained characterization tests](../tests/test_baseline_extension_admission_review_cpu_v1.py) reproduce these acceptances.
They do not test a later integrity supplement.
The combined suite passed 12 CPU tests before the analyzer interface changed.
Passing characterization tests preserve the gaps, not a fix.
The coordinator then added a separate integrity supplement and a required analyzer argument.
This bounded review does not inspect or approve that supplement.
Its new required argument is not a raw-record discrepancy.

The inspected pre-supplement analyzer hash is `f3c5d5bb0ddd10dd16a09cca0f4a43b132ab4d361f65a9633e963f6bb98cc731`.
[The archival source](../artifacts/analyze_baseline_extension_pre_integrity_review_v1.py) reconstructs the complete captured source text.
Its SHA256 matches that previously sampled hash exactly.
The fourth characterization test now loads that verified archive.
It does not call the changed analyzer without its required integrity argument.
The archive-based recheck passes 12 tests again from the project root.
A wrong-directory invocation first failed module discovery, before it could execute the tests.
The structured record separates that invocation from the successful recheck.

## Checks that work in the inspected scope

The analyzer requires runner and guard success, zero guard exit, and 64 completed requests.
It requires 64 gzip files, exact unique IDs, a matching saved protocol, and matching per-record protocol hashes.
It requires 32 requests per domain.
The synthetic full-arm test admits one complete arm and marks the other three unavailable.
A failed guard excludes that arm.
A success flag with a completion count of 63 causes rejection.
Malformed complete raw data causes an exception, not an accepted partial summary.
The analyzer opens raw gzip files read-only and writes a new analysis directory.

The launch contract is not read by the original analyzer.
The actual current protocol hash still matches the frozen launch hash.
The synthetic anchor failure must not be described as an actual protocol change.
Current scorer-file hashes also do not independently attest every executed source or dependency.

## Frozen completion, uncertainty, and work rules

EOS takes precedence over length at token 8192.
Only the declared length cap is answer truncation.
EOS and a valid frozen stop count as completed under the frozen override.
The original analyzer checks EOS and length metadata, but trusts the stop label.
The base scorer's non-EOS truncation flag is overwritten consistently with the analysis contract.

The bootstrap uses 4000 request resamples with seed 2026101911.
Its bounds use sorted indices 100 and 3899.
Paired accuracy uses one adaptive-minus-fixed value per shared request ID.
Both paired domains must contain the same 32 IDs.
Tokens and exit events do not become independent quality samples.
Degenerate intervals remain possible for unanimous small samples.
They do not establish noninferiority.

Decode work is the sum of recorded exit depths.
The analyzer checks it against recurrent, coda, histogram, graph, and full-R4 prefill counters.
Normalized decode work divides by four times the recorded decode positions.
The layer proxy multiplies that work by 24 or 48 physical layers.
It is not FLOPs, HBM traffic, elapsed time, or a ready frontier.
One-output requests have no decode events and remain in quality counts.
The depth summaries exclude them and report the remaining request count explicitly.
The emitted scope excludes prefill, first-output production, and other costs.

## Exact tokenizer and stop path

Use the existing `load_ouro_tokenizer` at the pinned local Base snapshot.
Its source is `looped_cdb/eval/model_loading.py:104–107`.
The Base1.4 revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The Base2.6 revision is `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
Both snapshots have identical tokenizer, configuration, and special-token file hashes in the structured record.

The runner encodes the frozen prompt and each frozen stop with `add_special_tokens=False`.
See `scripts/baseline_extension.py:58–60`.
It decodes output IDs with `skip_special_tokens=False` and no cleanup override.
See line 79.
The observed tokenizer is `GPT2Tokenizer`, with cleanup false and EOS 0.
The observed libraries are Transformers 5.9.0 and Tokenizers 0.22.2.
These observations occurred after launch.
Do not replace token-to-text equality with re-encoding of text, which need not reproduce a unique token sequence.

The CDB request class appends the sampled token before its completion check.
It keeps the matching stop suffix in generated tokens.
See `continuous_batching/requests.py:206–243`.
Thus, a declared stop requires a nonempty encoded frozen suffix in the output IDs.
Keep the existing EOS-first and cap-second classification.
Do not introduce a different string-based stopping rule after outcomes.

## Post-launch parser dependency record

The pre-outcome contract pins the scorer, interval helper, and runner.
It does not pin the imported `final_answer` extractor or mathematical parser closure.
The extractor resides in `poc_looped_exact_depth_chunk_pipeline/scoring.py`.
The current mathematical packages resolve under the existing `cpu_deps` root.
Their versions are `math-verify` 0.8.0, `latex2sympy2_extended` 1.10.2, SymPy 1.14.0, mpmath 1.3.0, and ANTLR runtime 4.13.2.
Three synthetic mathematical parser cases pass their expected match and mismatch checks.
The scorer catches parser exceptions as unsuccessful parsing, so this smoke test is not comprehensive parser validation.

[The structured review](../artifacts/baseline_extension_analyzer_review_v1.json) records metadata hashes and Python-source aggregates.
These are explicitly post-launch observations beginning at 2026-10-19 17:41:47 UTC.
They do not prove the dependencies were identical before launch.
They are not a complete binary or environment closure.
The review provides no new accuracy, quality, architecture-speedup, or candidate result.
No GPU ran during this audit.
The audit ends after this handoff.
