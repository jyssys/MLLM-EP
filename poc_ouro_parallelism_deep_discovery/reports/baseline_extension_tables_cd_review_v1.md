# Baseline extension and Tables C/D: bounded CPU review

## Result

The inspected workflow passes source and synthetic checks after two generator fixes.
No remaining blocker exists within this review's stated scope.
This result does not establish actual answer accuracy, timing values, or candidate performance.

This review reads no actual extension outcome.
It reads no partial outcome or protected factorial result.
The literature audit separately reviews the scorer's transitive package dependencies.

The [structured record](../artifacts/baseline_extension_tables_cd_review_v1.json) contains source hashes, test cases, and all 19 scope checks.

## Findings before the fixes

Two synthetic counterexamples exposed missing checks in the first inspected table generator.

| Finding | Synthetic input | Initial behavior | Current behavior |
|---|---|---|---|
| G1: incomplete-arm admission | A QUALITY_ONLY summary had N=32, but its arm status was NOT_AVAILABLE. | The summary entered the panel. | The generator rejects the summary. |
| G2: mismatched endpoints | TP1 and TP4 rows had different explicit boundary strings. | The generator emitted ratio 2.0 from their mode names. | The generator suppresses the ratio. |

These cases did not establish a discrepancy in an actual record.
The upstream native analyzer assigns the intended endpoint strings.
The review reported both findings to Main before publication.

Main patched both checks during this review.
The final source hash identifies the patched generator.
The initial generator hash was not recorded before that patch.
Thus, this report does not assign a fabricated hash to the initial source.

The earlier analyzer-boundary tests now use a hash-pinned archive for their historical integration case.
They preserve the earlier gaps as historical characterization.
They do not describe the current complete admission workflow.

## Prompt, text, stop, and launch checks

[check_record](../scripts/validate_baseline_extension_integrity.py#L19) re-encodes each frozen prompt with `add_special_tokens=False`.
It requires exact equality with recorded prompt IDs.
It decodes the recorded output IDs with `skip_special_tokens=False`.
It requires exact equality with recorded text.

The output IDs remain authoritative.
The validator does not assume that text re-encoding reproduces those IDs.
It rejects an empty generation.

The terminal checks preserve the runner's precedence.

1. A final EOS token selects EOS.
2. Otherwise, 8192 output tokens select the cap.
3. Otherwise, a frozen token suffix must justify stop completion.

The validator rejects empty encoded stop sequences.
It rejects an earlier EOS or an earlier frozen stop.
The suffix scan excludes only the final eligible suffix position.

[The integrity entry point](../scripts/validate_baseline_extension_integrity.py#L42) checks the frozen launch anchor and its source hashes.
It checks the 64-request count, 8192-token cap, physical mapping, model revision, and saved protocol.
It requires both runner success and outer-guard success.
A successful guard must also report exit code zero.

The validator hashes the pinned tokenizer files.
The production entry point requires Transformers 5.9.0 and local files.
This review uses a synthetic tokenizer for integrated tests.
It does not claim an independent production-tokenizer execution.

The supplement is explicitly post-launch.
It changes no answer extraction, grading, sample selection, gate, sampling, EOS, stop, or token cap.
It checks recorded provenance, not an independent model re-execution.

## Complete-arm scoring admission

[analyze](../scripts/analyze_baseline_extension.py#L111) requires the integrity supplement and matching protocol hash.
It checks the frozen scorer and runner source hashes.
It excludes failed, absent, partial-status, and non-PASS integrity arms.

Before scoring an admitted arm, the analyzer requires 64 raw files and 64 audited file entries.
It rehashes every raw file.
Each hash must match the supplement.
All prompt, text, finish, and earlier-terminal checks must remain true.

The analyzer also requires all 64 exact expected IDs.
Each domain must contain 32 requests.
It checks the saved protocol, runner count, output counts, positions, depths, graph coverage, and source counters.

[request_metrics](../scripts/analyze_baseline_extension.py#L25) keeps prefill work separate from recorded decode core-pass work.
The first output token does not enter the decode-depth denominator.
The request-weighted and token-weighted depth summaries remain separate.

The fixed and adaptive pairing requires both complete 32-request domain arms.
It pairs common request IDs.
It describes original operating points, not candidate noninferiority.

Synthetic integrated checks admit one complete 64-record arm.
They exclude the three absent arms.
They also reject a false complete count, changed raw hash, and changed launch source.
Failed guard status and non-PASS integrity status exclude the arm.

## Table-generation boundaries

[native_panels](../scripts/build_result_tables_cd.py#L16) now requires matching explicit boundary strings for TP1 ratios.
It still excludes DP4 and TP2×DP2 from those ratios.
Every strongest-same-P candidate ratio remains absent.

Native TP rows use the generate API wall.
DP4 and TP2×DP2 rows use the parent release-to-completion wall.
These endpoints remain descriptive, separate clocks.
A ratio of unmatched clocks cannot become an architecture speedup.

[extension_panels](../scripts/build_result_tables_cd.py#L47) now requires an AVAILABLE matching arm.
It requires 64 unique admitted records.
It also requires 32 admitted records for the summary's domain.

Post-fix synthetic checks reject missing, repeated, wrong-arm, and wrong-domain records.
A positive complete synthetic arm still enters the panel.
The generator preserves analyzer scores.
It does not independently recompute them from records.

The rendering keeps natural-output quality separate from the controlled constant-token hardware workload.
The original 8/domain cohort and the extension's 32/domain cohort remain separate.
All candidate live E2E and free-running quality cells remain NOT_RUN.

All four-GPU cohort controls reserve 208 GiB aggregate KV, with 52 GiB per rank.
Weight multiplicity and KV ownership still differ.
The separate 32K protocol's 52 GiB aggregate reservation must not enter that cohort comparison.

## Scope of all 19 headroom rows

The [headroom supplement](../artifacts/current_headroom_decomposition_v1.json) is consistent with the [negative-scope overlay](negative_scope_audit_v1.md).
It does not present 19 measured architecture failures.

| Candidate | Scope that controls the negative or missing result |
|---|---|
| C01 | Legal ready unions and a generic pooling prior do not establish a complete ownership cost or a universal residency limit. |
| C02 | Only the isolated-source remote full-MLP component loses at tested row counts. The AG/RS refinement and full native engine remain unmeasured. |
| C03 | The evidence concerns terminal address fixtures and skipped-depth adaptive publication. Fixed R4 has no terminal-copy saving from this rule. |
| C04 | The failures concern particular nonlinear partial-sum rewrites. Other layouts can change bytes, startup, or exposed synchronization. |
| C05 | The exact recovery diagnostic needs four expensive logical rounds in the tested initialization. The screen models distributed latency. It does not measure it. |
| C06 | The 2448 adjacent row pairs cover two prompts and three selected layers per model. No adaptive-decode or nonlinear down-input claim follows. |
| C07 | The P1 fixture failed bitwise qualification. Zero accepted service trials establish neither a branch architecture failure nor task-quality loss. |
| C08 | The GPU evidence covers a 4096-byte P2 broadcast and unfused transform. No test covers a fused codec or compressed serving collective. |
| C09 | Toy certificates do not provide a native final-hidden enclosure, useful checkpoint hit rate, or sampler-equivalence result. |
| C10 | Generic restoration has costs. Known-address prefetch does not remove a fresh-query dependency. No screen measures a complete cost advantage. |
| C11 | The naive BF16 checksum control fails. It does not reject other proof systems. No measurement identifies a qualified removable native straggler. |
| C12 | The fixed small linear-feature rank construction is narrow. Nonlinear summaries, indexes, and exact fallback remain open. |
| C13 | Four observer triplets fail timing validity guards despite functional checks. Their event spans cannot supply removable time or an Amdahl share. |
| C14 | The F1/4 prefill interpolation is a model, not a bound or measured PP stage. Seven negative budgets do not reject all pipelines. |
| C15 | Static RoPE absorption fails its position control. The audited retained-K variants lose their stated capacity saving, not all reassociations. |
| C16 | Equal ready sets do not imply equal physical costs. Cross-owner service, communication, and self-consistent future readiness remain unmeasured. |
| C17 | A generic norm identity and one BF16 counterexample do not reject every qualified norm/projection implementation or certificate. |
| C18 | Bounded sampled inputs have no whole-row identity hit. This is not a universal no-reuse result or a measured lookup cost. |
| C19 | The selected-key codec and uncached-read model do not measure full K+V latency. The named 32K native success does not exhaust capacity regimes. |

No row supplies a valid complete critical-path estimate from the failed CDB observer.
No row converts encoded size, operation counts, or an interpolation into measured latency.
The source links remain necessary for each row's model, mode, and hardware limits.

The C15 wording concerns the audited retained-K variants.
It must not become a rejection of all retained-state reassociations.
The C19 native 32K statement concerns the named reservation and completed baseline.
It must not become a claim that every larger or concurrent capacity regime is feasible.

## Tests and remaining limits

The final targeted CPU run passes 19 tests across four modules.
The historical characterization module passes four additional tests.
Thus, the final review records 23 passing unit tests.

The review also runs integrated synthetic admission checks and seven post-fix adversarial checks.
These are structural tests, not independent quality samples.
They establish no confidence interval or speedup.

The four frozen launch-source hash entries match.
The three frozen analysis-source hash entries match.
The final structured record pins 26 input hashes.

Some trust boundaries remain explicit.

- The integrity supplement is a trusted local analysis artifact, not a signed execution attestation.
- The analyzer rehashes raw records but does not rerun every tokenizer check after supplement creation.
- It does not recheck every launch-source hash after supplement creation.
- A partial JSON status can abort the integrity validator. This fails closed.
- The generator trusts analyzer summary arithmetic after its admission checks.
- This review does not establish actual output integrity, score values, clocks, or transitive scorer package behavior.

This reviewer edited no source or runtime code.
No frozen report, card, grading rule, raw file, or GPU state changed.

## Actual work

The actual interval is 2026-10-19 17:48:00 UTC through 2026-10-19 17:53:55 UTC.
The active interval through this checkpoint is 355 seconds.
No prior pause interval enters this segment.
Concurrent worker time does not add campaign elapsed time.

