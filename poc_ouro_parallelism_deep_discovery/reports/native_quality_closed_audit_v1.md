# Closed native TP4 baseline-quality audit v1

## Result and scope

The four Table C2b rows match the admitted native records and their analysis.
The counts, accuracy intervals, paired intervals, parser failures, and truncation counts pass independent CPU checks.
The current numerical table needs no change.
Both arms pass their recorded outer guards, but neither shutdown is warning-free.
The cleanup warnings must remain visible in the final dossier.

This is `QUALITY_ONLY` evidence for the existing fixed-R4 native TP4 baseline.
It is not candidate validation, a new held-out cohort, or a performance comparison.
The 128 native records cover 64 unique request IDs across two model arms.
Each arm has 32 GSM8K and 32 MATH-500 requests.
Repeated use of those IDs does not create 128 independent requests.

This audit reads the [admitted integrity record](../artifacts/native_quality_integrity_complete_v1.json) and [native analysis](../analysis/native_quality_extension_complete_v1/summary.json).
It also reads their exact closed launch receipts and bounded shutdown-warning lines.
It does not read any other experiment's raw outcomes or the original candidate held-out.
It does not change [the table](../analysis/result_tables_cd_complete_v2/tables_cd.md:69), source, or final report.

## Table C2b arithmetic

All intervals below are request-bootstrap 95% intervals.
Each domain contains N=32.
The CPU calculation uses 4000 resamples and seed 2026101911, with the original record order.

| Base / domain | Correct/N | Accuracy and 95% interval | Parse failures | Length-capped |
|---|---:|---|---:|---:|
| 1.4B / GSM8K | 25/32 | 78.125%, [62.500, 90.625]% | 1 | 0 |
| 1.4B / MATH-500 | 16/32 | 50.000%, [34.375, 65.625]% | 1 | 6 |
| 2.6B / GSM8K | 28/32 | 87.500%, [75.000, 96.875]% | 0 | 0 |
| 2.6B / MATH-500 | 25/32 | 78.125%, [62.500, 90.625]% | 2 | 6 |

Length-capped answers remain incomplete under the frozen scoring rule.
Parsing failure and length truncation are separate predicates.
They must not share one inferred count.
The summary's `correct_completed` predicate equals `answer_matches and not truncated` for all 128 records.

The generated panel objects equal the stored panel objects exactly.
The rendered C2b Markdown also matches the generator's output.
The CSV contains the same four domain numerators and denominators.
No quality-run wall value enters the table as architecture latency or speedup.

## Paired baseline changes and output identity

The comparison uses the same request IDs and staged fixed-R4 reference.
It does not compare native fixed R4 with staged adaptive output.
The following changes are native minus staged, in percentage points.

| Base / domain | Accuracy change and paired 95% interval | Staged wrong → native correct | Staged correct → native wrong | Identical output IDs/N |
|---|---|---:|---:|---:|
| 1.4B / GSM8K | +3.125 pp, [0.000, 9.375] pp | 1 | 0 | 20/32 |
| 1.4B / MATH-500 | −3.125 pp, [−15.625, 6.250] pp | 1 | 2 | 4/32 |
| 2.6B / GSM8K | 0.000 pp, [0.000, 0.000] pp | 0 | 0 | 25/32 |
| 2.6B / MATH-500 | 0.000 pp, [−9.375, 9.375] pp | 1 | 1 | 0/32 |

Equal aggregate accuracy does not imply equal per-request results or equal token sequences.
The zero-width GSM8K paired interval reflects unchanged correctness on this finite sample.
It does not establish numerical equivalence or a population noninferiority margin.
The intervals do not resolve the proposed 1 pp or 2 pp candidate-quality margins.

The output-identity counts come from admitted parity records.
I check their counts, lengths, common-prefix fields, and first-divergence consistency.
I inspect the exact list-comparison function and its use by the analyzer.
I do not reopen the staged raw token arrays or independently repeat every raw token comparison.
No model-logit, hidden-state, or full-KV parity claim follows.

The baseline paths use different BF16 backends and different stopping mechanisms.
Native string stopping differs from staged token-suffix stopping.
This audit does not assign a sole cause to any observed divergence.
No difference becomes a candidate's gain or loss.

## Ownership, source receipt, and cache checks

Both ownership files contain ranks0–3 in world size4.
The ordered mapping is physical GPUs4/5/6/7 for logical ranks0/1/2/3.
Each UUID matches the campaign's pinned physical-device identity.
Base1.4B retains 24 physical layers and 96 depth-indexed cache slots per rank.
Base2.6B retains 48 physical layers and 192 slots per rank.
Each physical layer has four local Q/KV heads and four recurrent cache modules.

Both launch receipts match all 14 frozen source hashes before native imports.
Both cache receipts identify fresh job-local vLLM, Inductor, and Triton roots.
The recorded engine settings retain BF16, fixed R4, TP4, native asynchronous scheduling, and graph enablement.
Prefix caching, chunked prefill, and EP remain disabled.
These quality arms reserve 20 GiB KV per rank.
They are not the 52 GiB/rank constant-token hardware protocol.

All 157 integrity input hashes match current files.
That count includes 128 compressed native request records.
The native analysis correctly hashes the integrity artifact, and Table C2b correctly hashes the native analysis.
These checks establish recorded provenance, not a complete binary-build receipt or independent model execution.

## Cleanup warnings and admission

Both arm statuses report 64 completed records.
Both outer guards report success, exit code0, and four restored burner workers.
These guard observations come from retained files, not a new live process or GPU check.

The [Base1.4B console](../outputs/native_quality_extension_base14_TP4_v2_isolated/console.log:249) records shutdown after SIGTERM.
Lines255–257 report one remaining process, a worker grace-period warning, and one leaked shared-memory object.
The [Base2.6B console](../outputs/native_quality_extension_base26_TP4_v2_isolated/console.log:324) records analogous shutdown.
Lines330 and340 contain TCPStore/NCCL connection-closure warnings.
Lines341–343 report one remaining process, worker SIGTERM count4, and one leaked shared-memory object.

Do not describe either run as warning-free graceful shutdown.
The warnings do not independently invalidate the 64 admitted request records or the successful outer guard.
They also do not prove that no cleanup defect exists.
Retain both the warning evidence and the guard outcome.

## Generator and CPU test coverage

The [generator](../scripts/build_result_tables_cd.py:65) checks quality-only labels, 64 unique records per arm, and 32 records per domain.
It recomputes each correctness numerator and rejects candidate relabeling.
The stored C2b rows pass those checks.

The two targeted CPU modules pass all 19 tests.
They cover partial arms, repeated IDs, incorrect numerators, candidate relabeling, completion rules, prompt binding, and text binding.
They also preserve the different hardware timer endpoints.
The test command disables CUDA visibility and Python bytecode writes.
No test loads a model or executes CUDA.

Coverage remains narrower than this audit:

- The panel function copies CI, parser-failure, and truncation fields without recomputing them.
- The native integrity validator hashes ownership, but does not independently repeat the ordered rank/UUID predicate.
- The targeted tests do not cover every mutation of those fields or every launch-receipt failure.

Those are future test-hardening opportunities, not failures of the current admitted table.
This audit supplies the missing aggregate and ordered-mapping checks for these two closed arms.
It does not certify every future input accepted by the generator.

## Actual work and limits

Work-segment initial timestamp: 2026-10-19T19:25:12Z.
The [JSON audit](../artifacts/native_quality_closed_audit_v1.json) records the accounting cutoff, checks, hashes, and warning locators.
No deliberate wait, GPU access, model run, source edit, or final-report edit occurred.
Concurrent worker time does not add campaign wall time.
