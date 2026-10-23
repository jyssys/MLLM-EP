# Failures and unresolved issues

## CPU input-source attempt 1

The initial Alpaca source slice had no inputs in the preregistered 180–350-token content window.
The generator stopped before it froze a workload or ran any policy.
A direct source diagnostic found input-token quantiles [5, 15, 31, 87, 170] at P0/P50/P90/P99/P100 for rows 10,000–11,999.
The input contained the instruction and optional context, not the reference answer.
No meaningless padding or target output tokens were added.

The revised source uses public Dolly instruction/context paragraphs for the chat-like lane.
The source hash, selected input rows, and actual length distribution remain mandatory.
This changes the workload proxy, not model weights, sampling, or a final serving outcome.

The first Dolly URL under the GitHub repository returned HTTP 404.
The source moved to the publisher's Hugging Face dataset file at revision `bdd27f4`.
No workload or policy result existed before this URL correction.

## Adaptive smoke v1: retained measurement wrapper

All four first policy cohorts saved their raw natural results and passed the Full-R4 assertion.
The next engine construction failed while allocating another 27-GiB KV tensor.
Each old engine owned a 54-GiB cache pool.
The CPU WorkLedger wrapper still referenced its closed engine after the first policy finished.
A weak-reference regression test reproduced this retained-engine condition without CUDA.
The test failed before the correction.

The close method now restores the native submit callback and clears its engine/bound-method references.
The dual controller also clears its closed engine references.
The correction does not change kernels, model state transitions, objective weights, or arrivals.
The v1 attempts, first cohorts, source snapshots, OOM traces, and failure markers remain immutable.
Reviewed v2 attempts repeat the same smoke inputs and policy cells.
No performance claim uses the failed v1 attempts.
All four reviewed v2 smoke jobs subsequently completed both policy cells with Full-R4 accounting.
The OOM message's CUDA0 is each worker's logical device. It maps to its authorized physical GPU4–7.

## Historical limits retained

The previous C-only accuracy regressions remain unresolved.
The frozen service lookup lacks exact support for many clipped spans.
Risk scores are empirical exposure proxies, not future complete-token predictions or SLO guarantees.
The initial checkpoint had no new quality or serving-capacity result.
The completed pilot's measured quality and remaining limits now appear in `quality.md`.
Qualified sustainable capacity remains NOT_RUN.

## CPU analysis command and schema corrections

An unqualified unittest discovery command could not resolve package-relative imports.
Discovery with the repository as the top-level package passed all 13 tests.
This was a test-launch error, not a failed model test.

The first constructed-state CSV writer rejected heterogeneous feasible/infeasible row keys.
The corrected analysis uses the union of those keys and preserves infeasible rows with empty action fields.
This correction changes no GPU job, frozen objective, or source split.

The first standalone artifact-audit command imported a helper with the parent's report root.
It reported a missing completed attempt at that incorrect root.
The corrected audit resolves each artifact directly under this campaign's completed marker.
The launcher already used the correct process-local root.
This analysis error was not evidence of raw-data corruption.

A standalone evaluator check initially put the older worktree first in Python's search path.
That order shadowed the current `poc_flashvep` package and failed before scoring.
The corrected import order matches the actual grader and passes the extractor SHA256 check.
Parser-only fixtures also pass the preserved API check for both domains.
These fixtures are not model generations or benchmark accuracy measurements.

The first portable export encountered an older completed smoke schema without `service_observations`.
It stopped with KeyError before publishing a portable-evidence manifest.
The corrected exporter records source-schema availability and retains missing optional audit fields as null.
It does not invent an event measurement for an unobserved old smoke.
Derived gzip exports are regenerated. Original raw attempts and their successful hashes remain unchanged.

The default Git whitespace check flags the standard CRLF endings emitted by the inherited CSV writer.
The scoped check with `cr-at-eol` accepts those line endings and checks other whitespace errors.
No raw CSV values or evidence bytes change to suppress that diagnostic.
