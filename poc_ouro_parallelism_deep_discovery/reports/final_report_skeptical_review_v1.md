# Skeptical review of the active synthesis draft

## Review outcome

The draft needs scoped corrections before it becomes a final synthesis.
The main dependency and evidence-class limits are sound.
This review does not assign a final campaign verdict or certify the duration gate.

The reviewed file is [final_report_draft_v1.md](final_report_draft_v1.md).
Its SHA256 is `ee94ed1bd2fccb8d21680a730aa36e6b5dc730c5980a8f34d35824d574ccb029`.
Line numbers below refer to that unchanged file.
The [structured review](../artifacts/final_report_skeptical_review_v1.json) records input hashes and read limits.

## Corrections

### F01. Separate each physical layout from its logical ranks

Draft lines 82–86 contain a factual error.
The completed standalone native TP2 jobs use physical GPUs 4,6, not 4,5.
Both [Base 1.4B](../outputs/native_cohort_base14_P2_v1/run_contract.json) and [Base 2.6B](../outputs/native_cohort_base26_P2_v1/run_contract.json) receipts record mask `4,6`.
The [long-context audit](native_long_context_observation_v1.md), lines 59–61, gives the same TP2 mapping.

Use method-specific mappings:

| Layout | Physical ownership | Logical mapping |
|---|---|---|
| Native TP4 | GPUs 4,5,6,7 | One group with ranks 0,1,2,3 |
| Standalone native TP2 | GPUs 4,6 | One group with ranks 0,1 |
| TP2×DP2 | Groups 4,5 and 6,7 | Ranks 0,1 inside each group |
| DP4 | One physical GPU per independent child | Visible GPU 0 inside each child |

The [hybrid audit](native_hybrid_cohort_observation_v1.md), lines 78–82, verifies the two TP2 groups.
Do not describe every four-device reservation as a four-rank TP world.
Keep the older P2 MLP fixture's 4,6 mapping separate from the collective-pair probe's mapping.
A participant UUID set alone does not prove rank order.
Name the ordered receipt if the draft claims an ordered UUID check.

### F02. Correct the earlier local context cap

Draft line 169 names a 4096-token cap.
The linked [long-context audit](native_long_context_observation_v1.md), line 12, names the earlier local cap as 10240.

Suggested replacement:

> These controls exceed the earlier local 10240-token runtime cap. That cap was not the checkpoint context limit.

Retain the existing limits.
Configured support through 65536 is not tested numerical or natural-quality validity at that length.
The completed control reaches 32768 prompt tokens plus 128 controlled outputs.

### F03. Correct the missing manifest link

The line-78 target `artifacts/model_runtime_manifest.json` does not exist.
This is the only missing target among 30 local link occurrences in the draft.

Use [semantics_manifest.json](../artifacts/semantics_manifest.json) for model and runtime semantics.
Use [environment_manifest.json](../artifacts/environment_manifest.json) separately for environment and device metadata.

### F04. Do not turn the mixed-clock rank count into a method winner

Draft lines 144–147 correctly distinguish the timing endpoints.
Lines 149–151 then rank raw medians from those different endpoints.
The 14/16 count is descriptive arithmetic.
It is not a common-endpoint performance ordering.

The [hybrid audit](native_hybrid_cohort_observation_v1.md) distinguishes native API wall from parent release-to-completion wall.
[Tables C/D](../analysis/result_tables_cd_complete_v1/tables_cd.md) exclude cross-endpoint speedup normalization.

Prefer to omit the rank count.
If the count remains, use this explicit label:

> With their different recorded timer endpoints, TP4 has the smallest raw median in 14/16 cells. This count does not establish a common-endpoint winner.

Keep the sequential-job and repeated-pattern limits.
This correction does not require a new run.

### F05. Name the adaptive runtime and its depth denominator

Draft line 192 says “native depth counters.”
In this section, those are CDB runtime counters.
The term can be mistaken for native vLLM adaptive TP validation.

Suggested replacement for lines 192–194:

> CDB runtime counters match the recorded adaptive decode events for every admitted request. Decode-depth accounting excludes full-R4 prefill and its first output token. Token-weighted mean decode depth ranges from 2.809121 to 3.059609 across the four size/domain cells.

Add that these single-GPU reference jobs do not establish adaptive TP2 or TP4 support.
The [completed quality report](baseline_extension_complete_v1.md), lines 80–96, states these limits.

The draft's quality counts are correct.
The completed summary contains 256 records across four arms.
Each arm has 32 GSM8K and 32 MATH-500 records.
There are 64 unique request IDs.
All recorded source-counter checks pass.
The eight correct/N entries and four MATH truncation counts in the draft match the completed summary.
No independent-token confidence interval or small-margin noninferiority claim follows.

### F06. Preserve the exact C06 operand and sample scope

Draft line 214 should name adaptive decode as unresolved, not all adaptive mode.
The [C06 screen](c06_sparse_fresh_delta_screen_v1.md), lines 11–23, includes full-R4 prefill in both modes.
Its R4/adaptive prefill values are bitwise identical.
Their separate record counts are not independent replication.

The 2448 total is a valid count of scoped adjacent-pass row-pair records.
The sampled scalar change fractions are approximately 99.50019%–99.63091%.
Thus, the samples support “nearly all sampled coordinates change.”

The captured operand is the 2048-wide MLP input.
It is not the 5632-wide nonlinear input to the down projection.
The screen states this at lines 89–91.

Suggested unresolved cell:

> Adaptive decode, the uncaptured down-projection input, projection parity and cost, and broader workloads.

Add the duplicate-prefill qualifier beside the 2448 denominator.
Do not transfer this result to an uncaptured operand or a complete sparse projection kernel.

### F07. State the depth-affinity model's hindsight assumption

Draft lines 256–261 retain the main cost limits.
They omit one material assumption.

The [complete depth-affinity screen](../analysis/depth_affinity_extension_complete_v1/screen.json) searches every permitted static partition using the observed exit histogram.
This is hindsight optimization.
It is not a causal online policy.

Add:

> The model optimizes the static partition in hindsight from the observed exit histogram. This is not a causal online ready-frontier schedule.

The four P4 records support the quoted range 0.702280–0.764902.
These ratios describe normalized saturated capacity inside the unit-cost model.
They are not measured GPU throughput, a native TP4 speedup, or a physical latency bound.
The earlier [capacity-screen report](depth_affinity_capacity_screen.md) has a smaller request scope.
Link the complete extension artifact when giving the new ratios.

### F08. Label logical KV slots, not physical allocations

Draft line 63 uses “Independent KV slots.”
Prefer “Logical depth/layer KV slots.”

Shared weights do not make history versions equal across four computed passes.
This does not require one separate allocator object for every slot.
A shared cache pool can contain disjoint logical versions.
Adaptive terminal-version aliasing has a different validity and lifetime contract.
The [exact frontier contract](exact_frontier_contract.md) keeps these state distinctions explicit.

The draft's 786432 and 1572864 bytes/token are correct full-R4 capacity quantities.
Do not turn them into measured reads or a universal physical-storage lower bound.

### F09. Narrow the opening control claim

Draft line 27 says that conventional controls “work.”
Use:

> Conventional controls complete the declared controlled hardware workload on the authorized H100 devices.

The controlled constant-token lane does not establish natural-generation quality or complete numerical equivalence.
The later sections already state this distinction.
The opening should retain it too.

### F10. Give active statements an explicit snapshot

Draft lines 7–9 and 197 contain changing duration and run-status facts.
Add the draft creation UTC and a specific coordinator-ledger snapshot link or hash.
Keep the native extension status explicitly as of that snapshot.

This review did not recompute coordinator duration.
It did not inspect pending native-extension outcomes.
It does not challenge or certify the quoted 8.776-hour snapshot.

## Boundaries that pass this review

The B1 dependency claim covers only the inspected graph.
It does not exclude every exact reformulation or quality-controlled algorithm.
Independent ready requests remain legal work.

The draft distinguishes 19 frozen dispositions from 19 measured architecture failures.
It does not convert generic controls into new architectures.
It makes no candidate LIVE_E2E or candidate free-running quality PASS claim.

C07 retains zero accepted service trials and an unresolved numerical cause.
The failed observer guards do not become component shares or Amdahl percentages.
The failed collective-pair runs remain NOT_ADMITTED.

C14 remains a model, not a measured pipeline or universal latency bound.
C19's capacity and read models remain separate from HBM counters and latency.
P8 remains PROJECTED_NOT_RUN.
The source corrections remain SOURCE_REPORTED, not campaign candidate measurements.

The proposed searched-space conclusion remains explicitly active and scoped.
It is not a universal impossibility claim.
This review does not turn that provisional conclusion into a final verdict.

## Validation and activity record

This review used only CPU source, metadata, and completed baseline-summary checks.
No GPU command, source edit, frozen-artifact edit, protected-factorial read, or candidate-held-out read occurred.
This review did not inspect raw response text or pending native-quality outcomes.

Input SHA256 values and exact check results are in the structured review.
Active work began at `2026-10-19T18:53:42Z`.
The structured review records the final validation timestamp and elapsed seconds.
This time record excludes prior work and pause intervals.

