# Independent review of the synthesis draft, version 1

## Result and scope

The draft supports zero current qualifying finalists as a selection outcome.
It does not support a universal architecture-failure claim.
The draft already preserves that distinction in most sections.
Seven bounded wording fixes remain before final publication.
No reviewed memory calculation requires a numerical correction.

This review covers the complete draft and both new Mermaid sources.
It checks closed source tables, selected arithmetic, model assumptions, and the negative-scope overlay.
It does not inspect active quality outputs or the protected factorial outcome.
It does not audit the draft's completed task-accuracy counts, source-count ledger, duration total, or final transport status.
No GPU, model execution, new engine, or candidate rescue occurred.
The reviewed files remain unchanged.

## Required fixes

### R1 — Qualify the TP2 physical mapping

Location: [draft line 84](final_report_draft_v1.md:84).

The unqualified statement assigns native TP2 to GPUs4/5.
That mapping applies to the constant-token cohort.
The separate long-context control uses GPUs4/6.
Its [report](native_long_context_observation_v1.md:59) states that mapping explicitly.

Suggested replacement:

“The constant-token native TP2 cohort uses physical GPUs4/5.
The separate long-context TP2 control uses GPUs4/6.
The isolated P2 MLP fixture also uses GPUs4/6.”

Keep per-run mappings authoritative.
Do not reinterpret the long-context TP2 rows as measurements on GPUs4/5.

### R2 — State the distributed baseline's depth mode

Locations: [draft line 28](final_report_draft_v1.md:28) and [Section 4](final_report_draft_v1.md:125).

The distributed controls use fixed R4.
The draft separately defines an adaptive reference, but does not state this boundary at the headline baseline claim.
The [hybrid report](native_hybrid_cohort_observation_v1.md:95) explicitly excludes an adaptive distributed-runtime result.
The [long-context report](native_long_context_observation_v1.md:64) also specifies fixed R4.

Add:

“These distributed controls use fixed R4.
They do not establish original-gate adaptive TP2/TP4 or distributed DP performance.”

This qualifier does not change the measured baseline results.

### R3 — Define the C14 substituted metric and percentage

Location: [draft lines 248–253](final_report_draft_v1.md:248).

The source model divides the complete TP1 first-token metric by four.
That metric is not isolated prefill compute.
The draft's phrase “assumed faster prefill” hides this distinction.
Its percentage also needs an explicit denominator.

Use the [source equations](c14_transpose_decision_screen_v1.md:72):

`M0 = W4 - F4 + F1/4`

Here, `W4` denotes measured TP4 API wall.
`F1` and `F4` are complete frontend first-token medians.
The modeled zero-transition API-wall reduction is `100*(F4-F1/4)/W4`.
It ranges from 4.13% to 10.14% at 128 outputs.
It is not a throughput increase or a measured prefill speedup.

Keep the seven negative 10% budgets and the 2.138374 ms positive budget.
The CPU check reproduces those values from all eight closed model rows.
Keep the existing statement that this model rejects neither all pipelines nor all output regimes.

### R4 — Narrow the C08 cost consequence

Location: [draft line 216](final_report_draft_v1.md:216).

“Does not pay” reads as a completed negative latency comparison.
The [underlying report](cost_and_headroom.md:98) measures separate raw-broadcast and unfused-transform primitives.
It does not measure a complete entropy-codec serving path.
The 4096-byte transform is approximately 0.00908 ms before coding or transport.
The raw P2 broadcast is approximately 0.01035 ms.
Those observations discourage this path, but do not determine every composed codec's latency.

Suggested replacement:

“Size reduction alone does not establish a net saving under the tested small-message costs.
The campaign did not time a complete codec serving path.”

Keep the fused-codec and exposed-message prerequisites.

### R5 — Separate required P8 projections from P8 measurements

Locations: [draft line 270](final_report_draft_v1.md:270) and [evidence-map line 12](../figures/final_evidence_map.mmd:12).

The specification requests three projections from P2/P4 behavior and a stated topology.
It does not require unauthorized P8 execution.
Thus, scenario names do not satisfy missing projections, rather than “missing measurements.”

Suggested draft replacement:

“The required candidate-level optimistic, central, and conservative P8 critical-path projections remain incomplete.
The component equations retain explicit unknowns.
No P8 measurement exists.”

Suggested figure destination:

“Candidate P8 critical-path projections incomplete.
No P8 measurement.”

Retain the [supplement's](p8_projection_feasibility_supplement_v1.md) topology and allocation qualifiers.
The hypothetical two-island case is not the archived same-host all-NV18 table.
Neither advertised bandwidth nor P4 medians establish eight-rank communication performance.

### R6 — State the hindsight condition of the depth-affinity model

Location: [draft lines 256–261](final_report_draft_v1.md:256).

The numeric range matches the source.
The [screen source](../scripts/depth_affinity_extension_screen.py:51) adds another material assumption.
It selects the best static partition using the complete observed depth histogram.
That search is hindsight analysis, not a causal online placement policy.

Add:

“The model selects its best static owner partition from the complete observed depth histogram.
It is not a causal scheduler.”

Keep the ideal-pool, unit-cost, zero-transfer, and non-throughput qualifiers.

### R7 — Make the shortlist gate non-circular

Locations: [draft lines 275–281](final_report_draft_v1.md:275) and [line 293](final_report_draft_v1.md:293).

Absence of LIVE_E2E alone cannot justify an empty research shortlist.
The [specification](spec_snapshot.md:962) permits expected gains stated as calibrated ranges or explicit hypotheses.
The current selection is defensible because no remaining proposal combines concrete distinct benefit, a complete cost contract, and qualified output semantics.

Add:

“A finalist need not already have LIVE_E2E evidence.
The current proposals lack the combined mechanism, cost, and numerical support needed to justify a targeted architecture PoC.”

Keep `CONDITIONAL_MECHANISM` and `NOT_RUN` separate from measured failure.
Do not convert the missing P8 projection or untouched held-out cohort into a candidate failure.

## Checks that passed

The closed constant-token CSV contains 80 cells and 400 admitted trials.
Its C1/N1024 TP1/TP4 ratios round to 1.340× and 1.267×.
The draft retains the API-versus-parent timing distinction.
The long-context CSV contains 24 cells and 144 trials.
Its N32768 mean paired ratios round to 2.1883× and 2.2098×.
These checks do not add workload-population evidence.

The frozen ledger contains 19 records.
Its counts are 12 `GENERIC_ONLY`, six `NO_NET_HEADROOM`, and one `CONDITIONAL_MECHANISM`.
Every candidate LIVE_E2E and free-running-quality field remains `NOT_RUN`.
The overlay, not the frozen label alone, defines the claim's scope.

The KV formula reproduces 786432 and 1572864 bytes per fixed-R4 historical position.
It also reproduces 96/192 logical block executions and depth-indexed slots.
The cohort reservation is 52 GiB per rank, or 208 GiB across four ranks.
The separate context reservation is 52 GiB per TP group, or 52/26/13 GiB per rank.
These are storage quantities, not HBM traffic or attention-time measurements.

The P8 supplement's Base2.6B C8/N8192 full-history replica requires 96 GiB KV per rank.
Its TP8-sharded history is 12 GiB per rank.
Only the first allocation exceeds the assumed device total before weights and scratch.
The capacity rejection does not apply to all owners or all distributed designs.

The dependency figure matches the inspected fixed-R4 recurrent norm boundaries.
Its caption scopes it to fixed R4 and excludes measured-overlap claims.
The opaque core nodes explicitly hide local forks.
The dependency figure needs no semantic fix.
The evidence figure already states that frozen labels are not 19 measured architecture failures.
I did not render either figure or test it with a Mermaid compiler.

## Actual work and evidence

Work started at 2026-10-19T18:53:42Z.
The [structured review](../artifacts/final_report_independent_review_v1.json) records the accounting cutoff, source hashes, checks, and inspection limits.
No pause or deliberate wait occurred.
This concurrent interval does not add campaign wall time.
Final stamping and handoff after the cutoff do not enter this interval.
