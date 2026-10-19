# Ouro parallelism deep discovery: synthesis draft

## Status

This is a review draft, not the final campaign completion record.
The campaign remains active.
The closed coordinator ledger currently records approximately 8.776 hours.
The ten-hour normal-completion gate remains unmet at this draft's creation.
The cache-isolated native TP4 quality extension is running.

No qualifying new distributed architecture is established.
The current shortlist contains zero candidates.
No candidate has a LIVE_E2E result or a free-running task-quality result.
The original candidate held-out cohort remains untouched.

The proposed scoped conclusion for review is NO_QUALIFYING_CANDIDATE_IN_SEARCHED_SPACE.
It is not a claim that no new parallelism can exist for Ouro.
The current conditional and unmeasured mechanisms remain explicitly unresolved.

## 1. What the campaign establishes

Shared weights do not supply future hidden states or equal depth-indexed KV.
The inspected B1 decode graph retains its recurrent chain.
Independent requests and ordinary operator forks still supply legal concurrency.
These observations do not rule out every exact reformulation or quality-controlled algorithm.

Physical conventional controls work on the authorized H100 devices.
The campaign measures native TP1/TP2/TP4, DP4, and TP2×DP2 on both Base sizes.
It also measures selected operators and transport fixtures.
Those controls are not new recurrence-specific architectures.

The strongest ideas encounter different limits.
Some have direct generic or looped priors.
Some lose in a named component implementation or conditional cost model.
Others lack a valid numerical contract, exposed-time observation, or self-consistent online schedule.
These are not interchangeable failure classes.

The 19 frozen dispositions contain 12 GENERIC_ONLY, six NO_NET_HEADROOM, and one CONDITIONAL_MECHANISM labels.
They are bookkeeping labels, not 19 measured architecture failures.
The [scope overlay](negative_scope_audit_v1.md) governs each interpretation.

## 2. Reproducible scope

| Item | Actual campaign scope |
|---|---|
| Models | Dense Base Ouro-1.4B and Base Ouro-2.6B |
| Precision | BF16 model execution |
| Fixed mode | R4 with full depth-indexed KV |
| Adaptive reference | Original gate, q=0.5, minimum R2, maximum R4, `last_exited`, full-R4 prefill |
| Authorized physical devices | H100 GPUs 4, 5, 6, and 7 only |
| Physical parallel measurements | P2 and P4 subsets documented in each receipt |
| P8 | Unmeasured capacity and topology scenarios only |
| Training | None |
| New engine | None |
| Protected factorial experiment | Preserved and NOT_EVALUATED |

The campaign does not repeat stale/delta hidden-state replacement.
It does not change the empirical target to a dLLM, Huginn, Raven, or a looped MoE.
It does not count adaptive-versus-fixed quality differences as a new mechanism's loss or gain.

### Loaded models

| Base model | Checkpoint revision | Physical blocks | R4 logical block executions | Independent KV slots | Hidden / FFN width |
|---|---|---:|---:|---:|---|
| 1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | 24 | 96 | 96 | 2048 / 5632 |
| 2.6B | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | 48 | 192 | 192 | 2048 / 5632 |

Both configs have 16 Q heads and 16 KV heads with head dimension 128.
The inspected maximum position is 65536, with RoPE theta 1000000 and no sliding window.
Configuration support does not prove every runtime allocation or workload at that length.
The actual native context controls reach 32768 tokens under their stated reservation.

Logical full-R4 KV is 786432 bytes/token for Base 1.4B.
It is 1572864 bytes/token for Base 2.6B.
These are capacity quantities, not measured HBM reads.
Adaptive work skipping does not automatically remove allocated depth slots.

See [model semantics](ouro_semantics_audit.md), [memory screen](interpass_cache_capacity_screen.md), and [runtime manifest](../artifacts/model_runtime_manifest.json).

### Device ownership

P4 maps physical GPUs 4/5/6/7 to visible logical ranks 0/1/2/3.
The UUID receipt checks this mapping.
Native TP2 uses physical GPUs 4/5.
The older isolated P2 MLP fixture uses physical GPUs 4/6.
Its visible ranks are therefore 0/1, not the P4 rank numbers.

All four authorized GPUs are H100 80 GB HBM3 devices with NV18 pair connectivity in the archived topology.
The campaign changes no power or clock setting.
Other users' jobs remain untouched.
Only verified owned utilization workers stop during guarded measurements.

## 3. Search and evidence coverage

The campaign registers 19 candidate cards across several mechanism families.
The initial register contains 14 cards across seeded areas A–G.
C09, C10, and C11 document independent alternatives outside seeded areas A–F.
C05–C08 include four initial execution-axis hypotheses.
Initial categories are hypotheses, not demonstrated novelty.

The current ledger checkpoint15 indexes 264 source observations and 88 distinct arXiv keys.
It indexes 230 work/document keys and 193 searches.
These counts include selected-section, code, metadata-only, and duplicate-version observations.
They are not complete-paper read counts.

The [nearest-work audit](literature_nearest_audit.md) and [claim re-audit](primary_claim_reaudit_v1.md) record the actual inspected extents.
The [final novelty audit](final_novelty_claim_audit_v1.md) reads all 19 cards and the current synthesis.
The coarse time categories cannot certify the exact fraction of effort spent on new axes or unseeded generation.
The dossier does not reconstruct unsupported retrospective percentages.

The source corrections matter:

- CDB computes model argmax feedback while controlling benchmark lengths and recorded exit traces.
- CDB Appendix B includes a Base 2.6B recurrent-step component experiment.
- KernelFlume includes real A6000 batch-2-to-64 evidence, not this campaign's H100 C1 result.
- SNLP labels its inspected verified-drafter E2E values theoretical.
- FlashLoop's BF16 option does not remove its sparsity approximations.
- Public RLT provides another original-gate runtime. CDB is not the only public implementation.
- RLT's inspected distributed path uses full-model P/D replicas, not validated adaptive TP2 or TP4 here.

These remain SOURCE_REPORTED facts.
They do not fill any campaign candidate-performance cell.
See the [RLT scope overlay](public_adaptive_runtime_scope_overlay_v1.md).

## 4. Physical conventional baselines

### Controlled constant-token workload

The completed cohort audit admits 400 clean trials across 80 cells and ten jobs.
Each cell has five trials and two reused prompt patterns at most.
Each request generates 128 copies of a fixed token for hardware control.
EOS does not shorten that workload.
These are physical inference measurements, not natural reasoning accuracy.

The following C1/N1024 medians use the same native API timing boundary.
The ratio divides the TP1 median by the corresponding TP median.
Sequential jobs do not establish randomized paired architecture confidence intervals.

| Base | TP1 API wall | TP2 API wall | TP4 API wall | TP1/TP4 ratio |
|---|---:|---:|---:|---:|
| 1.4B | 1089.762 ms | 929.606 ms | 813.429 ms | 1.340× |
| 2.6B | 2062.765 ms | 1826.399 ms | 1628.599 ms | 1.267× |

DP4 and TP2×DP2 use parent release-to-all-completed wall time.
That endpoint includes controller and receipt work.
It is not interchangeable with the native API endpoint.
The generator suppresses cross-endpoint speedup normalization.

TP4 has the lowest observed median in 14/16 four-GPU cells.
The other two C8/N4096 cells differ by less than 0.4%.
These small numerical differences do not establish a reliable winner.
The parent-minus-longest-worker residual also cannot identify every scheduling cost.

Every native rank reserves 52 GiB KV in this protocol.
The four-GPU arms therefore reserve 208 GiB aggregate KV.
Weight multiplicity and KV ownership differ across TP4, DP4, and TP2×DP2.
The separate long-context protocol reserves 52 GiB aggregate KV instead.

Full evidence: [conventional controls](native_hybrid_cohort_observation_v1.md) and [Tables C/D](../analysis/result_tables_cd_complete_v1/tables_cd.md).

### Long-context controls

The separate audit admits 144 trials across 24 cells.
It measures both Base sizes, TP1/TP2/TP4, and prefixes through 32768 tokens.
At 32768 tokens, mean paired TP1/TP4 request-wall ratios are 2.1883× and 2.2098× for Base 1.4B and Base 2.6B.
Only two frozen prompt patterns support those paired ratios.
They do not establish a population interval or an attention-only speedup.

These controls remove the old premise that the configured 4096-token cap was an unavoidable local capacity limit.
They do not reject every larger-context or higher-concurrency capacity opportunity.
See [the complete context audit](native_long_context_observation_v1.md).

## 5. Natural quality and actual adaptive work

All four staged baseline extension arms complete 64 requests.
The integrity validator admits all 256 records.
Each arm has the same 32 GSM8K and 32 MATH-500 IDs.
There are 64 unique request IDs, not 256 independent requests.

| Base | Original mode | GSM8K correct/N | MATH-500 correct/N | MATH length-capped/N |
|---|---|---:|---:|---:|
| 1.4B | Fixed R4 | 24/32 | 17/32 | 4/32 |
| 1.4B | Adaptive | 24/32 | 21/32 | 4/32 |
| 2.6B | Fixed R4 | 28/32 | 25/32 | 5/32 |
| 2.6B | Adaptive | 28/32 | 24/32 | 6/32 |

The [complete quality report](baseline_extension_complete_v1.md) retains percentages, request-bootstrap 95% intervals, parser failures, and paired differences.
Length-capped answers remain incomplete under the frozen rule.
The 32-request domains do not establish 1 pp or 2 pp noninferiority margins.
These are Base checkpoints, not a reproduction of the Thinking model's official reasoning protocol.

Actual native depth counters match the recorded adaptive events for every admitted request.
Token-weighted mean decode depth ranges from 2.809121 to 3.059609 across the four size/domain cells.
The selected reference physically skips some core passes.
It still allocates depth-indexed KV.

The independent native TP4 robustness extension remains pending in this draft.
Its cache-isolated protocol uses these same IDs, not a new held-out cohort.
Native string stopping and staged token-suffix stopping remain different backend mechanisms.
No cross-backend quality difference becomes a candidate's gain or loss.

## 6. Decisive candidate screens and limits

The full per-ID records are in [Tables A/B/E/F](current_tables_abef_overlay_v1.md).
The charged-cost view is in [Table D](../analysis/result_tables_cd_complete_v1/tables_cd.md).
The following rows explain the most consequential evidence without replacing those tables.

| Candidate or family | Actual evidence | Scoped consequence | What remains unresolved |
|---|---|---|---|
| C01/C02 operator pooling | Source priors, capacity screens, isolated physical MLP fixtures | No demonstrated recurrence-only advantage. The named remote-full-MLP fixture loses. | Fused pooling and complete source-owned attention/TP-FFN service cost |
| C03 terminal KV aliasing | Synthetic histories, source audit, bounded GPU address fixture | Direct prior and adaptive-only premise. No skipped fixed-R4 state to alias. | Complete live publication/reclamation and a separate fixed-mode saving |
| C04 distributed layouts | Positive algebra controls and nonlinear counterexamples | Naive unreduced-additive rewrites fail. Valid feature/statistic layouts remain possible. | Complete native numerical and physical cost qualification |
| C05 finite correction | Actual pretrained one-token hidden/KV diagnostics and optimistic cost models | The tested exact recovery offers no supported net advantage under its model. | A cheaper qualified correction, repair, and validation path |
| C06 fresh sparse delta | 2448 selected adjacent-pass row pairs, three layers, two prompts per model | Nearly all sampled coordinates change. The named exact-sparse premise is weak. | Population behavior, adaptive mode, actual projection kernels, and task quality |
| C07 verified branch fixture | Qualified admission followed by P1 bitwise failure | Zero accepted service trials. No architecture or task-quality failure follows. | First unequal operator boundary and complete concurrent-owner measurement |
| C08 exact-word transport | CPU bit reconstruction plus raw/transform components | Size reduction alone does not pay the tested unfused startup/transform costs. | A complete fused codec collective and exposed message |
| C09/C11/C17 certificates and structured work | CPU positive controls, counterexamples, and generic priors | No useful backend-qualified recurrent-only critical-path saving established. | Actual certificates, full native arithmetic, resource contention, and state publication |
| C12 historical-attention summaries | Rank constraints, selected keys, valid rational toy certificates | The conditional mechanism remains unqualified, not universally disproved. | Native FA3 envelope, useful checkpoint hits, full state and query costs |
| C13 gate overlap | Legal mandatory-R2 dependency contract and invalid timing observers | A legal overlap exists, but no valid removable-time percentage exists. | Clean exposed cost and same-policy overlap gain; separate fixed-mode benefit |
| C14 depth/chunk layout | Conditional physical-trace controls and a transition model | No recurrence-specific physical advantage established. Same DAG alone does not prove same arbitrary latency. | Complete measured prefill path and transition against tied-aware generic PP |
| C15 retained projection inputs | RoPE counterexample and charged storage models | The invalid static absorption is not exact. The valid variant loses its claimed saving in the model. | Position-aware representation with complete state and physical cost |
| C16 mixed-depth union | CPU frontiers and source-backed ownership contract | No additional ready-row width over the generic union in the tested graph. | Self-consistent online packets and full same-P cost/parity |
| C18 transition memoization | Selected actual MLP-input equality checks and generic CSE context | No useful whole-row identity hit in the bounded samples. | Full dynamic read-set equality, side effects, lookup cost, and lifetime |
| C19 lossless KV | Selected key coding and an uncached read model | Capacity reduction is not measured read or latency improvement. | Values, GPU decoder, retained bases, appends, and real admission boundary |

### Important numerical boundary

C07's selected P1 result differs in 1889/2048 hidden-state words.
Selected earlier-history and current-input checks pass.
The first unequal operator and its cause remain unresolved.
The campaign retains zero accepted timing trials and does not loosen the frozen criterion.

The native FA3 certificate audit also remains incomplete.
It lacks a useful range-qualified WGMMA error contract for the required QK and PV operations.
Its valid rational toy envelope cannot become a native attention certificate.
See [FA3 envelope obligations](native_fa3_tail_envelope_v1.md).

### Important timing boundary

All four CDB observer requests pass functional checks but fail the frozen timing-validity guards.
Their spans do not establish component shares or Amdahl percentages.
The failed collective-pair runs similarly remain NOT_ADMITTED after teardown stalls.
P4 of that new pair protocol is NOT_RUN.
See [observer validity](cdb_cost_trace_base14_adaptive_audit_v1.md) and [pair failures](hybrid_collective_pair_failure_v1.md).

## 7. What the cost models do and do not say

The C14 F1/4 interpolation is MODEL_ESTIMATE.
It grants an assumed faster prefill while keeping measured native decode unchanged.
At 128 outputs, its zero-transition request gains range from 4.13% to 10.14%.
Seven of eight modeled 10% transition budgets are negative.
The remaining budget is 2.138374 ms for Base 1.4B at N32768.
All modeled 15% transition budgets are negative.
None is a measured pipeline, a latency bound, or a rejection of all context/output regimes.

The depth-affinity screen is ANALYTIC_BOUND inside its declared unit-cost model.
It assumes complete shared weights on each worker and zero transfer, gate, head, publication, and contention cost.
Its ideal pool is not an implemented CDB baseline.
For the new adaptive traces, P4 pinned/pool saturated-capacity ratios range from 0.702280 to 0.764902.
This illustrates possible stranded service under that specific static-depth ownership rule.
It is not GPU throughput or a speedup against native TP4.

The C19 uncached common-base read ratios are 1.24077× and 1.15882× raw for the selected keys.
They exclude full values and a physical decoder.
They do not establish actual HBM traffic or a general KV-compression failure.

P8 remains PROJECTED_NOT_RUN.
The [P8 supplement](p8_projection_feasibility_supplement_v1.md) supplies conditional capacity rows and component equations.
It identifies missing candidate-level numerical latency calibration and matched TP8 costs.
Three scenario names alone do not satisfy those missing measurements.
The hypothetical two-island case is not the archived same-host topology.

## 8. Shortlist and next action

No candidate currently meets the required conjunction:

1. A distinct recurrence-specific execution or ownership mechanism.
2. A legal and fully costed state/dependency contract.
3. Credible incremental benefit over a competent same-P reference.
4. A valid numerical or matched task-quality path.
5. Supporting evidence across both sizes and both depth modes.

The campaign therefore should not start a large new engine from this evidence.
It should not rename a generic hybrid as Loop Parallelism.
It should not use prior dLLM communication percentages as Ouro speedup headroom.

The remaining conditional mechanisms have narrower prerequisites, not implementation-ready finalist status.
For C12, the cheapest prerequisite is a backend-qualified native numerical envelope and useful certificate-hit evidence.
For C13, it is a valid clean exposed-time observation under the unchanged gate arithmetic.
For C07, it is identifying the first unequal operator before any new distributed service trial.
These prerequisites are not recommendations to build three methods.

No full candidate quality or E2E confirmation runs because no primary architecture qualifies for that expense.
This preserves the original held-out rather than spending it on an undefined mechanism.

## 9. Limitations and completion obligations

- No new architecture LIVE_E2E or candidate free-running quality result exists.
- No actual HBM-counter measurement exists.
- CDB component attribution fails its observer guards.
- C07's numerical cause remains unresolved.
- Native/reference BF16 paths can diverge in natural generation.
- Controlled hardware cohorts use very few prompt patterns and sequential jobs.
- Quality pilots cannot prove the declared small noninferiority margins.
- Source counts include partial and metadata reads.
- Exact creative-effort fractions are not recoverable from the coarse coordinator ledger.
- P8 numerical critical-path projections remain incomplete.

The detailed evidence and failed attempts remain available.
No limitation becomes a fabricated speedup, a quality PASS, or a universal impossibility claim.
Final duration, resource release, current ledger publication, tests, and Git transport still require completion.

## 10. Artifact navigation

| Required view | Current artifact |
|---|---|
| Authoritative contract | [Frozen spec](spec_snapshot.md) |
| Prior project evidence | [Scope and historical evidence](scope_and_prior_evidence.md) |
| Search and primary claims | [Search map](search_map.md), [source audit](primary_claim_reaudit_v1.md) |
| Candidate Tables A/B/E/F | [Current overlay](current_tables_abef_overlay_v1.md) |
| Performance/quality Table C and cost Table D | [Current panels](../analysis/result_tables_cd_complete_v1/tables_cd.md) |
| Dependencies and ownership | [Exact frontier](exact_frontier_contract.md), [red team](dependency_red_team.md) |
| Both-size natural baseline extension | [Completed report](baseline_extension_complete_v1.md) |
| Raw outputs, commands, UUIDs, and source snapshots | `outputs/` and the per-run contracts |
| Resumable campaign state | [State](../research_state.json), [progress](progress_log.md), [actions](next_actions.md) |

Branch: `flashvep/ouro-parallelism-deep-discovery`.
The final transport commit is not assigned in this active draft.
