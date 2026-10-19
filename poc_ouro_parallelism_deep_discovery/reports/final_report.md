# Ouro parallelism deep discovery

## Publication status

The bounded research campaign is closed.
The [completion receipt](../artifacts/completion_receipt_v1.json) records36008.456322 coordinator seconds, or10.002349 hours.
Endpoint accounting excludes the explicit user pause and does not add concurrent worker durations.
Both native TP4 quality arms are complete and independently admitted.
The remaining specification shortfalls stay explicit in Section9 and the completion obligations.
Git transport receives a separate receipt after remote verification.

## Executive result

The current selection is **NO_QUALIFYING_CANDIDATE_IN_SEARCHED_SPACE**.
The shortlist contains zero new distributed architectures.
This conclusion concerns the searched transformations and retained evidence.
It is not a claim that no new parallelism can exist for Ouro.

The campaign registers 19 hypotheses across several execution and ownership families.
It tests mathematical dependencies, state ownership, numerical behavior, selected physical operators, and conventional baselines.
It also collects free-running baseline quality and actual original-gate decode traces for both Base model sizes.
No candidate has a LIVE_E2E result or a free-running task-quality result.
The original candidate held-out cohort remains untouched.

A finalist does not need an already completed LIVE_E2E experiment.
However, it needs a concrete distinct mechanism, a legal cost contract, and a credible numerical or quality path.
The remaining proposals do not establish that combination well enough to justify a new architecture implementation.
Some conditional mechanisms remain unresolved rather than disproved.

The frozen dispositions contain 12 GENERIC_ONLY, six NO_NET_HEADROOM, and one CONDITIONAL_MECHANISM labels.
These labels are bookkeeping, not 19 measured architecture failures.
The [negative-scope overlay](negative_scope_audit_v1.md) governs each interpretation.

## 1. Experiment contract and actual scope

The [frozen specification](spec_snapshot.md) matches the authoritative user document.
The campaign uses dense Base Ouro-1.4B and Base Ouro-2.6B.
It does not substitute a Thinking checkpoint, dLLM, Huginn, Raven, or looped MoE.
Training, new gates, reduced precision, and stale/delta hidden-state replacement remain excluded.

| Item | Actual scope |
|---|---|
| Precision | BF16 model execution |
| Fixed mode | R4 with full logical depth-indexed KV |
| Adaptive reference | Original gate, q=0.5, minimum R2, maximum R4, `last_exited` |
| Adaptive prefill | Full R4 |
| Authorized physical GPUs | H100 GPUs 4, 5, 6, and 7 only |
| Physical parallel controls | P2 and P4 subsets, with per-run mappings |
| Native distributed control mode | Fixed R4 |
| Original adaptive execution | Single-GPU CDB reference jobs |
| P8 | PROJECTED_NOT_RUN, with incomplete candidate-level latency projections |
| New engine | None |
| Protected factorial experiment | Preserved and NOT_EVALUATED |

The distributed controls do not establish original-gate adaptive TP2/TP4 or distributed adaptive DP performance.
Existing adaptive-versus-fixed quality differences remain operating-point observations.
They are not losses or gains caused by a new mechanism.

### Loaded architecture

| Base model | Checkpoint revision | Physical blocks | R4 logical block executions | Logical depth/layer KV slots | Hidden / FFN width |
|---|---|---:|---:|---:|---|
| 1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | 24 | 96 | 96 | 2048 / 5632 |
| 2.6B | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | 48 | 192 | 192 | 2048 / 5632 |

Both configs use 16 Q heads, 16 KV heads, and head dimension 128.
The inspected maximum position is 65536, with RoPE theta 1000000 and no sliding window.
Configured support does not establish every allocation or workload at that length.
The native context controls reach 32768 prompt tokens plus 128 controlled outputs.

Full-R4 logical KV requires 786432 bytes/token for Base 1.4B.
It requires 1572864 bytes/token for Base 2.6B.
These are capacity quantities, not measured HBM reads or universal physical-storage lower bounds.
A cache pool can store distinct logical versions without separate allocator objects.
Adaptive work skipping does not automatically remove allocated depth slots.

See [model semantics](ouro_semantics_audit.md), [semantics manifest](../artifacts/semantics_manifest.json), and [environment manifest](../artifacts/environment_manifest.json).

### Physical ownership and safety

| Method or fixture | Physical mapping | Visible logical mapping |
|---|---|---|
| Native TP1 | GPU4 | Rank0 |
| Standalone native TP2, including context controls | GPUs4/6 | Ranks0/1 |
| Native TP4 | GPUs4/5/6/7 | Ranks0/1/2/3 |
| TP2×DP2 | Groups [4,5] and [6,7] | Ranks0/1 inside each group |
| DP4 | Four independent singleton children | Visible CUDA device0 in each child |
| Older isolated P2 MLP | GPUs4/6 | Ranks0/1 |
| New failed collective-pair P2 fixture | GPUs4/5 | Ranks0/1 |

Ordered per-run receipts remain authoritative.
A four-device reservation does not always mean a four-rank TP world.
The authorized devices are H100 80 GB HBM3 GPUs.
The archived topology records NV18 connectivity between them.
The campaign changes no power or clock settings.
It stops only verified owned utilization workers during guarded measurements.
Foreign jobs and the protected experiment remain untouched.
Other GPUs can remain busy.
The guard enforces authorized-context ownership, not complete host or fabric exclusivity.

## 2. What recurrence does and does not make ready

The inspected current-token graph retains the recurrent hidden-state chain.
Pass r+1 needs the actual normalized result of pass r.
Its history also needs the matching request, depth, layer, position, and committed version.
Shared weights alone supply neither input.

Independent requests remain legal ready work.
Known causal prompt chunks can also supply a prefill frontier.
Ordinary operator forks remain possible inside the opaque core.
Thus, the B1 decode chain does not erase prefill concurrency or every exact reformulation.

The [editable dependency figure](../figures/final_dependency_summary.mmd) shows the fixed-R4 chain and depth-indexed history.
It does not imply measured overlap or a communication lower bound.
The [exact frontier contract](exact_frontier_contract.md) and [red-team analysis](dependency_red_team.md) record the fuller state obligations.
The [composition control](function_composition_screen_v1.md) separates a two-level expression construction from four ordered nonlinear state evaluations.
Its exact affine positive control does not qualify unchanged-Ouro composition or BF16 reassociation.

## 3. Search coverage and source claims

The initial register contains 14 cards across seeded areas A–G.
The final register contains 19 cards.
C09, C10, and C11 document agent-originated alternatives outside seeded areas A–F.
C05–C08 contain four initial execution-axis hypotheses.
These initial categories are hypotheses, not demonstrated novelty.

The versioned ledger records source observations, search queries, inspected extents, and experiment status.
Its counts include partial reads, metadata, code, and repeated versions.
They are not complete-paper read counts.
The published [current index](../artifacts/ledger_current.json) identifies the versioned source/evidence snapshot.

The source audits correct several consequential inherited claims:

- CDB computes model argmax feedback under controlled benchmark lengths and recorded exits.
- CDB Appendix B includes a Base 2.6B recurrent-step experiment.
- KernelFlume includes real A6000 batch-2-to-64 evidence, not this campaign's H100 C1 result.
- SNLP labels its inspected verified-drafter E2E values theoretical.
- FlashLoop's BF16 option removes quantization, not its sparsity approximations.
- Public RLT supplies another original-gate runtime. CDB is not the only public implementation.
- RLT's inspected distributed path uses full-model P/D replicas, not validated adaptive TP2/TP4 here.
- CacheFlow supplies historical-restoration and cached shard-boundary inputs, not uncomputed current-token hidden states.
- Static CSE proves identical IR operations, not complete dynamic tensor, KV, side-effect, and lifetime equality.

These are SOURCE_REPORTED facts or explicitly labeled transfer deductions.
They fill no campaign candidate-performance cell.
See [the primary-claim re-audit](primary_claim_reaudit_v1.md), [RLT overlay](public_adaptive_runtime_scope_overlay_v1.md), and [CacheFlow/CSE supplement](cacheflow_cse_source_supplement_v1.md).
The [camera-ready sampler supplement](recurrent_sampler_camera_ready_supplement_v1.md) resolves its earlier PDF-reader access limit.
Its final body does not supply unchanged-Ouro numerical or distributed-performance qualification.

The coarse coordinator categories cannot certify the exact fraction of effort spent on new axes or unseeded generation.
The dossier does not reconstruct unsupported retrospective percentages.
Read extent and implementation availability remain source-specific limitations.

## 4. Physical conventional baselines

### Constant-token cohort

The completed audit admits 400 clean trials across 80 cells and ten jobs.
The methods are native TP1/TP2/TP4, DP4, and TP2×DP2 for both Base sizes.
These controls use fixed R4.
Each cell has five trials after short warmups.
The cohort uses only two repeated prompt patterns.
It does not contain 400 independent natural reasoning requests.

Each request emits 128 copies of frozen token378.
EOS does not shorten this hardware-control workload.
The scripts check prompt IDs, output IDs, finish metadata, source identity, and timing denominators.
This lane measures controlled inference, not natural-generation quality.

The following C1/N1024 medians share the native generate-API timing boundary.
Speedup divides the TP1 median by the corresponding TP4 median.
The sequential jobs do not establish randomized paired architecture confidence intervals.

| Base | TP1 API wall | TP2 API wall | TP4 API wall | TP1/TP4 speedup |
|---|---:|---:|---:|---:|
| 1.4B | 1089.762 ms | 929.606 ms | 813.429 ms | 1.340× |
| 2.6B | 2062.765 ms | 1826.399 ms | 1628.599 ms | 1.267× |

DP4 and TP2×DP2 instead use parent release-to-all-completed wall time.
That endpoint includes controller and receipt work.
It is not interchangeable with the native API endpoint.
The analysis therefore suppresses cross-endpoint speedup normalization.
It assigns no common-endpoint winner from the raw median ranking.

Every native rank reserves 52 GiB KV in this protocol.
The four-GPU arms therefore reserve 208 GiB aggregate KV.
Weight multiplicity and KV ownership differ across methods.
The separate long-context protocol instead reserves 52 GiB aggregate KV per TP group.
Do not merge their absolute timings as one experiment.

See [the conventional-control report](native_hybrid_cohort_observation_v1.md) and [current Tables C/D](../analysis/result_tables_cd_complete_v2/tables_cd.md).
The earlier report's 14/16 raw-median count uses different timer endpoints.
That count is not a common-endpoint method ranking.

### Long-context controls

The separate audit admits 144 trials across 24 cells.
It measures both Base sizes, TP1/TP2/TP4, and prefixes through 32768 tokens.
At 32768 tokens, mean paired TP1/TP4 request-wall ratios are 2.1883× and 2.2098× for Base 1.4B and Base 2.6B.
Only two frozen prompt patterns support those ratios.
They do not establish a population interval or an attention-only speedup.

These controls exceed the earlier local 10240-token runtime cap.
That cap was not the checkpoint context limit.
The controls do not reject every larger-context or higher-concurrency capacity opportunity.
See [the complete context audit](native_long_context_observation_v1.md).
The editable [Base1.4B plot](../figures/controlled_context_latency_v1/base14.svg) and [Base2.6B plot](../figures/controlled_context_latency_v1/base26.svg) show those admitted medians.
They use a categorical context axis and no population interval.

## 5. Free-running quality and actual adaptive execution

### Completed staged extension

All four staged arms complete 64 requests.
The integrity validator admits all 256 records.
Each arm uses the same 32 GSM8K and 32 MATH-500 IDs.
There are 64 distinct problem IDs, not 256 independent problems.
These new extension IDs exclude the original calibration, original held-out, prior pilot, and protected-factorial metadata IDs.
They are additional baseline observations, not candidate confirmation.

| Base | Mode | GSM8K correct/N | MATH-500 correct/N | MATH length-capped/N |
|---|---|---:|---:|---:|
| 1.4B | Fixed R4 | 24/32, 75.000% | 17/32, 53.125% | 4/32 |
| 1.4B | Original adaptive | 24/32, 75.000% | 21/32, 65.625% | 4/32 |
| 2.6B | Fixed R4 | 28/32, 87.500% | 25/32, 78.125% | 5/32 |
| 2.6B | Original adaptive | 28/32, 87.500% | 24/32, 75.000% | 6/32 |

The [completed quality report](baseline_extension_complete_v1.md) retains request-bootstrap 95% intervals, parser failures, and paired differences.
The [output comparison](../analysis/baseline_output_comparison_complete_v1/summary.json) retains raw tokens, first divergence, and finish reasons.
Length-capped responses remain incomplete under the frozen grading rule.
These 32-request domains cannot establish 1 pp or 2 pp population noninferiority margins.
They do not reproduce the Thinking model's official reasoning protocol.

CDB runtime counters match recorded adaptive decode events for every admitted request.
Token-weighted mean decode depth ranges from 2.809121 to 3.059609 across the four cells.
This accounting excludes full-R4 prefill and its first output token.
The reference physically skips some core passes but retains depth-indexed KV capacity.
Core-pass counts are not total FLOPs, actual HBM traffic, or latency reductions.

The [descriptive depth profile](baseline_depth_profile_v1.md) adds request-weighted distributions and relative-position JSD.
Its quarter boundaries use completed trace length.
They are descriptive future information, not causal scheduling inputs.
No new predictor or DP placement policy follows from that supplement.

### Native TP4 baseline extension

The cache-isolated Base 1.4B arm completes 64 requests and passes integrity admission.
It scores 25/32 GSM8K, 78.125%, with bootstrap 95% CI [62.500, 90.625]%.
It scores 16/32 MATH-500, 50.000%, with CI [34.375, 65.625]%.
Parsing fails for one request per domain.
MATH has six length-capped responses, while GSM8K has none.

Native-minus-staged fixed accuracy is +3.125 pp on GSM8K, with paired CI [0.000, 9.375] pp.
It is -3.125 pp on MATH, with paired CI [-15.625, 6.250] pp.
Complete raw output IDs match staged fixed for 20/32 GSM8K and 4/32 MATH requests.
Equal intended mathematics therefore does not establish complete BF16 natural-output parity.

The Base 2.6B arm also completes 64 requests and passes integrity admission.
It scores 28/32 GSM8K, 87.500%, with bootstrap 95% CI [75.000, 96.875]%.
It scores 25/32 MATH-500, 78.125%, with CI [62.500, 90.625]%.
Its native-minus-staged fixed differences are 0.000 pp in both domains.
Their paired intervals are [0.000, 0.000] pp for GSM8K and [-9.375, 9.375] pp for MATH.
The degenerate GSM8K interval does not prove population equivalence.
Complete output IDs match for 25/32 GSM8K and 0/32 MATH requests.
Its MATH arm has two parsing failures and six length-capped responses.
GSM8K has neither.

The [complete native extension](native_quality_extension_complete_v1.md) retains the paired counts and raw-output examples.
Its [admitted analysis](../analysis/native_quality_extension_complete_v1/summary.json) is QUALITY_ONLY.
The fresh-cache v2 protocol uses the same extension IDs, not fresh held-out requests.
Native string-substring stopping and staged token-suffix stopping remain different backend mechanisms.
No cross-backend score or natural wall-time difference becomes a candidate's quality or speedup effect.
The [raw divergence diagnostic](backend_divergence_diagnostic_v1.md) finds first-token agreement for all128 pairs.
Of79 non-exact pairs,78 first differ before both terminal positions.
One pair first differs at native EOS, which requires a separate termination qualifier.
The first unequal operator and causal factor remain unresolved.

Both native outer guards complete successfully after internal worker-shutdown warnings.
The logs retain forced worker cleanup and shared-memory warnings.
Base 2.6B also records TCPStore/NCCL monitor warnings during shutdown.
This supports admitted completed outputs and qualified outer cleanup, not a perfectly graceful internal shutdown.
It is not a clean E2E timing result.

## 6. Decisive candidate screens

The [current Tables A/B/E/F](current_tables_abef_overlay_v1.md) contain every registered ID.
The [negative-scope overlay](negative_scope_audit_v1.md) distinguishes priors, measured component losses, structural facts, and missing evidence.
The rows below summarize the most consequential limits.
They do not replace the full per-ID records.

| Candidate or family | Actual evidence | Scoped consequence | Remaining prerequisite |
|---|---|---|---|
| C01/C02 pooled operators | Source priors, capacity screens, physical isolated MLP fixtures | Generic pooling already exists. The named remote-full-MLP fixture loses. | Complete fused owner/layout service with legal ready rows and all costs |
| C03 terminal aliases | Synthetic histories, source audit, bounded address fixture | Direct prior and adaptive-only premise. Fixed R4 has no terminal-copy saving. | Live publication/reclamation and a separate fixed-mode benefit |
| C04 distributed layouts | Positive algebra controls and nonlinear counterexamples | Specific unreduced-additive rewrites fail, not every valid layout. | Native numerical contract and complete exposed-cost comparison |
| C05 finite correction | Actual one-token hidden/KV diagnostics and optimistic models | Tested recovery offers no supported net gain under its model. | A cheaper qualified correction, verification, and finalization path |
| C06 fresh sparse delta | 2448 adjacent-pass records across selected layers and prompts | Nearly all sampled coordinates change. The named exact-sparse premise is weak. | Adaptive decode, uncaptured down inputs, and actual projection parity/cost |
| C07 verified branches | Qualified admission followed by failed P1 bitwise qualification | Zero accepted service trials. No architecture or task-quality failure follows. | Identify the first unequal operator before another service comparison |
| C08 exact-word transport | CPU reconstruction plus separate raw/transform timings | Size savings alone do not establish net savings at tested small-message costs. | Complete codec path, receiver-base lifetime, and exposed-message measurement |
| C09/C11/C17 certificates and structured work | Positive controls, counterexamples, and close generic priors | No qualified recurrent-only critical-path saving is established. | Native certificates/arithmetic and complete resource/state accounting |
| C10 historical restoration | Primary-source mechanisms and ownership obligations | Known saved history can overlap restoration, but does not create future current inputs. | Concrete useful restoration cost and full checkpoint/publication contract |
| C12 attention summaries | Rank constraints, selected keys, valid rational certificates | Conditional mechanisms remain unqualified, not universally disproved. | Native error envelope, useful checkpoint hits, and full query/state cost |
| C13 original-gate overlap | Legal mandatory-R2 contract and invalid timing observers | Legal generic baseline opportunity, with no valid removable-time percentage. | Clean exposed-time measurement under unchanged gate arithmetic |
| C14 depth/chunk layouts | Conditional physical-trace control and first-token transition model | No recurrence-specific physical advantage is established. | Complete native prefill/transition comparison against tied-aware generic PP |
| C15 retained projection inputs | RoPE counterexample and charged storage models | Invalid static absorption fails. A valid variant loses its claimed modeled saving. | Position-aware representation and full physical/numerical qualification |
| C16 mixed-depth union | CPU frontiers and source-backed ownership contract | No extra ready-row width over generic union in the tested graph. | Self-consistent online packets, future readiness, and full same-P service cost |
| C18 transition memoization | Selected actual MLP-input equality checks and CSE context | No useful whole-row identity hit appears in bounded samples. | Complete dynamic read-set proof, side effects, lookup, and lifetime costs |
| C19 lossless KV | Selected key coding and uncached common-base read model | Capacity reduction is not measured HBM or latency improvement. | Full K+V, GPU decoder, bases, append cost, and actual admission boundary |

C06 samples two continuity prompts and three layers per Base size.
Its fixed/adaptive full-R4 prefill copies are bitwise identical, not independent replication.
The captured operand is width2048, not the nonlinear width5632 down-projection input.
Its result does not measure a complete sparse projection kernel.

C07 differs in 1889/2048 selected hidden words.
Selected earlier-history and current-input checks pass.
The first unequal operator and its cause remain unresolved.
The campaign retains zero accepted service timings and does not loosen its frozen criterion.

The C12 [FA3 audit](native_fa3_tail_envelope_v1.md) lacks a useful range-qualified native WGMMA envelope for QK and PV.
Its rational toy certificate cannot become a native attention certificate.
The public instruction contract does not supply the assumed sequential-FMA guarantee.
This missing contract does not prove all exact selective-attention algorithms impossible.

## 7. Timing validity and modeled headroom

### Invalid attribution remains invalid

All four CDB observer requests pass functional checks but fail their frozen timing-validity guards.
Their spans do not establish component shares or Amdahl percentages.
The two new collective-pair runs retain complete diagnostic cell outputs but stall during teardown.
They remain NOT_ADMITTED.
Their P4 extension is NOT_RUN under the declared stopping rule.

The first native quality launch also remains NOT_ADMITTED.
It stopped for a cache-provenance precaution after two partial records.
That action does not establish a wrong-output or shared-cache defect.
The separate v2 jobs use fresh job-local caches and frozen source receipts.
All failures remain preserved.

See [observer validity](cdb_cost_trace_base14_adaptive_audit_v1.md), [pair failures](hybrid_collective_pair_failure_v1.md), and [the native stop record](../artifacts/native_extension_v1_pre_admission_stop.json).
No actual HBM-counter measurement exists.
Dense native TP also has no MoE EP Dispatch/Combine All-to-All to attribute.

The C02 operator uses three `F.linear` calls, separate SiLU/multiplication, and NCCL communication.
It is not the complete fused native vLLM MLP path.
Its P4 remote arm exchanges work only between ranks0/1 while the other ranks remain idle.
Each trial uses the maximum rank-local duration of the whole component.
It does not establish a globally aligned multi-layer critical path.

The C05 finite-correction model uses its archived changing-teacher native control.
It does not use the newer constant-token matrix.
Equal output counts do not make those protocols interchangeable.

### C14 conditional request model

Let W4 denote measured TP4 API wall.
Let F1 and F4 denote complete frontend first-token medians, not isolated prefill compute.
The candidate-favoring model substitutes F1/4 for F4:

```text
M0 = W4 - F4 + F1/4
modeled API-wall reduction = 100 * (F4 - F1/4) / W4
B_q = (1 - q) * W4 - M0
```

Here q denotes the requested fraction of API-wall reduction.
B_q is the modeled budget for every additional exposed phase cost.
It is not a KV-transfer-only allowance.
At 128 outputs, this zero-transition reduction ranges from 4.13% to 10.14%.
Seven of eight modeled 10% transition budgets are negative.
The remaining budget is 2.138374 ms for Base 1.4B at N32768.
All modeled 15% budgets are negative.
These are MODEL_ESTIMATE results, not measured pipelines, latency bounds, or rejection of every workload regime.

### Depth-affinity and coding screens

The depth-affinity screen is an ANALYTIC_BOUND inside its stated unit-cost ownership model.
It assumes full shared weights on each worker and zero transfer, gate, head, publication, and contention costs.
It chooses a static partition in hindsight from the complete observed depth histogram.
Its ideal pool is not an implemented CDB baseline or a causal scheduler.

For the new traces, P4 pinned/pool saturated-capacity ratios range from 0.702280 to 0.764902.
These are not measured GPU throughput, native TP4 speedup, or physical latency bounds.
See [the complete extension screen](../analysis/depth_affinity_extension_complete_v1/screen.json).

C19's selected-key uncached common-base read ratios are 1.24077× and 1.15882× raw.
They exclude full values and a physical decoder.
They do not establish actual HBM traffic or general KV-compression failure.
C08 similarly never times a complete codec serving path.

The [terminal-alias payload screen](adaptive_alias_capacity_v1.md) retains a positive adaptive capacity possibility.
Mean modeled committed-payload savings range from5.478% to25.336% across the four extension cells.
These are not allocated HBM, traffic, admission, or latency measurements.
They add no new novelty or fixed-R4 saving.
Whole-page-only alias models reduce mean savings to1.472–6.599% at page16 across those cells.
The inspected default and modeled page sensitivities are not an executed alias allocator.

### P8 obligations remain incomplete

P8 remains PROJECTED_NOT_RUN.
The [P8 supplement](p8_projection_feasibility_supplement_v1.md) adds capacity rows and conditional communication equations.
The required candidate-level optimistic, central, and conservative critical-path projections remain incomplete.
Their unknown terms include the matched native primitive, ownership schedule, interference, and publication cost.
Scenario names do not satisfy that missing calibration.

The archived host shows all-NV18 pair connectivity, but only GPUs4–7 participate in experiments.
The hypothetical two-island scenario is not that archived topology.
No advertised bandwidth or P4 primitive becomes a measured P8 result.
The absence of a qualifying candidate does not convert this projection shortfall into candidate failure.

### Final verification scope

The final isolated suite records44 modules and376 CPU tests with stable campaign Python sources.
Those tests do not execute a candidate or certify native numerical parity.
The versioned [source index](../analysis/ledger_final_v3/index.json) records275 observations,238 work/document keys,88 arXiv keys,193 search records, and136 experiment directories.
These counts include selected reads and failed runs.
They are not full-paper or successful-experiment counts.
The [late review receipt](../artifacts/late_auxiliary_review_v1.json) retains the EOS correction and auxiliary model limits.

## 8. Finalist selection and next action

Zero proposals meet the current conjunction:

1. A concrete distinct recurrence-specific execution or ownership mechanism.
2. A legal state/dependency contract with credible complete cost accounting.
3. Incremental benefit over a competent matched same-P reference.
4. A defensible numerical or task-quality path.
5. Supporting scope across both sizes and both depth modes, or a clearly limited exception.

This is a research-selection judgment under the retained evidence.
It is not a universal impossibility result or a requirement for every speculative idea to have LIVE_E2E first.
The campaign should not build a large engine or rename a generic hybrid as Loop Parallelism from these results.
The [independent qualification audit](qualification_audit_final_v1.md) checks the non-E2E prerequisite missing from each card.

The most informative remaining prerequisites are narrow:

- C12 needs a backend-qualified numerical envelope and useful certificate-hit evidence before a distributed kernel.
- C13 needs valid clean exposed-time attribution before a same-policy gate-overlap comparison.
- C07 needs its first unequal operator identified before any new distributed service trial.

These are diagnostic prerequisites, not three selected architectures or three implementation recommendations.
No candidate quality confirmation runs because no primary architecture qualifies for that expense.
The held-out cohort therefore remains preserved.

## 9. Limitations

- No candidate LIVE_E2E or free-running quality result exists.
- Both-model adaptive evidence uses the CDB reference, not native adaptive TP2/TP4.
- Invalid component observers cannot identify removable E2E shares.
- C07's numerical cause remains unresolved.
- Native/reference BF16 paths can diverge during natural generation.
- Controlled cohorts use few prompt patterns and sequential method order.
- Shared-host activity and clock variation are not a fully controlled experimental factor.
- Quality samples do not establish the small declared noninferiority margins.
- No actual HBM-counter result exists.
- Source counts include metadata and selected passages.
- Exact creative-effort fractions are not recoverable from the coarse time ledger.
- Candidate-level P8 critical-path projections remain incomplete.
- Tied-aware native cyclic PP and same-P established self-speculation are not measured serving controls here.

Stock native Ouro lacks the inspected PP support contract.
The pinned LoopSpec server rejects parallel sizes above one.
These implementation boundaries are not measured performance failures or proofs against distributed extensions.
No candidate receives an advantage from their absence.

These limits constrain the conclusion.
They do not become fabricated speedups, quality PASS claims, or global no-reuse theorems.

## 10. Reproduction and artifact map

| Required view | Artifact |
|---|---|
| Contract and previous evidence | [Frozen spec](spec_snapshot.md), [historical evidence](scope_and_prior_evidence.md) |
| Current candidate Tables A/B/E/F | [Scope-preserving overlay](current_tables_abef_overlay_v1.md) |
| Performance/quality Table C and cost Table D | [Separated panels](../analysis/result_tables_cd_complete_v2/tables_cd.md) |
| Dependencies and state ownership | [Exact frontier](exact_frontier_contract.md), [ownership supplement](contract_ownership_dag_supplement_v1.md) |
| Primary claims and source limits | [Re-audit](primary_claim_reaudit_v1.md), [nearest-work audit](literature_nearest_audit.md) |
| Positive structural limit on negative claims | [Adaptive terminal-alias payload model](adaptive_alias_capacity_v1.md) |
| Both-size natural baseline quality | [Completed staged extension](baseline_extension_complete_v1.md) |
| Native quality and raw-output limits | [Completed native extension](native_quality_extension_complete_v1.md), [divergence diagnostic](backend_divergence_diagnostic_v1.md) |
| Editable overview figures | [Dependency graph](../figures/final_dependency_summary.mmd), [evidence map](../figures/final_evidence_map.mmd) |
| Raw commands, UUIDs, failures, and outputs | Per-run `outputs/` contracts and receipts |
| Reproduction | [Final command index](reproduce_final_v1.md) |
| Completion limits and Korean summary | [Obligations](completion_obligations_v1.md), [Korean synthesis](final_summary_ko.md) |
| Actual released-device state | [Owned burner receipt](../artifacts/final_gpu_burner_receipt_v1.json) |
| Resumable state and actual time | [State](../research_state.json), [time ledger](../artifacts/time_ledger.jsonl), [progress](progress_log.md), [actions](next_actions.md) |

Branch: `flashvep/ouro-parallelism-deep-discovery`.
The publication receipt identifies the transported payload commit after remote verification.
