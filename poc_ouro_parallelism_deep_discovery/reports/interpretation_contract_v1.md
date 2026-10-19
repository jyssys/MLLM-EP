# Interpretation contract for the discovery dossier

## Purpose

This contract controls the final synthesis.
It preserves distinctions that can change a research decision.
It supplies no new model execution, measurement, architecture, or quality result.

## Computation and state

Shared parameter storage does not imply equal activations or equal KV.
The execution identity includes request, token position, physical layer, and recurrent depth.
The [Ouro contract](ouro_semantics_audit.md) separates these identities.

A mathematical DAG does not determine physical latency by itself.
Two ownership layouts can execute the same operators with different communication, cache behavior, batching, or contention.
The [negative-scope audit](negative_scope_audit_v1.md) retains this possibility.
It does not grant any unmeasured layout a gain.

An unchanged ready set does not prove unchanged service cost.
It also does not create an additional causal frontier.
The [attention/FFN refinement](dp_attention_tp4_ffn_refinement_v1.md) separates those two questions.

## Cache and memory

Logical KV capacity, physical KV allocation, and actual HBM traffic are different quantities.
The current adaptive path can skip work while retaining all depth slots.
The [capacity screen](interpass_cache_capacity_screen.md) grants favorable cache assumptions explicitly.
Its retained-byte bound is not a measured latency bound.

Native TP keeps its head shard's historical KV beside attention.
It does not move complete history between ranks at each layer.
A remote-KV proposal cannot remove a transfer that this baseline never performs.
See the [historical-KV screen](candidate_search_v9_certified_kv_prefetch.md).

The successful 32K native controls use a named cache reservation and workload.
They reject only the old premise that the local configured context cap defines an unavoidable capacity limit.
They do not establish every concurrency or larger-context operating point.
See the [native context report](native_long_context_observation_v1.md).

## Numerics and quality

Real-arithmetic equivalence does not establish native BF16 word identity.
Four matching top-1 positions do not establish natural-output parity.
The [quality audit](quality_and_exactness.md) retains actual cross-backend differences.

A frozen exact-word fixture failure remains a failure under that criterion.
It is not automatically a task-quality loss or an architecture impossibility result.
The [C07 audit](c07_p1_bitwise_failure_analysis_v1.md) records zero accepted service trials.
It does not loosen the criterion or identify an unobserved cause.

An original gate tensor does not guarantee an identical exit policy.
The arithmetic order, rounded hazards, accumulation, threshold, and decision horizon all matter.
The [gate contract](original_gate_overlap_contract.md) preserves the selected baseline's operations.
Newly trained lookahead gates remain outside this campaign.

A valid mathematical tail certificate still needs a native numerical envelope.
The [FA3 audit](native_fa3_tail_envelope_v1.md) does not set native error to zero.
Its missing bound is not proof that every certificate is impossible.

## Timing and statistics

A GPU microbenchmark times its stated operator and residency condition.
The C02 fixture uses three `F.linear` calls, separate SiLU/multiplication, and NCCL collectives.
It is not the complete fused native vLLM MLP path.
The P4 remote arm uses its two-rank peer group while the other ranks remain idle.
Its scoped isolated-source loss does not reject every pooled or fused implementation.
See [the measured code](../scripts/component_probe.py) and [cost report](cost_and_headroom.md).

Rank-local averages are not complete distributed stage latency.
The component fixture selects the maximum local duration for each complete operator trial.
It does not sum different ranks' independent component maxima.
This fixture does not establish a globally synchronized multi-layer critical path.

The CDB observer preserves functional outputs but fails its frozen timing guards.
Its event spans supply no valid removable-time share or Amdahl percentage.
See the [observer audit](cdb_cost_trace_base14_adaptive_audit_v1.md).

TP API wall and DP parent release-to-completion wall have different endpoints.
The [native hybrid report](native_hybrid_cohort_observation_v1.md) compares their descriptive medians without hiding that difference.
The result-table generator suppresses cross-endpoint speedup normalization.

Five repeated hardware trials are not five new natural requests.
Two frozen prompt patterns do not establish a population performance interval.
Natural-quality bootstrap samples use independent request IDs, not tokens or depth events.

## Evidence and novelty

A source result remains SOURCE_REPORTED.
Code support remains distinct from reproduced task quality and H100 performance.
The [primary-claim audit](primary_claim_reaudit_v1.md) records each selected source's inspected extent.
An absent public implementation in a bounded search does not prove absence.

The F1/4 prefill interpolation is MODEL_ESTIMATE, not a lower bound or measured PP stage.
The [C14 screen](c14_transpose_decision_screen_v1.md) changes no decode cost and charges a transition budget.
Its narrow negative budgets do not reject all legal pipelines or output lengths.

The 19 frozen card labels are bookkeeping, not 19 measured architecture failures.
No candidate has live-E2E or free-running-quality evidence at this checkpoint.
The [current synthesis](current_tables_abef_overlay_v1.md) separates initial hypotheses from demonstrated novelty.
Missing evidence cannot become a positive architecture result or a universal negative theorem.
