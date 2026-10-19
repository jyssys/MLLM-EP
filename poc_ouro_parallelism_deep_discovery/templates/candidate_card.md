# CXX: Descriptive mechanism title

Use explicit unknowns. A populated field does not establish correctness, physical overlap, quality, or novelty.

## 1. Identity

Record the ID, title, search family, revision, and initial category.

## 2. Mechanism

State one concrete execution transformation in one sentence. Avoid a marketing name.

## 3. Relevant Ouro property

State the required model property. Link its primary source or measured evidence and exact locator.
Distinguish shared weights from equal inputs or reusable state.

## 4. Target scope

Record both Base sizes, fixed/adaptive modes, prefill/decode stage, system-wide C, context, and P.
Separate targets from supported scope. Mark P8 projected or not run.

## 5. Baseline and exposed cost

Name the exact native and strongest applicable same-P baselines.
State the exposed cost that the mechanism could remove.
Match weights, precision, gate, KV, output policy, kernels, and admission budgets.

## 6. Operation sequence

Write the AS-IS and TO-BE sequences. Retain every necessary normalization, residual, state write, and publication guard.

## 7. Legal DAG and ownership

Link an editable DAG. Give each node its actual earliest inputs and predecessors.
Identify request, token/chunk, depth, layer/operator, device, layout, and state version.

| State | Creation point | Owner | Readers | Mutable? | Version/lifetime | Bytes | Transfer or recompute cost |
|---|---|---|---|---|---|---|---|
| Weight shard/tile | UNKNOWN | UNKNOWN | UNKNOWN | No | Model revision | UNKNOWN | UNKNOWN |
| Historical KV | UNKNOWN | UNKNOWN | UNKNOWN | Only before commit | Request/depth/layer/position | UNKNOWN | UNKNOWN |
| Current-token KV | UNKNOWN | UNKNOWN | UNKNOWN | Until commit | Candidate/version | UNKNOWN | UNKNOWN |
| Hidden activation | UNKNOWN | UNKNOWN | UNKNOWN | No after publication | Node/version | UNKNOWN | UNKNOWN |
| Gate/logit state | UNKNOWN | UNKNOWN | UNKNOWN | Declare | Request/token/depth | UNKNOWN | UNKNOWN |
| Candidate scratch | UNKNOWN | UNKNOWN | UNKNOWN | Yes | Discard/commit | UNKNOWN | UNKNOWN |

## 8. Source of concurrency

Name the useful concurrent units. Separate already-ready inputs from predicted or speculative inputs.
State which dependencies remain serial. Do not insert a global barrier into independent DP replicas.

## 9. Exactness and quality

Declare exact graph, numerical reordering, validated speculation, or approximate inference.
For speculation, distinguish greedy output from a preserved sampling distribution.
Record numerical, token, gate, KV, and mode-matched free-running quality tests.
Keep unavailable results explicit. Predeclare approximate-method margins before final evaluation.

## 10. State and memory

Count unique and replicated weights, activation state, historical/current KV, scratch, graph pools, and communication buffers.
Separate logical values, physical allocation, aliases, measured traffic, and estimated traffic.
State capacity effects on admission and context length.

## 11. Messages and topology

| Message | Producer | Consumer | Payload bytes | Required predecessor | Completion/publication rule |
|---|---|---|---:|---|---|
| UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

Record actual topology and devices for measured rows. Give future P8 assumptions separately.
Charge both transfer directions, metadata, staging, synchronization, and any moved or recomputed history.

## 12. Critical path and added costs

Write a resource-constrained cost model and its break-even inequality.
Include setup, compute, contention, communication, dispatch, validation, correction, rollback, commit, and fill/drain.
Do not sum overlapping durations or replace a serial sum with an unsupported maximum.
State assumptions, calibration error, and every unmeasured cost.

## 13. Nearest prior works

Name the nearest looped-model and generic mechanisms. Link pinned primary sources and inspected sections or functions.
Explain the substantive difference. Unavailable public multi-GPU code is not novelty.

## 14. Relation to prior project evidence

Link each relevant earlier negative and preserve its scope.
For a reopening, state the changed assumption, new fact, cheapest falsifier, tuning distinction, and abandonment budget.

## 15. Smallest falsification probe

Specify the smallest decisive probe, required data, and stop criterion.
State which result would reject the current mechanism and which wider questions would remain open.

## 16. Results and uncertainty

| Result | Evidence class | Model/mode/workload | Raw path | Metric and denominator | Uncertainty | Limitation |
|---|---|---|---|---|---|---|
| NOT_RUN | NOT_RUN | UNKNOWN | NONE | UNKNOWN | UNKNOWN | No result follows from this template |

Use the specification's evidence classes. Separate source results, structural accounting, models, components, and live requests.
Keep raw latency beside any speedup. State the independent resampling unit.

## 17. Both-size/both-mode/P compatibility matrix

| Scope | Compatibility | Physical or modeled evidence | Unresolved condition |
|---|---|---|---|
| Base1.4B fixed R4 | UNKNOWN | NOT_RUN | UNKNOWN |
| Base1.4B original adaptive | UNKNOWN | NOT_RUN | UNKNOWN |
| Base2.6B fixed R4 | UNKNOWN | NOT_RUN | UNKNOWN |
| Base2.6B original adaptive | UNKNOWN | NOT_RUN | UNKNOWN |
| P2 | UNKNOWN | NOT_RUN | UNKNOWN |
| P4 | UNKNOWN | NOT_RUN | UNKNOWN |
| P8 | Future target only | NOT_RUN | Topology and authorization |

Distinguish mode-specific or size-specific benefit from the full target.

## 18. Loop-specificity counterfactual

Select applicable R1, aliased unroll, physical-copy, sharing-disabled, and ownership-matched generic controls.
Give the generic control identical ready inputs, aliases, layout, and schedule freedom where applicable.
State what would remain without recurrence. Physical-copy memory changes are separate from traffic evidence.

## 19. Implementation complexity and next PoC

Identify the smallest useful implementation, files/functions, required infrastructure, and likely failure mode.
Give an estimate only with a stated basis. Otherwise record `ESTIMATE_UNAVAILABLE`.
Do not build an engine when a cheaper falsifier already rejects the premise.

## 20. Verdict and next action

Record a specification-defined scoped verdict, unresolved risks, and one most informative next action.
Do not infer a campaign verdict or finalist from this card alone.
Do not claim a universal impossibility from a bounded negative.
