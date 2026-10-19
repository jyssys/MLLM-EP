# Dependency and ownership red-team checkpoint

The tested schedules preserve distinct request, token, recurrent depth, physical layer, and state-version identities.
Weight identity does not imply input readiness.
Private scratch cannot update committed history before validation.
These rules apply to both Base model sizes.

## Screens with executable or explicit state checks

| Candidate | Available inputs | Required publication | Decisive dependency or cost | Evidence |
|---|---|---|---|---|
| C02 remote MLP | Completed normalized MLP input, immutable weights | Complete returned output before the next norm | Both activation directions remain serial at C1 | Physical P2/P4 operator controls |
| C04 distributed layout | Correct projection partials or disjoint features | Global values/statistics before nonlinear consumers | Partial softmax, SiLU, and RMSNorm cannot simply consume unreduced additive terms | CPU counterexamples and positive algebra controls |
| C05 finite correction | Causal initial guess and matching historical depth KV | Matching hidden and all current-token KV versions | Top-1 agreement alone does not validate internal state | Actual pretrained one-token emulation |
| C07 verified branches | Known draft token, its actual ancestor-state versions | Verified winner and every required cache copy | Future-depth tentative ancestor KV can require transfer before final winner selection | Explicit branch ownership/message ledger |
| C13 mandatory-prefix overlap | Actual h1 and original rounded hazard | R1/R2 hazards before the R2 exit decision | Mandatory R2 need not wait for R1 host consumption | 343 CPU policy cases and source event audit |
| C14 depth/chunk control | Predecessor hidden and causal same-depth KV | Completed writes and loop boundary norm | The tested graph is an aliased generic causal unroll | Eight DAG tests and structural matrix |
| C15 projection-input state | Exact normalized historical inputs and projection weights | Position-correct RoPE scores and head-specific value summaries | Static query absorption violates historical-position dependence | Exact/float64 controls and BF16 rounding counterexamples |
| C16 mixed-depth union | One legal current row per request | Per-row depth/request KV and original gate policy | Different depths do not create R times as many ready rows | Address/frontier positive and negative CPU controls |

This table does not certify a complete distributed implementation for every row.
It identifies the scope of the actual check and its remaining obligations.

## C1 exact frontier

One fixed-R4 decode token has a sequential recurrent boundary chain.
Each next pass consumes the previous pass's normalized output.
Each next token consumes the preceding sampled token.
The inspected graph therefore has no independent complete future pass at C1.

The graph still contains ordinary projection forks and feature/head parallelism.
TP or other generic operator parallelism can use those forks.
This observation is not a lower bound for every speculative, numerical, or approximate algorithm.

## Ready rows at higher concurrency

At system concurrency C, independent requests can supply up to C current non-speculative decode rows.
They can be at different recurrent depths.
Those rows retain separate depth-indexed historical KV and gate state.
A runtime may batch their shared dense operations when the inputs are ready.

The local CDB implementation groups its full-slot path by scalar KV slot.
WaveFront already implements a mixed-depth union with per-row slot selection.
A new multi-GPU design must compare against that generic alias-aware union, not only the restricted wrapper.
Cross-owner pooling adds data movement and waits.
It does not create another C current rows for each recurrent depth.

## Speculative ownership

A speculative branch may share an immutable committed prefix.
Its new token rows need private versions.
A later speculative token can require its tentative ancestor's deeper KV before final verification completes.
The required transfer depends on actual producer/consumer ownership.

The branch ledger charges each missing state-version/receiver pair once.
It does not charge a free complete prefix migration or assume a single final winner broadcast suffices.
No current artifact measures a distributed branch runtime.

## Numerical reordering

Real-arithmetic equivalence and BF16 execution equality are different claims.
The CPU controls retain counterexamples for changed norm, reduction, projection, and softmax order.
A correct mathematical identity can still change rounded intermediates.
Current native/reference natural outputs demonstrate why small logit probes cannot certify whole-generation parity.

## Diagram scope

`figures/ownership_dags.mmd` contains editable explanatory DAGs.
Its arrows identify required data/control messages.
Its parallel columns do not establish physical overlap or a measured speedup.
Only the actual GPU traces can support those claims.
