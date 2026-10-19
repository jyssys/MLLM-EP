# C15: Attention from stored normalized inputs

## 1. Identity

Revision 1. Family: exact state representation and operator reassociation.
Current verdict: `GENERIC_ONLY` for the valid retained-K variant.
The one-query RoPE absorption arm is algebraically invalid.

## 2. Mechanism

Store normalized attention inputs X, absorb K projection into queries, and apply V projection after head-specific weighted-X sums.

## 3. Relevant Ouro property

Ouro has 16 MHA heads, D=2048, d=128, and repeated physical weights.
It stores independent depth/layer state.
The official attention source projects Q/K before RoPE.
See [the source audit](../reports/residual_input_reassociation_redteam.md).

## 4. Target scope

Both Base sizes, fixed R4, and the unchanged original adaptive gate.
Decode at system-wide C=1/2/4/8 and contexts 1K/4K/8K.
P2/P4 require actual candidate tests.
P8 remains projected.
Prefill needs separate work accounting.

## 5. Baseline and exposed cost

Compare with competent same-P TP attention using cached rotated K and V.
The intended saving is cache capacity or exposed cache traffic.
No such saving received a new GPU measurement.

## 6. Operation sequence

AS-IS: normalize X, project Q/K/V, rotate Q/K, cache K/V, and compute the original softmax attention.
TO-BE: cache X and use reassociated score/value contractions.
The actual RoPE score requires a position-specific transformed query.
Retaining rotated K makes scores ordinary again but removes the unsharded capacity saving.

## 7. DAG and ownership

`current X -> current q -> score_h,j -> softmax_h -> weighted X summary_h -> Wv_h -> O projection`

Each score also reads committed historical X at its matching r/l/j version.
The retained-K arm reads the matching rotated K instead for scores.
All nonlinear consumers keep their actual predecessors.

| State | Owner and readiness | Lifetime/cost |
|---|---|---|
| Physical W shards | Model ranks after load | Immutable, same revision |
| X[r,l,j] | Historical-state owner after the actual norm | Retain all required depth/layer versions |
| Rotated K, if retained | Original head owner after projection/RoPE | D additional elements per historical node |
| Current q | Current head worker after Q/RoPE | Query-specific, not historical-state independent |
| Alpha and Z_h | Attention worker after score normalization | H separate summaries plus softmax state |
| Adaptive state | Original gate/runtime owner | Same q, exit depths, and KV propagation |

## 8. Source of concurrency

Independent heads, feature tiles, sequence tiles, and requests can run concurrently.
Historical X inputs are already ready.
Current alpha depends on the current query.
Repeated passes of one token remain a dependency chain.
No new independent recurrent frontier follows from this representation.

## 9. Exactness and quality

Value reassociation is exact in real arithmetic for the original alpha.
Position-specific key reassociation is also valid in real arithmetic.
One position-independent absorbed query is invalid with general Ouro RoPE.
BF16 projection casts break universal bitwise equivalence.
Model numerics, exits, KV, and free-running quality remain `NOT_RUN`.

## 10. Memory

Per logical token/layer, X occupies 4096 BF16 bytes.
K/V occupy 8192 bytes.
Rotated K plus X also occupies 8192 bytes.
For head TP, replicated X loses the aggregate saving at P2 and increases state at P4/P8.
Feature-sharding X needs a different communication plan.
Charge H*D summary values, weights, metadata, and scratch separately.

## 11. Communication

A direct feature-sharded plan reduces H*N partial scores before softmax.
At N=1024, its logical FP32 score payload is 65,536 bytes per query/logical layer.
It grows linearly with N.
Value partials also need assembly or an equivalent output layout.
Head replication and history all-gather are alternatives with explicit memory/transfer costs.
This accounting assumes no link bandwidth or eight-rank timing.

## 12. Critical path and added costs

Cached attention costs `4D^2+2ND` scalar multiplications at the stated attention boundary.
The no-RoPE X-only control costs `4D^2+2HND`.
Rotated K plus X costs `4D^2+(H+1)ND`.
The direct position-specific X-only path costs `(N+3)D^2+2HND`.
Rebuilding only K gives the cheaper fallback `(N+3)D^2+(H+1)ND`.
These counts include current-token projection replacements.
They are not measured latency or lower bounds for every algorithm.

Also charge norms, rotations, softmax, summary buffers, communication, layout changes, and kernel dispatch.
The bottleneck can change when arithmetic and storage change together.

## 13. Nearest work

DeepSeek MLA already absorbs projection matrices and explains the RoPE obstacle.
Its decoupled positional branch is a trained architecture, not unchanged Ouro.
The official V3 implementation forms head-specific latent summaries.
Residual-stream KV reconstruction and generic attention association are additional close controls.
The audit contains versioned primary links.

## 14. Prior evidence and changed assumption

C10 charged full-history K/V rematerialization.
C15 changes that assumption by testing projection reassociation.
The smallest test is the actual RoPE algebra, followed by head-aware costs.
This is not a stale/delta rerun or extra placement tuning.
The bounded screen ends here without an engine.

## 15. Smallest falsifier

The deterministic score test already rejects the one-query RoPE rewrite.
The retained-K storage count already removes its unsharded capacity claim.
A reopening needs a new exact position-aware representation with a lower complete resource cost.
It cannot remove RoPE or train a compressed latent.

## 16. Results and uncertainty

Six CPU assertion groups passed.
Valid reassociations matched float64 controls.
Static RoPE absorption and shared-head summaries failed their constructed controls.
Two explicit BF16 rounding counterexamples differ.
These are not model-quality failures.
Achieved latency, memory traffic, and quality remain `NOT_RUN`.

## 17. Compatibility matrix

| Scope | Result |
|---|---|
| 1.4B fixed/adaptive | Same RoPE and MHA obstacle, no candidate model test |
| 2.6B fixed/adaptive | Same dimensions, doubled physical layers, no new adaptive validation |
| P2 | Replicated-X capacity saving disappears |
| P4 | Replicated X uses 2 times baseline KV state per rank |
| P8 | Projected replicated X uses 4 times baseline KV state per rank |

## 18. Loop-specificity controls

Aliased logical unrolling preserves all X identities and matrix equations.
A generic executor can use the same representation.
An R1 source-level control also retains this opportunity.
Neither control needs recurrent weight identity for the algebra to hold.
This audit establishes no recurrence-specific advantage.

## 19. Complexity and next PoC

This audit recommends no engine.
A future exact representation needs a new algebraic proof before a kernel benchmark.
Its benchmark must compare the complete attention path with matched cached TP attention.
There is no supported implementation-time estimate.

## 20. Verdict

Do not shortlist C15 from this screen.
The cheap static-query form fails actual RoPE.
The legal retained-K form is generic and does not reduce unsharded state.
The next useful action is a new exact representation, not tuning this rejected rewrite.
