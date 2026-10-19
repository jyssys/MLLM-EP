# Independent discovery round 3

Worker: `/root/independent_axes`.
Segment began: 2026-10-19 11:50:20 UTC.
Scope: two provisional execution abstractions outside the frozen C01–C16 mechanisms.
Evidence: source inspection, algebra, and deterministic CPU controls only.

## Outcome

Neither proposal merits an engine from this screen.
R3-A exposes raw feature-tile work before a normalization scalar completes.
FlashNorm already supplies its central algebra.
The simple rewrite also changes an explicit BF16 rounding boundary.

R3-B removes a transition only after its complete read set repeats exactly.
This is valid conditional reuse under strict numerical and side-effect conditions.
Shared weights alone do not establish the required equality.
The screen supplies no checkpoint hit-rate evidence or recurrence-specific systems benefit.

These conclusions reject two proposed implementations and their current novelty claims.
They do not reject every component-ready or communication-avoiding algorithm.
No GPU, environment change, weight download, or model-quality experiment occurred.
The current candidate register and earlier reports remain unchanged.

## 1. Scope and separation from earlier candidates

The worker first read the current register and the specification's scope and constraints.
The register lists C01–C14.
Separate completed cards supply C15 and C16.
This report uses provisional IDs rather than changing coordinator state.

R3-A partitions a fresh input by final feature coordinates.
Each coordinate tile supplies part of one dense contraction.
It does not project an old complete hidden vector and a later dense delta.
Thus, it differs from C06's two-projection formula.
It also differs from C04's distributed layout closure.

R3-B requires exact operand equality before removing work.
It does not predict a hidden vector, truncate a residual, or solve an approximate trajectory.
It differs from C05 correction and C13 provisional continuation.
If its equality certificate fails, it executes the original computation.
No trained gate or learned predictor supplies that certificate.

## 2. Actual model boundaries

Both Base models use D=2048, I=5632, 16 MHA heads, and four recurrent passes.
They contain 24 and 48 physical layers.
The [semantics audit](ouro_semantics_audit.md) pins both official source revisions.
This round also inspected the local CDB model and original-gate adapter.

Each physical block applies four RMSNorm modules.
They surround attention and SwiGLU residual updates.
Each pass then applies a separate loop norm.
Fixed R4 therefore executes 388 and 772 RMSNorm calls before any candidate transformation.
These are structural counts, not measured time fractions.

The source norm computes the mean square in FP32.
It multiplies by the inverse RMS and casts the normalized coordinates to the input dtype.
It then applies the norm weight.
The following dense layer consumes that rounded result.
This BF16 boundary cannot disappear from an unrestricted bitwise-equivalence claim.

Q/K projection precedes RoPE.
Attention reads the matching depth-indexed historical KV.
The model has no added query/key norm that could cancel this dependence.
The original gate consumes the completed loop-normalized hidden state.
Its token-local survival/CDF update remains a separate operation.

## 3. R3-A: Component-ready contraction with delayed RMS scaling

### Mechanism and readiness

Let x contain final input coordinates, not unreduced TP contributions.
Partition its coordinates into tiles x_i.
Let `s(x)=1/sqrt(mean(x^2)+epsilon)` and let Gamma contain the norm weights.
In real arithmetic:

`W*RMSNorm(x) = s(x)*sum_i W[:,i]*Gamma_i*x_i`.

The raw partial product for tile i needs only x_i and its weight columns.
It can start before other final coordinate tiles arrive.
The norm reduction can accumulate squared coordinates concurrently.
The final projection waits for every partial product and the complete norm scalar.
This schedule performs one dense contraction, not two complete dense GEMMs.

No normalized coordinate becomes generally ready from its prefix alone.
The CPU control fixes the first coordinate at 1.
Changing the unfinished second coordinate from 0 to 3 changes its normalized BF16 value from 1.4140625 to 0.447265625.
An exact consumer of that normalized coordinate must retain the dependency.

### Source and genericity

[FlashNorm v5](https://arxiv.org/html/2407.09577v5), section 3.2, directly defers RMS scaling until after projection.
Its extensions include GLU and RoPE consumers.
Appendix A distinguishes mathematical equivalence from lossy-precision decoding behavior.
Its low-token stream fallback also prevents transferring large-batch results into this low-C target.

The inspected author revision is `c3cec7e8076926acd2a31c2e5d7eadfd47489318`.
Its [merge helper](https://github.com/OpenMachine-ai/transformer-tricks/blob/c3cec7e8076926acd2a31c2e5d7eadfd47489318/transformer_tricks.py#L125)
forms merged weights in FP32 and casts them to the original projection dtype.
This is source inspection only.
The worker did not run conversion, inference, installs, or the example's download commands.

Generic fusion also supplies the necessary implementation vocabulary.
The [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)
describes broadcast scaling, auxiliary outputs, and reductions.
That particular tutorial requires Blackwell.
It is not evidence for an H100 implementation or performance result.

### BF16 falsifier

Use BF16 inputs x=`[1,3]`, weights W=`[3,-1]`, and unit norm weights.
The FP32 norm scalar is 0.4472135603427887 in the software control.
The rounded norm output is `[0.447265625,1.34375]`.
The source-order projection gives -0.001953125.
The deferred raw projection gives zero because `3*1-3=0` before scaling.

This counterexample rejects universal BF16 identity for this simple rewrite.
It does not show a model-quality failure or reject a separately checked numerical-reordering method.
The software control does not emulate a complete CUDA kernel.
Its float64 positive control agrees to 4.440892098500626e-16.

### Ownership and complete costs

| Object | Readiness | Owner and added cost |
|---|---|---|
| Final x_i | Its actual producer completed | Feature owner, not a guessed partial sum |
| W columns and Gamma_i | Load or transfer completed | Projection worker, same checkpoint revision |
| Squared-sum partial | Matching x_i exists | Reduction worker, FP32 accumulation |
| Raw projection partial | x_i and weight columns exist | Projection worker, FP32 M-wide scratch |
| Norm scalar | All squared sums complete | Same token, any required scalar collective |
| Historical/current KV | Original attention dependencies | Unchanged depth/layer owners |
| Quantized Q/K/V or MLP input | Original casts complete, unless the method explicitly changes numerical order | Existing consumer owner |

For a D-to-M projection, the arithmetic remains D*M multiplications plus norm work and final scaling.
With input-feature partitioning, each worker can need an M-wide FP32 partial buffer.
Its logical size is `4*M` bytes before tiling.
For combined QKV, M=6144 gives 24,576 bytes per partial.
This is scratch capacity, not wire traffic or a required unsharded allocation.

Input-feature workers also need an output-partial reduction or another complete output layout.
A column-parallel baseline can have a different communication pattern.
Comparisons must match those layouts and include every new collective.
Tile flags, fences, kernel dispatch, packing, register pressure, and loss of GEMM efficiency remain additional costs.
Repeated weights do not make the full stack resident in registers or L2.

Two pre-linear norms per block are the direct targets.
The branch-output norms still precede residual addition and later nonlinear consumers.
Moving a scalar across that addition is not the same identity.
The loop norm also supplies the original gate.
Gate bias must remain after any valid output scaling.

### Mandatory proposal fields

| Field | R3-A screen |
|---|---|
| 1. Identity | Revision 1, component-ready tile dataflow |
| 2. Mechanism | Accumulate fresh feature-tile contractions before final RMS scaling |
| 3. Ouro property | Repeated physical layers and frequent RMSNorm boundaries, not activation reuse |
| 4. Target | Both Base sizes, fixed R4/original adaptive, decode C1/2/4/8, N1K/4K/8K, P2/P4 and projected P8 |
| 5. Baseline | Same-P TP with generic FlashNorm-style/fused norm-linear control and the same numerical contract |
| 6. Sequence | Norm then projection becomes raw tile products plus norm reduction, then scalar application |
| 7. DAG | Each raw tile waits for final coordinates and weights, final output waits for all partials and s(x) |
| 8. Concurrency | Ready coordinate tiles and independent norm/projection work, no future recurrent inputs |
| 9. Exactness | Real-arithmetic identity, universal BF16 identity fails, model quality NOT_RUN |
| 10. Memory | Same logical KV, dense weights, M-wide partial scratch, tile flags, optional merged-weight storage |
| 11. Messages | Tile transfer, norm reduction, and output-partial reduction depend on matched TP layout |
| 12. Critical path | Potentially removes exposed norm wait, adds joins and reductions, no measured saving |
| 13. Priors | Direct FlashNorm algebra and generic fusion, no demonstrated distinct runtime element |
| 14. Prior NO-GO | Not a C06 dense-delta rescue, but nonlinear and cast dependencies still remain |
| 15. Falsifier | BF16 cancellation test rejects bitwise claim, generic deferred-norm control rejects novelty |
| 16. Evidence | Deterministic CPU algebra and source inspection only |
| 17. Compatibility | Both sizes/modes share the boundary, P2/P4 NOT_RUN, P8 projected |
| 18. Counterfactual | R1 and alias-aware unrolling retain the same opportunity |
| 19. Complexity | No engine, native fused numerics and layout work would need separate qualification |
| 20. Verdict | GENERIC_ONLY for valid numerical reordering, simple universal-BF16 arm rejected |

The single next informative action is a matched generic deferred-norm control, only if this engineering optimization becomes a separate task.
It is not a new campaign finalist.

## 4. R3-B: Exact transition coalescing with a complete read-set certificate

### Mechanism and legal inputs

Cache a completed deterministic transition and its side effects.
Reuse them only when the new transition has an equal complete read set.
A hash can select a candidate entry.
Exact comparison or a proven immutable-object identity must establish the hit.
A cryptographic digest alone is not a mathematical collision-free certificate.

For one full recurrent pass, the key must cover at least:

- All input hidden bytes and their dtype/layout.
- Physical weights and every norm/gate parameter revision.
- All historical KV values that the pass reads, with causal lengths and versions.
- Positions, RoPE convention, masks, and attention scaling.
- Backend, reduction/cast path, batch shape, and any relevant deterministic execution settings.

Full-pass equality does not require identical request names.
It requires equal numerical inputs and a legal mapping of replayed side effects into the new request's namespaces.
A returned hidden vector alone is insufficient.
The replay must also reproduce every current-token KV write that skipped operators would create.

For a pure MLP component, the key is smaller.
Its normalized input, weights, and numerical execution signature can suffice.
Historical KV does not directly enter that component.
This cheaper arm is generic operator memoization rather than a recurrent-state theorem.

### Why shared weights do not prove a fixed point

Write a pass as `h_r = F(h_(r-1), KV_r, positions, numerical_signature)`.
Even with one shared physical F, the historical bank KV_r can differ across r.
Equality of two hidden vectors therefore does not establish equal transition inputs.
Full-depth prefill explicitly creates different recurrent-depth banks.
Original adaptive terminal copies do not make every historical bank identical.

The CPU counterexample uses equal hidden input and equal keys.
Changing values from `[0,2]` to `[0,4]` changes attention output from 1 to 2.
A second control changes query position with the same raw hidden input.
Its outputs change from 0.5378828427399902 to 1.4621171572600098.
Both differences reject an incomplete cache key, not exact memoization itself.

A complete finite-state repetition can justify a fixed-point or cycle shortcut.
That requires stationary future read sets and the same deterministic numerical transition.
It also requires replaying the original transition side effects.
The CPU positive control constructs a fixed point at hidden value 2.
It computes one transition, reuses three, and still commits four separate depth writes.
This is a constructed positive control, not a discovered model fixed point.

### Original adaptive gate

An equal gate input under the same numerical signature permits reuse of the same gate result.
It does not permit reuse of the previous token's CDF or continuation decision.
The original token-local policy must consume that hazard at each actual depth.
Minimum depth and forced maximum depth remain unchanged.

For repeated hazard 1/4, the constructed CDF values are 0.25, 0.4375, and 0.578125.
At threshold 1/2 and minimum depth 2, the original policy exits at depth 3.
It does not exit at the first repeated state merely because the hidden value repeats.

The real-arithmetic geometric formula also needs a finite-precision caution.
For a constructed FP32 hazard, sequential double CDF updates give 0.9744000061035152.
The geometric shortcut gives 0.9744000061035151.
A threshold at the first value changes the decision.
The candidate must retain the original sequential policy arithmetic rather than assume bitwise geometric equivalence.
This constructed threshold is not the campaign's model operating point.

### Generic priors and non-priors

[vLLM's official prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
already uses hashes of complete prefix identity, token blocks, and extra configuration keys.
Its shared computation and state reuse are the nearest production control.
This source does not implement full dynamic hidden/KV equality coalescing across recurrent depths.
The distinction is a possible runtime extension, not evidence that the extension has useful hits.

[OpenXLA's compiler documentation](https://github.com/openxla/xla/blob/main/docs/hlo_to_thunks.md)
describes common-subexpression elimination.
Static duplicate-expression removal does not establish equality of dynamic input values.
It still supplies the valid alias-aware generic control when those values are already identical objects.

[Fuzzy RNN memoization](https://arxiv.org/abs/2202.06563) predicts similar outputs with an added bitwise network.
That mechanism is not this exact certificate and is outside this campaign's no-new-learned-component contract.
[Approximate RNN fixed points](https://arxiv.org/abs/2106.02417) concern iterative dependency recovery rather than observed bitwise transition repetition.
[UCNN](https://arxiv.org/abs/1804.06508) exploits repeated scalar weights, not full repeated transition inputs.
None of their performance results transfer to this proposal.

### Ownership, memory, and communication

| Object | Readiness | Owner and required action |
|---|---|---|
| Input hidden | Actual preceding pass completed | Request worker, exact bytes or safe immutable identity |
| Historical KV | Original commit/propagation completed | Matching request/depth/layer owner |
| Signature entry | All key components available | Cache owner, configuration and version namespaced |
| Cached hidden/output | Original transition completed | Immutable result owner until eviction |
| Cached KV write log | Every source layer completed | Replay matching values into separate target identities |
| Hit decision | Exact comparison completed | Dispatcher, no guessed equality |
| Gate CDF/survival | Previous policy update completed | Token owner, update at each actual depth |

At one physical layer, a BF16 MLP input and output each contain 4096 bytes.
A full pass's historical KV read set contains `2*L*N*D*2` bytes.
Comparing two independent banks directly can read twice that amount.
An immutable canonical object can avoid that scan only after a sound identity check.
Incremental hashes reduce repeated scanning but do not prove byte equality by themselves.

| Physical L | N | One pass historical KV | Two-bank comparison | Current KV replay per skipped pass |
|---:|---:|---:|---:|---:|
| 24 | 1024 | 192 MiB | 384 MiB | 192 KiB |
| 24 | 4096 | 768 MiB | 1536 MiB | 192 KiB |
| 24 | 8192 | 1536 MiB | 3072 MiB | 192 KiB |
| 48 | 1024 | 384 MiB | 768 MiB | 384 KiB |
| 48 | 4096 | 1536 MiB | 3072 MiB | 384 KiB |
| 48 | 8192 | 3072 MiB | 6144 MiB | 384 KiB |

These are structural byte counts, not measured memory traffic or latency.
They exclude inputs, weights, metadata, allocation overhead, and result scratch.
Actual replay can use an equivalent safe alias protocol instead of physical copies.
That protocol must preserve each target identity and lifetime.

Under TP, every relevant shard must agree on a hit before dependent work skips.
Charge local comparison or identity lookup, distributed consensus, result routing, and replay.
A remote memo owner adds transfer and queue costs.
At C=1, only cross-depth or existing-cache reuse can help.
At C=2/4/8, cross-request duplicates are also available to generic prefix/component caching.
P8 has no measured result.

The critical-path condition is conditional, not automatic:

`lookup + exact comparison + agreement + replay < skipped complete transition cost`.

The inequality must hold on actual hits, and misses must charge their overhead too.
No hit-rate or break-even estimate comes from a constructed fixed point.
Keeping a small cache does not remove the cost of constructing a valid full-pass key.

### Mandatory proposal fields

| Field | R3-B screen |
|---|---|
| 1. Identity | Revision 1, certified exact work elimination |
| 2. Mechanism | Reuse transition output and side effects after complete read-set equality |
| 3. Ouro property | Repeated physical function allows a possible cross-depth lookup, distinct KV remains mandatory |
| 4. Target | Both Base sizes, fixed R4/original adaptive, C1/2/4/8, N1K/4K/8K, P2/P4 and projected P8 |
| 5. Baseline | Same-P generic TP/DP/CDB with prefix caching and equally safe component memoization |
| 6. Sequence | Complete normal transition becomes exact-key check, cached output/replay on hit, normal transition on miss |
| 7. DAG | Hit waits for all key inputs and agreement, readers wait for replayed committed state |
| 8. Concurrency | Key metadata can prepare from committed history, hidden comparison still waits for actual hidden input |
| 9. Exactness | Conditional bitwise reuse under matched deterministic path, no similarity substitution |
| 10. Memory | Result cache, complete signature metadata, write logs, distinct logical depth KV |
| 11. Messages | Shard agreement, remote result routing, state replay, eviction/lifetime coordination |
| 12. Critical path | Only actual certified hits remove compute, misses add lookup/check overhead |
| 13. Priors | Generic memoization, exact prefix caching, CSE, and non-equivalent approximate RNN reuse |
| 14. Prior NO-GO | No stale/delta rescue, full input equality is a new strict condition rather than assumed similarity |
| 15. Falsifier | Count exact full-key or pure-component repeats in an existing committed calibration trace |
| 16. Evidence | Constructed positive and negative CPU controls, model hit rate and performance NOT_RUN |
| 17. Compatibility | Both sizes/modes require the same certificate, P2/P4 need shard agreement, P8 projected |
| 18. Counterfactual | Alias-aware unrolling and R1 cross-request memoization can exploit the same certified duplicates |
| 19. Complexity | No engine, first build a read-only exact-hit census if an adequate existing trace exists |
| 20. Verdict | CONDITIONAL_MECHANISM, no evidence of useful recurrence-specific repeats |

The single next informative action is the exact-hit census, not another inference engine.
This conditional status does not put the proposal on the shortlist.
Do not rerun model workloads merely to rescue this hypothesis without a changed-evidence plan.
Stop if the inspected trace has no sufficient exact repeats or if generic reuse accounts for all hits.

## 5. CPU evidence and reproduction

Run from the campaign root:

`python scripts/candidate_round3_cpu.py`

The [CPU artifact](../artifacts/candidate_round3_cpu.json) preserves the actual values and structural counts.
Its script SHA256 is `68fc29186ec78409a566d0fe99255651f938932ba1dfd60ce3a442e04ea9b98f`.
The first CPU run occurred at 11:54:55 UTC.
The final control uses byte keys and distinguishes signed zero.
Its actual execution ran from 11:58:47.621510 to 11:58:47.621906 UTC on 2026-10-19.
This tiny runtime is not an inference performance measurement.

The controls use fixed constructed tensors and software BF16 rounding.
They contain no random-weight model, checkpoint tensor, GPU, or trained component.
They check algebraic and state-key claims only.
Model numerics, gate agreement, quality, latency, throughput, and incremental speedup remain `NOT_RUN`.

## 6. Source observations and time ledger

The [source observations](../artifacts/candidate_round3_source_observations.json)
record queries, source locators, pinned code, failed retrieval, and scope limits.
One guessed FlashNorm filename returned HTTP 404.
The subsequent official tree lookup found the actual example and helper paths.
No failed retrieval supplies a technical claim.
The searches do not establish a complete prior-work inventory.

The source and scope block began at 11:50:20 UTC.
The source/derivation checkpoint was 11:53:54 UTC.
The CPU-control checkpoint was 11:54:55 UTC.
Source checks and report synthesis continued after that checkpoint.
The final evidence checkpoint was 12:00:01 UTC.
The active worker interval to that checkpoint was 581 seconds.
The report and individual source-observation prose fields passed the clarity checks.
Both JSON artifacts parsed, and the saved script identity matched.
Final publication followed that checkpoint.
Count this worker interval once within concurrent campaign wall time.
Do not add overlapping worker intervals as extra elapsed hours.
