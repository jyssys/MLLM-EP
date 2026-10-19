# Adversarial final search: discharge arithmetic obligations

## Outcome

This search adds zero candidate cards.
It examines two execution families through four explicit design-space cells.
The first family tries to omit nonzero FFN contributions after a sound output certificate.
The second tries to replace expensive projection rechecking with static algebraic checks.
Neither family uses a previous hidden state, predicts a future query, or changes trained weights.

The first family has a valid conditional opportunity under a declared rounding contract.
It lacks a qualified native certificate, useful hit evidence, and a recurrence-only ownership discriminator.
The second has a direct finite-field verification prior.
Its naive linear checksum does not check the BF16-rounded function, and it does not remove the prover's dense work.
These boundaries apply only to the stated constructions, not every possible algorithm.

This report does not reopen the completed C02/C16 or C14 handoffs.
It preserves the [negative-scope overlay](negative_scope_audit_v1.md) and [alias caution](alias_equivalence_interpretation.md).
Generic DAG expressibility alone proves no zero-benefit claim.
No GPU, model, held-out workload, serving engine, package change, or trained component occurs here.

## 1. Cells checked

| Cell | Concrete work unit or state | Difference from earlier cells | Result |
|---|---|---|---|
| A1 | A current down-projection tail whose nonzero contribution cannot change any required BF16 output word | Does not require R5's zero channels or C18's equal inputs | Valid real/rounding construction, native envelope and hit rate missing |
| A2 | An uncomputed gate/up/down channel group enclosed from fresh u and static weight bounds | Can avoid more than down work, without a stale reference | Valid coarse real bound, tighter qualified certificate and physical benefit missing |
| B1 | A precomputed weight-side check vector that checks a fresh projection over a field | Verification rather than C11's redundant coded workers/quorum | Direct Slalom prior, prover still executes the linear operation |
| B2 | A rounding-aware adaptation of B1 for the actual BF16 projection function | Targets native words rather than a field-linear surrogate | Naive checksum fails a positive rounded-output control, complete verifier unspecified |

The [19-card matrix](interim_candidate_evidence_matrix_v1.md) still supplies the broader search context.
Rounds [5](candidate_search_v5_independent_axes.md), [8](candidate_search_v8_precomputed_arithmetic.md), and [9](candidate_search_v9_certified_kv_prefetch.md) delimit nearby arithmetic and certificate work.
This search does not count those earlier cells again.

## 2. A1: certify a nonzero down-projection tail

### Protocol: Native-Word Tail Discharge

Let u be the actual fresh N3 output after attention and its residual.
Let `a = SiLU(Wg*u) * (Wu*u)` with the actual intermediate cast sites.
The down projection computes `m = Wd*a` before N4.
Divide its channels into a completed set S and an uncomputed set T.
For output i, let `z_S[i]` be the completed contribution.

Once the complete fresh a exists, Cauchy–Schwarz supplies this real bound:

```text
abs(m[i] - z_S[i]) <= norm2(Wd[i,T]) * norm2(a[T]) = E[i]
```

Static weight-tail norms can precede the current input.
The numeric activation-tail norm cannot precede the actual SwiGLU result.
Prefix/suffix norm calculations can share work across outputs.
The protocol accepts only when the complete result interval fits the candidate word's rounding cell.
Otherwise it computes another channel group or completes the ordinary operation.

This predicate can omit nonzero terms.
For the declared exact-sum/one-BF16-cast control, partial value 1 and absolute tail cap 1/4096 fit one cell.
That cell has strict boundaries 511/512 and 257/256.
The certificate therefore preserves the stored BF16 value 1 without requiring the tail to equal zero.
This is a positive algebra control, not an Ouro hit observation.

At partial value 257/256, any positive tail can leave the tie's rounding choice.
The constructed tail 2^-30 does so.
Thus, a small absolute tail alone does not establish word equality.
The distance to each rounding boundary matters.

### Native contract and costs

The inspected runtime uses BF16 operands, intermediate casts, and backend accumulation/reduction choices.
A real interval does not automatically enclose that executed operation graph.
The certificate needs conservative errors for its partial result, omitted work, casts, and native dense result.
It also needs exceptional-value handling and a complete output predicate before N4 consumes m.
No native envelope or certificate hit rate exists in this search.

All D=2048 down outputs affect the next norm.
Certifying one output coordinate does not release the complete nonlinear consumer.
Some coordinates can finish early, but tiled norm/output streaming already needs the C17 generic control.
The protocol must not publish an unreduced partial as a complete FFN output.

The protocol can omit selected down-channel arithmetic and weight reads.
All gate/up work already occurred in A1.
Down projection accounts for one-third of the three dense FFN contractions' arithmetic at these shapes.
That fraction is an operation count, not a latency bound or measured speedup.

Added work includes activation-tail norms, static bounds, interval checks, incomplete-output masks, irregular kernels, and fallback.
Distributed owners also need versioned partials, matching accumulator semantics, stop decisions, and completion fences.
Cancellation can save a tile only before execution or through an explicitly supported interruption point.
Already-running work and final drain remain charged.

## 3. A2: discharge a complete nonzero channel group

The second cell asks whether a certificate can precede the omitted gate/up work too.
It uses only fresh u and static weight bounds, not a previous recurrent activation.
For omitted channel j, define:

```text
G[j] = norm2(Wg[j,:]) * norm2(u)
U[j] = norm2(Wu[j,:]) * norm2(u)
abs(SiLU(g[j])) <= abs(g[j]) <= G[j]
abs(a[j]) <= G[j]*U[j]
E[i] <= sum(j in T) abs(Wd[i,j])*G[j]*U[j]
```

These are real-arithmetic inequalities for finite values.
Actual rounded gate/up/activation products need an outward extension of those bounds.
The static coefficient `sum_T abs(Wd[i,j])*norm2(Wg[j,:])*norm2(Wu[j,:])` can precede u.
The query-time bound then scales that coefficient by `norm2(u)^2`.
Tighter current-input intervals can reduce conservatism but require more actual work.
This search measures neither tightness nor savings for checkpoint operands.

For an illustrative channel width 64, I=5632 gives 88 nonempty suffix-bound vectors.
One D×88 table with eight-byte entries occupies 1,441,792 bytes per physical layer.
It occupies 33/66 MiB for 24/48 layers.
This width is a checker design choice, not a native GPU tile claim.
Eight-byte storage alone does not establish directed rounding or soundness.
The gate/up row norms, interval scratch, identities, and control are additional.

The static coefficients can serve every actual visit to a physical layer.
Ordinary autoregressive inference can also reuse them for every token of its fixed weights.
The numeric u, g, up result, and a remain fresh at every recurrent visit.
R4 therefore supplies no right to reuse their values or skip the current predicate.

### Nearest primary mechanisms and the missing discriminator

[ConvReLU++, §3 and Algorithm 1](https://yuanchun-li.github.io/static/files/MobiSys23_ConvReLU%2B%2B.pdf) supplies exact bound-driven work omission before a nonlinear consumer.
Its ReLU has a constant zero region.
Its reference-patch construction is not imported as a stale/delta mechanism here.
The paper also charges bound/reference work and discusses lost matrix-multiplication efficiency in §4.
Those are generic control requirements, not H100 cost predictions.

[DEX v1, §§6.1–6.4](https://arxiv.org/html/2610.11748v1#S6) separates exact sign/ReLU termination from calibrated low-digit skipping.
Its exact negative result fixes a ReLU output to zero.
Its segmentation head needs a sign-only decision.
Its reduced-digit magnitude paths are approximate under its INT8 model contract.
None establishes unchanged BF16 SwiGLU output words or a free distributed transfer.

Ouro's SiLU does not have ReLU's negative constant region in real arithmetic.
R5 already tested a separate bounded zero-channel predicate.
That zero result does not reject A1/A2's nonzero-tail predicate.
Conversely, A1/A2's positive toy does not establish useful checkpoint hits.

The cheapest prerequisite is a backend-qualified interval at one actual output boundary.
The next discriminator is whether it omits useful native tiles after charging its own work and fallback.
A generic same-P certificate kernel needs the same opportunity.
No evidence here identifies an additional benefit from recurrent ownership or a new concurrency frontier.
No engine or quality run follows from this cell.

## 4. B1: immutable-weight algebraic verification

### Protocol: Static-Check Projection Admission

For a field-linear projection `y=W*x`, choose a check vector c and precompute `t=transpose(W)*c`.
The current check compares `dot(c,y)` with `dot(t,x)`.
For suitable random challenges, Freivalds-style checking bounds false acceptance over that field.
Precomputation uses immutable W before the current input exists.
The check still needs the actual fresh x and claimed y.
It does not compute y or release its nonlinear consumer before the claim exists.

The exact finite-field positive control uses modulus 257:

```text
W = [[1,2],[3,4]], x = [5,6], c = [7,11]
y = [17,39], t = [40,58]
dot(c,y) mod 257 = dot(t,x) mod 257 = 34
corrupt y = [18,39] gives 41 and fails this check
```

This is one deterministic arithmetic control, not a soundness experiment or a model evaluation.
The general probabilistic guarantee needs its stated challenge, secrecy/reuse, and arithmetic assumptions.
Its error budget must account for every actual checked projection.

### Direct prior and complete cost

[Slalom, §§2.3, 3.1, and 3.2](https://arxiv.org/pdf/1806.03287) directly supplies weight-side preprocessing for online linear-layer checks.
Its implementation contract quantizes operands into a prime field.
Its baseline also includes a slower trusted environment and a faster untrusted co-processor.
This campaign supplies neither that baseline cost nor permission to replace BF16 with its fixed-point model.

The [official code](https://github.com/ftramer/slalom/tree/96ff15977b7058b96d2a00a51c6aabbe729cc6d5) documents preprocessing controls.
Its `SGXDNN/layers/dense.hpp:52` asserts `!verif_preproc` in the inspected constructor.
Its `fwd_verify_impl:193–210` returns through `apply_impl` rather than exposing the claimed cheap preprocessed check there.
This source boundary prevents a claim that the inspected dense fast path is a drop-in implementation.
The paper's algebraic prior remains direct.
No Slalom code executes or installs here.

For k checks, a direct verifier stores O(k*(input width+output width)) field data per projection.
It also charges offline dense check-vector preparation, online input/output dot products, and challenge management.
Shared physical weights can reuse that data across R4.
Fixed ordinary weights can reuse it across tokens too.
This fact alone does not supply a recurrence-specific physical optimum.

If a remote worker computes the entire FFN, checking its three linear stages requires suitable intermediate witnesses.
One explicit combined protocol sends fresh u and receives g, up, and m.
At BF16 word width, that payload is `(2D+2I)*2 = 30720` bytes per row.
It excludes checks, versions, headers, and fences.
The verifier reconstructs the specified SwiGLU intermediate before checking the down claim.
Native intermediate casts and nonfinite behavior remain part of that obligation.
This payload is a protocol example, not a lower bound for every proof system.

The prover still performs the dense projections unless a separate cheaper algorithm exists.
Verification does not remove their arithmetic or weight reads by itself.
Native same-P inference already trusts its ordinary arithmetic path without recomputing every layer for integrity.
Replacing an absent integrity tax cannot establish a serving speedup.
This differs from C11's quorum/straggler protocol, but supplies no new useful execution benefit here.

## 5. B2: adapt the check to BF16 outputs

The function `round_BF16(W*x)` is not field-linear or real-linear.
A direct decoded-word check can reject an output that obeys its declared rounding contract.
Use BF16-representable weights and inputs:

```text
W = [[1, 1/256], [1, -1/256]]
x = [1,1], c = [1,1]
exact W*x = [257/256,255/256]
BF16 RNE output = [1,255/256]
dot(c,rounded output) = 511/256
dot(transpose(W)*c,x) = 2
```

This example uses exact products/sums followed by the stated BF16 cast.
It does not emulate an unidentified Tensor Core reduction tree.
The checksum mismatch already exists without that extra accumulation uncertainty.
Rounding the checksum itself or adding a tolerance changes its soundness contract.
It does not prove exact model-state equality automatically.

An exact encoding of finite BF16 operands into a field does not remove output rounding.
A stronger verifier could check a specified floating-point circuit with adequate witnesses.
That is a separate algorithm with proof generation, intermediate state, validation, and communication costs.
This report proves neither its impossibility nor its efficiency.
It identifies no such complete protocol for unchanged Ouro.

The cheapest disproof of the naive B2 adaptation is the exact control above.
A reopening needs a sound BF16 function check and a cheaper complete prover/verifier path than native same-P inference.
No current observation supplies that path.

## 6. Both sizes, modes, and concurrency

Both Base models have D=2048, I=5632, and 16 MHA heads.
Their 24/48 physical layers execute 96/192 fixed-R4 block visits per token.
A1/A2 operate only on each visit's fresh N3/SwiGLU values.
B1/B2 operate only on a fresh input and completed projection claim.
Neither family makes the next recurrent hidden state or its current KV ready early.

At C1, conditional channel work or check dot products exist inside the current operation.
At C2/C4/C8, up to C request fronts add independent current inputs before operator tiling.
These task counts do not establish simultaneous physical execution or speedup.
P2/P4 require complete ownership and communication costs, even if a certificate later becomes sound.
P8 remains a structural target with no physical result.

An exact matching m or projection output must still pass through all required norms and residual operations.
Original-gate adaptive execution retains q=0.5, minimum R2, maximum R4, and `last_exited`.
No interval heuristic can silently change selected hidden states, gates, hazards, or depth-indexed KV.
Base 2.6B adaptive remains `ADAPTIVE_2P6B_UNVALIDATED`.
Every quality and mechanism-latency cell remains `NOT_RUN`.

## 7. Evidence and conservative work ledger

Measured observations in this round concern only deterministic constructed arithmetic and source identities.
Structural bounds and payloads are analytic.
Native certificate soundness, checkpoint hit rates, task quality, distributed latency, and recurrence-specific benefit remain missing.
The [JSON source ledger](../artifacts/adversarial_final_axis_search_v1.json) retains queries, pins, hashes, exact control operands, and results.
No earlier report, card, raw evidence, held-out artifact, or coordinator state changes.

The first segment begins at `2026-10-19 17:06:34 UTC`.
Its last recorded clock before interruption is `17:07:20 UTC`.
This report counts only those 46 seconds.
Additional pre-interruption source reads lack a final clock and do not add counted work.
The main resume time is `17:11:01 UTC`.
This worker's resumed segment begins at its recorded `17:11:39 UTC` clock.
The conservative excluded interval is therefore `17:07:20–17:11:39 UTC`, or 259 seconds.
The resumed validation cutoff is `2026-10-19 17:18:55 UTC`.
The resumed interval spans 436 seconds.
Total conservatively counted work is 482 seconds, or 8 minutes 2 seconds.
JSON parsing, eight input hashes, four rational controls, scalar accounting, and report lint pass.
Final timestamp insertion and handoff follow that cutoff.
Concurrent worker intervals do not add campaign elapsed time.
