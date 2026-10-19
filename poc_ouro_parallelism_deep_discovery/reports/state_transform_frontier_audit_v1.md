# Exact state transforms and the recurrent frontier

## Decision

This bounded audit produces zero new candidate cards.
The inspected transforms do not supply a cheaper exact state summary for unchanged Base Ouro at fixed R4.
They also do not establish exact original-gate adaptive execution.
The obstruction is a missing cheap, state-complete transform, not a universal impossibility theorem.

This audit excludes the Jacobi/Newton correction already screened in C05.
It adds no stale hidden prediction, training, quantization, KV approximation, or new engine.
All executed checks use Python scalar algebra on CPU.

## Required operands and the dependency at issue

For one token and frozen authoritative historical cache, write:

```text
(h_r, V_r) = F_r(h_(r-1), C_r), r=1,...,4
```

`C_r` contains the actual depth-indexed historical KV, positions, and masks.
`V_r` contains the current token's required internal-layer KV writes.
The operator also includes the actual loop boundary norm.
Tied weights do not imply identical `F_r`, because their historical read states can differ.

A qualifying transform must remove the wait for the actual predecessor hidden state.
It must still produce the same hidden states and authoritative KV required by later tokens.
Original adaptive execution also needs the original gate hazards, their ordered accumulation, and terminal-state publication.
The [existing frontier contract](exact_frontier_contract.md) preserves these requirements.

An exact affine summary would provide a genuine positive example:

```text
h_r = A_r h_(r-1) + b_r
(A_after,b_after) compose (A_before,b_before)
  = (A_after A_before, A_after b_before + b_after)
```

All coefficients must exist before the scan.
Their construction must not require the missing predecessor states.
Associativity then permits a parallel prefix computation, subject to its actual matrix and communication costs.
This is a generic affine-scan mechanism, not a consequence of equal weights alone.

## 1. Nonlinear parallel-in-time: what Parareal adds

The author-hosted [Gander–Hairer manuscript](https://www.unige.ch/~gander/Preprints/gander_plenary2007.pdf) analyzes nonlinear Parareal.
Its university publication record dates the proceedings chapter to 2008.
The author's preprint list uses 2007.
This audit preserves that metadata difference.

Section 2, equation 6 uses:

```text
U_n^(k+1) = F_n(U_(n-1)^k)
          + G_n(U_(n-1)^(k+1)) - G_n(U_(n-1)^k)
```

Fine evaluations use the old boundary guesses in parallel.
The new coarse boundary values still propagate sequentially.
Section 3 assumes a controlled fine/coarse error expansion and a Lipschitz condition for its convergence bound.
These ODE assumptions do not establish useful convergence for four frozen Transformer passes.

For Ouro, an exact cheap coarse propagator remains unspecified.
Choosing the complete original pass as the coarse propagator restores its serial cost.
Using an approximate coarse propagator requires correction, state validation, and KV finalization.
That transfer belongs to the [previous C05 audit](parallel_sampling_nearest_audit.md), not a new candidate here.
No Parareal implementation or model executes in this round.

## 2. State-space lifting: exact closure is the missing mechanism

[Brunton et al.](https://arxiv.org/html/1510.03007v2) define Koopman evolution by composition in equations 4–6.
Equations 9–10 require the chosen observable space to remain invariant.
The [published PLOS article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150171) appeared on 2016-02-26.
Its appendix includes reproducible MATLAB examples.
The inspected arXiv version is v2, dated 2015-12-23.

A relevant exact construction would require:

```text
phi_r(T_r(S,C_r)) = A_r(C_r) phi_(r-1)(S)
D_r(phi_r(S_r)) = S_r
```

Here `S_r` includes the hidden state, required current KV, and gate/publication state.
`T_r` advances that complete state through one pass.
The coefficients must depend only on already available operands.
The lift, matrix representation, and decoder must cost less than the dependency they replace.

The paper gives exact special systems, not a universal cheap lift.
Its logistic-map example requires progressively higher polynomial observables.
Its nonlinear scalar example also shows that nonpolynomial coordinates can succeed where polynomial truncation fails.
Therefore, failed polynomial closure does not reject every possible transform.
This audit does not apply the paper's global fixed-point discussion as a theorem about Ouro.

A finite R4 horizon is another important exception.
For `F(x)=x²`, four outputs require only powers through `x^16`.
An infinite-horizon nonclosure argument alone does not reject that finite expression.
Likewise, a formal lift containing `h,F(h),...,F^4(h)` exists.
Its construction already computes the missing trajectory unless a cheaper exact representation exists.
No inspected source provides such a representation for Ouro, including its depth-indexed cache side effects.

## 3. Reversible recurrence: past reconstruction is not future readiness

[Reversible Recurrent Neural Networks](https://papers.neurips.cc/paper/8117-reversible-recurrent-neural-networks.pdf) is arXiv:1810.10999v1, dated 2018-10-25.
Section 3 changes GRU/LSTM updates into reversible grouped updates.
Equations 4–6 reconstruct old states from available later states.
Section 3.3 identifies finite-precision information loss.
Algorithm 1 and Section 5 retain lost information in integer buffers.

The official code pin is `0d4b07f6f0c69e35f2b112d18cd7402161ef2094`.
[`revgru.py:28–89`](https://github.com/matthewmackay/reversible-rnn/blob/0d4b07f6f0c69e35f2b112d18cd7402161ef2094/revgru.py#L28) updates the first group before computing the second group's gates.
Lines 91–136 reverse those updates using an output state and information buffers.
The implementation uses fixed-point hidden values, not an unchanged BF16 Ouro state representation.

The transferable benefit concerns activation storage during training.
An inverse still requires its later-state operand.
Launching inverse work from an unknown final state needs a boundary guess and matching solve.
That does not eliminate the initial-value dependency.
Saving previous inputs can make an augmented computation reversible, but it does not produce later inputs earlier.

## 4. Reversible Transformer execution

[Reformer](https://arxiv.org/html/2001.04451v2), v2 dated 2020-02-18, uses a changed two-stream residual structure.
Section 3, equations 7–9 give:

```text
Y1 = X1 + Attention(X2)
Y2 = X2 + FeedForward(Y1)
```

The feed-forward operation still needs the actual `Y1`.
The inverse subtracts recomputed residuals from available outputs.
This supports training-memory savings, not independent execution of successive nonlinear updates.
The paper trains its models on eight GPUs or eight TPU v3 cores.
It does not validate an unchanged Ouro checkpoint.

The paper links the official Trax implementation.
The inspected modern pin is `31022d6cd7dd525ed11a04d84cd3936228499173`.
[`ReversibleHalfResidual.forward()`](https://github.com/google/trax/blob/31022d6cd7dd525ed11a04d84cd3936228499173/trax/layers/reversible.py#L296) computes its residual before addition.
Its `reverse()` raises `NotImplementedError` at lines 318–319.
The implementation uses `reverse_and_grad()` for reconstruction during differentiation.
This current code pin is not an asserted original-paper release.

Replacing Ouro's residual stream with Reformer coupling changes the function.
No inspected source proves an exact conversion for the released weights.
Reformer attention approximation is also outside this audit's exact-KV contract.

## 5. Invertible residual maps do not give a free inverse

[Invertible Residual Networks](https://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf), ICML 2019, supplies a sufficient invertibility condition.
Section 2, Theorem 1 requires `Lip(g)<1` for blocks `F(x)=x+g(x)`.
The paper explicitly states that this condition is not necessary.
Algorithm 1 evaluates `x^(k+1)=y−g(x^k)` to recover an input from an available output.
Its error bound depends on the contraction constant and iteration count.

The official code pin is `630e886bcd50cc21dee0769d1a5b35100d3ff569`.
[`conv_iResNet.py`](https://github.com/jhjacobsen/invertible-resnet/blob/630e886bcd50cc21dee0769d1a5b35100d3ff569/models/conv_iResNet.py#L56) constructs spectrally normalized residual blocks.
Its inverse evaluates the residual repeatedly from the supplied output.
This is not evidence that Ouro satisfies the condition or lacks an inverse.
Enforcing a new weight constraint would violate the unchanged-weight contract.
No exact finite-R4 forward shortcut follows from the inspected inverse algorithm.

## Factorization and finite-precision checks

An invertible basis change `z=P h` gives `z'=P F_r(P^-1 z,C_r)`.
It does not remove the current-input nonlinear evaluation by itself.
Replacing this expression with a matrix power requires an additional exact closure identity.
Ouro still contains norms, attention, SiLU, products, and residual updates between repeated matrix uses.
Current attention coefficients depend on the actual current query and key operands.
Freezing them from another state creates an approximation or a solver guess.

Function composition is associative, but a balanced composition object need not provide cheap composition evaluation.
The CPU example constructs a height-two expression for four calls to `F(x)=x²+1`.
Evaluation still consumes inputs `1,2,5,26` in order and returns 677.
This is a counterexample to syntax-only parallelism, not a lower bound for every algebraic implementation.

Real-arithmetic invertibility and implemented BF16 recovery are separate claims.
The CPU rounding example maps BF16 inputs 0 and `2^-8` to the same rounded result after addition of 1.
This uses BF16 round-to-nearest-even semantics, not a model or kernel observation.
It does not prove that the full Ouro transition collides on campaign inputs.
Any exact inverse must preserve required implementation information or supply another sufficient certificate.

Conversely, normalization is not automatically noninvertible in real arithmetic.
For positive epsilon and nonzero gains, RMSNorm has an inverse on its range:

```text
u_i = y_i / gain_i
a = mean(u_i²), a < 1
x_i = u_i sqrt(epsilon / (1-a))
```

The CPU positive control checks this identity with exact rational operands and a perfect-square scale.
It establishes no full-core or BF16 inverse.
Finite BF16 state spaces also admit formal exhaustive transition tables.
Their existence is not evidence of a practical cheap summary.
Thus this audit does not infer universal impossibility from smooth nonlinear dynamics or real-arithmetic arguments.

## Operand screen and reopening conditions

| Mechanism screen | Required available operand | Missing exact mechanism | Outcome |
|---|---|---|---|
| Affine or Koopman prefix | Input lift and composable coefficients before the missing states | Cheap certified closure and complete KV/state decoder for the actual read context | `NO_QUALIFYING_TRANSFORM` |
| Inverse or bidirectional propagation | Authoritative later boundary and matching read state | A way to obtain that boundary without prediction or the original forward chain | `NO_DEPENDENCY_REMOVAL` |
| Coarse/fine multiple shooting | Cheap coarse sweep and provisional fine states | A cheaper exact correction with authoritative KV | Existing C05 family, no new card |
| Pure basis change or composition tree | Current exact operands | A cheaper closed composition representation | Representation change alone |

A future reopening must provide an explicit identity on the stated input domain.
It must identify which predecessor wait disappears.
It must reconstruct all current KV, original gate operands, and publication state.
It must include transform construction, inverse, correction, communication, scratch, validation, and commit costs.
Its baseline must permit the same transform on an aliased generic unroll and same-P TP of the resulting legal frontier.
Shared weights can amortize valid preprocessing, but that saving needs explicit construction and capacity costs.

No inspected source or CPU check supplies the missing Ouro-specific identity.
This is a scope-limited obstruction for the examined transfers.
It does not reject an undiscovered exact factorization or a proved restricted-domain certificate.

## Reproduction, records, and actual time

```bash
python scripts/check_state_transform_frontier.py
```

The [CPU artifact](../artifacts/state_transform_frontier_cpu_v1.json) records six passing checks and the script hash.
These are algebra and interpretation checks, not model correctness, quality, traffic, or latency results.
The [source ledger](../artifacts/state_transform_frontier_sources_v1.jsonl) preserves versions, code pins, locators, and reading limits.
The [search ledger](../artifacts/state_transform_frontier_searches_v1.jsonl) records the actual new queries.
The [decision record](../artifacts/state_transform_frontier_decision_v1.json) records the verification cutoff.

Worker start: 2026-10-19 15:08:37 UTC, which is 2026-10-20 00:08:37 KST.
Verification cutoff: 2026-10-19 15:16:46 UTC.
The worker segment spans 489 seconds through that cutoff.
Metadata publication follows the cutoff.
Concurrent worker intervals do not add campaign elapsed time.
