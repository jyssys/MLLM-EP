# Certified operator screen

Status: bounded primary-source audit complete. GPU execution, Ouro execution, and implementation-specific numerical certification are `NOT_RUN`.

## Outcome

This screen found no verified, profitable certificate that removes dense Ouro SwiGLU or unmasked attention work without changing its computation. This is a scope-limited finding, not an impossibility result.

TEAL, CATS, and WINA provide training-free pruning mechanisms. They can discard nonzero contributions. Their quality evidence does not establish exact operator values, original-gate decisions, or persistent KV values.

FlashAttention provides the stronger generic exact baseline for attention evaluation. Its exactness theorem concerns the mathematical attention operator. It does not promise bitwise identity across different floating-point kernels.

A primary SSA repository provides explicit attention bounds and exact-output exceptions. Its numerical reference does not provide a directed-rounding proof. It is an author-maintained research artifact, not a verified published paper in this screen.

Finite-precision zeros remain possible. BF16 underflow, rounding, and kernel behavior can produce exact zeros even when the real-arithmetic result is nonzero. A skip needs a conservative certificate for the actual implementation. This worker measured neither an Ouro certificate nor a profitable zero frequency.

## Sources and versions

The [source and search notes](../artifacts/certified_operator_sources.jsonl) preserve identities, versions, code revisions, access windows, and rejected leads.

| ID | Primary source | Latest version observed | Pinned public code |
|---|---|---|---|
| F01 | [Training-Free Activation Sparsity in Large Language Models](https://arxiv.org/html/2408.14690v3) | 2408.14690v3, 2025-02-25 | [TEAL, fb7373c](https://github.com/FasterDecoding/TEAL/tree/fb7373c93ac3594817c9ee64d4e08b47430a1822) |
| F02 | [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](https://arxiv.org/html/2404.08763v4) | 2404.08763v4, 2024-11-03 | [CATS, 679959c](https://github.com/ScalingIntelligence/CATS/tree/679959c3323790be0a7d7d9af78e0580f8b88ed8) |
| F03 | [WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference](https://arxiv.org/html/2505.19427v2) | 2505.19427v2, 2026-02-18 | [WINA, f31fcc3](https://github.com/microsoft/wina/tree/f31fcc3b27663b2bca066fde7ac1a7e85380d50d) |
| F04 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2) | 2205.14135v2, 2022-06-23 | [FlashAttention, e7a6402](https://github.com/Dao-AILab/flash-attention/tree/e7a6402a9ff099de5eb628ce207a88a3f0aa5378) |
| F05 | [Subquadratic Sparse Attention, repository manuscript](https://github.com/jonsmirl/ssa/blob/0613e78d65db5b90ddba840d069fa66307faefed/paper/subquadratic_attention.md) | Repository revision, 2026-09-11. Published-paper identity not established. | [SSA, 0613e78](https://github.com/jonsmirl/ssa/tree/0613e78d65db5b90ddba840d069fa66307faefed) |

All five sources had readable method bodies and public code at inspection. This worker inspected code but executed none of these repositories.

## Mechanism audit

### F01: TEAL removes small values, not only exact zeros

Section 4.2, Definition 1 sets `s_t(x_i)=0` when `|x_i| <= t`. An empirical activation distribution supplies the threshold. A positive threshold can remove a nonzero value. This is approximate pruning without new training, not a zero certificate.

The [splitk_sparse_gemv_kernel](https://github.com/FasterDecoding/TEAL/blob/fb7373c93ac3594817c9ee64d4e08b47430a1822/kernels/sparse_gemv.py#L44) masks weight loads with `abs(x0) > threshold`. It accumulates in FP32 and allocates FP16 output. `SparseGEMV.forward` selects this path for one-token decode and dense matmul otherwise. Replacing its threshold with zero does not, by itself, prove equality with the original dense reduction.

### F02: CATS computes the gate before selective up/down work

Equations 4–5 replace SiLU output with a thresholded activation. The custom-kernel algorithm first computes `v=SiLU(x W_gate)`. It then masks the up and down weights. The expensive gate projection already occurred before this decision.

The [SparseSiLU.forward](https://github.com/ScalingIntelligence/CATS/blob/679959c3323790be0a7d7d9af78e0580f8b88ed8/experiments/models/sparse_mistral/sparse_silu.py#L242) implementation applies `nn.Threshold` to both activation signs. The code zeros the equality boundary, whereas Equation 4 retains that boundary. This difference needs an explicit choice in any reproduction. Neither rule proves unchanged dense-model outputs at positive thresholds. The paper separates direct inference from additional fine-tuning.

### F03: WINA bounds approximation under stated assumptions

Equation 4 keeps the largest `|x_i| ||W[:,i]||_2`. Lemma 3.1 gives optimal approximation for a column-orthogonal linear matrix. Theorem 3.2 minimizes an error upper bound for column-orthogonal linear networks. It does not prove zero error for nonlinear SwiGLU transformers.

The paper also describes an offline orthogonal tensor transformation. Its functional-capacity statement is not a bitwise certificate for unchanged stored weights. The [MLP code](https://github.com/microsoft/wina/blob/f31fcc3b27663b2bca066fde7ac1a7e85380d50d/wina/mlp.py#L121) masks current inputs before projections and masks intermediate states before the down projection. It does not certify that every removed contribution equals zero. Quantization compatibility remains separate from exactness.

### F04: FlashAttention preserves the operator through online reduction

Algorithm 1 combines block maxima, exponential sums, and weighted outputs. Theorem 1 returns `softmax(Q K^T) V` with `O(N^2 d)` FLOPs. Theorem 2 reduces HBM traffic. This is not a proof that positive, unmasked contributions can disappear.

The pinned [forward kernel](https://github.com/Dao-AILab/flash-attention/blob/e7a6402a9ff099de5eb628ce207a88a3f0aa5378/csrc/flash_attn/src/flash_fwd_kernel.h#L78) limits tile ranges with causal or local masks. These are structural exclusions. The [softmax code](https://github.com/Dao-AILab/flash-attention/blob/e7a6402a9ff099de5eb628ce207a88a3f0aa5378/csrc/flash_attn/src/softmax.h#L118) uses FP32 state, `exp2f`, and online rescaling. A new numerical certificate must match that actual arithmetic, not only the real-valued formula.

### F05: explicit bounds do not turn positive error into exactness

Manuscript Section 5.7 bounds omitted mass and output error from key/value block summaries. The [reference](https://github.com/jonsmirl/ssa/blob/0613e78d65db5b90ddba840d069fa66307faefed/ssa/certified_attention.py#L49) builds an immutable NumPy snapshot. It charges summary construction, query bounds, ordering, and value checks separately. It has no GPU kernel or incremental-cache implementation.

`_positive_exp` prevents underflow from satisfying a zero-mass tolerance. `read` can return `certified=False` at a block cap. Equal omitted values can yield zero output error despite positive omitted mass. Its floating-point guard explicitly does not establish directed-rounding certification. Thus this source supplies useful equations, not a certified Ouro kernel or established speedup.

## Analytic screen for unchanged dense operators

The following statements are `ANALYTIC_IDENTITY` or stated sufficient conditions. They are not model measurements.

### SwiGLU

For one intermediate channel, write:

```text
g_j = w_gate,j dot x
u_j = w_up,j dot x
z_j = SiLU(g_j) u_j
y   = W_down z
SiLU(a) = a / (1 + exp(-a)).
```

For finite real `a`, SiLU is zero exactly when `a=0`. Unlike ReLU, it has no negative half-space of exact zeros. This real-arithmetic fact does not exclude machine-level zeros.

An established current zero can remove a channel contribution under appropriate finite-input and arithmetic conditions. An implementation must also preserve relevant NaN, infinity, signed-zero, and reduction behavior. For example, `0 * infinity` is not a finite zero. A numerical zero test after gate evaluation cannot recover the gate computation already spent.

For omitted channels `D`, the triangle inequality gives:

```text
||W_down (z - z_hat)||_2 <= sum_{j in D} |z_j| ||W_down[:,j]||_2.
```

This bound needs no learned predictor. A positive bound permits error rather than proving equality. Zero terms or a separate cancellation certificate can give a stronger conclusion.

An interval wholly inside one rounding cell can establish a rounded value only when its error bounds match the actual kernel. Accumulation precision, exponential approximation, cast position, rounding mode, and subnormal handling matter. This screen establishes none of those kernel-specific bounds.

### Attention

For finite real unmasked logits, every softmax probability is positive. A masked `-infinity` logit is a different case. Positive support does not rule out exact output cancellation or equal value vectors.

Let `S` be the retained set, `D` the omitted set, and `o_hat` the normalized retained output. Direct expansion gives:

```text
o - o_hat = sum_{i in D} exp(a_i) (v_i - o_hat) / (Z_S + Z_D).
```

This residual can vanish even when `Z_D > 0`. A valid certificate must prove that fact, rather than assume small mass means zero error. It can also prove that a numerical contribution rounds to zero under a specified implementation. Online rescaling means a final-exponential argument alone may not suffice.

The generic bound `Z_D <= A` gives omitted mass at most `A/(Z_S+A)`. A positive result is an error certificate, not operator equality. A top-key certificate alone does not bound the total mass of many omitted keys.

## Can recurrence reuse a certificate?

Tied weights permit reuse of static column norms. An unchanged historical KV store permits reuse of its summaries. Neither fact makes current activations or current attention scores equal to their previous-depth values.

For a known current drift bound, Cauchy–Schwarz gives:

```text
||x_new - x_old||_2 <= epsilon
|w dot x_new - w dot x_old| <= ||w||_2 epsilon.
```

A ReLU zero survives if the resulting upper bound remains nonpositive. This example does not transfer to SiLU. A nonpositive SiLU input need not produce zero. A zero linear value also need not survive nonzero input drift. The CPU example kept weight 1 fixed while input changed from -0.1 to 0.1. Tied weights did not preserve the activation sign.

For a fixed key block, let `K_max` bound its key norms. A valid old upper-score bound can expand by `epsilon_q K_max` under query drift. If keys also change, the bound needs all three drift terms:

```text
||delta_q|| ||k|| + ||q|| ||delta_k|| + ||delta_q|| ||delta_k||.
```

Ouro's different depth caches need their own summaries or proved relations between caches. Current-token appends also change the store. A prior-depth certificate cannot silently stand for the new query, new KV, or expanded causal prefix.

The certificate must include the cost of its current drift bounds. It must also include summary maintenance, masking, gathers, checks, synchronization, and dense fallback. Repeated weights alone establish no favorable R4 amortization or recurrence-specific novelty.

## Greedy decisions are a weaker target

Suppose a full logit certificate proves `||logits - logits_hat||_infinity <= E`. A unique approximate winner with margin greater than `2E` has the same argmax. This condition does not preserve probabilities, sampled outputs, intermediate states, or KV tensors.

The checkpoint gate has its own threshold decisions. A token-output margin certificate does not certify those gate decisions. Approximate committed KV can affect later tokens even when the current greedy token matches. A same-gate, same-KV claim therefore needs stronger certificates or exact repair before commit.

## CPU sanity checks and limits

The worker ran a standard-library Python check at 11:07:49 UTC. It used seed 0 and 100 small attention-bound trials. Every trial satisfied the tested mass and output inequalities within a `1e-12` assertion tolerance. The minimum observed output-bound slack was `0.37737913962509273`.

The scalar examples returned `SiLU(-1)=-0.2689414213699951` and `SiLU(-0.01)=-0.00497500020833125`. Threshold 0.1 removes the latter nonzero value. With logits `[0,0]` and values `[0,2]`, dense output was 1 and first-key output was 0. Equal values `[3,3]` instead gave 3 in both cases.

These are `ANALYTIC_SANITY_CPU` results. They are not directed-rounding proofs, repository reproduction, model-quality results, or GPU timings. The notes preserve the exact output and reproduction code. This worker measured neither BF16 zero frequency nor a profitable Ouro skip rate.

## Rejected lead and disposition

A search result associated “Certified Logit-Faithful Sparse Tracing via Softmax Tail Bounds” with arXiv 2510.19875. The [primary record](https://arxiv.org/abs/2510.19875) instead identifies *Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention*. The alleged certificate title is a rejected identity mapping. It does not count among the five qualifying sources.

The scope-limited disposition is:

- Structural mask skipping and exact attention evaluation are generic baseline mechanisms.
- Positive-threshold pruning changes the operator, even without training or a learned predictor.
- Positive local error bounds do not prove unchanged gate, KV, or sampling behavior.
- Exact cancellation and implementation-specific zero certificates remain possible but unimplemented here.
- A new Ouro candidate remains blocked on a conservative certificate and measured profitable cases, not on a mathematical impossibility claim.

## Time accounting

This worker segment ran from 2026-10-19T11:02:28Z to the accounting cutoff at 2026-10-19T11:10:54Z. Its observed duration was 506 seconds, or 8 minutes 26 seconds. Worker minutes overlap the coordinator's elapsed time. They do not add campaign elapsed hours. No GPU job, environment change, installation, model change, or deliberate wait occurred. Final stamping and validation after the cutoff are outside this duration.
