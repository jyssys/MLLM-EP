# C15: Attention from stored normalized inputs

Worker: `/root/independent_axes`.
Segment began: 2026-10-19 10:55:43 UTC.
This report records algebra, source inspection, and deterministic CPU controls.
It contains no GPU measurement or model-quality result.

## Outcome

The proposed representation can halve unsharded logical state by storing each normalized attention input X instead of K and V.
Reassociation avoids full historical projections in a no-RoPE counterfactual.
It does not make that counterfactual cheap for Ouro's 16 distinct MHA heads.
Each head needs a separate 2048-wide weighted-X summary.

Actual Ouro RoPE prevents one position-independent absorbed query from scoring all historical X vectors.
The direct position-specific rewrite restores a history-sized dense contraction.
Retaining rotated K avoids that problem, but K plus X uses the same unsharded storage as K plus V.
Head-TP replication can increase its distributed storage further.

The static RoPE rewrite is algebraically invalid.
The legal retained-K variant is generic and lacks the claimed capacity saving.
No arm merits a rescue engine from this screen.
This conclusion does not reject every exact latent-cache or communication-avoiding algorithm.
The complete candidate card is [C15](../candidates/C15_residual_input_reassociation.md).

## 1. Changed assumption relative to C10

C10 stores projection inputs and rebuilds historical K/V with full dense projections.
That screen charges history-sized projection work.
C15 tests a different algebraic representation rather than repeating that implementation.

The changed assumption is that key projection can move onto the query.
Value projection can move after the weighted sum.
The smallest falsifier is a RoPE score identity test plus head-aware work accounting.
This is not stale-state replacement, low-rank truncation, or a fitted predictor.
The budget is one bounded source/algebra screen with no engine build.
The screen stops when the proposed absorbed query fails or the retained-state benefit disappears.

## 2. Actual Ouro attention

The Base checkpoints use D=2048, H=16, d=128, and 16 KV heads.
Thus, `D=H*d` and the attention is MHA rather than GQA.
The physical layer counts are 24 and 48.
The fixed recurrent count is R=4.
The source hashes match across the two inspected model revisions.

X is the output of `input_layernorm` immediately before attention projection.
It is not one token embedding shared across all physical layers and recurrent depths.
Each X needs its own request, token, recurrent depth, physical layer, and version.

The attention source applies bias-free Q/K/V projections.
It applies RoPE to projected Q and K, but not V.
It then writes the rotated K and unrotated V into the depth-indexed cache.
The eager path computes scaled scores, FP32 softmax, and a cast back to the query dtype.
It multiplies these probabilities by V, joins the heads, and applies O projection.
Other backends can use different arithmetic paths.

The RoPE helper uses split-half rotation.
For a vector `[a,b]` with equal-sized halves, it uses `rotate_half([a,b])=[-b,a]`.
The rotary module computes frequencies and trigonometric values in FP32.
It returns cosine and sine in the input dtype.
The inspected configs use theta=1,000,000 and no RoPE scaling.
This model description adds no query/key normalization.

The installed vLLM path also projects QKV before calling the rotary module.
It selects separate attention/cache modules by recurrent depth.
This source order confirms that the RoPE obstacle applies to the actual layout.

## 3. Exact real-arithmetic equations

Suppress request, physical-layer, and recurrent-depth indices for this section.
Let X_j be the D-wide normalized input at historical position j.
For head h, let Wk_h and Wv_h have shape d by D.
Let q_h be the already-rotated current query.

`K_h,j = R_j * Wk_h * X_j`

`V_h,j = Wv_h * X_j`

`score_h,j = q_h^T * K_h,j / sqrt(d)`

`alpha_h = softmax(score_h + mask_h)`

`out_h = sum_j alpha_h,j * V_h,j`.

All equations retain the original causal mask and every logical KV identity.

### Value-side positive control

For the actual alpha values, linearity gives:

`out_h = Wv_h * Z_h`, where `Z_h = sum_j alpha_h,j * X_j`.

This identity remains valid with RoPE because V does not rotate.
It does not move softmax across a product.
It preserves the already computed head-specific alpha.

The H heads usually have different alpha vectors.
Therefore, this representation needs H separate D-wide summaries Z_h.
One shared Z does not produce all head outputs in general.
The CPU control includes a counterexample to that collapse.

### Key-side no-RoPE positive control

If `R_j=I`, define `u_h=Wk_h^T*q_h` once per query.
Then `score_h,j=u_h^T*X_j/sqrt(d)` for every position j.
This identity avoids historical K projection.
It increases the score dot width from d=128 to D=2048 for each head.
This is a mathematical control, not a permitted change to Ouro's RoPE.

### Actual RoPE

For Ouro, the valid query transform is:

`u_h,j = Wk_h^T * R_j^T * q_h`.

It depends on historical position j.
Using one u_h for all j drops the required position transform.
Moving R_j past a generic dense Wk_h is also invalid.

The direct valid rewrite computes u_h,j separately.
It costs `N*H*D*d = N*D^2` projection multiplications.
It therefore returns to history-sized key-side dense work.
This count describes that direct implementation.
It is not a lower bound for every possible structured algorithm.

The CPU noncommutation example uses `W=diag(2,1)` and a 90-degree rotation R.
For q=`[1,0]` and X=`[1,1]`, the direct score is -1.
The valid expression `X^T*W^T*R^T*q` also gives -1.
The invalid expression `X^T*R^T*W^T*q` gives -2.

## 4. Charge current-token projections too

The following counts cover one new query at one logical attention layer.
They include the current Q/K/V/O projections or their stated replacements.
They exclude softmax, RoPE/scaling arithmetic, norms, metadata, and runtime costs.
They count scalar multiplications, not measured GPU operations or latency.

The cached baseline costs `4D^2 + 2ND` multiplications.
The four D-squared terms are current Q, K, V, and O projections.
The `2ND` term counts score and value contractions on cached tensors.

For X-only without RoPE, the count is `4D^2 + 2HND`.
The absorbed query and projected summary replace the baseline's current K/V projections.
This avoids double-counting those dense costs.
The H full-width score and weighted-X contractions remain.

For rotated K plus X, the count is `4D^2 + (H+1)ND`.
The ordinary score dot remains d-wide.
The value-side weighted input becomes D-wide for every head.
The summary projection replaces the new-token V projection.

For direct position-specific X-only RoPE, the count is `(N+3)D^2 + 2HND`.
The N dense transforms replace current K/V projection work as specified above.

| N | Cached baseline multiplications | X-only, no-RoPE ratio | Rotated K plus X ratio | Direct X-only RoPE ratio |
|---:|---:|---:|---:|---:|
| 1024 | 20,971,520 | 4.0000 | 2.5000 | 208.6000 |
| 4096 | 33,554,432 | 8.5000 | 4.7500 | 520.3750 |
| 8192 | 50,331,648 | 11.0000 | 6.0000 | 693.5833 |
| 16384 | 83,886,080 | 13.0000 | 7.0000 | 832.1500 |

These ratios describe arithmetic counts for the listed algorithms.
They are not slowdowns measured on a GPU.
They also do not prove that no memory-constrained operating point can benefit.

For contraction-only accounting, the no-RoPE ratio is `H+D/N`.
That ratio excludes current-token baseline projections and cannot describe whole attention.
The whole-attention table uses a stronger and fairer boundary.

An obvious exact fallback rebuilds only K and keeps the value-side reassociation.
Its count is `(N+3)D^2+(H+1)ND`, below the direct position-specific query path.
Its ratios are 207.1/516.625/about 688.5833/826.15 at the same four N values.
This fallback still performs history-sized dense work.
It reduces C10's leading full-K/V reconstruction count, but does not eliminate that cost family.
No claim depends on the more expensive query-side path being the best algorithm.

Absorbing V projection into O projection also needs accounting.
It forms H separate D-by-D products `Wo_h*Wv_h` if materialized densely.
Their weights and dense application cost grow to H*D-squared rather than the original two D-squared factors.
Retaining the rank-d factors recovers the factored cost already counted above.

The new summaries contain `H*D=32,768` values rather than H*d=2048.
FP32 summaries therefore require 128 KiB before partitioning or tiling.
Ordinary head-output accumulators require 8 KiB at that same aggregate boundary.
A kernel can tile these values, but tiling does not remove their arithmetic.

## 5. Storage and traffic are different

For one token and one logical layer, the baseline K/V state contains `2D` BF16 elements.
It occupies 8192 bytes.
X alone occupies 4096 bytes.
Rotated K plus X occupies 8192 bytes.
Thus, the retained-K variant provides no unsharded state reduction.

Full depth-indexed R4 capacity is:

`KV = 2*L*4*N*D*2 bytes`.

`X = L*4*N*D*2 bytes`.

| Model | N | Full KV, GiB | X only, GiB | Rotated K plus X, GiB |
|---|---:|---:|---:|---:|
| 1.4B, L24 | 1024 | 0.75 | 0.375 | 0.75 |
| 1.4B, L24 | 16384 | 12 | 6 | 12 |
| 2.6B, L48 | 1024 | 1.5 | 0.75 | 1.5 |
| 2.6B, L48 | 16384 | 24 | 12 | 24 |

These capacities exclude weights, metadata, allocation overhead, and scratch.
They do not measure HBM traffic.
They retain every required depth/layer X instead of assuming one X serves the whole model.

An ideal fused X-only kernel might read one X tile and reuse it across local heads.
That could reduce bytes relative to separate K and V arrays.
A two-pass implementation can instead read X once for scores and again for values.
Its input bytes then equal the ordinary K/V total before other traffic.
Separate head kernels can read X repeatedly.
No traffic reduction follows from the capacity count alone.

## 6. Distributed ownership changes the memory result

Ordinary head TP stores `2ND/P` KV elements per rank.
A head-local worker needs all D input coordinates to form its weighted-X summary.
Replicating X therefore stores ND elements per rank.
Rotated K plus replicated X stores `ND/P + ND` elements per rank.

| P | Replicated X / baseline KV per rank | Rotated K plus replicated X / baseline KV per rank |
|---:|---:|---:|
| 2 | 1.0 | 1.5 |
| 4 | 2.0 | 2.5 |
| 8 | 4.0 | 4.5 |

P8 is a capacity projection only.
The same ratios apply to aggregate bytes across ranks under this replicated-X design.
X-only loses its aggregate capacity advantage already at P2.

Feature-sharding X can preserve its half-sized aggregate state.
It creates a different communication plan.
In the direct plan, ranks compute partial scores across the D feature dimension.
They reduce those scores before the original softmax.
The logical FP32 score array contains H*N values per query and logical layer.
It occupies 64/256/512/1024 KiB at N=1K/4K/8K/16K.
The collective moves additional wire bytes.

Value reconstruction also needs projection-partial assembly or an equivalent output layout.
That plan must include weight redistribution and the ordinary O projection.
Alternatively, gathering all historical X moves history-sized state.
Neither alternative gives free communication avoidance.
A comparison needs a conventional executor with the same feature/head layout.

No current-token X can enter the historical store before its actual normalization completes.
Adaptive `last_exited` propagation must copy or reference the same selected source version as the baseline.
An exit decision is not known before its gate computation.
The fixed-depth capacity screen assumes no adaptive slot reduction.

## 7. BF16 breaks unrestricted reassociation

Real linear identities do not preserve arbitrary intermediate casts.
The cached baseline stores rounded projected K and V.
The reassociated path does not necessarily reproduce those rounded tensors.
The eager softmax path also has its own dtype boundaries.
An implementation must preserve or explicitly test these differences.

The value control uses W=`[1,1]` and X rows `[1,1/256]`, `[1,3/512]`.
It uses attention weights `[1/2,1/2]`.
All input values are exactly representable in BF16.
The cached projected values round to 1 and 1.0078125.
Their weighted output rounds to 1.
Projecting the exact weighted X instead rounds to 1.0078125.

The key control uses the same W and X=`[1,1/256]` with q=1.5.
The cached projected K rounds to 1.
The score then rounds to 1.5.
The reassociated score rounds to 1.5078125.
These controls use explicit software rounding, not an H100 kernel.
They reject a universal bitwise-equivalence claim, not a model-quality claim.

The intended exactness class is at best `Numerically reordered` until backend tests establish the required contract.
Original-gate values, exit depths, KV commits, and free-running outputs need mode-matched tests.
No such model test ran for C15.

## 8. Primary near priors

DeepSeek-V2 describes key/value matrix absorption and the RoPE noncommutation obstacle in sections 2.1.2–2.1.3.
Its solution uses a trained low-dimensional latent and a separate positional branch.
That architecture is not the released Ouro MHA graph.
Changing Ouro to decoupled or partial RoPE is outside this screen.
[DeepSeek-AI, arXiv:2405.04434v5](https://arxiv.org/html/2405.04434v5).

The official DeepSeek-V3 implementation supplies a direct generic code control.
Its absorbed path caches a latent and a separate positional key.
It transforms the non-positional query, computes attention, forms a head-specific latent summary, and projects that summary to values.
This is not newly invented recurrence execution.
[Pinned author code](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L478).

The residual-stream paper reconstructs layer-specific K/V from the matching residual input in equations 13–15.
Its reconstruction identity is relevant generic prior art.
Its later single-checkpoint storage formula does not justify deleting C15's per-layer/per-depth inputs without a reconstruction schedule.
Its reported bit identity also assumes preserved operation order.
C15 changes that order.
[Qasim et al., arXiv:2603.19664v1, sections 3.2–3.4](https://arxiv.org/html/2603.19664).

Efficient Attention uses association to reduce dot-product attention cost.
Its exact proof uses scaling normalization.
Its softmax factorization is approximate, not unchanged Ouro softmax attention.
C15's value identity does not require that approximation because it retains the original alpha.
[Shen et al., arXiv:1812.01243v10, section 3.3](https://arxiv.org/pdf/1812.01243).

MHA2MLA changes RoPE and uses low-rank approximation plus fine-tuning.
It is useful nearby work, but its conversion is outside this training-free, unchanged-RoPE contract.
[Ji et al., ACL 2025](https://aclanthology.org/2025.acl-long.1597/).

No source-reported latency, quality, or compression result becomes an Ouro result here.

## 9. Deterministic CPU results

Run:

`python poc_ouro_parallelism_deep_discovery/scripts/residual_input_reassociation_cpu.py`

The script uses two constructed heads with d=2 and D=4.
The RoPE case uses historical positions 0/1/2 and query position 3 for one frequency pair.
The separate integer rotation example isolates noncommutation.
It imports only the standard library.
It uses no random tensors or model weights.

| Control | Result | Meaning |
|---|---|---|
| No-RoPE absorbed scores | Matched to float64 tolerance | Valid positive control only |
| Position-specific RoPE scores | Maximum absolute errors 0 and 0 | The position-specific formula is valid |
| Static absorbed RoPE scores | Maximum absolute errors about 3.4480 and 0.9904 | One static transformed query is invalid |
| Value reassociation with RoPE | Maximum absolute errors about 7.63e-17 and 0 | Original head-specific alpha preserves the real identity |
| One shared summary | Failed the constructed head-output test | Keep separate Z_h values |
| BF16 K/V boundaries | Both explicit rounding counterexamples differ | No universal bitwise guarantee |

All six assertion groups passed.
The work and memory tables use the actual Ouro dimensions, separate from the tiny tensors.
The raw artifact is [residual_input_reassociation_cpu.json](../artifacts/residual_input_reassociation_cpu.json).
Evidence class: `CPU_ALGEBRA_CONTROL_AND_STRUCTURAL_ACCOUNTING`.
Latency, achieved traffic, and model quality: `NOT_RUN`.

## 10. Decision and reopening rule

Reject the single-query absorption arm for actual Ouro RoPE.
Retain no-RoPE only as a positive algebra control.
The legal K-plus-X arm is generic and provides no unsharded state reduction.
Its direct arithmetic count also increases.
These facts do not establish a GPU slowdown, but they remove the proposed cheap memory-saving mechanism.

A reopening needs a concrete exact representation that handles every historical position.
It must reduce state after head/feature ownership and preserve the original softmax and KV semantics.
It must charge all H summaries, current-token replacements, and BF16 boundaries.
Novelty tests must include aliased logical unrolling and a generic absorbed-attention implementation.
This audit recommends no rescue engine.

## 11. Source and time ledger

Two changed query batches searched the live literature during this segment.
The artifact preserves all query strings and source locators.
Two tool scripts initially failed to parse before any nested tool ran.
The revised scripts then ran the intended read-only source calls.
The CVF PDF endpoint failed, so the arXiv primary PDF supplied Efficient Attention.

Official model revisions:

- 1.4B: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- 2.6B: `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
- Model source SHA256 for both: `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.

Relevant source anchors are `OuroAttention.forward`, `apply_rotary_pos_emb`, `OuroRotaryEmbedding.forward`, and `OuroDecoderLayer.forward`.
The official DeepSeek-V3 code revision is `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`, dated 2025-08-28.
The source inspection did not install or execute that code.

This worker used no GPU, process control, installation, downloaded weights, or git mutation.
Concurrent worker time must not multiply campaign elapsed time.
The final artifact records the actual CPU run and end checkpoint.

The recorded end checkpoint is 2026-10-19 11:06:35 UTC.
The checkpoint span is 652 seconds.
The final CPU run occupied 2026-10-19T11:06:34.370326+00:00 through 2026-10-19T11:06:34.370693+00:00.
This duration is not a model or GPU latency result.
The JSON parsed and both report files passed the prose test with zero findings.

Script SHA256: `ae7459b6d06fbccede4c23af353aaf6ef86135c47142a08ce961dddf677a2f50`.
