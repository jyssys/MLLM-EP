# Layout propagation and structured residual controls

Status: `PASS_CPU_ALGEBRA_CONTROLS`. This work establishes no new architecture or speedup.

The tested local-normalization rewrite is invalid. Correct feature-sharded layouts remain possible when they preserve global statistics and dense-projection dependencies. Exact base-plus-delta algebra also remains possible in real arithmetic. It does not remove a dense projection without additional structure.

The [script](../scripts/layout_residual_cpu.py) and [result artifact](../artifacts/layout_residual_cpu.json) contain the constructed tensors and outputs. These are CPU algebra controls. They are not model-quality, model-convergence, GPU-performance, or H100-equivalence results.

## Scope, precision, and activity

This segment started at `2026-10-19T10:12:07Z`. It read the pinned public layer source and official generic controls. It created only the three assigned files. It did not change earlier artifacts.

The default Python had no `torch` package. No package installation followed that check. The implementation uses Python binary64 operations and an explicit software BF16 rounding model.

The first successful script execution ran from `2026-10-19T10:16:30.243022+00:00` to `2026-10-19T10:16:30.244948+00:00`. A later execution added a cached-base rounding control. The saved artifact records its own actual execution times. These execution intervals are provenance, not inference benchmarks.

The BF16 path rounds FP32 values to BF16 with round-to-nearest, ties-to-even. It uses declared FP32 product and accumulation steps. Its norm path retains the source's intermediate BF16 cast before gamma multiplication. It does not reproduce a particular GPU reduction tree, square-root instruction, fused kernel, or NCCL implementation.

All arrays are explicit, small, constructed inputs. No random weights or checkpoint tensors appear. The causal control uses two tokens, width 4, two heads, MLP width 6, two physical layers, and two loops. It is a structural arithmetic control, not a small substitute for Ouro quality evaluation.

## Pinned operator sequence

The source revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. The relevant functions are `OuroDecoderLayer.forward`, `OuroMLP.forward`, `OuroAttention.forward`, `OuroRMSNorm.forward`, and `OuroModel.forward`. [Pinned source](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py).

The source has four RMSNorm modules inside each decoder block. It normalizes each sublayer's output before residual addition. It also normalizes the stack output after each recurrent pass. The config selects SiLU, so the gate/up product is SwiGLU. [Pinned config](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json).

Write the real-arithmetic normalization as `N_gamma(z)=s(z)*D_gamma*z`, where `s(z)=(||z||²/d+epsilon)^(-1/2)`. Each norm has its own gamma. The formulas retain epsilon.

For a block input x, the exact intended sequence is:

```text
z1 = N1(x)
q,k,v = QKV(z1)
q,k = RoPE(q,k,position)
c = causal_attention(q,k,v,matching historical KV)
a = W_o c
u = x + N2(a)
z3 = N3(u)
g = W_gate z3
p = W_up z3
m = SiLU(g) elementwise_multiply p
b = W_down m
y = u + N4(b)
```

There is no invented q/k norm in this sequence. Attention scaling remains inside the attention operator. After the last physical block, the loop norm produces the next pass's input.

The two-layer/two-loop dependency is:

```text
x -> Block_0(r0) -> Block_1(r0) -> N_loop
  -> Block_0(r1) -> Block_1(r1) -> N_loop -> output
```

Physical weights repeat in the order `[0,1,0,1]`. The intermediate states and K/V identities do not repeat. The control computes separate causal K/V for each logical block. It does not simulate cached decode, an exit gate, sampling, or speculative commit.

## Replicated TP and feature-sharded layouts

Here, `row-TP partial` means a contribution to every output coordinate from a disjoint input partition. It does not mean a complete output-feature shard.

For `a=W_o*c`, partition c by attention heads. Rank j computes `a_j=W_o[:,J_j]*c[J_j]`. The complete output is `a=sum_j a_j`. Every `a_j` spans all d output coordinates.

The following layouts preserve the intended real-arithmetic graph. The feature-sharded control is illustrative, not a proposed best implementation.

| Consumer | Replicated 1D TP | Illustrative feature-sharded layout |
|---|---|---|
| N1 | Local norm on complete x | Reduce feature-square statistics, then normalize local features |
| Q/K/V | Complete input, head-local output rows | Gather normalized features before head-local projections |
| Attention | Each complete head uses its matching K/V | Same head-local contract |
| W_o output | Full-width row-partials, then all-reduce | Full-width row-partials, then reduce-scatter |
| N2 | Local norm after complete reduction | Global square statistic over disjoint completed feature shards |
| First residual | Add x exactly once | Add the matching x feature shard exactly once |
| N3 | Local norm on complete u | Global square statistic over disjoint u features |
| Gate/up projections | Split complete output neurons | Gather normalized features, then split complete output neurons |
| SwiGLU | Local operation on complete gate/up pairs | Same complete-neuron requirement |
| Down projection | Row-partials, then all-reduce | Row-partials, then reduce-scatter |
| N4 and residual | Normalize complete b, then add u | Reduce norm statistic, normalize b features, then add u features |
| Loop norm | Local norm on replicated y | Global statistic, then local feature scaling |

Residual ownership matters. Adding x to every rank's partial output and then summing creates `P*x`, not x. A valid partition assigns each residual feature once. An equivalent additive representation must preserve that accounting.

The toy uses a central CPU representation for all layouts. It checks their arithmetic. It does not execute messages or check device resource limits.

### Float64 control results

The script compares every block output and both loop boundaries. Across these six boundaries, the largest absolute difference is `1.7763568394002505e-15`. The comparison threshold is `1e-12`.

The largest difference occurs after physical layer 1 in the second loop. Both replicated TP and the feature-sharded arithmetic differ by that amount. The two paths preserve all 18 norm applications per token: 16 block norms and two loop norms.

This positive control prevents an overbroad conclusion. Feature sharding is not inherently invalid. Treating an unreduced additive contribution as a complete feature value is the invalid shortcut.

### Structural communication accounting

| Two layers × two loops | Vector all-reduces | Vector reduce-scatters | Vector all-gathers | Scalar-statistic all-reduces |
|---|---:|---:|---:|---:|
| Replicated 1D TP | 8 | 0 | 0 | 0 |
| Illustrative feature layout | 0 | 8 | 8 | 18 |

The feature layout starts and ends in feature-sharded form. Its initial/final interface costs are not a matched E2E comparison. Each statistical reduction contains per-token scalars, not necessarily one scalar for a whole batch.

Reduce-scatter followed by all-gather implements all-reduce semantics. Moving work between those operations does not prove that vector communication disappeared. [NCCL collective operations, documentation 2.32.3](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

A width-2048 BF16 vector contains 4096 bytes per token. This is a logical payload, not measured wire traffic. The counts above are not a communication lower bound or a latency estimate.

## Why local nonlinear evaluation fails

### RMSNorm needs cross-rank terms for additive partials

For overlapping full-width contributions a and b:

```text
||a+b||² = ||a||² + 2*a_dot_b + ||b||²
```

The script uses `a=[1,2]` and `b=[3,-1]`. The complete square norm is 17. The sum of local square norms is 15. The missing cross term is 2.

Thus, an all-reduce of each full-width partial's local square norm is insufficient. Disjoint completed feature shards have no such missing cross term. Their local square sums can form the correct global statistic.

The explicit local-normalization rewrite also fails:

```text
N([1,0]+[0,1]) != N([1,0]) + N([0,1])
```

With epsilon `1e-6`, the maximum absolute difference is `0.4142126481612789`. This counterexample rejects a universal equivalence claim for that rewrite. It does not reject algorithms that compute the required global statistic differently.

### SwiGLU needs complete preactivations

The control splits one gate preactivation into 1 and -1. It splits the corresponding up value into 2 and 3.

```text
SiLU(1-1)*(2+3) = 0
SiLU(1)*2 + SiLU(-1)*3 = 0.6552928931500245
```

The local nonlinear results cannot replace the nonlinear result after reduction. Standard column TP avoids this error. Each rank owns complete gate/up neurons, so it can apply SwiGLU locally.

### Softmax score partials differ from disjoint KV segments

The control uses score contributions `[2,0]` and `[0,1]`. Softmax after score addition gives `[0.7310585786300049,0.26894142136999516]`. Averaging separate softmax results gives `[0.5748692496739387,0.42513075032606124]`.

The maximum difference is `0.1561893289560662`. This rejects that score-partial rewrite. It does not reject exact attention-state merging over disjoint KV segments. That method retains the normalization information required for a correct merge.

## Positive real-arithmetic norm fusion

Nonlinearity does not require a separate full-vector materialization at every boundary. A small exact statistic can sometimes represent the missing dependency.

Consider `u=x+N_gamma(a)` followed by `N_eta(u)`. After reduce-scatter, x and the complete a have matching disjoint feature ownership. Define:

```text
M0 = sum_i a_i²
M1 = sum_i x_i²
M2 = sum_i x_i*gamma_i*a_i
M3 = sum_i gamma_i²*a_i²
s  = (M0/d + epsilon)^(-1/2)
||u||² = M1 + 2*s*M2 + s²*M3
t  = (||u||²/d + epsilon)^(-1/2)
N_eta(u) = t*D_eta*(x + s*D_gamma*a)
```

One reduction can combine four local moments. In real arithmetic, this can replace two sequential scalar-statistic reductions for this particular chain. The float64 control reports zero difference on its constructed input.

The loop norm followed by the next layer's input norm has another exact identity. Let `v=N_gamma_loop(x)` and let the next norm use eta.

```text
s = (||x||²/d + epsilon)^(-1/2)
t = (s²*||D_gamma_loop*x||²/d + epsilon)^(-1/2)
N_eta(v) = s*t*D_eta*D_gamma_loop*x
```

Two global moments suffice in real arithmetic. The float64 control again reports zero difference on its constructed input. Neither identity depends on repeated weights. Ordinary adjacent norms can admit the same transformation.

### BF16 limits of those identities

The source casts the normalized vector to the input dtype before gamma multiplication. Residual addition also changes rounded values. Removing these intermediate rounding points changes the numerical graph.

The software BF16 compound-chain examples have maximum absolute differences of `0.001953125`, `0.00390625`, and `0.00390625`. The corresponding loop-chain differences are `0.00048828125`, `0.0`, and `0.0078125`.

These constructed values refute universal bitwise equivalence for the tested fusion. They do not estimate an Ouro error distribution or demonstrate task-quality loss. A numerically reordered implementation needs separate tolerance and free-running checks.

## Exact normalized base-plus-delta projection

Let `y=x+a`. Let `D_gamma` denote the norm's diagonal scale. Let W be a dense projection. Define `s_x=(||x||²/d+epsilon)^(-1/2)` and `s_y=(||y||²/d+epsilon)^(-1/2)`.

In real arithmetic:

```text
W*N_gamma(y) = s_y*(W*D_gamma*x + W*D_gamma*a)
```

If `b=W*N_gamma(x)` is genuinely available, the equivalent cached formula is:

```text
W*N_gamma(y) = (s_y/s_x)*b + s_y*(W*D_gamma*a)
```

The changed norm scalar is necessary. Replacing it with the old scalar is not this identity. The norm statistic also requires current data:

```text
||x+a||² = ||x||² + 2*x_dot_a + ||a||²
```

Caching `||x||²` does not provide the current cross term or delta norm. Feature-sharded values still require correct global statistics. Forming y and reducing its squares is another valid method.

The float64 test reports zero difference for the two-projection identity. It reports `4.440892098500626e-16` for the cached-base form. Both use the same constructed x, a, gamma, and W recorded in the artifact.

### Dense work and state costs

For a dense W with m outputs and d inputs, one direct projection uses approximately `2*m*d` FLOP per token. A multiply and an add count separately. Without a ready base, the two-projection rewrite uses approximately `4*m*d` FLOP. It also adds scalar and vector work.

With a genuinely ready base, the delta projection still uses approximately `2*m*d` FLOP. It matches the original dense projection count before extra work. Small delta magnitude does not change dense GEMM dimensions.

| Projection shape | Direct dense FLOP/token | Uncached base+delta dense FLOP/token |
|---|---:|---:|
| QKV stack, 6144×2048 | 25,165,824 | 50,331,648 |
| Gate/up stack, 11264×2048 | 46,137,344 | 92,274,688 |
| Attention output, 2048×2048 | 8,388,608 | 16,777,216 |
| MLP down, 2048×5632 | 23,068,672 | 46,137,344 |

These are structural dense-projection counts. Only projections with the stated normalized-input contract directly use the norm identity. The down-projection row illustrates dense linear cost, not a claim that its SwiGLU input has a new RMSNorm.

The columns exclude norm statistics, square roots, gamma multiplication, vector addition, output scaling, cache reads, communication, and storage. A stacked operator describes arithmetic accounting. It does not establish that a fused kernel exists.

The base must match the same projection, norm gamma, request, token representation, and state version. An older recurrent state is not automatically the required x. Storing base outputs also consumes memory and can change traffic.

An exact sparse delta or another cheap exact representation could change this cost argument. This work establishes no such structure. Truncating a dense delta introduces approximation and a new quality obligation.

### BF16 projection counterexample

Use `x=[256,1]`, `a=[-256,0]`, `W=[1,1]`, and `gamma=[1,1]`. The software source-order evaluation gives `1.4140625`.

The two separately rounded projections cancel to zero before norm scaling. Their final result is `0.0`, with absolute difference `1.4140625`. The rounded cached-base formula gives `2.0`, with absolute difference `0.5859375`.

The software control also gives `0.0` for `(256+1)+(-256)`. It gives `1.0` for `256+(1-256)`. These examples locate a rounding hazard. They do not predict any particular GPU collective's output.

## Generic controls and novelty boundary

[Megatron-LM v4](https://arxiv.org/html/1909.08053v4), Section 3, already explains why row-partitioned partials need synchronization before nonlinear consumers. Its column/row composition permits local nonlinear work on complete output neurons. This is a training paper, but its linear-algebra distinction applies to the control here.

[PyTorch TP documentation 2.14](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html) exposes column, row, and input/output layout controls. Its `SequenceParallel` norm path shards tokens, not hidden coordinates. A token shard retains a complete feature vector. It therefore differs from the feature-statistic reductions in this report.

[FlashInfer communication documentation 0.7.2](https://docs.flashinfer.ai/api/comm.html) lists all-reduce/RMSNorm fusion and residual/norm variants. API existence does not establish support for every Ouro ordering. In particular, Ouro normalizes a sublayer output before residual addition. A baseline must retain that norm. Installed support and performance remain unchecked.

Logical unrolling with aliased weights permits the same layout propagation. The loop boundary does not create an automatic layout conversion in the replicated baseline. A new method cannot claim to remove a conversion that the matched baseline never needed.

The rejected claims are narrow:

- Apply RMSNorm separately to additive row-partials, then sum.
- Apply SwiGLU to incomplete gate/up preactivations, then sum.
- Average softmax over additive score partials.
- Treat exact real-arithmetic fusion as universal BF16 graph equivalence.
- Treat two dense base/delta projections as less dense work without another property.

The controls do not reject fused collectives, valid feature statistics, sequence parallelism, two-dimensional TP, alternative exact representations, or quality-controlled numerical reorderings. They establish no communication optimum. They establish no measured headroom.

## Reproduction

Run the dependency-free script:

```bash
python poc_ouro_parallelism_deep_discovery/scripts/layout_residual_cpu.py
```

It prints JSON and checks assertions. The saved artifact records the final run, source revision, precision contract, and source links. All GPU and model-quality cells remain `NOT_RUN`.

Final checkpoint: `2026-10-19T10:20:14Z`. The recorded segment span was 487 seconds of wall time. Artifact and timestamp annotations followed that checkpoint. This segment does not establish the campaign's ten-hour duration.
