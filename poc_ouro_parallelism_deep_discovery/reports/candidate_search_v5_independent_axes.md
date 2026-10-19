# Independent candidate search, round 5

Worker: `/root/independent_axes`.
Segment began at 2026-10-19 13:36:53 UTC.
Scope: two new units of useful work, not stale-state correction or request placement.
Evidence classes: primary-source inspection, constructed CPU algebra, and a bounded CPU checkpoint screen.

## Outcome

This round adds zero candidate cards.
R5-A targets exact conditional channel work inside SwiGLU.
Its bounded CPU screen exposes no zero channels to remove.
R5-B targets independent lossless arithmetic components of a fresh dense contraction.
Its tested representation expands both operands and does not preserve an arbitrary FP32 accumulation order automatically.
Neither proposal has a supported net-cost argument for a distributed engine.

These negative results apply only to the stated proposals and controls.
They do not prove that all activation sparsity or all arithmetic decompositions fail.
No GPU execution, task-quality evaluation, trained component, or runtime modification occurred.

## 1. Interpretation and actual dependencies

The worker read [the alias interpretation](alias_equivalence_interpretation.md) before this round.
An aliased conventional executor can represent a schedule without implementing its ownership protocol efficiently.
That fact alone does not reject a concrete ownership or memory benefit.
This round instead tests prerequisites and charges the new work.

Both Base models use D=2048 and I=5632.
Their 24/48 physical layers execute four mandatory passes under fixed R4.
The inspected block applies four RMSNorms, attention, residual updates, and SwiGLU.
Each recurrent boundary applies its loop norm.
Historical KV remains distinct by request, depth, layer, and position.
The original adaptive gate and terminal-state policy remain unchanged.
See [the exact frontier contract](exact_frontier_contract.md).

Neither proposal supplies a future hidden state or KV.
Both operate only after the exact current input and required norm result exist.
Their possible work is therefore available at C=1 as well as C=2/4/8.
Availability does not establish efficient physical concurrency.

## 2. R5-A: Certified-zero SwiGLU channels

### Mechanism

For channel j, let `g_j = W_gate[j,:] x` and `u_j = W_up[j,:] x`.
The intermediate channel is `a_j = SiLU(g_j)*u_j` under the loaded rounding contract.
If that contract certifies a zero activated gate and a finite up result, the channel contributes zero.
A runtime could omit its up row and down column.
It must still compute the gate and all required surviving contributions.

This is conditional arithmetic elimination, not whole-transition identity from C18.
It does not approximate a small nonzero value as zero.
For real finite inputs, SiLU has no nonzero root.
Floating-point underflow and exact cancellation need separate treatment.
The certificate must refer to the actual activation and cast sites, not a threshold chosen for speed.

### Readiness, ownership, and costs

| Required field | R5-A contract |
|---|---|
| Causal inputs | Complete fresh normalized x, checkpoint gate weights, and the actual activation/cast semantics |
| Concurrent unit | Certified channel tiles, then surviving up/down tasks from the same current MLP |
| State owners | Original attention owner retains KV, hidden/residual owner retains the exact current version, MLP owners retain immutable channel weights |
| Exactness class | Algebraically exact zero elimination for finite unused operands, with rounding and reduction qualification still required |
| Saved cost | Selected up-row and down-column reads and arithmetic |
| Added costs | Gate-first delay, certificate scan, finite-result guard, masks, sparse indexing, packing, irregular loads, and output merge |
| Same-P baseline | Native fused gate/up projection and dense down projection, plus a generic exact zero-skip kernel |
| Cheapest disproof | Count certified zero channels and removable tiles before implementing sparse execution |

The native path already fuses gate and up projection.
Waiting for a complete gate first can serialize work that the baseline computes together.
Channel-level pipelining can reduce that delay, but requires more launches or persistent task control.
The following RMSNorm still waits for the complete down-projection result.
No channel mask permits arbitrary partial results to cross that norm.

A zero gate alone does not justify an uncomputed nonfinite up result.
IEEE arithmetic gives `0*Inf = NaN`.
A conservative finite-result bound or exceptional fallback must therefore accompany the certificate.
For example, cached row norms and a current-input bound can bound the unused up projection.
The runtime must charge that check and its rounding margin.

### Priors and source check

TEAL describes magnitude-based approximate activation sparsity and sparse kernel/layout costs.
Its output-sparsity discussion already describes gating the up and down projections.
It does not supply exact removal of arbitrary nonzero SwiGLU channels.
[TEAL v2, sections 3, 4.2, 4.4, and 5.4.1](https://arxiv.org/html/2408.14690v2).

Model Casting gives the same gate-first, skip-up/down structure explicitly.
It obtains sparse gates through an activation change and further training.
Those changes are outside this campaign.
Its existence separates a known execution mechanism from an unestablished exact-zero opportunity in unchanged Ouro.
[Model Casting v1, sections 3.1–3.2](https://arxiv.org/html/2609.31975v1).

The installed Python wrapper dispatches CUDA SwiGLU to a custom operation.
The pinned current official CUDA source casts the activated gate back to its scalar or packed dtype before multiplication.
That source is not a binary identity check for the installed package.
The attempted release-tag source URL returned 404, so the audit retains this qualification.
[vLLM current source, revision 81f7c69, lines 28–83 and 164–177](https://github.com/vllm-project/vllm/blob/81f7c69577433b938ee16539020792dae37c7d0c/csrc/activation_kernels.cu#L28).

### Bounded result

The CPU screen uses earlier captured MLP inputs and actual released gate weights.
For each model, it selects two prompts, three physical layers, and four first-decode recurrent calls.
This gives 24 rows and 135,168 channel values per model.
The calculation uses FP64 CPU projection, BF16 gate rounding, FP32 CPU SiLU, and BF16 activation rounding.

| Model | Reconstructed rows | Gate channels | Zero gate values | Zero rounded SiLU values | All-zero width-32 tiles |
|---|---:|---:|---:|---:|---:|
| 1.4B | 24 | 135168 | 0 | 0 | 0 |
| 2.6B | 24 | 135168 | 0 | 0 | 0 |

This is not a measurement of the native GPU activation kernel.
It is not a task-quality result or population-level sparsity estimate.
The rows come from two bounded prompts, not independent benchmark requests.
The selected fixed-R4 screen supplies no adaptive hit-rate evidence.
Width 32 defines a logical checker tile, not an identified native kernel geometry.
Within this screen, even a zero-cost sparse runtime would find no certified-zero work to remove.
That result does not support an engine.

The constructed CPU control also distinguishes two rounding contracts.
For gate `2^-133` and up `2^16`, BF16 activation rounding first gives zero.
Keeping the activation in FP32 through multiplication gives a nonzero BF16 result of approximately `3.0092655381e-36`.
This control uses FP32 subnormals and does not establish GPU FTZ behavior.
It does not refute the pinned source's explicit intermediate cast.
It shows why the certificate cannot silently change cast sites.

## 3. R5-B: Lossless arithmetic-component contraction

### Mechanism

Represent a fresh BF16 contraction using signed integer slices and exact power-of-two scales.
For one shared-scale representation, write `W = sum_i alpha_i W_i` and `x = sum_j beta_j x_j`.
Each pair contraction `W_i x_j` uses the same already-ready input, not a future hidden prediction.
Independent GPUs could compute different pair terms and combine the results.
Every retained term contributes useful arithmetic under the declared representation.

The proposal does not permit mantissa truncation as its source of gain.
It must preserve all required input bits and the matched accumulation contract.
Losing input bits would be a reduced-precision shortcut, not this hypothesis.

| Required field | R5-B contract |
|---|---|
| Causal inputs | Complete fresh BF16 x after its actual norm, immutable W, exact slice/scaling metadata |
| Concurrent unit | Integer slice-pair contractions and their explicit reconstruction tasks |
| State owners | Existing request/KV owners remain unchanged, arithmetic workers own immutable W slices and versioned partial sums |
| Exactness class | Input representation can be lossless, but real-dot reconstruction is not automatically the native FP32 operation graph |
| Saved cost | A hypothesized cheaper operand/arithmetic path, not fewer recurrent evaluations |
| Added costs | Static expanded weights, dynamic x conversion, multiple contractions, partial sums, communication, reconstruction, and exceptional-value handling |
| Same-P baseline | Native BF16 GEMM/TP on the same fresh rows and generic integer-emulation methods |
| Cheapest disproof | Measure exact representation width and pair count, then test the claimed accumulation contract |

Weight slices can remain static across passes.
Input slices cannot become ready before the corresponding new x exists.
Repeated physical weights may amortize setup or favor selected residency.
That possibility deserves a cost test rather than an alias-only dismissal.
It does not remove expanded storage or repeated operand reads during execution.

### Priors and numerical boundary

The Ozaki/IMMU method already decomposes higher-precision contractions into integer slices.
Its paper charges splitting, workspace, component GEMMs, and higher-precision reconstruction.
Its main target is FP64 GEMM, not replacing native BF16 inference with fewer operations.
[Ootomo, Ozaki, and Yokota, v3, sections 2.3–3.2 and 4.3](https://arxiv.org/html/2306.11975v3).

Official cuBLAS also documents floating-point emulation.
The inspected version targets FP32 with BF16x9 and FP64 with fixed-point methods.
Those paths do not establish a drop-in H100 acceleration for the campaign's BF16 operand contract.
This audit changes no library or environment setting.
[cuBLAS 13.0.3, section 1.5](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html#floating-point-emulation).

The constructed CPU control uses BF16-representable operands `[4096,3,4096]` and `[8192,1,-8192]`.
Their real dot is 3.
A declared serial FP32 accumulation gives 4.
That serial order is not an identified cuBLAS or Tensor Core reduction order.
The control rejects the inference that lossless input bits imply bitwise native output.
It does not prove that every arithmetic implementation fails numerical qualification.

### Representation screen

The checker measures the lowest and highest occupied binary positions in each BF16 row.
It counts base-128 signed-digit slices for one exact shared-scale format.
The positive unit control fits `[1,1.5]` into one slice.
The counterexample `[1,2^-16]` needs 17 magnitude bits and three such slices.
Both controls pass.

The same bounded checkpoint rows require 3–4 input slices.
Gate-weight rows require 3–6 slices across the three selected layers of each model.
Even the best weight row needs 9–12 pair contractions for complete pair expansion on these inputs.
These counts do not establish physical GEMM latency.
They also do not bound exponent grouping, sparse slices, or every alternative exact representation.

They do refute a one-slice lossless INT8 replacement for these operands under the tested format.
Storing three or more byte slices can also exceed the original two-byte weight storage before metadata.
No measured cheaper memory or compute path offsets this expansion.
The representation and numerical gaps therefore do not support a distributed engine.

## 4. Both-mode and ownership conclusion

Both proposals target a current MLP or projection and can apply structurally to either layer count.
Neither relies on early-exit heterogeneity for its possible work.
The CPU checkpoint observation covers fixed R4 only.
Adaptive performance, exactness, and gate/KV compatibility remain untested.
P2/P4/P8 require their own communication and resource account.
No GPU-count speedup follows from the component counts.

This round does not reject a memory optimum merely because a generic executor can represent it.
R5-A lacks its required zero-work supply in the bounded screen.
R5-B's specified representation adds substantial component work and lacks a matched numerical contract.
Those are the scoped reasons for returning zero cards.

## 5. Reproduction and effort

Run the CPU-only checker with the existing local checkpoints and captures:

```text
/home/esjung/.venvs/cp-composition-vllm026/bin/python scripts/candidate_round5_cpu.py --checkpoint-screen
```

The [CPU artifact](../artifacts/candidate_round5_cpu_v1.json) retains all row-level records and limits.
The [source observations](../artifacts/candidate_search_v5_source_observations.json) retain search queries, pins, and actual checkpoints.
The final checker ran from 13:44:18.524252 to 13:44:21.258761 UTC.
An earlier checker run preceded two added representation unit tests.
The final run passed those tests and reproduced the bounded statistics.
No raw capture or earlier report changed.
The evidence and validation checkpoint was 13:48:19 UTC.
The segment lasted 686 seconds to that checkpoint.
Concurrent worker intervals do not multiply campaign elapsed time.
