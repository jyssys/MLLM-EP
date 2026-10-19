# Round 8: precomputed arithmetic and shared subset work

## Outcome

This round examines two small-granularity execution hypotheses and adds zero candidate cards.
Neither hypothesis uses stale hidden states, trained predictors, or a changed target model.
Both expose genuine arithmetic identities or finite-domain constructions.
Neither supplies a distinct recurrent ownership advantage after the source and dependency screen.
This conclusion does not reject every exact table implementation or common-subexpression kernel.

No GPU, checkpoint, model, or serving engine executes in this round.
The only executed checks use constructed CPU operands and symbolic size formulas.
Their results establish no task quality, native numerical parity, HBM traffic, or latency.

## Starting evidence and rejected duplicate

The starting matrix contains C01–C19 and no qualifying architecture shortlist.
This round reads the [matrix](interim_candidate_evidence_matrix_v1.md), [negative checkpoint](negative_results.md), and [alias caution](alias_equivalence_interpretation.md).
It also reads the recent state-transform, projected-attention, certified-operator, and lagged-FFN screens.
Their conclusions retain their existing scopes.

An initial terminal residual/head decomposition overlaps C06, C09, and C17.
That proposal starts a known residual's head projection before the FFN branch finishes.
Its extra projections and norm join do not establish a separate axis.
This round stops that duplicate and does not count it among the two hypotheses below.

## A. Replace input-dependent arithmetic with finite-domain lookup

### Hypothesis and causal inputs

BF16 has a finite word domain.
An owner could prepare exact product tables before the current hidden state exists.
After a normalized input coordinate arrives, workers could replace some multiplications with lookups.
The possible benefit replaces arithmetic, not a future coordinate's producer.

This differs from C18's whole-input/result cache.
Neither a repeated activation row nor a previous trajectory supplies the result.
It also differs from C19 because the table supplies arithmetic rather than reconstructing KV words.
These distinctions establish a screening question, not novelty.

A scalar lookup needs the current activation word and the fixed weight word.
QKV still needs current `N1(x)`.
FFN gate/up projections still need current `N3(y)` after attention and its residual.
The down projection still needs the current SwiGLU result.
No lookup obtains those operands before their causal producer.
Every output accumulation and subsequent normalization still needs its required contributions.

### Primary generic mechanisms

[T-MAC v2, §3.1, Algorithm 1](https://arxiv.org/html/2407.00088v2#S3.SS1) separates static weight preparation from activation-dependent table construction.
Its small tables enumerate bit patterns applied to the current activation group.
They are not precomputed outputs for an unknown future activation.
The scoped algorithm targets low-bit weight-quantized models, not unchanged BF16 Ouro.

[LUT-GEMM v4, §3.1, Equations 1–2](https://arxiv.org/html/2206.09557v4#S3.SS1) gives the same relevant readiness distinction.
Its partial-dot tables require the current activation subvector.
Its reduced bit alphabet comes from binary-coded weight quantization.
This campaign cannot assume that representation for the released BF16 checkpoint.

The [pinned T-MAC code](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/ops/qgemm.py#L129-L187) indexes an activation-derived LUT with packed weight groups.
The [table constructor](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/intrins/lut_ctor.cc#L38-L84) reads current activation values before its additions and quantization.
No external repository executes here.
Its activation-table quantization is not imported into this proposal.

### Positive capacity controls

Do not multiply a scalar product table by every model weight.
A global BF16-word-pair table can serve all matrices and all physical layers.
At four bytes per entry, its direct size is `65536² × 4 = 16 GiB`.
That size is substantial, but it does not exceed one H100's full HBM capacity by itself.
This statement is capacity accounting, not a feasible kernel or measured cache benefit.

A smaller positive construction factors the floating-point encoding.
A finite BF16 value consists of sign, integer significand `m`, and power `e`, giving `x = sign × m × 2^e`.
The significand fits eight bits, including the implicit leading bit for normal values.
All 256×256 unsigned significand products fit in 16 bits.
A direct table therefore occupies only 128 KiB.
Sign/exponent handling, normalization, exceptions, and final rounding remain runtime work.
This table is universal arithmetic data, not a weight-specific recurrence summary.

The CPU control checks all 65536 integer entries.
It also checks exact rational reconstruction on 14 distinct finite-word pairs, cycled through 1024 evaluations.
The check intentionally does not preserve signed zero or nonfinite behavior.
It does not implement FP32 FMA or a native GEMM.

An unshared table for every two-input partial dot is a different construction.
For one 2048-to-5632 projection, its direct size is `5632 × 1024 × 65536² × 4 = 88 PiB`.
This rejects that direct layout, not every factored or shared table.
The small significand construction prevents a universal capacity rejection.

### Ownership, complete costs, and falsifier

A local table retains current activations and ordinary weight ownership.
It adds table loads, bit extraction, exponent handling, exceptions, and accumulator instructions.
It does not reduce the number of required output contributions automatically.
It must compete with tensor-core GEMM, not a scalar multiply loop.
The 128 KiB table fits different hardware storage levels than the 16 GiB table.
Neither capacity implies useful lookup throughput.

A remote scalar service introduces a different communication pattern.
One simple protocol sends a two-byte activation and a two-byte weight, then returns a four-byte product.
It transfers 88 MiB for all products of one 2048-to-5632 projection.
Sending that projection's complete BF16 input and output transfers 15360 bytes instead.
These counts exclude headers, indexing, queueing, reductions, and retries.
They describe this protocol, not a lower bound for every encoding.

Keeping the weights at the remote owner can amortize activation sends.
Returning only complete outputs then becomes ordinary remote GEMV or TP.
That path must use the C02/generic-disaggregation controls.
A replicated universal table instead changes no model/KV owner by itself.
At P2/P4/P8, replicating the 16 GiB table consumes 32/64/128 GiB in aggregate.
Replicating the 128 KiB table consumes 256/512/1024 KiB in aggregate.

An FP32 product table also needs a numerical contract.
Stored rounded products can differ from a single-rounding multiply-add at underflow.
The CPU case uses BF16 operands `2^-75` and FP32 accumulator `2^-149`.
Stored-product then add returns `2^-149`, while the single-rounding exact-sum control returns `2^-148`.
See [NVIDIA's floating-point guide, §2.3](https://docs.nvidia.com/cuda/floating-point/index.html#the-fused-multiply-add-fma).
This is not an assertion about the selected H100 GEMM's underflow behavior.

The cheapest disproof of unrestricted bitwise equivalence is this constructed arithmetic counterexample.
The cheapest source discriminator already rejects a new recurrence mechanism: static arithmetic tables do not create future model inputs.
No distributed benefit distinct from generic lookup arithmetic or remote projection survives this formulation.
A restricted-domain exact kernel remains possible but unimplemented.
This round proposes no kernel or quality run for it.

## B. Compile repeated coefficients into shared current-input subsets

### Hypothesis and exact arithmetic

Repeated BF16 weight values can expose shared arithmetic even when complete activation rows never repeat.
For output row `o`, group input positions by identical coefficient `c`:

```text
S[o,c] = {j: W[o,j] = c}
z[S]   = sum(j in S) x[j]
y[o]   = sum(c) c * z[S[o,c]]
```

Identical or overlapping subsets across output rows can share intermediate sums.
This is an arithmetic DAG derived from weights, not an activation-result cache.
Neither a stale value nor equality between recurrent states supplies the result.
The row-group map can exist before current inputs arrive.
Its numeric subset sums cannot.

The positive toy has weights `[[2,2,3,4],[5,5,6,7]]` and input `[1,2,3,4]`.
Both rows reuse the fresh sum of the first two coordinates.
The exact outputs remain `[31,61]`.
Multiplications decrease from eight to six, and additions decrease from six to five.
These are operation counts for the constructed toy, not an Ouro measurement.

### Direct priors and limits

[UCNN, §III-A, Equation 2](https://d1qx31qr3h6wln.cloudfront.net/publications/UCNN_ISCA18_FINAL.pdf) directly factors repeated coefficients into activation-group sums.
Section III-B and Figure 4 share intersecting activation subsets across filters.
The paper explicitly charges indirection tables and their access cost.
Its reported hardware evaluations use fixed-point configurations, not this native BF16 pipeline.

[CREW, §IV-A](https://arxiv.org/html/2107.09408v1#S4.SS1) provides a second direct generic control.
It identifies repeated weights per input offline, then computes current-input unique products online.
It retains index tables and output accumulations.
Section VII-A reports that FC indexing costs limit UCNN's factorization benefit in its studied setting.
Those results do not establish H100 performance or repetition statistics for Ouro.
CREW's additional product approximation in §IV-B is outside this exact screen.

The static plan can serve all four visits to a physical matrix.
It can also serve every token of an ordinary dense model with that matrix.
Repeated physical weights do not make the numeric subset sums reusable across different current inputs.
This round measures neither current-checkpoint weight histograms nor useful subset-hit rates.
C18's zero whole-row matches do not measure these weight-derived subsets.

### Readiness and state owners

For QKV, each subset consumes the current normalized attention input.
For gate/up, each subset consumes the current normalized attention residual.
For down, it consumes current SwiGLU channels.
All subsets needed by an output must complete before its norm or later nonlinear consumer.
Moving raw pre-norm sums before normalization would reopen C17's separate cast/normalization contract.
This hypothesis does not assume that rewrite.

The physical matrix owner retains coefficient/group maps and their revision.
A subset owner retains current input coordinates and token/pass/layer tags.
Output owners retain pending contributions and accumulator state.
KV stays with its authoritative attention owner and logical depth.
The original gate still needs the completed loop-normalized hidden state.

### Added costs and a conditional useful frontier

Let `U` count distinct subset nodes and `R` count coefficient/subset references across output rows.
Subset construction costs at least its explicit gathers and reductions under the chosen DAG.
The final stage still needs `R` coefficient products and their output accumulations.
Without cross-row sharing, coefficient grouping can save multiplies while retaining roughly one input reference per original matrix entry.
Dense tensor-core execution already reuses its input through structured tiles.

Static costs include subset search, maps, coefficient storage, and any code generation.
Dynamic costs include gathers, subset scratch, output references, synchronization, and fallback.
For an index-per-input layout, both 2048 and 5632 coordinates need more than eight index bits.
An ordinary uint16 index already costs two bytes, before coefficients and group boundaries.
This is one layout's cost, not a universal compression lower bound.

Remote subset ownership additionally requires input delivery and sum delivery to every consumer owner.
An FP32 sum costs four payload bytes per delivered reference unless the protocol aggregates it further.
Co-location can avoid messages but then resembles generic local CSE or a UCNN/CREW-style kernel.
No measured shared subset currently justifies a new cross-owner frontier.
Fewer multiplications alone do not establish shorter latency or lower HBM traffic.

The cheapest numerical disproof uses BF16-representable operands.
For weights `[1,2,1]` and input `[2^24,0.5,-2^24]`, the real result is one.
A specified left-to-right FP32 calculation returns zero.
Grouping the equal coefficients first returns one, which changes the final BF16 word.
This rejects universal cast/reduction equivalence for that rewrite.
It does not emulate native Tensor Core accumulation or reject every safe common-subexpression plan.
Preserving the original reduction tree can avoid this particular reordering but needs its own complete cost and numerical contract.

The source discriminator identifies direct UCNN/CREW overlap before any model run.
This screen finds no recurrence-specific ownership discriminator beyond those generic mechanisms.
It therefore adds no new card and requests no GPU test.

## Both model sizes, depth modes, and C/P scope

Both Base models use D=2048, I=5632, 16 MHA heads, and the same layer dependencies.
Base-1.4B has 24 physical layers. Base-2.6B has 48.
The universal arithmetic table does not multiply by layer count.
Weight-specific subset metadata does, unless genuine inter-matrix structure exists.
This round measures no such structure.

At C1, neither hypothesis creates an independent future recurrent input.
It changes work inside the one ready current operator.
At C2/C4/C8, at most 2/4/8 independent request fronts supply current operands before additional tiling.
This is not a utilization prediction.
Both mechanisms must compare against same-P asynchronous TP with graph replay and an equally allowed generic arithmetic transformation.
P8 remains a structural projection only.

Fixed R4 charges four current operator visits, every norm, all depth-specific KV, and the final head.
An exact arithmetic transformation must preserve the original adaptive gate operands and ordered hazards too.
A changed BF16 hidden word can change that gate and later persistent KV.
Matching one current greedy token would not certify those states.
Any claimed exact path needs its backend numerical contract before timing or quality conclusions.

The alias caution does not reject either hypothesis merely because a generic graph could express it.
The actual stopping reasons are direct mechanism priors, retained operand dependencies, specified extra costs, and missing ownership benefit.
An undiscovered concrete ownership optimum remains possible.
Neither this round nor the 19-card matrix establishes universal impossibility.

## Reproduction and actual work

```bash
CUDA_VISIBLE_DEVICES='' python scripts/precomputed_arithmetic_cpu_v8.py --output artifacts/NEW_PRECOMPUTED_ARITHMETIC.json
```

The script refuses an existing output path.
The retained [CPU artifact](../artifacts/precomputed_arithmetic_cpu_v8.json) records six passing assertion groups and the script hash.
The [source record](../artifacts/candidate_search_v8_precomputed_arithmetic_sources.json) records actual queries, source identities, locators, hashes, and access limits.
It also records the actual segment times after validation.
No raw evidence, existing report, runner, installed package, or coordinator state changes.

Worker start: `2026-10-19 15:51:34 UTC`.
Validation cutoff: `2026-10-19 16:02:14 UTC`.
The observed segment spans 640 seconds, or 10 minutes 40 seconds.
Final timestamp insertion and handoff follow that cutoff.
Concurrent worker intervals do not add campaign elapsed time.
JSON parsing, seven local hash checks, and report lint passed.
