# Round 9: current-query KV certificates and known-address prefetch

## Result and scope

This round examines two execution protocols and adds zero candidate cards.
The protocols expose different legal work.
The first conditionally avoids opening a KV page after the current query exists.
The second moves committed historical KV before that query exists.
Neither supplies a future hidden state, guesses a query, or changes the target weights.

The first protocol has direct query-bound and omitted-tail priors.
It lacks a qualified native numerical certificate and a distinct recurrent ownership advantage.
The second is a legal generic memory-overlap mechanism.
Its physical benefit remains unmeasured, rather than disproved.
Neither screen warrants a new distributed engine or GPU proposal.

No GPU, research model, or installed runtime executes in this round.
The executed tests use deterministic rational arithmetic and a toy versioned store.
They establish no native parity, task quality, memory traffic, or latency.

## Starting contract

This round reads rounds [5](candidate_search_v5_independent_axes.md), [6](candidate_search_v6_independent_axes.md), [7](candidate_search_v7_projected_attention_states.md), and [8](candidate_search_v8_precomputed_arithmetic.md).
It also reads the [alias caution](alias_equivalence_interpretation.md), [negative-scope audit](negative_scope_audit_v1.md), and [19-card matrix](interim_candidate_evidence_matrix_v1.md).
Generic DAG expressibility alone is not a rejection criterion.
A concrete physical retention or ownership optimum can still matter.

Both Base models use D=2048, I=5632, 16 MHA heads, and head width 128.
They have 24 or 48 physical layers and four fixed recurrent passes.
The inspected native source first computes fresh Q/K/V, applies RoPE, and selects `attn[current_ut]`.
The four block RMSNorms and the recurrent boundary norm remain unchanged.
See `ouro.py:198–209,260–281,355–374` and the [frontier contract](exact_frontier_contract.md).

The original adaptive mode retains q=0.5, minimum R2, maximum R4, and `last_exited`.
The gate consumes the completed loop-normalized hidden state.
Neither protocol changes that dependency or permits uncommitted KV publication.
Adaptive Base 2.6B remains `ADAPTIVE_2P6B_UNVALIDATED`.

An initial selective-weight-retention lead stops as a duplicate screen.
The [DWDP/SiDP audit](dwdp_dense_transfer_v1.md) already charges remote weight staging and partial retention.
The [favorable cache-capacity model](interpass_cache_capacity_screen.md) already leaves selected physical retention open.
No new named weight subset or removed cost distinguishes that lead here.
This is not a proof that selective retention cannot help.

## A. Bounded-Tail BF16 Admission

### Named protocol and legal frontier

`Bounded-Tail BF16 Admission` is a screening protocol, not an implemented candidate.
Each authoritative historical page retains exact coordinate minima/maxima of its stored, rotated K.
It also retains coordinate-wise maximum absolute V values.
Metadata identities include request, generation, logical depth, physical layer, head shard, and committed range.
The current incomplete page follows an exact visible-prefix path.
No metadata can borrow a different depth's K or expose uncommitted positions.

Once the actual current query q exists, page-bound tasks can run independently.
For head width d and page c, a real-arithmetic score cap is:

```text
U_c(q) = sum_j max(q_j*Kmin[c,j], q_j*Kmax[c,j]) / sqrt(d)
```

This cap can choose which pages to open first.
It cannot, by itself, authorize dropping the others.
The owner evaluates a nonempty kept set S and computes its partition mass Z_S and normalized output a_S.
For unopened page c with n_c tokens, use `A_c = n_c*exp(U_c)` as a mass cap.
In an actual implementation, these calculations need overflow-safe scaling and outward error control.

For output coordinate t, let `M[c,t]` bound every unopened `abs(V[j,t])`.
The following conservative bound includes the changed denominator:

```text
abs(a[t] - a_S[t])
  = abs(sum_unopened exp(score_j)*(V[j,t]-a_S[t])) / (Z_S+Z_unopened)
 <= sum_c A_c*(M[c,t]+abs(a_S[t])) / Z_S
  = E[t]
```

The protocol accepts a page omission only if its complete output interval fits one rounding cell.
Otherwise it opens more pages or falls back to the complete attention read.
All authoritative K/V remains allocated for fallback.
This protocol therefore claims possible read elimination, not KV capacity reduction.

### Exactness boundary

The bound is valid over the stated real-arithmetic operands.
It does not establish bitwise identity to the installed attention backend.
That backend includes rounded scores, exponentials, online normalization, value accumulation, and a BF16 output cast.
It can have intermediate casts and a reduction order that differ from a one-rounding reference.

Let `eta_S` enclose the computed kept-output error and `eta_D` enclose the native dense-output error.
One sufficient check encloses the native dense result around the computed kept result with radius `E+eta_S+eta_D`.
That entire interval must round to the actual candidate BF16 word.
The required envelopes must cover the selected backend and exceptional values.
This audit establishes no such native envelope.
Setting both errors to zero would silently change the contract.
Merely comparing a tail score with an underflow threshold is also insufficient for an arbitrary online reduction order.

The CPU control deliberately uses exact rational masses and one final BF16 RNE cast.
It does not emulate native attention or claim that its logits are BF16 model operands.

| Constructed case | Kept BF16 output | Dense BF16 output | Strict rounding-cell certificate |
|---|---:|---:|---|
| Kept mass/value 1/1, tail mass/value 2^-12/2 | 1 | 1 | Pass |
| Kept mass/value 1/1, tail mass/value 2^-20/2^20 | 1 | 2 | Fail |
| Tail value zero, tail mass 2^-8 | 1 | 255/256 | Fail |
| Kept output exactly 257/256, positive tail mass 2^-20 | 1 | 129/128 | Fail |
| Kept mass/value 1/1, tail mass/value 63/0 | 1 | 1/64 | Fail |

The second case rejects a score-only small-mass rule.
The third case isolates a denominator change.
The fourth case shows why a positive but arbitrarily small tail can cross a rounding boundary.
The first case preserves a valid positive opportunity under the declared control contract.
These are not observed Ouro attention distributions.

### State ownership and complete costs

The page owner maintains metadata when committed K/V changes.
The query owner retains the fresh query and output certificate.
An owner can keep metadata local, send it to the query owner, or replicate it.
Every choice charges its reads, updates, transfers, and leases.
Keeping it local can require a query broadcast and returned page bounds.
Central metadata can require page-fetch requests after the certificate chooses a page.

For all 16 heads, BF16 K minima/maxima plus V absolute maxima occupy `3*D*2 = 12288` bytes per page.
The logical metadata-to-KV ratio is 9.375% at page size 16 and 2.34375% at page size 64.
TP head sharding divides the logical per-rank bytes, before alignment and control data.
FP32 interval arithmetic, page identities, ordering, and certificate scratch add costs.
Each appended token requires the relevant K/V summary updates.
No summary is reusable across recurrent depths merely because the projection matrices use shared weights.

For N historical positions and page size b, page scoring costs O((N/b)*D) scalar bound work per ready row.
Selected pages still need ordinary scores, softmax, and PV work.
Output checks can cost O((N/b)*D) or more across repeated admission rounds.
Fallback retains the metadata work in addition to a complete attention read.
Sorting, irregular page access, synchronization, native-layout compatibility, and conservative error margins also remain charged.

Native TP already keeps each rank's assigned head KV beside its attention work.
It does not transfer complete historical KV across ranks for every layer.
A new remote page-owner layout therefore cannot count an absent baseline KV transfer as a saving.
It must justify its query/page messages, joins, and output ownership against local head-sharded TP.
This screen identifies no such incremental distributed saving.

### Primary controls, falsifier, and disposition

[Quest v2, §3.4 and Algorithm 1](https://arxiv.org/html/2406.10774v2) directly supplies page extrema and current-query score caps.
Its fixed top-K page budget is an approximation, not a dense-output certificate.
The [pinned evaluation code, lines 146–218](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py#L146) exposes that distinction.
This inspected Python evaluation computes full attention scores before applying its selection mask.
It is not evidence of the optimized system's actual page-read savings.

The author-maintained [SSA source, §5.7](https://github.com/jonsmirl/ssa/blob/0613e78d65db5b90ddba840d069fa66307faefed/paper/subquadratic_attention.md#L600) already derives omitted-mass and value-output certificates with full-scan fallback.
Its [CPU reference, lines 138–145](https://github.com/jonsmirl/ssa/blob/0613e78d65db5b90ddba840d069fa66307faefed/ssa/certified_attention.py#L138) explicitly limits its floating-point cushion.
This repository is a primary author artifact, not an independent serving evaluation.
No performance result from it transfers to Ouro here.

[SoftAct, §III-B, Equations 18–20](https://yuzhe-fu.github.io/papers/TCSVT.pdf) supplies a nearby finite-output zero-detection prior.
Its quantized nonlinear hardware contract is not the native BF16 Ouro attention contract.
This round imports neither its quantization nor its activation change.

The cheapest numerical disproof of a score-only shortcut is the table above.
The cheapest next prerequisite for the stricter protocol is a source-backed native error envelope.
Only then would one archived current-query/K/V case test useful page admission under that envelope.
This round assembles and evaluates no paired complete archived case.
If the certificate opens every page, that operating point has no removed KV read to pay for the new work.

Disposition: `GENERIC_CERTIFICATE_WITH_UNQUALIFIED_NATIVE_BOUND`.
This is narrower than a claim that exact selective attention is impossible.
It adds no candidate card.

## B. Versioned Historical-KV Lookahead

### Named exact protocol and actual independent work

`Versioned Historical-KV Lookahead` acquires a read lease for the next known `(request,generation,depth,layer,head-shard)` history.
It copies only committed positions below the current token position into bounded staging storage.
The lease protects the source pages against reclamation or generation reuse.
A completion event protects the staged buffer before attention reads it.
A consumer-completion event protects that buffer before reuse.
Current-token K/V follows the ordinary fresh producer and append path.

The copy can start before the future hidden state and query exist.
The addresses and immutable historical words, rather than a guessed query, make it legal.
This is useful memory work if it hides a transfer or read latency that the matched baseline would expose.
It is not progress on QK, softmax, or PV.
It cannot select query-important pages before the query exists unless selection uses a separate valid query-independent policy.

Byte-preserving staging can leave the attention arithmetic unchanged if its layout, strides, and backend path remain identical.
A changed kernel or dispatch path still requires numerical qualification.
The toy CPU store tests source versions and ranges only.
It accepts a committed prefix before any query and joins it with fresh current K/V.
It rejects an uncommitted current position, a wrong depth, and a reused generation.
It does not implement CUDA streams, paged attention, or a serving cache.

### Local L2 hints and remote HBM staging are different

Local L2 prefetch moves no authoritative KV owner.
It attempts to warm selected future cache lines while current work executes.
It can evict useful weights, current KV, or another request's state.
Prefetch reads and intervening misses still consume memory service.
The measured 50 MiB total L2 is not a promise of 50 MiB available retention.
The supported persisting set-aside and effective contention budget remain separate properties.

Peer staging changes the data path.
For M bytes, charge a source read, M link bytes, a destination write, and the attention operand read.
Caches can change physical HBM transactions, so these are logical operations.
Immutable history needs no copy-back of those same bytes.
Remote ownership of newly appended K/V needs its own publication path if the compute owner does not retain authority.
Opposite-direction traffic for other readers is additional traffic, not a free return channel.

Native head-sharded TP has no mandatory peer-KV staging to hide.
Introducing a peer-owned cache can make capacity placement feasible under another workload.
This round establishes no such admission boundary at the current low-C target.
Removing a transfer from an artificially remote baseline would not demonstrate improvement over same-P native TP.

### Capacity and all-layer accounting

One future physical layer's historical K+V for all C active rows occupies `M = 8192*N*C/P` bytes per rank.
Two complete staging buffers occupy 2M.
All fixed-R4 history occupies `4*L*M`, before current tokens, padding, metadata, and runtime scratch.

| N / C | P2 one future layer | P4 one future layer | P8 one future layer |
|---|---:|---:|---:|
| 1024 / 1 | 4 MiB | 2 MiB | 1 MiB |
| 1024 / 8 | 32 MiB | 16 MiB | 8 MiB |
| 8192 / 1 | 32 MiB | 16 MiB | 8 MiB |
| 8192 / 8 | 256 MiB | 128 MiB | 64 MiB |

The [CPU artifact](../artifacts/candidate_round9_certificates_prefetch_cpu_v1.json) retains all 48 size/C/P/N accounting rows.
At N1024/C1/P2, all R4 history is 384 MiB for Base 1.4B and 768 MiB for Base 2.6B per rank.
At N8192/C8/P2, those values become 24 GiB and 48 GiB.
Those are logical history capacities, not measured allocations or traffic.
An individual 66 MiB MLP also competes through its actual TP shard and kernel tiles.
This round measures no coexistence or cache-hit rate.

If fraction f of every required history is remote and staged once per fixed visit, wire payload per rank is `f*4*L*M`.
The aggregate is P times that expression under the balanced all-active layout.
Actual source/destination ownership, request placement, and head sharding determine the traffic matrix.
P8 is structural accounting only.
No advertised bandwidth becomes a latency estimate.

The four recurrent visits use different logical-depth KV addresses.
Keeping R1 history does not make R2 history resident.
Retaining a selected prefix across tokens can help repeated reads of that same logical KV.
Ordinary autoregressive layers have that same opportunity.
Shared projection weights do not merge the four histories.

### Fixed, adaptive, and ready-work supply

At C1, one active decode row can prefetch one or more future histories without producing another ready query.
At C2/C4/C8, up to C request fronts supply addresses, not four times C independent future queries.
More lookahead can increase staging lifetime and contention without increasing useful arithmetic.
Base 2.6B doubles the number of physical-layer histories and boundaries, not their per-layer dimensions.

Fixed R4 makes the next depth/layer sequence predictable.
Within an adaptive pass, later layers remain required if the gate admits that pass.
At an exit-capable boundary, a future-pass prefetch is provisional memory work.
An exit wastes those bytes but must not publish the copied buffer as new authoritative KV.
Minimum R2 makes the first transition mandatory, not later transitions.
The original gate and `last_exited` publication still determine actual use.

### Primary controls, cheapest falsifier, and disposition

Official [Transformers cache documentation](https://huggingface.co/docs/transformers/v5.5.4/en/kv_cache#cache-offloading) already describes asynchronous next-layer KV prefetch.
The [pinned code, lines 1361–1382 and 1415–1426](https://github.com/huggingface/transformers/blob/b19c4a6669eab3d446789405d5bd00c3ace65863/src/transformers/cache_utils.py#L1361) uses a separate stream and explicit waits.
Its CPU-offload design is not a drop-in peer-paged Ouro adapter.
It is a direct mechanism prior for moving known KV before its next consumer.

Official [PTX ISA 8.8, §9.7.9.15](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu) defines cache-line prefetch and eviction-priority variants.
The [CUDA 12.9.1 guide, §§6.2.3 and 6.2.9.5](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html) supplies persistence limits and asynchronous peer-copy primitives.
These APIs establish legal primitives, not measured overlap or retained cache residency.

The cheapest ownership discriminator is the existing native head-sharded trace.
If the proposed saving is a removed peer-KV transfer, that trace already has no such required transfer.
That fact rejects the stated saving, not every prefetch benefit.
The CPU lease/range control is the cheapest correctness discriminator for the toy protocol.
A physical reopening needs a named exposed memory stall or real capacity boundary and an exact retained subset.
It must compare against native async/graph TP and the same generic prefetch facility.

Disposition: `LEGAL_GENERIC_PREFETCH_WITH_UNMEASURED_PHYSICAL_COST`.
This audit neither demonstrates nor rules out a recurrence-specific ownership optimum.
This round adds no card and requests no GPU test.

## Evidence separation and work ledger

Protocol A differs from C12 by retaining complete authoritative KV and making a query-dependent read decision.
It differs from C19 by not compressing KV and from C09 by certifying an attention output, not a final vocabulary winner.
Its nearest generic certificate prior is nevertheless direct.
Protocol B differs from C10 by transferring existing words rather than reconstructing historical K/V.
It differs from C01/C16 by preparing data before an arithmetic task becomes ready.
Those differences do not establish novelty or profit.

Both Base sizes and fixed/original adaptive receive source/structural analysis only.
Every model-quality and mechanism-latency cell remains `NOT_RUN`.
Neither a failed heuristic nor a generic prior establishes zero physical headroom for every layout.

Reproduce the CPU controls with a new output path:

```text
CUDA_VISIBLE_DEVICES='' python scripts/candidate_round9_certificates_prefetch_cpu.py --output artifacts/NEW_ROUND9_CPU.json
```

The script refuses an existing output.
Six assertion groups pass, including positive certificate and versioned-prefix controls.
The [structured observations](../artifacts/candidate_search_v9_certified_kv_prefetch_sources.json) preserve queries, primary pins, hashes, access limits, and UTC checkpoints.
No earlier artifact, candidate card, coordinator state, package, runner, or raw file changes.

Work began at `2026-10-19 16:34:54 UTC`.
The CPU control ran from `16:40:33.910064` to `16:40:33.911359 UTC`.
The validation cutoff is `2026-10-19 16:44:19 UTC`.
The observed segment spans 565 seconds, or 9 minutes 25 seconds.
Report lint, Python syntax, JSON parsing, 12 input hashes, and the CPU script hash pass.
Final timestamp insertion and handoff follow that cutoff.
Concurrent worker intervals do not add campaign elapsed time.
