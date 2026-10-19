# Round 7: project a current attention state before its merge

## Decision and scope

This round examines one sub-stage ownership hypothesis and produces zero new cards.
It asks whether output projection can start on completed KV partitions before complete attention becomes available.
The input is a current, query-specific partial attention state.
The mechanism uses neither future hidden states nor historical activation predictions.

A head-preserving transformation is valid in real arithmetic.
A cheap merge of already-mixed output vectors is not generally valid.
The direct valid representation expands the vector state by 16× for Ouro.
It also repeats output-projection work across KV partitions.
Removing the native attention-output cast changes a deterministic BF16 control.

These results do not reject every fused attention or distributed basis transformation.
They reject the proposed cheap compact merge and its unrestricted bitwise-equivalence claim.
The remaining valid paths have generic CP/operator-streaming controls and no demonstrated recurrence-specific benefit.
No GPU, model, serving runner, or installed package changes occur in this round.

## Mechanism and actual input readiness

Both Base models use H=16 attention heads, head width d=128, and D=2048.
Their physical layer counts are 24 and 48.
The fixed path executes four visits to each physical layer.
The [semantics audit](ouro_semantics_audit.md) pins the checkpoints and cache semantics.

The native sequence remains:

```text
current normalized input -> Q/K/V -> RoPE Q/K
current Q plus matching committed/current K/V -> attention
attention output, in query dtype -> Wo -> required output reduction -> N2
```

The installed `ouro.py:200–212` passes attention output directly to the row-parallel output projection.
The installed `attention.py:512–531` defaults that output allocation to the query dtype.
For the scoped BF16 path, Wo consumes a BF16 attention tensor.
All four block norms and the loop norm remain in the graph.

Partition the required KV positions into disjoint sets s=1…S.
Each partial task requires the actual current query and its exact depth/layer KV versions.
Let `a[s,h]` be the normalized partial output for head h on partition s.
Let `lse[s,h]` be that partition's log-sum-exp.
These are fresh current-query values, not a summary that can be prepared before the query.

The global coefficients and head output are:

```text
beta[s,h] = exp(lse[s,h] - logsumexp_t(lse[t,h]))
a[h] = sum_s beta[s,h] * a[s,h]
o = sum_h Wo[:, h*d:(h+1)*d] * a[h]
```

Every beta generally depends on the other KV partitions for the same head.
A completed local partial therefore does not imply that its final global weight exists.
The proposed early task is `g[s,h] = Wo_h * a[s,h]`.
It can execute before other partitions finish, if its complete weight block is available.
The final result still needs all required coefficients and contributions.

## The head coefficient barrier

In real arithmetic, preserving the head index gives:

`o = sum_s sum_h beta[s,h] * g[s,h]`.

The tempting compact state instead stores only `g[s] = sum_h g[s,h]`.
Different heads generally have different beta values.
One scalar coefficient for g[s] cannot implement those different weights.
Thus, the ordinary scalar attention-state merge does not simply transfer into the mixed output basis.

The deterministic full-rank example uses two one-dimensional heads:

```text
Wo = [[1,2], [3,5]]
a[0] = [1,1], a[1] = [0,0]
beta[0] = [0.75,0.25]
g[0] = [3,8]
required o = [1.25,3.5]
```

The two coordinates require different scalar ratios.
Even the best least-squares scalar gives maximum error 0.0547945205479452 in this example.
This is a CPU algebra control, not model error.

### Why this is not a universal impossibility proof

An invertible Wo permits a larger operation on the compact state:

```text
B_s = blockdiag(beta[s,1]*I_d, ..., beta[s,H]*I_d)
Wo * B_s * a_s = (Wo * B_s * inverse(Wo)) * g_s
```

The CPU positive control recovers the exact target with this basis transform.
It prevents a claim that every compact representation must retain H full output vectors.
The transform still needs current beta and a dense operator or an equivalent structured implementation.
Applying `inverse(Wo)` and Wo directly adds dense contractions.
Precomputing H projector matrices instead adds up to H dense matrices and their application costs.
Conditioning, finite precision, storage, and setup also require evidence.
This round does not inspect checkpoint ranks or construct those matrices.

A second example uses a rank-deficient 2×2 Wo.
Two different head states give the same projected state and the same LSE metadata.
Their required globally weighted outputs differ.
This rejects a universally sufficient compact projected state for arbitrary Wo.
It says nothing about the rank of an actual checkpoint matrix.

## Direct valid ownership plan and complete costs

Consider the direct plan with all heads and all output coordinates on every KV-partition worker.
Each worker retains its assigned KV positions and a complete Wo.
Each completed local head output can start its projection while other workers finish attention.
The join retains g[s,h], beta metadata, and the matching request/depth/layer identity.

| State | Owner/readiness | Cost or unresolved requirement |
|---|---|---|
| Historical KV partition | Assigned worker after committed prefill/decode publication | Native head-sharded KV needs redistribution or a different initial ownership plan |
| Current query for all heads | Each sequence worker after Q/RoPE | Query broadcast/all-gather or duplicated Q projection |
| Current K/V | Owner of the current position and matching depth/layer | Gather head shards or duplicate projections, then append with explicit versioning |
| Local a[s,h] and LSE | Sequence worker after its exact local attention | Original partial state plus metadata |
| Wo_h | Projection worker before its task | Replicated weights or charged weight access |
| Early g[s,h] | Projection worker after local a[s,h] | D values per head rather than d |
| Global beta and output | Merge owner after required partition statistics/results | LSE exchange, coefficient distribution, result merge, fences and scratch |

For FP32 vector states, ordinary partial attention stores `H*d*4 = 8192` bytes per row and partition.
The head-preserving projected state stores `H*D*4 = 131072` bytes.
Each representation also needs 64 bytes for 16 FP32 LSE values.
The vector-state expansion is 16×.
At C8, this direct projected state occupies 1 MiB per partition before metadata and other scratch.
These values describe logical state capacity, not measured wire or HBM traffic.

One complete BF16 Wo occupies 8 MiB per physical layer.
All physical Wo matrices occupy 192 MiB for 1.4B and 384 MiB for 2.6B.
Replicating across S sequence workers multiplies aggregate Wo storage by S relative to one aggregate copy.
It also performs `2*S*D^2` projection FLOPs per ready row, instead of `2*D^2`.
The direct work factors are 2/4/8 at S=2/4/8.
No weight tile becomes permanently resident merely because R4 repeats its address.

Output-feature tiling can reduce a worker's retained vector width.
Head/sequence/output meshes can change the duplication factors.
They also change parameter ownership, result routing, joins, and collective costs.
Those alternatives require their own complete 2D/3D accounting.
The direct counts are not lower bounds for every possible algorithm.

Waiting for global beta before projection avoids storing H projected vectors.
It also restores the statistics dependency before that projection work.
A QK-only prepass could make statistics available before PV completes.
It must retain attention weights or repeat score work and associated reads.
This is an additional algorithm and cost, not a free compact merge.

## Useful concurrency and model/mode coverage

At C1, current-query heads and KV partitions already supply legal sub-stage work.
The upper task counts for one ready operator are H*S attention partitions and corresponding projected-head tasks.
At C2/4/8, the ideal row supply scales those counts by 2/4/8.
Actual readiness can be smaller.
These are task counts, not simultaneous H100 execution or speedup estimates.

This work needs no future recurrent hidden state.
It also creates none.
R4 does not multiply the ready frontier of one causal token by four.
Distinct ready depths from different requests must retain their individual KV identities.

Both Base sizes have the same head coefficient barrier and per-layer dimensions.
Fixed R4 encounters the attention boundary 96 or 192 times per token.
Adaptive execution encounters it at the actual selected depths.
The original gate still needs a complete loop-normalized hidden state.
Changing a BF16 hidden value can change that gate's decision.
The method supplies no right to skip its hazard ordering or `last_exited` publication.

All four model/mode cells have source/algebra analysis only.
Adaptive Base 2.6B remains `ADAPTIVE_2P6B_UNVALIDATED`.
No P2/P4 implementation or performance result exists for this hypothesis.
P8 has structural accounting only, not a measured or fitted latency projection.

## BF16 cast falsifier

The native path consumes a rounded attention result before Wo.
Moving Wo before the merge can remove that cast even when the real-arithmetic identity holds.
The software control uses one head, two value coordinates, equal partition weights, and BF16-representable operands:

```text
V0 = [1,1]
V1 = [1.0078125,1]
Wo = [1,-1]
unrounded merged attention = [1.00390625,1]
BF16 merged attention = [1,1]
```

The source-order output is 0.
Projection before the merge gives 0.00390625 after the final BF16 cast.
This rejects universal bitwise identity for the simple transformed path.
It does not measure native accumulation order, task quality, or gate divergence.
A numerically reordered proposal would require those separate tests.

An exact implementation can retain each head's native rounding before projecting that completed head.
It can overlap that projection with unfinished other heads when resources permit.
This is a legal generic head-streaming/fusion opportunity.
It does not project an unfinished head as if its global normalization were complete.

## Nearest generic mechanisms and existing cards

FlashInfer explicitly defines partial attention outputs with per-head LSE and their merge.
Its [recursive-attention tutorial](https://docs.flashinfer.ai/tutorials/recursive_attention.html) also describes KV partition work and result merging.
The [merge API](https://docs.flashinfer.ai/generated/flashinfer.cascade.merge_state.html) preserves the head axis.
The installed `cascade.py:42–95` uses an output tensor matching the input value dtype and FP32 LSE.
This is a direct generic state-merge control, not a new recurrent mechanism.

ClusterFusion gives a strong H100 operator-streaming control.
Its [paper](https://papers.neurips.cc/paper_files/paper/2025/file/3760d0ea4709a913a4804f4b4c073836-Paper-Conference.pdf), §3.2 and Algorithm 3, keeps each head's dependencies within a cluster.
It completes statistics, rescales/reduces attention, then applies output projection on chip.
The inspected [H100 kernel](https://github.com/xinhao-luo/ClusterFusion/blob/8755446cb495a09e8bbaaf62fdda333c82ee7ade/include/H100/llama/kernel.cuh#L507-L618) follows that order.
It uses FP16 intermediates and half output atomics.
The paper's §4 evaluates other checkpoints on one H100, not BF16 Ouro distributed execution.
Its intracluster primitives are not evidence of a cross-GPU ownership protocol.

| Existing cards | Difference or direct overlap |
|---|---|
| C01, C16 | Those pool ready physical operations/rows. This question concerns a fresh head/KV-partition state inside one attention operation. |
| C02, C03, C14 | Remote MLP, depth arenas, and causal PP do not implement this projected-state merge. Their state/ownership accounting remains relevant. |
| C04 | Generic layout propagation is a direct risk. The new derivation isolates head-specific softmax coefficients after output-basis mixing. |
| C05, C06 | No trajectory correction or dense base-plus-delta prediction occurs. |
| C07, C09, C13 | No token branch, partial LM-head decision, or gate-control speculation occurs. |
| C08, C19 | No compressed transport or changed KV representation supplies the claimed saving. |
| C10, C12, C15 | Historical K/V and RoPE remain unchanged. The state is current-query dependent, not rematerialized history or a committed feature summary. |
| C11 | The work is not encoded quorum computation. Extra projections come from sequence partitioning. |
| C17 | No RMS scalar moves across a projection. Multiple head-specific softmax coefficients create a different algebraic barrier. |
| C18 | No equal-input cache hit or transition reuse occurs. |

These distinctions establish a separate question, not a new architecture claim.
Native TP already exposes independent heads and projection shards.
Generic split-KV attention already exposes sequence-partition work.
Generic head streaming can connect completed head outputs to Wo without a global materialization.
Those controls remain stronger than a baseline with deliberately serialized operators.
Native asynchronous scheduling and whole-R4 graph replay must also remain enabled where applicable.

Shared physical Wo weights can reduce capacity relative to four materialized logical copies.
An alias-aware generic control can retain the same copies.
That observation does not alone reject a different physical ownership optimum.
This screen instead names concrete state expansion, extra contractions, missing coefficients, and BF16 costs.
It finds no evidence that those costs produce a profitable recurrence-specific optimum.

## Falsification, limitations, and actual work

The [CPU script](../scripts/attention_projection_merge_cpu_v1.py) passed four deterministic assertion groups.
The [raw artifact](../artifacts/attention_projection_merge_cpu_v1.json) records inputs, outputs, times, and structural counts.
The head-preserving identity and invertible-basis positive control both have maximum error 0.0 in their constructed cases.
The scalar merge, rank-deficient compact-state claim, and omitted BF16 cast fail their stated controls.
None of these controls contains random model weights or a research-model quality test.

Current verdict: reject the cheap mixed-state scalar merge.
Keep valid head streaming in the generic-engineering category.
Do not build a distributed engine from this screen.
A reopening needs a specified representation or ownership plan that removes the named costs under a matched numerical contract.
It also needs an incremental comparison with native TP and the applicable generic streaming/CP control.

Reproduction:

```text
python scripts/attention_projection_merge_cpu_v1.py
```

Work began at 2026-10-19 14:29:42 UTC.
The CPU control ran from 14:33:56.191410 to 14:33:56.191629 UTC.
The structured source observations record the segment end after report validation.
Concurrent worker intervals do not add campaign elapsed time.
