# C02/C16 refinement: source-owned attention and shared TP4 FFN

## Result and limits

Source-owned full-head attention can feed a shared TP4 dense FFN legally.
Different recurrent depths can share an FFN packet when they use the same physical layer weights.
Every row must retain its own KV, residual, original gate state, and continuation guard.
This is a concrete ownership protocol, not merely an expressible DAG.

The protocol has a direct generic DP-attention/TP-FFN prior in SGLang.
No new card follows.
The ownership tradeoff remains unmeasured for Ouro.
Neither the earlier remote full-MLP loss nor generic DAG expressibility proves that this hybrid loses.
This report refines C02/C16 without changing their frozen cards or reports.

The tradeoff has real costs and possible savings.
The hybrid retains one aggregate FFN copy and can amortize it over rows from multiple sources.
It replicates attention weights and changes two per-layer all-reduces into an input all-gather and an output reduce-scatter.
It does not reduce the number of collective calls automatically.
Padding, row readiness, graph execution, and attention layout determine whether the changed physical path helps.

This round executes source reads and scalar structural calculations only.
It executes no GPU, model, serving engine, or task-quality test.
All latency, quality, and native numerical-parity cells remain `NOT_RUN`.

## 1. Named protocol and actual dependencies

Call the scoped protocol `Fresh-Row AG/RS`.
The four GPUs retain complete attention matrices for all physical layers.
Each source owns all 16 heads of its requests' depth-indexed KV.
The GPUs retain ordinary TP4 channel shards of each dense FFN.
The physical FFN weights repeat across actual recurrent visits, not across distinct physical layers.

For each ready row at physical layer ell:

1. The source applies N1 to the fresh hidden/residual input.
2. It computes full Q/K/V, applies RoPE, and publishes current K/V into the matching logical-depth slot.
3. It computes attention using that slot's required committed history and fresh current K/V.
4. It applies the output projection, N2, the attention residual, and N3.
5. It submits the complete normalized FFN input u to a packet for physical layer ell.
6. All ranks agree on the packet order, owners, lengths, padding, and state versions.
7. All-gather replicates u over the TP4 FFN ranks.
8. Each rank computes its gate/up channels, SwiGLU channels, and partial down-projection output.
9. Reduce-scatter sums the down contributions and returns each complete output row to its source.
10. The source applies N4 and its retained residual before continuing.

N1–N4 are Ouro's four distinct block RMSNorms.
The source retains the attention residual y while the packet executes.
The packet needs u, not a transfer of historical KV or y.
N4 cannot consume an unreduced down-projection contribution.
The recurrent boundary norm still follows the complete 24/48-layer pass.
These dependencies follow native `ouro.py:68–102,125–158,198–209,260–281,355–374`.

A packet key includes physical layer, weight revision, dtype, numerical contract, and operator stage.
Its row identities include source, request, generation, token position, recurrent depth, and input version.
Depth need not be equal because the physical FFN matrices are equal.
Rows at different physical layers cannot share that same GEMM without a different grouped-weight operation.

The protocol is exact in real arithmetic.
All-gather and row routing can preserve BF16 words.
Full versus head-sharded attention can select different GEMM shapes and reduction paths.
AG/RS can also change down-reduction order relative to native all-reduce.
Mixed packets can change GEMM dispatch even with identical inputs and weights.
Therefore, the protocol does not establish native bitwise parity.
It requires matched numerical, gate, KV, free-running-token, and quality checks before performance claims.

## 2. Legal ready supply and mixed depth

Let B_i be source i's ready FFN rows for the same physical layer.
Let `B=sum_i B_i`, with `B<=C` for non-speculative decode.
One request supplies one active token front, not four simultaneous recurrent inputs.
Attention can execute independently across different ready source rows.
The shared FFN then exposes TP4 work over the packet's actual rows.

| System C | Balanced source row counts | Example depths at one physical layer | Maximum packet rows |
|---:|---|---|---:|
| 1 | [1,0,0,0] | [1] | 1 |
| 2 | [1,1,0,0] | [1,2] | 2 |
| 4 | [1,1,1,1] | [1,2,3,4] | 4 |
| 8 | [2,2,2,2] | [1,2,3,4,1,2,3,4] | 8 |

Staggered arrivals can produce the fixed-R4 examples.
They do not establish how often those fronts coincide at one physical layer.
Synchronous fixed-R4 requests already give native TP a common-depth C-row batch.
Delaying them to create depth diversity supplies no benefit.
At 24/48 different physical-layer queues, the ready packet can also be smaller than C.

Adaptive packets require each row's actual execution guard and the established terminal-KV policy.
Minimum R2 admits the second pass without an R1 continuation decision.
Exit-capable boundaries require the actual continuation decision unless a separate provisional-state contract applies.
No future exit trace can supply rows.
The original gate uses the completed loop-normalized hidden state, q=0.5, minimum R2, maximum R4, and `last_exited`.
A source can stop its row, but the other ranks must still follow the agreed collective sequence.
Empty sources may need zero-row or padded participation.
Terminal publication must preserve the baseline's selected state in all required deeper logical slots.
Local ownership removes no required publication or version check.

At C1, the same token's attention and FFN remain causally serial.
The idle ranks provide FFN shards, not future attention work.
At C2/C4/C8, attention from another row can overlap a packet only if dependencies and resources permit it.
All-rank FFN execution and collectives can contend with that source's attention.
Nano-batch overlap therefore needs an interference cost, not only a legal DAG.

## 3. Same-P controls and the changed critical path

| Control | Attention and KV | FFN | Per-layer model collectives |
|---|---|---|---|
| Native TP4 | Four Q/K/V heads per rank for every row | TP4 over the ready batch | Attention AR plus FFN AR |
| All DP4 | Source owns full attention and KV | Source executes full local FFN | None inside the block |
| Fresh-Row AG/RS | Source owns full attention and KV | Shared TP4 over gathered rows | FFN-input AG plus FFN-output RS |
| Generic DP-attention/TP-FFN | Same proposed ownership when configured | Same shared packet | Same AG/RS or explicitly charged equivalent |

AR means all-reduce, AG means all-gather, and RS means reduce-scatter.
Native TP already exposes the complete ready batch to its FFN.
The hybrid's FFN batch gain is relative to independent DP sources, not automatically relative to native TP.
All-DP sources remain independent and must not receive an artificial all-source barrier.
Keep native asynchronous scheduling and whole-R4 graph replay enabled in the fixed-depth native control.

For a synchronized packet, a conditional hybrid path is:

```text
max(source attention/N2/residual/N3 finish times)
  -> packet agreement/packing -> AG -> TP FFN partials -> RS
  -> source N4/residual -> source continuation
```

Each included row also pays its wait for packet admission.
Unequal attention lengths, source loads, depths, and gate work can increase that wait.
The best generic control can release smaller packets instead of waiting for all sources.
That trades batching against startup and queue delay.

Against native TP, the removed attention AR must pay for changed attention kernels, AG, and the changed FFN completion path.
The FFN AR becomes RS, not zero communication.
Against all DP, pooled/sharded FFN savings must pay for both exchanges and new synchronization.
The earlier isolated-source remote full-MLP test measures neither complete comparison.
It sends work to a full-MLP owner, rather than to TP4 channel shards with a larger gathered packet.

The hybrid also needs an execution strategy for packet-size changes, row ownership, and unequal source progress.
Its scheduling and graph costs cannot disappear from the comparison.
Native scheduling already covers much of the measured cached-decode window.
No additional host-launch saving follows automatically from this hybrid.

## 4. Collective bytes and startups

For P=4 and D=2048, a B-row BF16 activation contains `S=4096*B` bytes.
In a simple equal-size ring model, two ARs send `4*(P-1)/P*S = 3*S` bytes per rank.
AG plus RS sends `2*(P-1)/P*S = 1.5*S` bytes per rank when rows balance without padding.
These are logical send bytes, not measured link traffic or a latency model.
Native custom all-reduce need not use this ring algorithm.

Standard equal-count AG/RS needs a common per-owner count.
With balanced low-C assignment, `B'=P*max_i(B_i)` rows enter the padded global packet.
Only B rows are useful.

| C=B | Useful owner counts | Padded B' | Native ring bytes/rank/layer | Hybrid padded ring bytes/rank/layer | Hybrid/native byte ratio |
|---:|---|---:|---:|---:|---:|
| 1 | [1,0,0,0] | 4 | 12288 | 24576 | 2 |
| 2 | [1,1,0,0] | 4 | 24576 | 24576 | 1 |
| 4 | [1,1,1,1] | 4 | 49152 | 24576 | 0.5 |
| 8 | [2,2,2,2] | 8 | 98304 | 49152 | 0.5 |

Padding also makes the FFN process extra rows unless the kernel has an explicit valid-row path.
This illustration grants native TP its B-row shape and the hybrid minimal equal-count padding.
Actual graph buckets need their own accounting.
Variable-count AG/RS can avoid this padding.
An ideal aggregate accounting then gives half the two-AR payload at every B, but not balanced per-rank work or lower latency.
It still needs supported variable-count collectives, matching counts/order, graph buffers, and metadata.
Broadcast/reduce for one owner is another generic special case, not free AG/RS support.

Both native and hybrid have two collective calls per physical-layer visit under this protocol.
Fixed R4 gives 192 calls for Base 1.4B and 384 for Base 2.6B in both plans.
AG and RS each need a startup and completion dependency.
Their internal ring steps can differ from AR, but call count alone does not predict startup latency.
At C4, the illustrative fixed-R4 payload becomes 4.5/2.25 MiB per rank for native/hybrid Base 1.4B.
Base 2.6B gives 9/4.5 MiB.
These small-message volume examples establish no speedup.

## 5. Weight, compute, KV, and terminal costs

One physical attention stack slice contains 32 MiB of BF16 Q/K/V/O weights.
One FFN contains 66 MiB.
The following counts cover only those seven dense matrices.

| Base | Physical layers | Native TP4 core/rank | Hybrid core/rank | All-DP core/rank | Hybrid extra versus TP4 |
|---|---:|---:|---:|---:|---:|
| 1.4B | 24 | 588 MiB | 1164 MiB | 2352 MiB | 576 MiB |
| 2.6B | 48 | 1176 MiB | 2328 MiB | 4704 MiB | 1152 MiB |

Native TP retains one aggregate attention/FFN copy.
The hybrid retains four aggregate attention copies and one aggregate FFN copy.
All DP retains four aggregate copies of both.
Every control can reuse its physical weights across recurrent visits.
Comparing the hybrid with four materialized logical cores would be an invalid capacity advantage.

For one ready batch per active source, let a count active attention owners.
Under an explicit one-read-per-matrix-per-batch operand model, native TP uses 32 MiB of aggregate attention operands.
The hybrid and all DP use `32*a` MiB.
This is not measured HBM traffic.
Caching, kernel tiling, and repeat loads can change actual transactions.

| C | a | Native attention operands | Hybrid attention operands | Native core operands | Hybrid core operands | All-DP core operands |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 32 MiB | 32 MiB | 98 MiB | 98 MiB | 98 MiB |
| 2 | 2 | 32 MiB | 64 MiB | 98 MiB | 130 MiB | 196 MiB |
| 4 | 4 | 32 MiB | 128 MiB | 98 MiB | 194 MiB | 392 MiB |
| 8 | 4 | 32 MiB | 128 MiB | 98 MiB | 194 MiB | 392 MiB |

Do not multiply these weight operands by row count again.
Two rows in one source GEMM can share its weight reads.
The aggregate useful attention FLOPs remain unchanged for the same C rows.
The hybrid changes where those FLOPs execute and the shapes of the contractions.
It does not duplicate the same row's attention across all sources.
At C1, one source executes full-head attention while native TP spreads that row's heads and projections.
At balanced C4/C8, aggregate attention/KV work can balance, but attention weights have less row amortization per source.

Ouro has 16 MHA KV heads.
TP4 partitions them into four heads per rank without the MLA duplication that motivates some DP-attention systems.
Hybrid request sharding and native head sharding retain the same aggregate logical KV words.
For balanced C divisible by four and equal histories, they also retain the same KV capacity per rank.
At C1, the hybrid source instead retains four times native TP4's per-rank KV, while other sources retain none.
Unequal request lengths and allocated adaptive depth slots can further unbalance source capacity and memory work.
No aggregate KV capacity saving follows from changing ownership alone.

The tables exclude norms, loop norms, gate, embeddings, LM head, metadata, padding, communication buffers, and graph pools.
Each complete BF16 embedding or LM-head matrix is 192 MiB for vocabulary size 49152 and D=2048.
Both configs set `tie_word_embeddings=false`.
Replicating those matrices at each source costs capacity and reads.
Keeping their native vocabulary sharding instead requires the corresponding row dispatch and output/decision path.
Neither choice is free or specified by the FFN protocol alone.

The original gate remains at the source after loop norm.
Its selected output, ordered hazards, and terminal KV publication need the established reference semantics.
All sources and FFN ranks also need consistent retirement and packet sequence information.
No graph may skip a required collective merely because one local source exits.

## 6. Primary generic controls and actual source limits

The inspected official SGLang checkout is revision `78da62519012a06833ae37ea9a214d101c2963b8`.
Seven inspected files match that commit exactly.
The source artifact retains their hashes.
No SGLang module executes.

The [communicator, lines 1258–1285](https://github.com/sgl-project/sglang/blob/78da62519012a06833ae37ea9a214d101c2963b8/python/sglang/srt/layers/communicator.py#L1258) normalizes source rows before DP gathering when attention TP size is one.
Its lines 1504–1523 provide the output-scatter path, including permitted RS and RSv cases.
This directly covers the proposed ownership boundary.

The [dense DeepSeek FFN constructor, lines 2340–2359](https://github.com/sgl-project/sglang/blob/78da62519012a06833ae37ea9a214d101c2963b8/python/sglang/srt/models/deepseek_v2.py#L2340) leaves the FFN TP rank/size at their default unless full-DP dense mode applies.
Its dense MLP uses column gate/up and row down projections at lines 249–284.
The decoder enables RS and publishes its reduction mode around the FFN at lines 2371–2379 and 2471–2509.
This is a dense-FFN mechanism inside an MLA/MoE model, not a dense Ouro adapter.

The [linear layer, lines 1670–1695](https://github.com/sgl-project/sglang/blob/78da62519012a06833ae37ea9a214d101c2963b8/python/sglang/srt/layers/linear.py#L1670) honors the decoder's reduction flags.
It can leave partial results for the communicator instead of performing an extra AR first.
A proposed Ouro path must do the same explicitly.
AR followed by a local slice is not the claimed RS saving.

The [DP helpers, lines 568–601 and 823–917](https://github.com/sgl-project/sglang/blob/78da62519012a06833ae37ea9a214d101c2963b8/python/sglang/srt/layers/dp_attention.py#L568) distinguish equal-count gather, variable counts, fallback, and output reduction.
Its padding enum at lines 88–98 distinguishes MAX_LEN and SUM_LEN.
The exact selected path matters because some alternatives use AR rather than AG/RS.
No FP8 communication option from this code belongs in the unchanged BF16 comparison.

The [scheduler metadata path, lines 152–214 and 451–497](https://github.com/sgl-project/sglang/blob/78da62519012a06833ae37ea9a214d101c2963b8/python/sglang/srt/managers/scheduler_components/dp_attn.py#L152) exchanges counts and graph eligibility.
It can create idle batches when other sources still have work.
This is evidence of real synchronization costs, not permission to omit empty ranks.

The inspected Llama attention still uses global TP size at `models/llama.py:156–198`.
Thus, these sources do not establish that stock SGLang already enables this hybrid for every dense MHA model.
They establish concrete generic primitives and a supported dense-FFN coupling in the inspected DeepSeek path.
Ouro still needs its four norms, recurrent state identities, and original gates.

The official [SGLang design discussion](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/#dense-ffns) compares DP/TP attention and dense FFNs.
Its MLA cache-duplication argument does not transfer to Ouro's head-sharded MHA.
Its throughput workloads and TP32 alignment example do not establish H100 low-C Ouro costs.
The inspected I=5632 gives 1408 FFN channels per TP4 rank, which is divisible by 128.
Do not import its quoted FFN collective count instead of auditing Ouro's actual two-AR block.

[Ulysses, §3.1](https://arxiv.org/html/2309.14509v1) exchanges sequence/head layouts with all-to-all around attention.
It is not this request-owned full-head attention plus FFN AG/RS protocol.
It is a relevant generic layout control, not a direct cached-decode ownership implementation or an inference speed prediction.

[NanoFlow, §4](https://arxiv.org/html/2408.12757v2) supplies resource-aware operator/nano-batch overlap and an interference model.
Its pinned [split code](https://github.com/efeslab/Nanoflow/blob/f179a907828b87e042126e585880301ff5b2c62a/core/nanobatchSplit.py) partitions rows.
Its [executor](https://github.com/efeslab/Nanoflow/blob/f179a907828b87e042126e585880301ff5b2c62a/core/executor.py) event-orders the declared dependencies.
They do not supply an unchanged Ouro implementation or a low-C latency result.

## 7. What recurrence changes and what remains open

Shared FFN weights create a real compatibility fact.
Rows at different recurrent depths can join one same-physical-layer GEMM without switching matrices.
Rows at different untied layers cannot claim that same weight-read amortization.
That fact can matter physically, even though a generic aliased executor can represent the graph.

WaveFront already supplies a direct mixed-depth union control.
A generic DP-attention/TP-FFN executor can use the same compatible packet once its per-row state protocol exists.
Native TP can also consume those same FFN rows under an appropriate mixed-depth attention/cache adapter.
Neither stock-runtime support nor graph expressibility proves equal implementation cost.

This refinement identifies no additional recurrence-only owner or batch discriminator beyond shared-weight compatibility.
It also establishes no zero-benefit theorem.
The remaining concrete question is whether local full-head KV ownership plus AG/RS lowers the complete exposed path enough to offset replication and queue costs.
That is an unmeasured generic ownership question within C02/C16.

A bounded reopening needs one packet trace with actual per-source ready times, depths, layer identities, and owner loads.
The comparison must retain the same useful rows and count padding, norm/residual work, exchanges, gate publication, and packet waits.
It must compare all-DP4, native TP4, and the same generic DP-attention/TP-FFN protocol.
Only an incremental cost that depends on physical sharing can support a separate recurrent claim.
This source-only refinement authorizes no GPU test or engine.

## 8. Reproducibility and actual work

The [structured artifact](../artifacts/dp_attention_tp4_ffn_refinement_v1.json) retains formulas, four C rows, both core-capacity rows, queries, primary locators, hashes, and scope limits.
Scalar Python calculations reproduce the byte and operand tables.
They import no Torch, vLLM, SGLang, or CUDA module.
The calculations do not test model quality or native arithmetic.

Work began at `2026-10-19 16:55:09 UTC`.
The validation cutoff is `2026-10-19 17:03:17 UTC`.
The observed segment spans 488 seconds, or 8 minutes 8 seconds.
JSON parsing, 16 local hashes, all structural table calculations, and report lint pass.
Final timestamp insertion and handoff follow that cutoff.
Only new scoped report/artifact files change.
Concurrent worker intervals do not add campaign elapsed time.
