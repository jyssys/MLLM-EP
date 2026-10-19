# Hydragen: bounded C07 transfer audit

Revision 1. Decision: `GENERIC_ONLY`, with Ouro integration and numerical parity `UNTESTED`.
Hydragen strengthens the required baseline for distributed branch ownership.
Prefix sharing alone does not establish a new candidate.
No GPU, model, package, or environment changes occur in this audit.

## Primary source and inspected implementation

The latest observed paper is arXiv:2402.05099v2, revised on 2024-05-13 at 08:49:44 UTC.
The primary record links the official repository.
Its current `HEAD` remains `1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd`.
The implementation targets Llama-family models, not Ouro.
It requires no trained predictor or weight update. [Primary record](https://arxiv.org/abs/2402.05099), [official repository](https://github.com/ScalingIntelligence/hydragen/tree/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd)

| Inspected locator | Mechanism or limit |
|---|---|
| Paper §3.1, Eqs. 4–5, App. A | Partition attention and merge partial outputs with their log-sum-exp values. |
| Paper §§3.2–3.3 | Batch different queries over identical KV. Extend sharing to a prefix tree. |
| `attention.py:250–280,340–354` | Reshape queries across sequences for shared attention. Compute private suffix attention separately. Merge the results. |
| `attention.py:21–43,46–151,154–174` | Stable merge using a maximum LSE. Two partial results use Triton. Other counts use Torch. |
| `llama.py:58–115,173–234,335–346` | Preallocated shared caches and per-completion caches. Explicit append and truncate operations. |
| `tp.py:73–132` | Shard attention heads and MLP dimensions. Reduce attention and MLP outputs across ranks. |
| `README.md:483–490` | No full serving engine or continuous batching. Partial outputs introduce rounding differences. |

Source links: [attention operation](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/hydragen/attention.py#L250), [cache implementation](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/hydragen/llama.py#L173), [TP implementation](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/hydragen/tp.py#L73), [limitations](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/README.md#L483).

## Execution and numerical contract

Let `O_i(q)` denote attention over one disjoint KV partition.
Let `l_i(q)` denote its log-sum-exp value.
For each query independently, the stable merge is:

```text
m(q) = max_i l_i(q)
a_i(q) = exp(l_i(q) - m(q))
O(q) = sum_i a_i(q)*O_i(q) / sum_i a_i(q)
```

This identity is exact in real arithmetic.
It does not require the queries to match.
It requires each query to retain its own partial output, denominator, mask, and key coverage.
Identical prefix KV allows a matrix of different queries to reuse that prefix.
The suffix remains private.
The algorithm still computes each query's scores and weighted values.
It does not reuse a previous query's attention result. [Versioned paper, §§3.1–3.3](https://arxiv.org/html/2402.05099v2)

The non-padded code groups contiguous query rows through a reshape.
The shared batch size must divide the query batch size.
Irregular branch trees therefore need grouping metadata or an adapter.
They are not an existing Ouro integration. [Function contract and reshape](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/hydragen/attention.py#L177)

The official README documents FP32 LSE values and FP16/BF16 partial attention outputs.
The merge cannot recover precision that partial-output rounding removes.
Thus mathematical exactness does not establish bitwise equality with native BF16 attention. [Numerical precision](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/README.md#L488)

The attention test uses FP16, absolute tolerance `2e-3`, and mean relative difference at most `5e-3`.
The end-to-end test uses teacher-forced reference tokens on Sheared-LLaMA-1.3B.
It checks logits with absolute tolerance `0.75` and mean relative difference below `0.05`.
These checks do not establish free-running token identity, BF16 identity, or original-gate identity for Ouro. [Attention test](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/tests/test_attention.py#L34), [end-to-end test](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/tests/test_e2e.py#L23)

## Which Ouro queries can share a prefix?

The following rules are transfer deductions, not measured integration results.
The safe grouping identity includes the request generation, prefix lineage, KV version, physical layer, recurrent depth, head shard, and visible range.
The query vectors themselves can differ.
Every grouped query must see the whole shared range under the same score semantics.
A causal mask must not expose another branch's private append.

| Query relation | Permitted shared work |
|---|---|
| Same depth, same layer, same immutable committed prefix | Shared prefix attention, even with different query vectors or future token positions. |
| Same depth, sibling branches after a fork | The common ancestor prefix only. Their private appends remain separate. |
| Same depth, descendants with a longer common lineage | A verified common subtree segment can form another shared level. |
| Same depth but different histories or state versions | No sharing merely because token positions or depths match. |
| Different depths under fixed R4 | No automatic prefix equality. Shared physical weights are insufficient. |
| Adaptive depths with explicit terminal `last_exited` copies | Entrywise equality can follow from the copy provenance. It does not establish equality of the whole prefix. |
| Different visibility limits | Use only a valid common range, or a causal grouped operator that preserves each row's cutoff. |

LoopSpec already borrows a parent's prefix at the same role and depth.
Its cursor dictionaries use recurrent depth, not batch row, as the key.
Forks defer prefix-mapping copies to the first graph.
They do not copy KV tensors at fork.
Private append slots remain owned until pruning and reader completion permit release.
Hydragen adds an attention-computation optimization to that storage sharing.
It does not replace branch lifetime control. [LoopSpec manager:207–260,357–429](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/branching/manager.py#L207)

LoopSpec already advances mixed-depth lanes through one recurrent batch.
An integration can preserve this shared projection/MLP batch while grouping attention by compatible KV.
Splitting the entire transformer execution by depth would unnecessarily weaken that control. [LoopSpec executor:293–339](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L293)

WaveFront addresses KV as `depth * physical_layers + layer`.
Its existing grouped path already combines same-slot queries into a causal varlen segment.
Distinct slots use separate segments.
Thus same-slot query grouping is also direct prior art.
Cross-branch immutable-prefix decomposition differs from that contiguous causal grouping. [WaveFront:247–304,345–372](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks.py#L247)

For fixed R4, later-depth inputs contain the preceding loop's output.
Their projected K and V need not equal shallow-depth K and V.
Full-R4 prefill therefore does not provide one depth-independent prompt cache.
For adaptive execution, terminal copies can create limited exact aliases.
The source copy path fills skipped deeper slots for each exited token.
The campaign must still preserve its original gate and exact terminal-copy semantics. [WaveFront depth addressing and terminal copies](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks.py#L108)

This audit does not validate original-gate speculative verification.
Attention rounding can change hidden states, gate decisions, and branch acceptance.
Neither prefix ownership nor a real-arithmetic proof removes those tests.

## Distributed ownership screen

The relevant alternatives are composable, not mutually exclusive:

1. One mixed-depth batch with ordinary attention.
2. The same batch with ordinary TP at the same P.
3. The same TP batch with generic shared-prefix attention.
4. Full replicas with branch ownership, each with the same prefix optimization when applicable.

Hydragen already combines shared-prefix attention with head-sharded TP.
Its TP code creates local head shards before cache allocation.
Attention and MLP outputs then use all-reduces.
No per-query remote KV transfer is inherent in that head-sharded baseline. [TP code](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/hydragen/tp.py#L90)

The following accounting assumes equally sized immutable prefix blocks.
It assumes ideal reuse within each group and complete head sharding across TP ranks.
It is not a measurement of kernel reads or HBM traffic.

```text
g = compatible depth/layer/prefix group
n_g = number of ready queries in g
K_g = bytes in g's complete prefix block
a_g = number of replica owners with at least one query in g

Independent logical prefix exposure = sum_g n_g*K_g
Grouped single-device exposure       = sum_g K_g
Grouped replica-owner exposure       = sum_g a_g*K_g
Grouped TP cluster exposure          = sum_r sum_g K_g/P = sum_g K_g
```

Actual kernels can reread KV across query tiles.
L2 can also eliminate some redundant HBM reads without this transformation.
Grouping adds partial-output, LSE, layout, and merge costs.
The expressions above identify a reuse opportunity, not a speedup or traffic lower bound.

The CPU check uses one block unit per group:

| Synthetic ready set | Independent units | Grouped units | Replica-owner units | Grouped TP cluster units |
|---|---:|---:|---:|---:|
| One query at each of four depths | 4 | 4 | 4 | 4 |
| Four same-depth queries on four replicas | 4 | 1 | 4 | 1 |
| Two branches at four depths, branch ownership | 8 | 4 | 8 | 4 |
| Same eight queries, depth ownership | 8 | 4 | 4 | 4 |

The first case has no inter-query prefix reuse.
The second case refutes an automatic replica advantage from prefix sharing.
The third case shows how branch ownership can fragment each shared group.
The fourth preserves locality, but adds cross-owner recurrent dependencies that this block count excludes.
None of these counts determines the fastest placement.
External concurrency C also does not specify any group's width or prefix identity.

A stationary prefix service is another possible generic decomposition.
It must receive each query and return that query's partial output and LSE.
Its caller still computes private attention and merges the results.
Queueing, transport, synchronization, and merge rounding all enter the comparison.
It cannot count a saved remote KV read against TP when TP already holds that head's KV locally.

An ownership optimum therefore needs more than a reduction in prefix copies.
It must beat TP with the same branch DAG and the same generic prefix optimization.
Charge tentative ancestor KV delivery, hidden-state movement, publication, pruning, and full weight replication.
The source does not provide that distributed Ouro protocol or a campaign-width optimum.
The campaign's existing [branch ownership contract](branch_ownership_cost_contract.md) remains applicable.

## Measured source scope and audit checks

The paper reports CodeLlama-13B decode throughput with TP8 on eight A100-40GB GPUs.
Its main attention microbenchmark uses one A100-40GB.
Appendix C.2 adds H100 and L40S attention microbenchmarks.
These are not distributed Ouro branch measurements. [Paper §§4.1–4.2 and App. C.2](https://arxiv.org/html/2402.05099v2)

Appendix D.1 excludes prefill through a one-token timing subtraction.
Appendix D.2 uses CUDA graphs, 1,000 warmup iterations, and 1,000 timed trials.
It flushes L2 between microbenchmark iterations.
These protocols do not establish low-width branch latency on the campaign runtime. [Experiment details](https://arxiv.org/html/2402.05099v2)

The official microbenchmark allocates BF16 queries and KV.
End-to-end loading instead uses checkpoint `dtype="auto"`.
This audit does not assign one precision to all paper datapoints. [BF16 microbenchmark](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/scripts/microbenchmark.py#L58), [end-to-end loader](https://github.com/ScalingIntelligence/hydragen/blob/1ea2255db0778f5cf47a4f72a535dc77a8b6b7bd/scripts/synth.py#L155)

The new CPU script checks two scalar decomposition cases and four grouping cases.
It uses Python binary64, not BF16 or the official CUDA kernels.
Both decomposition errors are below `1e-14`.
The script also checks the table's logical counts.
Run: `python scripts/check_hydragen_transfer_screen.py`.
This check supports the algebra only.
No latency or quality result follows.

## Changed-assumption decision and time

Hydragen supplies a stronger generic operator baseline for C07.
It does not change the recurrent dependency frontier.
It does not make different-depth prefixes equal.
It supplies no new original-gate guarantee or distributed branch commit protocol.
The C07 architectural decision remains `GENERIC_ONLY`.
Performance and adaptive correctness remain `UNTESTED` or `UNVALIDATED`, respectively.

Worker segment start: 2026-10-19 14:06:24 UTC.
Source retrieval and CPU-check cutoff: 2026-10-19 14:09:40 UTC.
Final verification cutoff: 2026-10-19 14:12:22 UTC.
The worker segment spans 358 seconds through that cutoff.
This worker interval is not additional campaign elapsed time.
