# Round 6: redundant compute instead of a collective

## Decision and scope

This round produces zero new candidate cards.
It tests one distinct memory/ownership hypothesis, rather than another stale-state or compression variant.
The hypothesis replicates each physical MLP across the TP ranks.
Every rank computes the same complete ready rows.
This removes the MLP-output reduction, but duplicates dense work and weight operands.

The inspected dependency graph permits this transformation in real arithmetic.
The tested unchanged component implementation does not justify an engine.
Full local MLP execution loses all 16 low-row median comparisons against TP plus reduction.
Those comparisons cover both Base checkpoints, P2/P4, and rows 1/2/4/8.
The component latency ratios are 0.739699–0.807682×, defined as TP latency divided by full-local latency.
These ratios are not full-model speedups.

This is a scoped negative result, not a proof against every replication layout.
Native fused kernels, full-stack reuse distance, adaptive execution, and P8 remain untested for this transformation.
No new GPU or model execution occurs in this round.

## New assumption and nearby rejected families

C02 sends a ready activation to a remote MLP owner and returns its result.
This hypothesis instead uses the complete MLP input already present on every TP rank.
It introduces no activation round trip.
Each rank owns a complete MLP output before the next norm.

C04 examined carrying partial contributions through later consumers.
This hypothesis carries no partial contribution across a norm.
It computes every missing contribution locally before N4.
It changes physical redundancy, not mathematical input readiness.

The hypothesis uses no memoization, predicted hidden state, correction round, compressed KV, or zero-channel test.
It does not need different requests to reach the same physical layer.
It also does not claim that redundant arithmetic creates new future-depth work.

## Actual causal inputs and operation sequence

Both Base models use hidden width D=2048 and MLP width I=5632.
They have 16 query heads, 16 KV heads, and four recurrent passes.
Their physical layer counts L are 24 and 48.
The pinned [semantics audit](ouro_semantics_audit.md) records the checkpoint identities.

The official block executes these semantic operations:

```text
z = N1(x)
a = Wo(attention(RoPE(Wq z), RoPE(Wk z), Wv z, required depth/layer KV))
y = x + N2(a)
u = N3(y)
m = Wdown(SiLU(Wgate u) * Wup u)
output = y + N4(m)
```

All four norms remain distinct.
The native runtime retains part of the residual separately.
Its stored hidden tensor alone is not the semantic block output.
The loop norm remains after each complete physical stack.

The conventional TP MLP uses column-sharded gate/up weights and row-sharded down weights.
Its down projection produces partial sums, then a reduction completes the output.
The proposed island instead applies full gate/up/down weights to the complete u on every rank.
Each rank then applies the unchanged N4 and residual operation.

This schedule requires these conditions:

1. The current attention output reduction completed.
2. Each rank holds the complete current residual and normalized u.
3. Each rank holds the same complete physical MLP weights.
4. Every rank uses the same row identities and actual continuation decisions.
5. The next attention reads the required committed KV for that row, depth, and layer.

Inspected local source locations are `ouro.py:68–102,260–281,367–374` and `linear.py:1680–1697`.
The official [block source](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) contains the four norms at lines 370–423.
No invented q/k normalization enters this derivation.

## Concurrent work and state owners

The concurrent unit is one complete MLP computation per rank for the same ready row batch.
Its result supplies that rank's next TP-attention input.
The rank-local results are useful, but their arithmetic is redundant across ranks.
The construction exchanges a collective for duplicated computation.

| Object | Owner and lifetime | Changed cost |
|---|---|---|
| Physical MLP weights | Complete immutable copy on every rank, reused across all actual passes | More HBM capacity and repeated weight operands |
| Attention weights | Ordinary TP shards | Unchanged |
| Historical/current KV | Ordinary head-sharded depth/layer state | Unchanged identities and capacity |
| Complete MLP input u | Replicated current row, after its predecessors | No new input transfer |
| MLP gate/up/product scratch | Local full-width scratch on every rank | More intermediate elements than TP |
| MLP output and residual | Complete local result and unchanged residual | Omit only the MLP-output reduction |
| Gate and selected output | Original model/runtime owners | No policy change or early decision oracle |

At C1, the MLP has one ready row on each rank.
R2 cannot supply another row before R1 supplies its complete next-pass input.
At C2/4/8, the ideal ready batch has at most 2/4/8 rows for one physical operator.
Arrival times, different depths, and gate decisions can make that batch smaller.
Four passes do not multiply these bounds by four.

Assigning different rows to the full replicas would create a different DP/TSP construction.
It would need the relevant activation distribution and collection.
This report does not use that change to rescue this hypothesis.

## Fixed and original-gate adaptive modes

Fixed R4 removes L×4 MLP reductions per token.
These counts are 96 for Base 1.4B and 192 for Base 2.6B.
The same numbers of attention-output reductions remain.
These are operation counts, not exposed-time measurements.

The adaptive reference uses the original gate, q=0.5, minimum R2, maximum R4, and `last_exited`.
The gate consumes the completed loop-normalized hidden state.
Only actual continuation decisions supply additional committed rows.
The island cannot assume that a request will reach R3 or R4.

For one request with actual depth d, the structural reduction count becomes L×d.
Different request depths can divide the ready frontier.
The runtime still needs ordered hazard accumulation and the original selected output.
It must preserve terminal-depth publication into uncomputed deeper KV slots.
`last_exited` retains its allocated logical slots.

| Model/mode | Legal real-arithmetic rewrite | Current physical evidence | Remaining validation |
|---|---|---|---|
| 1.4B fixed R4 | Yes, with complete current input | Layer0 component comparison only | Native fused and full-stack numerics/performance |
| 1.4B original adaptive | Yes, subject to actual gates and state publication | No adaptive island run | Gates, selected output, free-running quality, performance |
| 2.6B fixed R4 | Yes, with complete current input | Layer0 component comparison only | Native fused and full-stack numerics/performance |
| 2.6B original adaptive | Source-level feasibility only | `ADAPTIVE_2P6B_UNVALIDATED` | Baseline support and all island validation |

P2/P4 component evidence exists for both checkpoint sizes.
Every P8 performance cell is `NOT_RUN`.
The capacity equations below apply to P8 without predicting its latency.

## Physical sharing and charged capacity

One full BF16 MLP occupies `3×D×I×2 = 69,206,016` bytes, or 66 MiB.
All physical MLP weights occupy 1584 MiB for 1.4B and 3168 MiB for 2.6B.
Replication adds `(1−1/P)` times those bytes per rank relative to TP.

| Model | P | TP core per rank, MiB | Replicated MLP plus TP attention, MiB | Added MLP bytes per rank, MiB |
|---|---:|---:|---:|---:|
| 1.4B | 2 | 1176 | 1968 | 792 |
| 1.4B | 4 | 588 | 1776 | 1188 |
| 1.4B | 8 | 294 | 1680 | 1386 |
| 2.6B | 2 | 2352 | 3936 | 1584 |
| 2.6B | 4 | 1176 | 3552 | 2376 |
| 2.6B | 8 | 588 | 3360 | 2772 |

These core counts include the seven dense block matrices only.
They exclude embeddings, LM head, norms, gate, KV, graph buffers, allocator overhead, communication buffers, and scratch.
The island retains full KV capacity at 786432 and 1572864 bytes per cached token before division across TP ranks.
No allocated-capacity value is an HBM-traffic measurement.

Physical recurrence matters to a possible capacity optimum.
Four materialized MLP copies would occupy 6336 or 12672 MiB per rank in this design.
The shared physical copies occupy only 1584 or 3168 MiB.
An alias-aware unrolled control can retain exactly those same physical copies.

That representability does not, by itself, reject a useful memory optimum.
The missing result is a capacity or latency advantage against the alias-aware conventional control.
The present design adds capacity relative to ordinary TP.
This round establishes no admission boundary where recurrence makes that added replication profitable.
It does not use a needlessly materialized baseline to manufacture such a boundary.

The measured H100 L2 capacity is 50 MiB per rank in the frozen component records.
One full MLP already exceeds that capacity.
The whole physical stack creates a much longer revisit interval.
This does not establish a cache-miss rate or measured HBM traffic.

## Critical path and extra work

Let `T_full(B)` denote complete local MLP latency for ready batch B.
Let `T_TP(B,P)` include the sharded MLP and its output reduction.
With unchanged neighboring operations, the island needs:

```text
T_full(B) + extra_island_control(B,P) < T_TP(B,P)
```

The removed cost is the MLP-output collective on the current path.
No attention collective, gate, norm, head, sampling operation, or future dependency disappears.
Native asynchronous scheduling and whole-R4 CUDA Graph replay remain enabled in the proper fixed baseline.
There is no additional host-dispatch saving to count automatically.

Each complete MLP requires `6×B×D×I = 69,206,016×B` dense FLOPs per rank.
The aggregate MLP work becomes P times ordinary TP work.
Each rank reads complete logical weight operands rather than a 1/P shard.
These operand counts are not measured off-chip reads.
Batching can reuse weight operands within a GEMM, so C must not multiply weight traffic automatically.

Gate/up and product scratch also become full-width on every rank.
Their exact allocation depends on fusion and buffer reuse.
Model loading must create the complete immutable replicas before execution.
Changing native graph captures, weight loaders, and numerical tests is additional engineering work.

## Cheapest screen and actual result

The existing `component_probe.py:74–96` already executes the nearest unchanged implementation on every rank.
Its local arm uses the full layer0 MLP weights and the same complete input on each rank.
Its TP arm uses exact weight slices and a BF16 all-reduce.
Each arm uses 31 trials and 16 operations per captured graph replay.
The recorded metric uses the slowest participating rank per complete operator trial.

The new [CPU analysis](../scripts/candidate_round6_replica_screen.py) reads those frozen records.
It runs no GPU library or model.
Its [output](../artifacts/candidate_round6_replica_screen_v1.json) retains all tested row sizes and source hashes.

| Base model | P | Rows1 full / TP, ms | Rows1/2/4/8 TP/full ratio range | Full-local faster low-row medians |
|---|---:|---:|---:|---:|
| 1.4B | 2 | 0.037074 / 0.029944 | 0.768144–0.807682× | 0/4 |
| 1.4B | 4 | 0.037246 / 0.027600 | 0.741019–0.758929× | 0/4 |
| 2.6B | 2 | 0.037048 / 0.029016 | 0.755779–0.783200× | 0/4 |
| 2.6B | 4 | 0.037134 / 0.027468 | 0.739699–0.752241× | 0/4 |

This is a failed unchanged-component cost screen at the priority row counts.
The trials were not randomized paired full-stack runs.
The report supplies no new confidence interval or E2E estimate.
The component inputs repeat selected captured rows.
They do not constitute diverse request-quality evaluation.

The outside-target result must also remain visible.
At rows512 and P2, full-local MLP beats TP in both checkpoint records.
The TP/full ratios are 1.099925× and 1.100417×.
P4 does not show that rows512 result.
This prevents a universal claim that full replication always loses.
It does not supply 512 ready decode rows at C≤8 or a validated prefill gain.

The native MLP uses merged gate/up projection and `SiluAndMul`.
The component probe uses separate gate/up `F.linear` calls and `F.silu`.
Both component arms share that implementation difference.
They do not settle the best native fused implementation.
Any future kernel improvement also belongs in the matched TP control.

## BF16 boundary control

The real-arithmetic MLP function remains identical, but bitwise parity is not guaranteed across TP layouts.
The executable contains a deterministic non-model down-projection example.
Its activation is `[1,1,1,1]` and its weights are `[256,1,−256,0]`.
The complete real sum is 1.
Two shard sums are 257 and −256.
BF16 rounds those shard outputs to 256 and −256 before reduction.
The resulting output is 0, while one complete projection rounds to 1.

This CPU control shows why shard-boundary casts matter.
It does not reproduce a native GEMM order or measure model quality.
The frozen real-input component records also report nonzero full-versus-TP numerical differences.
Adaptive gates can amplify such differences through changed depth decisions.
Any engine would require matched free-running tokens, gate traces, KV validation, and task-quality tests.

## Primary generic controls and novelty limit

GSPMD explicitly supports replicated tensors and mixed per-operator partitioning.
This provides a generic vocabulary for full MLP replication beside partitioned attention.
It does not establish that an existing serving engine chooses the best Ouro ownership policy.
See Xu et al., [GSPMD v2](https://arxiv.org/html/2105.04663v2), §§1,3.1,3.2, December 23, 2021.

Ong's [dynamic-partitioning report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-108.pdf) gives a closer concrete tradeoff.
Its §4.3 replicates the attention output projection and replaces its reduction with an all-gather.
Its weight-gathered MLP instead gathers weights transiently and partitions input rows.
That MLP is not the redundant same-row island examined here.
Sections 4.3–4.4 and 5.1 directly charge added compute, weight memory, and communication.
This is a generic inference precedent, not an Ouro measurement.

The [PyTorch 2.14 API](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module) supports per-submodule TP plans.
The API supports module-level composition, not evidence of a validated Ouro implementation or optimum.
This bounded source screen verifies no official implementation of this exact island.

Recurrence changes the physical replication bill compared with four independent parameter sets.
It does not change the duplicate-work factor or produce independent future hidden states.
No observed capacity boundary or incremental latency gain distinguishes this island from generic selective replication.
The current verdict is `GENERIC_ONLY` plus `NO_NET_HEADROOM` for the tested unchanged low-row component path.

## Stop and reopening rule

Do not build a new engine from this screen.
Do not rerun a full benchmark to repeat the same unchanged-kernel comparison.
Do not infer a universal impossibility result.

A reopening needs one specific new fact:

- A matched native fused MLP result reverses the low-row cost inequality.
- A measured full-stack ownership effect reverses it without unequal capacity or kernels.
- A real admission boundary makes the tied physical replicas useful against an alias-aware same-P baseline.

The cheapest new probe would then test that changed fact at one shape.
It must include complete neighboring norms/residuals, both required communication directions, and unchanged BF16 semantics.
P8 advertised bandwidth alone does not supply the changed fact.

## Reproduction and time

```text
python scripts/candidate_round6_replica_screen.py
```

The published artifact command adds its fresh `--output` path.
The program refuses to overwrite that output.
The source observations contain query text, primary locators, hashes, and scope limits.
The report does not change frozen evidence or coordinator files.

Actual discovery began at 2026-10-19 13:57:08 UTC.
The CPU screen ran from 14:02:39.420149 to 14:02:39.436628 UTC.
The final segment end and elapsed time appear in the structured source observations.
This is one worker's elapsed interval.
Concurrent worker intervals do not add campaign elapsed time.
