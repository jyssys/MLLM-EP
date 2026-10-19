# C07 branch ownership and cost contract

## Decision and comparison

C07 assigns verified speculative Ouro branches to full model replicas on P2 or P4.
The mandatory comparison applies ordinary TP2 or TP4 to the same LoopSpec branch batch.
Both methods must use the same target, proposal rules, branch DAG, verification, and output policy.
Internal branch count is not system-wide request concurrency.

The inspected evidence establishes no additional recurrence-specific execution axis for this replica placement.
LoopSpec already shares physical weights across mixed-depth lanes and shares immutable branch prefixes.
TP on that same batch retains both properties.
Replica placement adds full weight copies, history placement, cross-owner dependencies, and commit traffic.
It may still help a measured operating point, but no such result exists here.

Current verdict: `GENERIC_ONLY` as an architectural distinction, with distributed feasibility and performance `UNTESTED`.
Adaptive original-gate branch verification remains `UNVALIDATED`.
No GPU or model execution occurs in this audit.

## Pinned prior evidence and limits

The LoopSpec pin is `41aa710ed7c9ce908cffa03597be5d1df19d7b56`.
Its branch manager shares prefix KV through mapping indices.
Branches own new decode slots. Release excludes borrowed prefix slots.
See `sglang_recurrent/loopspec/branching/manager.py:106–144,207–250`.

Its executor advances active lanes through one runner with per-lane depths, branch IDs, and state slots.
See `sglang_recurrent/loopspec/executor.py:293–339`.
Its server rejects TP, PP, DP, EP, attention CP, and MoE DP values other than one.
It also rejects the inspected disaggregated modes at `sglang_recurrent/server/cli.py:88–118`.
Thus same-batch TP is a required control, not an already available measured baseline in that checkout.

The Ouro adapter retains but does not evaluate the original gate at `sglang_recurrent/modeling/ouro.py:85–98`.
The paper describes target-preserving rejection sampling under its stated assumptions.
This audit does not independently inspect a complete rejection or distributed commit implementation.
It defines the required distributed contract below instead of attributing that contract to existing code.

The WaveFront pin is `434520932897fcc1e26898891c7e2feaa6ab099f`.
Its Ouro implementation addresses each row's recurrent depth and replicates terminal KV after early exit.
See `src/decoding/ouro_blocks.py:128–132,264–267,294–297,381–397`.
Its scheduler verifies positions in order and rolls back after rejection at `src/decoding/scheduler.py:155–224`.
This evidence does not establish matched original-gate adaptive distributed correctness.

These source findings come from the campaign's pinned literature inspection and its supplied excerpts.
No local LoopSpec or WaveFront execution occurs here.
Primary sources: [LoopSpec branch manager](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/branching/manager.py#L106), [LoopSpec executor](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L293), and [WaveFront scheduler](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/scheduler.py#L155).

## Matched work unit

The work unit is a versioned branch node, not one recurrent pass treated as an independent model.
Each node records request generation, lineage, input position, recurrent depth, physical layer, operator, and state version.
The matched batch contains the same nodes in both placements.
The matched comparison includes discarded nodes and target verification work.

A prefix can be shared only when its token IDs, state versions, and relevant conditioning history match.
Shared weights do not make different-depth KV equal.
The same input position at R1 and R4 retains distinct state under fixed R4.
An accepted output token does not necessarily add its own KV immediately.
KV appears when the runtime processes that token as an input.
Therefore, every byte count below uses newly cached input positions, not raw speculative or accepted output counts.

## Ownership and lifecycle

The coordinator owns the authoritative request generation and committed input interval.
Each replica owns a full model and a declared local history version.
Each branch owns private append rows and hidden states on its assigned replica.
It borrows immutable local prefix rows through reference-counted mappings.
An ordinary device pointer cannot borrow another GPU's state without a real remote-access or transfer protocol.

1. Pin the target, tokenizer, gate, q, depth limits, KV policy, precision, and sampling contract.
2. Establish the required immutable history on each participating owner.
3. Create a branch version with its parent version and explicit input span.
4. Allocate private append rows before branch computation.
5. Execute a node only after its matching predecessor hidden state and historical depth KV exist locally.
6. Verify proposed tokens in causal order against the declared target.
7. Select only the accepted contiguous input interval and its valid target state.
8. Publish that state into the authoritative version and update required replica histories.
9. Release losing appends only after all their readers finish.
10. Retain borrowed prefix rows until every branch reference finishes.

The coordinator cannot select a branch merely because its confidence or answer quality is higher.
Here, "winner" means the lineage and interval that target verification permits.
After the first rejection, descendants conditioned on the rejected token cannot become authoritative.
A correction token requires the target's corresponding sampling or greedy rule.

Commit publication uses an epoch and an input interval.
Each receiving replica stages payload and metadata before acknowledging that epoch.
Its subsequent tasks can read the new version only after its own update completes.
Unrelated ready owners need no artificial global DP barrier.
The source retains sent rows until all copy and reader events finish.

## Validation contract

For greedy output, the target must compute its next-token decision on the accepted conditional prefix.
The proposed token must match that decision before acceptance.
A trusted target worker can send token decisions rather than complete vocabulary logits.
This reduces result traffic, but not target computation.

For sampling, an exact rejection protocol needs the proposal and target probabilities in the required precision.
An acceptance test can use the proposed token's probability ratio.
A rejection can require sampling from the normalized positive part of the target-minus-proposal distribution.
Moving only the proposed token's scalar probabilities does not automatically supply that residual distribution.

Co-locating both distributions and sampling state can avoid a vocabulary-vector transfer.
Otherwise, transferring a FP32 vector of 49152 probabilities costs 196608 bytes per position.
Recomputing the missing distribution is an alternative computation cost.
This is a conditional implementation cost, not a universal communication lower bound.
The request's RNG counters and rejection decisions must remain independent of rank scheduling.

BF16 reordering across batch shapes or TP ranks requires separate numerical tests.
A mathematical sampling guarantee does not erase those implementation differences.
The existing fixed-depth guarantee cannot automatically extend to a different adaptive target.

## The missing tentative-history messages

Committed history replication is insufficient for every speculative lane.
A future-token lane at depth r can require an ancestor token's depth-r KV before that ancestor becomes committed.
If those nodes have different owners, the required tentative state must move when it becomes ready.
The receiving lane cannot substitute a different depth or an uncomputed ancestor value.

For each state version, list every remote receiver that needs it.
Charge its required payload once for each receiver lacking that exact version.
If a receiver already holds the accepted version from tentative execution, do not charge it again at commit.
Wrong-lineage states do not satisfy that condition.

Moving a branch between owners also requires its private ancestor state and hidden boundary value.
A normalized BF16 hidden vector has 4096 bytes.
A native fused hidden/residual pair can require 8192 bytes.
Those values do not replace historical KV.
Recomputing ancestor state instead of transferring it changes the work ledger and must enter both cost and timing.

## Explicit message protocol

This is a proposed ABI, not an existing LoopSpec distributed implementation.
Each control header contains eight uint64 fields and occupies 64 bytes.
Its fields are generation, epoch, branch ID, parent ID, input begin, input count, depth mask, and opcode.
Token IDs use int64 and occupy 8 bytes each.
An adaptive terminal-depth entry uses uint8 and occupies 1 byte per cached position.
These choices specify the ledger. They are not information-theoretic minimum encodings.

| Message | Producer → consumer | Payload | Completion dependency |
|---|---|---|---|
| `FORK` | Coordinator → branch owner | 64-byte header, new input IDs, optional boundary hidden state | Required parent history and input IDs must exist before execution |
| `HISTORY` | Existing KV owner → new replica owner | Missing committed BF16 rows, address spans, adaptive depth entries when needed | Receiver installs the declared history version before its first dependent read |
| `DEPENDENCY` | Producing branch owner → consuming branch owner | Missing tentative BF16 KV rows and optional hidden boundary | Producer completes the row before transfer, receiver waits before attention |
| `VERIFY_RESULT` | Target verifier → coordinator | 64-byte header, target token decisions or declared sampling data | No acceptance before matching target evaluation |
| `COMMIT` | Verified state owner → missing replica owners | Header, accepted input IDs, terminal depths, missing accepted KV | Receiver publishes the epoch after payload and metadata complete |
| `ACK` | Receiving owner → coordinator/source | 64-byte header | Source can retire transfer leases after acknowledgement and reader completion |
| `PRUNE` | Coordinator → losing branch owners | 64-byte header and any extra branch IDs | Free only private rows with no remaining readers or descendants |

Different full passes can stream one physical layer's K/V row at a time.
One such row pair contains 8192 bytes for either Base size.
A complete token/pass across all layers contains 196608 bytes at L24 and 393216 bytes at L48.
The receiver still needs every required layer row before its corresponding read.

## Required state and capacity

Let B denote BF16 KV bytes for one input position and one recurrent pass across all physical layers.
Let K=4B denote the full-depth KV bytes for one cached input position.
Let N denote the committed prefix length.
Let U count unique computed speculative token/pass states after shared-prefix deduplication.
Let F count private cached input positions in a simple expanded four-slot implementation.

```text
B = 2 * L * 16 * 128 * 2
K = 4 * B
valid speculative payload = B * U
expanded speculative capacity = K * F
replica cluster payload = P*W + P*K*N + sum_owner(expanded_private_capacity)
TP cluster payload approximately = W + K*N + expanded_private_capacity + replicated_norm/gate_overhead
```

The source policy `last_exited` allocates all four depth slots.
Early exit reduces executed state generation, not that allocation.
Its private valid payload can be smaller than its expanded physical capacity.
A compact alias allocator is a separate change and must also be available to the matched baseline.
Graph pools, attention scratch, activation buffers, sampling data, metadata, and allocator padding add further memory.
No equation here reports a measured HBM allocation or HBM traffic.

| Quantity | Base 1.4B | Base 2.6B |
|---|---:|---:|
| Checkpoint tensor payload W | 2869305346 bytes | 5335949314 bytes |
| W in GiB | 2.672248842 | 4.969490053 |
| B | 196608 bytes | 393216 bytes |
| K | 786432 bytes | 1572864 bytes |
| Full-depth prefix at N=1024 | 0.75 GiB | 1.50 GiB |
| Full-depth prefix at N=4096 | 3.00 GiB | 6.00 GiB |
| Full-depth prefix at N=8192 | 6.00 GiB | 12.00 GiB |
| Four new full-depth input positions | 3 MiB | 6 MiB |

Both local Base checkpoint headers supply these weight payload counts.
The 2.6B checkpoint is Base at `1ed04250da1a9936042725d302e81c8fa2ab5abd`, not Thinking.
This audit reads its header after the coordinator's acquisition. It does not load the model.

For P2/P4, all 16 KV heads divide evenly across TP ranks.
The ideal TP cache payload per rank is `(K*N + private_payload)/P`.
It does not require KV-head replication at these P values.
Norm parameters remain replicated.
Adaptive original-gate support can replicate the small gate rather than incorrectly invoking the unused native gate path.
That support remains a proposed implementation, not a source-proven adaptive TP engine.

## Concrete payload comparison

The following capacity example uses C=1, N=8192, J=P branches, and four private full-depth positions per branch.
It assumes no shared private positions after the fork.
Both placements execute exactly that branch work.
The TP estimate includes replicated norm weights and the fixed native gate-bias term.
It excludes runtime memory and rounding/padding overhead.

| Model | P | Replica payload per rank | TP payload per rank |
|---|---:|---:|---:|
| Base 1.4B | 2 | 8.675179 GiB | 4.339239 GiB |
| Base 1.4B | 4 | 8.675179 GiB | 2.171269 GiB |
| Base 2.6B | 2 | 16.975349 GiB | 8.490973 GiB |
| Base 2.6B | 4 | 16.975349 GiB | 4.248784 GiB |

These values are tensor-payload accounting, not measured GPU memory.
They show why replicated history and weights cannot disappear from the capacity ledger.

## Minimum state-transfer payload under the declared placement

Assume one existing copy of each state item and no receiver recomputation.
The following bounds use the declared uncompressed BF16 payload representation.
To give P replicas a full-depth prefix requires at least `(P-1)*K*N` delivered BF16 payload bytes.
The same aggregate replication payload applies when the initial state is head-sharded across P ranks.
A tree broadcast changes link scheduling, but not each receiver's missing payload.
Repeating prefill on every replica avoids that KV transfer by adding duplicate prefill computation.

For k newly committed full-depth input positions, complete replication requires `(P-1)*K*k` delivered payload bytes.
Tentative transfers of those same accepted versions count toward this total.
Tentative states from rejected lineages add traffic beyond it.
The initial prefix and incremental commit are separate intervals, so they do not overlap in this accounting.

| Model | P | Initial full-depth prefix, N=8192 | Commit of four full-depth input positions |
|---|---:|---:|---:|
| Base 1.4B | 2 | 6 GiB | 3 MiB |
| Base 1.4B | 4 | 18 GiB | 9 MiB |
| Base 2.6B | 2 | 12 GiB | 6 MiB |
| Base 2.6B | 4 | 36 GiB | 18 MiB |

For adaptive history, known terminal-copy equality can reduce wire payload without reducing expanded cache capacity.
With a retained terminal-depth ledger, send only computed depths and reconstruct deeper terminal copies locally.
For a full-R4 prompt and adaptive decode history, the required computed payload is:

```text
adaptive_wire_history = B * (4*N_prompt + sum_decode D(t))
adaptive_wire_commit = B * sum_new_cached_positions D(t)
replication_payload = (P-1) * corresponding_wire_payload
```

Each D(t) is an actual completed original-gate depth, not a predicted future exit.
The receiver must also receive the depth metadata and execute the terminal-copy work.
Without the ledger or another verified equality protocol, use the full expanded payload bound.
Fixed R4 has D(t)=4 and offers no cross-depth wire deduplication.

For four new positions that all exit at R2, adaptive payload is half the fixed value.
At P4, this gives 4.5 MiB for Base 1.4B and 9 MiB for Base 2.6B, plus metadata.
These are structural conditional values, not observed exit frequencies or bandwidth measurements.

TP on the same branch batch does not require full accepted KV replication across all ranks.
Its KV remains head-sharded after verification.
It does require the model's TP collectives, logits handling, and coordinated branch metadata.
Compare their actual costs rather than declaring either placement communication-free.

## Latency and recurrence-specificity

The replica schedule must include initial state placement, tentative dependency transfers, verification, commit, and losing work.
It may avoid some TP collectives but lose shared-batch GEMM efficiency.
Its P full weight copies also change memory capacity and possible weight traffic.
Weight payload does not determine actual HBM traffic or kernel latency.

The TP schedule must execute the same branch DAG and preserve the same prefix sharing.
Its critical path includes the collective boundaries that feed normalization and later operators.
Neither schedule can replace a dependency sum with a maximum unless those inputs are ready.
Initial transfer amortization must state its accepted-work denominator.
First-request latency cannot silently exclude replica initialization while the baseline includes prefill.

Tied weights make shared-depth branch batching useful, but LoopSpec already uses that fact.
Ordinary TP of that batch also retains tied weights.
Generic copy-on-write prefix sharing and distributed commit do not supply a new recurrence-specific distinction.
Replica branch placement is therefore an execution placement choice, not an established new axis.

Reopening C07 requires a concrete changed ownership protocol that reduces a measured exposed cost beyond that TP control.
It must preserve the same verified branch work and charge every missing history item.
It also needs independent fixed and adaptive validation on both Base sizes.
The inspected LoopSpec distributed restriction is an engineering gap, not proof that the required control cannot exist.

## Bounded tests and next action

The CPU accounting reads the two cached safetensors headers and tests the formulas above.
A symbolic example charges an accepted ancestor transfer only once when commit later needs that same state.
It excludes a losing suffix from the authoritative state set.
These tests do not implement distributed references, validation, or rollback.
Their raw output and source ledger appear in `artifacts/branch_ownership_cost_ledger.json`.

The next decisive test remains the same-work comparison of co-batched branches, replica placement, and TP placement.
Before GPU engineering, require a lane trace with each cross-owner hidden and KV dependency.
If those dependencies require repeated full-prefix movement or duplicate ancestor computation, reject that placement before an engine build.
