# Ownership and dependency supplement v1

## Status and sources

This supplement documents existing execution contracts. It proposes no architecture and reports no measured overlap. The [editable figure](../figures/contract_frontier_and_c14_control_v1.mmd) supplements the four existing explanatory diagrams. A figure does not replace an executable schedule or numerical checks.

The source is the [exact frontier contract](exact_frontier_contract.md). Its pinned implementation locators include `modeling_ouro.py:316–322,535–608` and native `ouro.py:260–281,355–374`. The [C14 audit](c14_cyclic_prefill_reopening_audit_v1.md) supplies the physical-owner control. The [branch contract](branch_ownership_cost_contract.md) supplies the separate proposed C07 message protocol.

## Reference state ledger

This table instantiates the six state categories required by specification Section 5.4. It describes the reference or a named proposed contract. It is not a measured allocation table.

| State | Creation and earliest availability | Owner and readers | Mutability and lifetime | Logical payload and transfer boundary |
|---|---|---|---|---|
| Physical weight shard | Checkpoint load completes | Native TP feature/head rank, or C14 physical-layer owner | Immutable for the pinned model | Native TP and balanced C14 retain one aggregate dense core. They partition it differently. Setup, non-core tensors, and workspace remain separate. |
| Historical KV | Actual matching projection and RoPE complete, then permitted publication | Native local head rank, or C14 owner of that physical layer | Immutable after commit until request retirement | Full-head K+V has 8192 BF16 bytes per token, depth, and physical layer. TP2/TP4 own 4096/2048 logical bytes per rank. Allocation padding differs. |
| Current-token KV | Actual current QKV/RoPE completes | Same logical depth/layer owner | Writable before publication. A speculative branch needs an exclusive private version | Same per-row payload as history. A C07 receiver needs each missing ancestor version before its attention read. |
| Hidden activation | Its actual predecessor completes | Producing rank or stage, then the named consumer | Immutable after publication until all readers finish | A full H2048 BF16 row has 4096 bytes. Native internal boundaries can retain an additional 4096-byte residual. No stage may drop that residual silently. |
| Gate and output state | Original loop norm, gate, and selected head complete in order | CDB token owner for gate/CDF, output owner for token selection | Retain each required hazard and possible selected hidden until routing finishes | A proposed FP32 hazard packet has 4 bytes per row. Events, metadata, host state, and logits add storage. A gate signal alone is not an exit decision. |
| Branch or solver scratch | Branch fork or correction initialization | Declared scratch owner, with exclusive append leases | Mutable until validation. Release losers only after their readers finish | No fixed size follows without a branch graph or solver state. Missing payload, decoder, validation, and commit costs remain `UNKNOWN`. |

These counts use BF16, H2048, 16 KV heads, and head dimension 128. Both Base models share those dimensions. Their physical layer counts are 24 and 48. Aggregate full-R4 KV remains 786432 and 1572864 bytes per cached token, respectively. These quantities are capacities, not measured HBM traffic.

## Earliest-input and guard boundaries

The decode diagram retains the actual pass chain. Its gate branch describes policy guards, not simultaneous execution of all adaptive depths. Mandatory R2 does not wait for an R1 exit decision under minimum R2. R3 and R4 require continuation or a separately isolated provisional-state contract.

The figure shows only the R2 adaptive decision to keep it compact. R3 still needs its own original ordered hazard decision. Maximum R4 forces completion. Fixed R4 uses h4. Neither mode can use a future input token before the preceding token decision exists.

The C14 chunk panel is a schematic contraction of physical partitions. It does not enumerate every internal layer. A complete executable graph must expand each layer's earlier-position KV dependencies. The panel shows data independence, not physical concurrency or kernel timing.

C14 keeps one physical weight partition per owner across recurrent visits. It retains distinct KV banks for every required depth. The strongest generic cyclic control receives the same aliases, owners, chunks, KV, kernels, messages, and scheduling freedom. Its constructed physical trace can then match C14. No new mechanism follows from prefix sharing or cyclic labels alone.

## Remaining candidate-specific work

The table does not assign an unimplemented C01 pool, C04 layout, C10 restoration path, C11 code, or C12 summary to invented ranks. Those designs remain incomplete. C07 has a proposed ABI, not a validated distributed engine. C03 has address tests, not a compact alias allocator. C19 has key-only coding, not a GPU decoder or full K+V protocol.

For a screened negative, preserve the scoped obstruction without building an unnecessary engine. For a proposed survivor, replace every unknown owner, lifetime, payload, and guard with a concrete contract before substantial GPU implementation.
