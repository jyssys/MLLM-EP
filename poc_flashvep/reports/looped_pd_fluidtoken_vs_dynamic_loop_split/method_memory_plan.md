# Common memory plan for split qualification

Status: planned configuration. GPU validation remains pending.

The interrupted `static_gate_v3_huginn_atomic_native8` diagnostic used 16,384 D KV blocks.
That pool can reserve only three complete 2048/16 requests. It is not a tuned serving baseline.
The v4 qualification uses 26,752 D KV blocks for every Huginn method and transport control.

## Structural accounting

A Huginn 2048/16 request reserves 4,128 recurrent KV blocks in the current native layout.
Both the previous 28,288-block pool and the planned 26,752-block pool can reserve six complete requests.
The planned pool removes 1,536 blocks. It retains the integer full-request concurrency of the previous pool.
The measured storage size of the previous pool is 76,472,647,680 bytes.
At the same block layout, the removed blocks release 4,152,360,960 bytes, or 3.8671875 GiB.
This is structural accounting. It is not a measured allocator or peak-HBM reduction.

The extra memory is intended for the larger D Prefill graph and continuation workspace.
The qualification must measure actual allocation and peak HBM before clean method timing.
An allocation failure remains a failed configuration. It does not authorize silent pool reduction or output truncation.

All Huginn comparison arms use the same pool, graph limits, and D token budget.
The native and full-snapshot controls remain necessary after this memory change.
Ouro retains the planned 16,384-block pool because its block geometry differs. Its actual reservation limits require separate validation.

Evidence for the previous Huginn pool:
`raw/recovery_huginn_slots8_graph8_tail512/memory_inventory_initial_1654056.json`.

No claim of safe large-Prefill workspace follows from free coordinator KV blocks alone.
Coordinator credit and physical free HBM are different resources.
