# Stage memory and communication ownership

Native TP4 owns one quarter of each layer's attention heads and weights on each GPU.
Every rank has all 192 logical cache slots, with four local KV heads per slot.

In PP4, each GPU owns one recurrent pass: 48 full-head KV slots and one full copy of the shared physical core.
The aggregate logical KV remains unchanged. Each rank's logical KV is the same quarter as native TP4.
Core-weight storage increases because different GPUs cannot share HBM parameter storage.
The conservative cost model charges the complete measured TP1 parameter footprint on each stage.

In PP2×TP2, each stage owns two recurrent passes and half of the core weights on each of its two GPUs.
Each GPU owns 96 slots with eight KV heads each. Its logical KV remains one quarter of the full request.
The conservative boundary sends one full hidden activation to each TP consumer rank.

The BF16 hidden payload is `chunk_rows × 2048 × 2 bytes` per single boundary edge.
PP4 crosses three such boundaries per prompt chunk. Stage-local KV does not cross those edges.
TP4 decode after PP prefill would require an actual head/depth cache redistribution.
That unimplemented transition cannot be free in an achievable-speedup claim.
Zero-transition estimates are optimistic bounds only.

Decode in PP layout traverses all four depths sequentially for each new token.
The next real token is unknown until the preceding token completes. Prefill concurrency does not remove that dependency.
