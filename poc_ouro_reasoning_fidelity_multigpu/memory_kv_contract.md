# Weight and KV ownership

The pinned model contains 48 physical layers and four recurrent passes.
The native runtime registers 192 independent logical KV slots.
The model card's 24-layer description does not match the loaded config.
The campaign preserves the loaded checkpoint rather than changing its layer count.

Each token has independent KV state at each logical slot.
The model has 16 KV heads and head dimension 128.
The aggregate BF16 KV cost is 1,572,864 bytes per token.
TP1, TP2, and TP4 have 16, 8, and 4 local heads respectively.
Loaded tensors and storage ownership must confirm those values.

Native registration order is physical-layer-major.
The HF reference uses loop-major cache indices.
Comparisons map each native module name to its logical slot.
An array-position comparison without this mapping is invalid.

Weight storage, KV pool reservation, filled KV, graph workspace, and device memory are separate accounts.
An allocator pool is not fully occupied request KV.
The ownership ledger records cache shape, stride, storage pointer, rank, and GPU UUID.
No capacity claim assumes that TP always divides every allocation by four.

The config permits 65,536 positions.
The primary campaign uses at most 32,768 total positions.
This limit follows the documented Thinking context and validated runtime boundary.
Quality batches use four requests to support the predeclared longer caps.
The actual ownership ledger, not this planning estimate, determines feasibility.
