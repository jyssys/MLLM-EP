# Static4 design and installed-API constraints

The implementation patches the installed modular fused-MoE call only for the
one physical mixed prefill/decode iteration. Attention and the router run once
over the full scheduled token set. Each source token's full top-8 route stays
in one slice. Rows are greedily assigned by
`unique_destination_ranks × hidden_size × BF16_bytes`.

The physical sequence in every MoE layer is:

1. DeepEP prepare/dispatch S1 and S2.
2. Resident decode prepare/dispatch, sparse expert execution, and finalize/combine.
3. DeepEP prepare/dispatch S3 and S4.
4. Concatenate the four received expert-major buffers and metadata.
5. Run the prefill sparse expert kernel once.
6. Split expert output by received-slice layout, finalize through each saved
   DeepEP handle, and scatter to original source-row order.

The installed DeepEP API exposes a separate opaque dispatch handle for each
prepare, so the bounded prototype uses `PROTO_PER_SLICE_COMBINE`. It does not
claim a native unified combine. It records repack, four finalize calls, and
scatter separately.

An instrumentation-only EP all-reduce discovers asynchronous external-DP
admission. An arrival barrier then excludes CPU slice-build rank skew from the
reported CUDA dispatch quantum. Both remain charged to end-to-end wall time.
This Python/D2H partition and barrier are deliberately not presented as the
future fast path.

Implementation type: `STOCK_DEEPEP_PYTORCH_COMPOSITION`.
Result scope: `NATIVE_SERVING_PROTOTYPE_MOE_PATH`.
