# Existing static partition control: compatibility before measurement

Pinned vLLM 0.6.4.post1 `get_pp_indices` supports `VLLM_PP_LAYER_PARTITION`;
FastPP `make_layers` calls it. However the pinned FastPP `ModelConfig` computes
local KV layer count as `total_layers // pp_size`, ignoring the explicit partition.
`MHATokenToKVPool` addresses layers modulo this count. With 14 owned layers and
12 slots, distinct layers alias KV storage. An unchecked flag sweep would therefore
produce invalid outputs and cannot be performance evidence.

Bounded isolated-source compatibility patch: when the existing partition variable
is explicitly set, use the same pinned vLLM indices to allocate the exact owned
layer count. The default path is unchanged. Reject empty/mismatched partitions.
CPU regression against the extracted actual function fails before this patch
(2/3 failures) and passes after; also checks unique per-owned-layer modulo slots.
Native output sanity remains required before any timing claim.

The native memory estimator averages only PP ranks 0 and 1. Do not assume this is
safe for an uneven model. All control arms must use the same conservative explicit
`--max-total-tokens 262144`, below available H100 capacity for the chosen model.
No production environment upgrade, routing change, or assertion bypass.

The proposed 8/12/14/14 existing partition comes from the modal measured-cost cut
in a *separate instrumented* run. This is an existing static engineering control,
not a successor method. Cost portability across physical ranks is unproven;
only fresh uninstrumented request-level A/B can judge it. A stage-max proxy is
not an E2E upper bound.
