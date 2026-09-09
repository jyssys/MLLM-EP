# Layered Prefill transfer scope

Native Qwen3-30B-A3B BF16 TP2 is directly measured with official graph-enabled
layered scheduling and chunked controls. All logical experts remain on each TP
rank with sharded matrices; this is not EP. See CODE_AUDIT and REPRODUCTION.

Qwen3-VL's native encoder, MRoPE and DeepStack are not implemented in this pinned
nano-vLLM model. A source-faithful full port is therefore not established.
Instead, fresh real-image vLLM TP2/DP2/DeepEP4 observations supply layer/group
diagnostics and directly joined request timelines. The larger matched control
has 120 exact token-count pairs; see `../COMMON/LARGE_MATCHED_TRANSFER.md`.

Measured-cost contiguous partitioning is solved exactly for the supplied cost
vector, not for the true changed serving execution. Native instrumentation
changes timing and group changes alter prefill/decode composition. No direct
MLLM request-level successor oracle has been validated. This boundary is a port
limitation, not a finding that Layered Prefill fails on MLLMs.
