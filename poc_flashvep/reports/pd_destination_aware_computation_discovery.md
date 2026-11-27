# Destination-aware KV generation

## Decision

`NO_GO_COMMUNICATION` for forwarding a single layer's normalized hidden rows and generating its keys and values on D. `NO_GO_INCREMENTAL` for the application of the same idea at a suffix boundary without a mechanism beyond existing layer placement. Keep the direct paged KV handoff as the serving control.

## Physical cost

The source is BF16 Qwen3-8B with hidden size 4,096, 8 KV heads and head dimension 128. A normalized layer input is 8,192 bytes per row. Its packed K and V are 4,096 bytes per row. Moving the hidden rows doubles the traffic. P still needs those keys and values for its own causal attention. D runs a second K/V projection and the associated K normalization and positional rotation.

| Original live representation | BF16 values per token | Bytes per token |
|---|---:|---:|
| Embedding, layer residual, normalized input, projected attention output, layer output, each | 4,096 | 8,192 |
| Query, concatenated over 32 heads | 4,096 | 8,192 |
| Key or value, 8 KV heads | 1,024 | 2,048 |
| Q/K/V before splitting heads | 6,144 | 12,288 |
| One SwiGLU gate or up projection, or fused activation after gating | 12,288 | 24,576 |
| Joined gate and up projections | 24,576 | 49,152 |
| Full 36-layer K+V | 73,728 | 147,456 |

These are logical tensor footprint calculations from the loaded configuration. Physical backing, descriptors and request metadata are recorded separately. The one-layer payloads below were transferred between the actual GPU buffers.

| Model and prefill rows | Direct P→D KV, median ms | Hidden P→D and D projection, median ms | Candidate / direct |
|---|---:|---:|---:|
| Qwen3-8B 1,024 | 0.0529 | 0.1970 | 3.72× |
| Qwen3-8B 4,096 | 0.0835 | 0.3404 | 4.08× |
| Qwen3-8B 8,192 | 0.1285 | 0.6085 | 4.74× |

Timing uses CUDA events on two H200s connected by NV18. The direct arm copies already generated K and V. The candidate uses the same unquantized loaded weights and transfers only the normalized hidden rows. Key and value correctness was bitwise for these probes. At 8,192 rows, pinned host→D KV was 0.616 ms, pinned host→D hidden alone was 1.220 ms, and D→P hidden was 0.188 ms. These events cover local copies and projection. Full native NIXL handoff also pays descriptor setup and ownership protocol. Layout-related costs depend on the connector configuration.

Qwen2.5-7B-Instruct provides a different GQA geometry. The hidden row is 7,168 bytes and K+V is 2,048 bytes, a 3.5× payload ratio. In the 4,096-row probe, hidden transfer and duplicate projection took about 3.96× the direct peer copy. The full A/B streams, shapes and actual sample populations remain in the raw operator artifacts.

## Full-model accounting and the strongest alternative

Qwen3-8B has 36 layers. At 16,384 input rows, logical BF16 prompt K+V is 2.25 GiB across all layers. Native NIXL received 144 GiB for the 64-request cohort. This verifies transfer volume. Connector timing includes the actual receiver-side software and descriptor costs, which differ from one CUDA peer copy. The separate online comparison evaluates native cross-layer block packing.

One BF16 hidden boundary can amortize its bytes across at least three Qwen3 layers, because after two layers the payloads are equal. For Qwen2.5-7B the minimum is four layers. This bound requires D to continue the later layers exactly for the affected rows, including the sequential attention and residual dependencies. A token subset also requires the earlier prefix state for each attention layer. This recovers suffix or token placement with a different payload accounting. It does not establish a separate operation that can be skipped on both devices. The budget must include D work, any reverse transfer, and the decode deadlines. Prior work already explores layer splitting and the related transfer–recomputation trade-off. [Prior work audit](pd_resource_complementary_prior_work.md)

An MLP row move of Direction A completes after attention for that layer and has already consumed the KV. It cannot erase that layer's P→D KV payload. Combining A and B has no demonstrated independent saving. A+B serving, streaming beyond the native connector control and task-level candidate correctness are `NOT_RUN`.

[Primary measured samples](../pd_resource_complementary_parallelism/results/operator_primary.jsonl.gz) · [Secondary measured samples](../pd_resource_complementary_parallelism/results/operator_secondary.jsonl.gz) · [Transfer cost figure](../pd_resource_complementary_parallelism/figures/destination_payload_cost.png)
