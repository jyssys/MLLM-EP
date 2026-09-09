# Current vLLM upstream status

Audit date: 2026-09-09 (Asia/Seoul)

Source checkout: `/home/esjung/external/vllm-cp-composition-audit`

Commit: `5acd95906b7c7a54dde89396d2bb06fe28ebeed0`

## Required upstream items

| Item | State at audit | Relevance |
|---|---|---|
| `context_parallel_deployment.md` | Present on current main | Documents PCP and DCP as distinct axes; support is backend/model dependent. |
| #50391 | Open RFC | Direct collision with DCP-aware speculative verification. It identifies per-query rank-local KV-length semantics and proposes exact strategies including virtual `q_len=1`, context/current LSE merge, custom masking, and direct CP-aware attention. |
| #46358 | Open RFC | Directly proposes orthogonal PCP and DCP. PCP-perpendicular-DCP by itself is not a new contribution. |
| #53673 | Open RFC | Decouples draft-model parallelism from target PCP/DCP; adjacent to DCP+spec composition. |
| #45425 | Open correctness issue | Reports silent MTP+DCP corruption when an unsupported backend/graph path processes variable-length verification rows. |
| #51429 | Open support issue | `ModelConfig.verify_with_parallel_config` rejects some DCP-over-PCP configurations admitted by `ParallelConfig`. Current source retains the relevant GQA check. |
| #53573 | Open correctness issue | Combined MLA PCP+DCP may derive divergent DCP KV-gather collectives from rank-local PCP metadata. |

## Source-level findings

- `ParallelConfig` admits PCP/DCP compositions only for restricted divisibility cases.
- Non-MLA GQA DCP requires `TP > total_kv_heads` and `DCP <= TP / total_kv_heads` in current `ModelConfig` validation.
- Current PCP manager rejects pipeline parallelism, encoder-decoder models, multimodal inputs, and speculative decoding; PCP is MLA-only.
- DCP speculative verification is backend-specific. If an attention backend lacks `supports_dcp_with_varlen`, vLLM forces multi-token verification off the decode fast path by setting the reorder threshold to one.
- FlashAttention MLA advertises DCP-varlen support only for interleave size one; FlashInfer has a DCP-aware path. Therefore “DCP+spec support” is not a single portable capability.
- The current PCP manager computes global DCP-aware KV slot mappings during prefill and masks writes after PCP gathering. This is evidence that the supported MLA path already writes toward canonical final ownership instead of performing an obvious standalone prefill-to-decode KV redistribution.

## Environment and measured-version status

- Current-main requirements: PyTorch 2.13.0 and FlashInfer 0.6.18. Its CUDA-13 wheel cannot execute on this host's CUDA-12.8 driver.
- Stable local runtime: vLLM 0.20.0, PyTorch 2.11.0+cu129.
- Isolated compatible runtime built for the decisive runs: `/home/esjung/.venvs/cp-composition-vllm026`, vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129.
- Importing current-main Python over an older binary installation fails at the compiled-extension ABI. The study therefore uses current-main source as the support/novelty source of truth and v0.26 as the newest driver-compatible measurement path.
- N-gram speculation in v0.26 falls back from Model Runner V2 to Model Runner V1. This is recorded as a runtime limitation, not hidden.
