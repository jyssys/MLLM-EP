# Stock vLLM DeepEP DBO correctness investigation

Date: 2026-08-07

Base commit: `4051afd32dacdafd2b2dd27e42d1d4dfa1f8b655`

Branch: `flashvep/deepep-overlap-revalidation`

## Scope and setup

This investigation changes no FlashVEP schedule or performance optimization.
It uses vLLM 0.20.0, Qwen3-VL-30B-A3B-Instruct, TP2/DP2/EP4, DeepEP high
throughput, eager execution, and physical GPUs 4,5,6,7. Each run generates one
greedy output token from identical prompts. The matrix covers DBO off/on,
global requests 2/4/8, a 790-token text prompt, and an 896x896 image prompt
with 799 decoder tokens (784 image tokens).

The probe records scheduler request IDs before splitting, request and token
slices for each ubatch, one decoder attention shape per model call and ubatch,
FA3 scheduler-metadata addresses, sampled-token order, and the final restored
output order. Raw results are under `poc_flashvep/deepep_revalidation/results`.

## Baseline result

The first failing global request count is **2** for both modalities.

| Input | DBO off token | Stock DBO on at request 2 | Stock DBO on at request 8 |
|---|---:|---:|---|
| Text | 2132 | 151645 | 2132 / 6303 / 151645 |
| Vision | 1986 | 7836 / 8695 / 100486 | 198 / 1986 / 12210 / 20869 / 64012 / 121430 |

Final restoration was not the cause. For example, a rank-local request-8 wave
submitted engine ID prefixes `[35, 36, 37, 38]` and returned
`[35, 36, 37, 38]` in the same order. This held for DBO off/on, text/vision,
and requests 2/4/8. The public submission IDs add UUID suffixes while the
returned `RequestOutput.request_id` contains the numeric engine prefix; the
normalized order is unchanged.

## Exact failure locations

There are two independent faults.

1. **DBO attention metadata cache aliases ubatch 1 to ubatch 0.**

   `GPUModelRunner._build_attention_metadata()` in
   `vllm/v1/worker/gpu_model_runner.py:2211-2271` caches by
   `(KVCacheSpec, builder type)`, without the ubatch ID. DBO has a separate
   builder for each ubatch, but ubatch 1 reuses ubatch 0's metadata through
   `FlashAttentionMetadataBuilder.update_block_table()`.
   `vllm/v1/attention/backends/flash_attn.py:579-588` copies the object and
   changes only block table and slot mapping, retaining query/sequence fields
   and the mutable FA3 scheduler buffer.

   In the stock vision request-2 trace, ubatch 0 has 399 tokens and ubatch 1
   has 400. Nevertheless ubatch 1 reports `num_actual_tokens=399`,
   `query_start_loc=[0,399]`, and `sequence_lengths=[399]`; both ubatches use
   the same scheduler metadata address. This is the direct cause of the
   request-16 FA3 `batch_size must be equal to batch_size_k` failure when the
   aliased request dimensions diverge further. Text request 2 has equal
   395-token halves, so its shapes appear valid, but it still shares the same
   mutable FA3 scheduler metadata and produces the wrong token.

2. **Qwen3-VL DeepStack embeddings are not ubatch-sliced.**

   After isolating attention metadata, text correctness is restored but vision
   still returns token 198 at request 2. Qwen3-VL stores DeepStack vision
   embeddings in one model-side buffer. In
   `vllm/model_executor/models/qwen3_vl.py:1702-1764` each concurrent forward
   reads `buffer[:num_tokens]` and clears the same buffer; calls at
   `qwen3_vl.py:2817-2834` do not carry the DBO token offset. Thus both
   ubatches consume the prefix instead of `[0:399]` and `[399:799]`.

The first fault affects text and vision. The second is vision-specific, so the
observed behavior is neither solely a Qwen3-VL metadata problem nor an output
reordering problem.

## Minimal fix

`dbo_correctness_probe.py` installs two runtime-scoped workarounds only when
`FLASHVEP_DBO_CORRECTNESS_FIX=1`:

- Disable FlashAttention's block-table-only metadata reuse for ubatching so
  each ubatch builds independent query/sequence and FA3 scheduler metadata.
- Slice Qwen3-VL DeepStack embeddings by current ubatch ID and mark the shared
  payload consumed only after both ubatches finish. No asynchronous zero-fill
  is issued across the two CUDA streams.

The corrected vision request-2 trace has:

| Field | ubatch 0 | ubatch 1 |
|---|---:|---:|
| token / DeepStack slice | `[0,399]` | `[399,799]` |
| query/key shape | `[399,16,128]` / `[399,2,128]` | `[400,16,128]` / `[400,2,128]` |
| `num_actual_tokens` | 399 | 400 |
| `sequence_lengths` | `[399]` | `[799]` |
| FA3 scheduler metadata | distinct address | distinct address |

After both fixes, text produces 2132 and vision produces 1986 for every
request on both DP ranks, for requests 2/4/8 and all three measured repetitions.

## Clean post-fix latency

Tracing was disabled. Values are rank-0 end-to-end medians after 2 warmups and
7 measured iterations; the DP ranks are barrier-synchronized and report
effectively identical wall time. Speedup is `DBO-off / DBO-on`.

| Input | Requests | DBO off (ms) | DBO on (ms) | Speedup |
|---|---:|---:|---:|---:|
| Text | 2 | 3478.693 | 3420.287 | 1.017x |
| Text | 4 | 3548.152 | 4079.021 | 0.870x |
| Text | 8 | 3526.316 | 4070.329 | 0.866x |
| Vision | 2 | 2761.519 | 3514.604 | 0.786x |
| Vision | 4 | 2790.298 | 4088.033 | 0.683x |
| Vision | 8 | 2855.602 | 4109.111 | 0.695x |

Correctness is recovered, but this small one-token end-to-end matrix does not
show a useful DBO speedup. That performance result is observational only; this
task intentionally adds no scheduling or performance optimization.

## Reproduction

```bash
# Instrumented baseline or fixed correctness matrix
WARMUPS=1 ITERATIONS=3 \
  poc_flashvep/deepep_revalidation/run_dbo_correctness_matrix.sh RESULT baseline

WARMUPS=1 ITERATIONS=3 \
  poc_flashvep/deepep_revalidation/run_dbo_correctness_matrix.sh RESULT fixed

# Clean post-fix latency, no trace hooks
ENABLE_TRACE=0 WARMUPS=2 ITERATIONS=7 \
  poc_flashvep/deepep_revalidation/run_dbo_correctness_matrix.sh RESULT fixed
```

The runtime workaround is deliberately narrow. A production upstream fix
should include ubatch identity in attention-metadata reuse and pass DeepStack
embeddings as explicitly sliced model inputs rather than infer offsets from a
shared two-ubatch buffer.
