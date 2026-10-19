# Long-context C1 capacity source audit

## Outcome and evidence boundary

Both pinned Base checkpoints configure a 65536-token context limit.
C1 prompts of 16384 or 32768 tokens plus 128 outputs stay within that limit.
The installed native Ouro, RoPE, and FA3 interfaces contain no inspected source restriction that excludes these cases.
This finding concerns configured support, not a long-context numerical, quality, or latency result.
This audit ran no GPU work and changed no existing runner.

The requested C1 cases do not establish a capacity need for C19 beyond ordinary uncompressed native TP.
The largest case needs a 48.2109375 GiB native KV pool per TP1 rank, including the null block.
Recorded Base2.6B TP1 parameter storage adds about 4.96949 GiB.
Other allocations still need runtime evidence.
This audit does not certify an OOM-free configuration.

The [metadata ledger](../artifacts/long_context_capacity_source_audit.json) contains exact byte counts, hashes, source locators, and actual segment timestamps.
Its capacity tables cover fixed R4, BF16 KV, PP1, no speculative lookahead, and C1 only.
They do not establish support for a long-context adaptive engine.

## Checkpoint and positional support

The pinned [Base1.4B configuration](/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json:44) sets `max_position_embeddings=65536`.
The pinned [Base2.6B configuration](/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json:68) sets the same value.
Both use full attention, 16 query heads, 16 KV heads, and head dimension 128.
Both set `rope_theta=1000000`, `rope_scaling=null`, and `sliding_window=null`.
Neither checkpoint config requests a new extrapolation scheme for these cases.

Both tokenizer configurations set `model_max_length=131072` at line 165.
That tokenizer value does not extend the model's 65536-position RoPE cache.
The installed [length calculation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/model.py:2177) takes the smaller model and tokenizer limits.
The architecture converter derives the model limit from its configuration at [lines 326–359](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/transformers_utils/model_arch_config_convertor.py:326).
The old benchmark's `max_model_len=10240` is a local cap.
It is not the checkpoint's configured context limit.

Native [Ouro initialization](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:231) passes `config.max_position_embeddings` to `get_rope`.
The [RoPE compatibility path](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/transformers/modeling_rope_utils.py:765) retains the configured theta and selects default RoPE.
The [RoPE factory](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/__init__.py:101) constructs the default rotary implementation.
Its [cache construction](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:94) covers positions 0 through 65535.
Its [native lookup](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:171) indexes that cache by actual token positions.

For 128 ordinary generated outputs, the last output does not require another model call.
The last computed position is therefore `N+126`.
It is 16510 for N16384 and 32894 for N32768.
Even the conservative total-position reserves, 16512 and 32896, remain below 65536.
No `VLLM_ALLOW_LONG_MAX_MODEL_LEN` override is necessary.
No RoPE setting needs a change.

The [RoPE source comment](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:80) notes possible numerical differences from HF cache construction.
Position validity does not remove that numerical question.
Constant allowed-token outputs also do not prove finite logits or natural long-context quality.
Those properties require separate evidence.

## Native attention and KV ownership

Prior native TP1 consoles for both sizes report FlashAttention version 3 at line 27.
See the [Base1.4B console](../outputs/control_async_base14_P1_v1/console.log) and [Base2.6B console](../outputs/control_async_base26_P1_v1/console.log).
The [installed selector](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/fa_utils.py:132) prefers FA3 on supported Hopper hardware.
The [backend](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:168) accepts head dimension 128 and BF16 KV.
It passes actual sequence lengths, maximum sequence length, and page tables to the [FA3 interface](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/vllm_flash_attn/flash_attn_interface.py:341).
This source inspection does not execute the installed binary kernel at 16K or 32K.

Native [Ouro creates four Attention objects per physical layer](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:170).
Each pass has a separate cache prefix.
The [forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:198) writes and consumes the selected pass's cache.
The four recurrent passes reuse block parameters, not their historical KV values.
No fixed-depth cache alias follows from shared weights.

The preserved ownership files report 96 cache slots per rank for Base1.4B and 192 for Base2.6B.
Each reported cache slot has a distinct storage pointer in those processes.
At TP1/2/4, the local KV head counts are 16/8/4.
No KV-head replication occurs at these TP degrees.
The actual cache shape is `[num_blocks, local_KV_heads, 16, 256]`.
K and V share the last dimension without a dtype change.
The [backend shape contract](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:130) and [cache update](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:1084) match this layout.

## Exact request capacity and native pool minimum

For C1, let `T=N+128`, `L=24/48`, and `P=TP1/2/4`.
The reserved request KV payload is:

`K_request = 2 * L * 4 * T * (16/P) * 128 * 2` bytes.

The first factor represents K and V.
The last factor is bytes per BF16 value.
All requested T values are multiples of the observed 16-token block size.
The actual `N+127` written positions require the same rounded block count as this reserve.
Thus these cases require exactly 1032 or 2056 request blocks under this layout.

| Checkpoint | Prompt N | Raw reserved KV, TP1 | TP2 | TP4 |
|---|---:|---:|---:|---:|
| Base1.4B | 16384 | 12.09375 GiB | 6.046875 GiB | 3.0234375 GiB |
| Base1.4B | 32768 | 24.09375 GiB | 12.046875 GiB | 6.0234375 GiB |
| Base2.6B | 16384 | 24.1875 GiB | 12.09375 GiB | 6.046875 GiB |
| Base2.6B | 32768 | 48.1875 GiB | 24.09375 GiB | 12.046875 GiB |

The native [BlockPool reserves one null block](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/core/block_pool.py:187).
It removes that block from the free queue.
The exact minimum pool therefore needs 1033 or 2057 blocks, not merely the request payload.
The [allocator](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/core/kv_cache_utils.py:990) rounds available bytes down to complete pages across all depth slots.

`K_pool_min = (ceil((N+127)/16) + 1) * 16 * 2 * L * 4 * (16/P) * 128 * 2` bytes.

| Checkpoint | Prompt N | Minimum native pool, TP1 | TP2 | TP4 |
|---|---:|---:|---:|---:|
| Base1.4B | 16384 | 12.10546875 GiB | 6.052734375 GiB | 3.0263671875 GiB |
| Base1.4B | 32768 | 24.10546875 GiB | 12.052734375 GiB | 6.0263671875 GiB |
| Base2.6B | 16384 | 24.2109375 GiB | 12.10546875 GiB | 6.052734375 GiB |
| Base2.6B | 32768 | 48.2109375 GiB | 24.10546875 GiB | 12.052734375 GiB |

These values cover device KV tensors only.
They exclude block tables, model parameters, RoPE buffers, activations, CUDA Graph allocations, allocator slack, and communication workspaces.
They assume one full-attention cache group, block size 16, and no added lookahead.
The metadata retains all exact byte counts.

## Capacity-feasible H100 settings and required checks

The preserved [clean-start snapshot](../outputs/control_async_base26_P1_v1/clean_start.json) reports an H100 80GB HBM3 device with 81559 MiB physical memory.
That value is 79.6474609375 GiB, not 80 GiB.
It is historical evidence, not a current free-memory measurement.

A uniform 52 GiB KV pool covers all requested C1 cases and the null block.
For the largest TP1 case, that pool plus recorded parameters totals about 56.96949 GiB.
About 22.67797 GiB of the recorded physical capacity remains before other allocations.
This arithmetic is not an activation-peak measurement or an OOM guarantee.
The metadata also proposes smaller pools rounded upward to 4 GiB boundaries.
Those pools range from 4 to 52 GiB per rank.
They are capacity-feasible starting points, not tested safe settings.

The [manual KV-budget path](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_worker.py:462) still executes `profile_run`.
It skips automatic memory sizing and does not respect `gpu_memory_utilization` for that pool.
Do not treat a utilization setting as extra protection after a manual pool setting.
The long full-prefill profile, runtime activations, attention workspace, and graphs can still fail allocation.

Before any authorized diagnostic, freeze these conditions:

- Use C1, `max_num_seqs=1`, BF16 KV, fixed R4, and no speculative lookahead.
- Use `max_model_len=16512` for N16384, or `max_model_len=32896` for N32768.
- Set `max_num_batched_tokens` to at least the corresponding model-length cap.
- Keep chunked prefill disabled if the control requires one full prefill.
- Do not use 32768 as the token budget with a 32896 model-length cap.
- Keep original checkpoint RoPE settings and leave long-length override flags disabled.
- Use a declared C1 graph configuration, such as capture size `[1]`, rather than silently retaining larger cohort captures.
- Retain asynchronous scheduling and the native singleton allowed-token path.
- Use fresh VLLM/AOT, Inductor, Triton, and compilation cache roots for each job.
- Check the actual UUID, FA3 selection, BF16 cache dtype, block size, 4L slots, and minimum available block count.
- Retain initialization failure, post-warmup allocated/reserved memory, peak memory, and preemption evidence.
- Check finite logits or selected hidden values in a separate diagnostic. Forced token identity alone is insufficient.

The [scheduler source](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/scheduler.py:272) requires the token-budget condition for a configuration without chunked prefill.
An old 10240-token cap or 20 GiB pool can reject a case that the checkpoint and hardware otherwise permit.
Do not reinterpret those local settings as a model or H100 limit.

## Native TP weights versus KV/weight decoupling

The preserved ownership measurements show one aggregate copy of the large model weights under native TP.
They do not show one complete checkpoint per TP rank.
The [physical block list](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:331) contains L layers.
The [R4 loop](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:367) invokes those same layer objects four times.

| Checkpoint | TP1 aggregate parameter bytes | TP2 aggregate | TP4 aggregate |
|---|---:|---:|---:|
| Base1.4B | 2869305346 | 2869702660 | 2870497288 |
| Base2.6B | 5335949314 | 5336739844 | 5338320904 |

The small aggregate increase comes from replicated normalization weights and gate bias.
It equals `(P-1) * 397314` bytes for Base1.4B.
It equals `(P-1) * 790530` bytes for Base2.6B.
All six ownership files and hashes appear in the metadata.
These are prior process observations, not measurements from a new long-context run.

KV/weight decoupling can move a local weight shard away from a KV owner.
The receiving compute owner must still retain or obtain that shard.
It cannot claim removal of `(P-1)` complete weight copies relative to native TP.
Native TP does not contain those copies.
Removing every replicated small parameter would save only 1191942 or 2371590 aggregate bytes at TP4.
That upper bound is not a proposed executable transformation.

Native TP also keeps its historical KV locally.
No native remote-KV transfer exists for this proposal to remove.
A new ownership layout must charge its actual activation messages, compute-owner storage, and any added base retention.
The requested C1 capacity arithmetic alone supplies no such advantage.

## Capacity, read volume, and latency remain separate

KV capacity counts retained state once.
Attention still consumes the required historical keys and values for each dependent pass.
For 128 outputs, ordinary generation performs 127 incremental decode calls after prefill.
Their logical KV operand coverage is:

`bytes_per_position_per_rank * sum(N+j for j in 1..127)`.

For Base2.6B TP1 at N32768, this equals 6558314397696 logical bytes.
That figure excludes prefill and is not measured HBM traffic.
Cache retention, kernel tiling, repeated reconstruction, and transaction layout can change physical reads.
No latency follows from this byte count alone.

The [existing C19 screen](lossless_kv_measurement_screen.md) covers selected keys, not complete K+V at these long contexts.
It measured smaller CPU-coded frames but modeled larger reads under uncached reconstruction.
It did not measure GPU decoding, long-context admission, or latency.
Do not apply its key-only ratios to full model KV capacity.
Do not count original fixed-depth state aliases or native remote transfers that do not exist.

Disposition: source-feasible long-context C1 control, with no established C19 capacity necessity beyond native TP.
Long-context numerical validity, task quality, GPU memory peaks, and latency remain `NOT_RUN`.
This bounded result is not a universal impossibility claim for other cohorts, contexts, or lossless representations.

## CPU checks and actual time

The CPU calculation passed 114 structural checks with zero failures.
The checks cover configuration limits, prior slot ownership, TP head sharding, pool rounding, null-block capacity, and parameter replication.
The checks also reproduce all 14 prior per-rank KV pool byte counts from the source formula.
No native model or GPU kernel executed.
The segment began at 2026-10-19 14:23:23 UTC.
The metadata records the actual final timestamp and elapsed seconds.
