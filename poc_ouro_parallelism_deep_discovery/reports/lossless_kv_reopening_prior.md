# Lossless KV reopening: prior and read-path audit

Status: source and mathematical audit complete. Codec execution, decoder implementation, GPU work, and Ouro integration are `NOT_RUN`.

## Outcome

Large resident KV is a different cost target from C08's tested small hidden messages. The earlier startup-cost result does not reject this reopening. However, lower stored capacity does not imply lower attention-read traffic.

A raw shared base plus residuals can increase logical reads across sequential R4. A compressed base or actual reuse must overcome repeated base reads. Generic independent compression is therefore a necessary baseline, not only raw BF16 storage.

Current ZipServ code already contains BF16 compressed K/V decoding inside FlashAttention tile loads. Its published paper describes KV adaptation as future work, but the current repository contains a later implementation. Neither statement substitutes for the other. The inspected integration lacks paged KV and in-place append support. It is not a verified online Ouro path.

The reopening remains `GENERIC_LOSSLESS_STORAGE_WITH_UNMEASURED_READ_COST`. A distinct architecture mechanism is `NOT_ESTABLISHED`. Cross-loop correlation could improve a generic codec's statistics without exposing new independent recurrent work.

## Actual capture scope

The two capture directories contain `actual_key_words.npz`. Both contain **keys only**, not values. The arrays have shape `[12,3,257,128]` and unsigned 16-bit storage. They cover four depths for three physical layers and three heads.

The 1.4B physical layers are 0, 12, and 23. The 2.6B layers are 0, 24, and 47. Both select heads 0, 8, and 15. The fixture comes from a controlled native R4 TP1 request. Its status labels it `QUALITY_ONLY`, not a codec or performance result.

`CampaignProbe.export_key_rank_fixture`, lines 11–39, extracts the key half of native KV storage and preserves BF16 words through an integer view. This worker read metadata and array headers only. It computed no new compression ratio. It did not generalize key statistics to values, other layers, larger contexts, TP2/TP4, or adaptive exits.

The [source/search observations](../artifacts/lossless_kv_reopening_sources.jsonl) preserve capture hashes, source revisions, locators, and access windows.

## Primary nearest sources

### SplitZip: exact BF16 representation, dense decode destination

[SplitZip, 2605.01708v3](https://arxiv.org/html/2605.01708v3), dated 2026-06-23, targets prefill–decode transfer. Section 3 preserves sign/mantissa bits and codes frequent exponents. Escapes retain every uncommon exponent. Calibration changes codec tables, not model weights.

Its storage expression is `1.5N + 3M` bytes for N words and M escapes, before additional framing. This is an independent-tensor baseline. It does not establish equivalent statistics for XOR residual words.

The pinned [decode function](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/codec_gpu.py#L561) allocates dense int16 output, reconstructs words, applies escape corrections, and returns a BF16 view. This public path is not fused into attention. Its transfer results do not establish faster repeated resident-KV reads.

### ZipServ: current code has an on-read BF16 path

[ZipServ, 2603.17435v1](https://arxiv.org/html/2603.17435v1), dated 2026-03-18, studies compressed weights and fused GEMM. Section 7 names KV adaptation as future work. That paper is not evidence that current code lacks KV support.

Official revision `ecf205f` contains a modified [FlashAttention loader](https://github.com/QuiverDance/ZipServ_ASPLOS26/blob/ecf205f865d316712f7ddf5c7690b66abf46f864/kernel_benchmark/third_party/flash_attn_v283/csrc/flash_attn/src/flash_fwd_kernel.h#L220). It reconstructs BF16 words from exponent codes, bitmaps, exact sign/mantissa bytes, and fallback full values. `has_zipserv_kv` routes K/V tile loads through this decoder before attention. This avoids a separate full dense-cache destination in that path.

The [API](https://github.com/QuiverDance/ZipServ_ASPLOS26/blob/ecf205f865d316712f7ddf5c7690b66abf46f864/kernel_benchmark/third_party/flash_attn_v283/csrc/flash_attn/flash_api.cpp#L1406) requires BF16 and rejects paged KV, in-place append, rotary append, and left padding. The benchmark restricts KV heads to eight. The lower API has broader layout checks, so that benchmark limit is not a universal kernel limit. This is direct generic on-read prior code, not validated Ouro support or cross-loop XOR coding.

### ExANS/OpenLake: exact blocks restored after retrieval

The provider's [ExANS method description](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache) preserves sign/mantissa bytes. It codes reversibly centered exponent symbols with ANS. Independent block frames support external KV retrieval.

Current OpenLake revision `0aceab0` provides BF16 GPU codec source. [openlake_expans_decode](https://github.com/openlake-project/openlake/blob/0aceab0086896578364b326a331803c668240c53/crates/openlake_kv_client/cuda/src/expans_codec.cu#L557) runs ANS decoding and `merge_bf16_position_phase` into destination buffers. The [connector](https://github.com/openlake-project/openlake/blob/0aceab0086896578364b326a331803c668240c53/external/connectors/vllm/openlake_expans.py#L157) passes destination addresses. This is source-visible GPU reconstruction, not an inspected compressed-attention tile loader. The blog's externally cached H100 snapshot does not establish resident Ouro decode benefit.

### Glyd: payload exactness is not attention parity

The project's pinned [DETAILS.md](https://github.com/surya-koritala/Glyd/blob/6c9ff3d517cf7ea2c2b80fd9d49b9f30833cb7f7/DETAILS.md#L228) describes lossless BF16 KV for full attention with head size 128. It explicitly states that compressed-cache attention is not bit-equal to vLLM and that greedy outputs can diverge. Its `auto` mode retains stock KV when `exact` or `verify` is active.

The public [GPU module](https://github.com/surya-koritala/Glyd/blob/6c9ff3d517cf7ea2c2b80fd9d49b9f30833cb7f7/bindings/python/glyd/gpu/__init__.py) delegates to a separate `glyd_gpu` package. This worker did not retrieve or audit that package's kernels. Direct on-read support is therefore a project-described lead, not source-verified here. The README's `exact=True` statement cannot establish parity for compressed KV when the detailed path retains stock KV.

### LMCache RFC: load completion differs from KV readiness

[RFC 4498](https://github.com/LMCache/LMCache/issues/4498) proposes compressed storage/transfer followed by GPU decompression into paged KV. It distinguishes record arrival from usable KV completion. Its ownership discussion covers streams, destination buffers, events, locks, and lifetimes.

The issue remained open at inspection. This report treats its selected design passages as a proposal, not a validated implementation. It does not infer that all current LMCache code lacks compression. Its external-load target also supplies no nonexistent native-TP transfer saving.

## Exact cross-loop representation

The following equations are `ANALYTIC_IDENTITY`, not codec measurements. Let `W_r` contain the exact words of one depth's aligned KV block.

For a shared-base representation:

```text
D_r = W_r XOR W_1
W_r = D_r XOR W_1, for r=2,3,4.
```

For a predecessor chain:

```text
D_r = W_r XOR W_(r-1)
W_r = W_1 XOR D_2 XOR ... XOR D_r.
```

XOR alone does not reduce byte count. A codec must compress the residual words and record framing. Small numerical differences do not prove small XOR entropy. A codec that suits raw BF16 exponent distributions need not suit the residual distribution.

Every base needs an exact identity and retained lifetime. The identity includes checkpoint, request generation, prefix block, physical layer, head shard, position, layout, dtype, depth, and version. Post-RoPE position identity matters. No decoder may substitute an uncomputed depth or a numerically close base.

Each appended residual becomes encodable only after its actual source words exist. Encoding adds reads and writes even if later historical reads amortize them. Block indexing, random access, ragged tails, metadata, fallback, and allocation also need accounting.

## Capacity and repeated-read accounting

Let S be the raw size of one equal-sized depth bank. Let b be the encoded base size. Let d2, d3, and d4 be encoded residual sizes. Let c1 through c4 be independent compressed sizes.

The table assumes separate cold depth reads. It excludes metadata, scratch, encode writes, and decoder computation. It counts logical payload reads, not measured HBM traffic.

| Representation | Stored bytes | Payload reads for all four depths |
|---|---:|---:|
| Raw independent banks | 4S | 4S |
| Independently compressed banks | c1+c2+c3+c4 | c1+c2+c3+c4 |
| Shared base, independent residuals | b+d2+d3+d4 | 4b+d2+d3+d4 |
| Predecessor chain, reconstructed separately | b+d2+d3+d4 | 4b+3d2+2d3+d4 |

With a raw base `b=S`, the shared-base path reads at least 4S before positive residual traffic. The chain repeats early residual reads as well. Capacity savings therefore cannot establish bandwidth savings under this read model.

For an illustrative constructed case, set `b=S` and each residual to `S/4`. Storage falls to `1.75S`, a 56.25% decrease. Shared-base reads become `4.75S`, an 18.75% increase. Chain reads become `5.5S`, a 37.5% increase. These are symbolic examples, not results from the captures.

A compressed base can change the result. Shared-base reads beat raw storage only if `4b+d2+d3+d4 < 4S`, before added costs. They beat independent compression only if that expression is below `c1+c2+c3+c4`. A better capacity ratio can still lose the second comparison.

Actual cache hits can reduce HBM traffic below these cold-read counts. They require a named retained set and measured behavior. R4 visits intervening physical layers before reusing a corresponding base. This report assumes neither perfect retention nor universal eviction.

Retaining decoded predecessors in HBM introduces full-size state reads and writes. Retaining them on-chip requires a feasible lifetime and capacity. Retaining all decoded banks can remove the intended hot-cache capacity saving. None is a free solution to chain rereads.

## Decode placement matters

An uncached staging design reads compressed bytes, writes a dense bank, then reads that bank for attention. Its logical movement includes the compressed input plus a full-size write and read. Actual HBM behavior depends on cache and scheduling.

Fused tile decode can avoid that full-bank intermediate. It still pays compressed loads, metadata, reconstruction, scratch, and register pressure. Current ZipServ code demonstrates this generic placement. It does not demonstrate cross-loop base reuse or this campaign's state ownership.

Restoring identical KV words is necessary for a word-exact cache. It does not prove identical attention arithmetic after a kernel change. A same-policy result needs unchanged or qualified reductions, masking, casts, original gate inputs, and terminal-state publication.

Native TP keeps each rank's historical KV shard local in the relevant baseline. This reopening has no per-token remote KV transfer to remove. A separate migration or offload proposal must name a real transfer before claiming its compression saving.

## Adaptive aliases and stronger controls

Under `last_exited`, skipped historical depths read the terminal depth's actual KV. A page alias can represent that identical payload once. The [dynamic-runtime audit](dynamic_runtime_prior_boundary.md) already identifies this prior boundary.

Compress unique payloads after aliasing. Do not encode copied terminal values into all-zero residual streams and count those savings against a baseline that could use aliases. Refcounts and base leases must prevent release while any logical depth or decoder still needs a block.

Adaptive execution reads only its actually required current depths. Historical tokens can have different exit depths within one block. The fixed-R4 table therefore cannot become an adaptive traffic result by multiplying it by an average depth. Full-R4 prefill also remains unchanged under the selected contract.

The minimum controls are:

1. Raw, alias-aware KV under the same gate and prefill policy.
2. Independent lossless compression of the same unique blocks.
3. Shared-base and predecessor-chain formats with every base read charged.
4. Dense staging versus fused tile decode, with scratch and writes counted.
5. Aliased logical unrolling with the same codec and exact block identities.

An R1 control can test generic independent compression. Cross-depth coding needs the same logical depth states even in an unrolled control. A compression-statistics advantage alone does not establish a new recurrent execution axis.

## Reopening decision and next evidence

The changed regime is valid: large KV can make capacity or bandwidth relevant where small-message startup dominated. Existing measurements do not establish either benefit for this new format.

The next source/math gate is an encoded-byte and read-amplification comparison on the bounded key fixture. Values require separate captures. Any later implementation needs online append, depth-aware ownership, exact reconstruction, and qualified attention arithmetic. The task excludes decoder implementation and GPU tests.

Classify a surviving result by its actual saving. Lower capacity without faster reads is a storage result. Faster generic compressed reads are kernel engineering. A distinct architecture claim needs an additional ownership or scheduling benefit against those matched controls.

## Time

This worker segment ran from 2026-10-19T12:24:53Z to the accounting cutoff at 2026-10-19T12:34:25Z. Its observed duration was 572 seconds, or 9 minutes 32 seconds. Worker time overlaps coordinator time and does not add campaign elapsed hours. Final stamping and validation after this cutoff are outside that duration.
