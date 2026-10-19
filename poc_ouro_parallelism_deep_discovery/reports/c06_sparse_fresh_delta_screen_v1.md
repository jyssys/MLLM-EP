# C06 exact sparse fresh-delta screen v1

Decision: the retained samples do not support the tested sparse-update premise. Nearly every input coordinate changes between adjacent passes. Every tested aligned tile contains a changed coordinate. This is a CPU structural result, not a speedup or quality result. No candidate card changed.

## Scope and pair identity

The [artifact](../artifacts/c06_sparse_fresh_delta_v2.json) contains 2448 row pairs across both Base sizes. Each size contributes 1224 row pairs. Each size uses two continuity prompts and three selected physical layers. This is not held-out or population evidence. Row pairs share prompts, layers, positions, and recurrent state. No independent-token confidence interval applies.

The Base1.4B layers are `[0,12,23]`. The Base2.6B layers are `[0,24,47]`. The checkpoint revisions remain `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` and `1ed04250da1a9936042725d302e81c8fa2ab5abd`.

The [capture hook](../scripts/capture_and_sanity.py:25) retains the final 32 rows of each prefill call. Calls 1 through 4 cover full-R4 prefill in both modes. The screen compares only depth pairs `0→1`, `1→2`, and `2→3` at the same position. The two prompt lengths are 304 and 318. Their retained prefill positions are 272–303 and 286–317.

Fixed decode uses calls 5 through 20. Each four-call group belongs to one position. The screen excludes each group boundary. It covers positions 304–307 and 318–321. The screen excludes adaptive decode because its call groups have variable lengths.

Positions are source-backed reconstructions, not retained position tensors. The mapping uses the capture hook, one unchunked prefill per request, and the fixed four-pass policy. CDB's [prefill completion](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1407) leaves the next position after the prompt. Its [decode item](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1561) uses that position. The checked CDB revision is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.

Every artifact row records these identities:

- Checkpoint size, mode, dataset ID, request ordinal, and scoped native request ID.
- Physical layer, absolute position, retained tensor row, and adjacent depths.
- Both exact capture keys.

The native request ID is `request-0` after each reset. The scoped key prevents collisions between requests and modes. All 2448 keys are unique. Fixed and adaptive prefill tensors are bitwise identical within each size. Their separate denominators do not provide independent replication.

## Measured scalar and tile sparsity

Each row has 2048 BF16 elements. Numeric equality treats `+0` and `-0` as equal. Word identity compares all 16 storage bits. Nonfinite inputs cause rejection. No retained input is nonfinite. No sampled pair has a signed-zero-only word change. Thus, numeric-equal and word-identical counts coincide in these samples.

The prefill entries below apply separately to fixed and adaptive mode.

| Size and phase | Row pairs | Equal scalars / all scalars | Equal fraction | Changed coordinates per row |
|---|---:|---:|---:|---:|
| 1.4B prefill, each mode | 576 | 4354 / 1179648 | 0.36909315% | 2021–2048 |
| 1.4B fixed decode | 72 | 629 / 147456 | 0.42656793% | 2021–2047 |
| 2.6B prefill, each mode | 576 | 4926 / 1179648 | 0.41758219% | 2020–2048 |
| 2.6B fixed decode | 72 | 737 / 147456 | 0.49981011% | 2019–2047 |

No whole row remains unchanged. All tested tiles also change. Each tile uses an aligned, non-overlapping coordinate group within one row.

| Tile width | Unchanged / all prefill tiles, each size and mode | Unchanged / all fixed-decode tiles, each size |
|---|---:|---:|
| 8 | 0 / 147456 | 0 / 18432 |
| 16 | 0 / 73728 | 0 / 9216 |
| 32 | 0 / 36864 | 0 / 4608 |
| 128 | 0 / 9216 | 0 / 1152 |

Both numeric and word-based tile tests give these counts. This result directly tests fresh coordinate changes. It does not reuse the earlier whole-row memoization criterion.

## Structural byte accounting

Dense BF16 input costs 4096 bytes per row. The optimistic indexed model packs each coordinate index into 11 bits. It charges 16 bits per replacement word or 32 bits per FP32 delta. Each row rounds its payload upward to whole bytes. Counts, headers, alignment, discovery, and codec costs are absent.

For `k` changed words, replacement bytes are `ceil(27k/8)`. For `k` numerically changed coordinates, FP32-store bytes are `ceil(43k/8)`. These formulas assume that the consumer already owns the matching base row. These explicit encoding models are not lower bounds over all encodings.

| Size and phase | Dense BF16 bytes | Indexed BF16 replacement bytes | Indexed FP32-store bytes |
|---|---:|---:|---:|
| 1.4B prefill, each mode | 2359296 | 3966879 | 6317467 |
| 1.4B fixed decode | 294912 | 495572 | 789226 |
| 2.6B prefill, each mode | 2359296 | 3964947 | 6314391 |
| 2.6B fixed decode | 294912 | 495211 | 788649 |

BF16 replacement costs 1.67918226–1.68138250 times the dense payload across these groups. The artifact also charges ordinary uint16 indices. That control costs more. These are representation sizes, not measured HBM traffic or network transfers.

## Exact subtraction is a separate condition

The Base1.4B differences all fit FP32 exactly. Its FP32-store counts therefore describe exact mathematical deltas. They still exceed dense BF16 payload by 2.67614068–2.67769157 times.

Base2.6B contains mathematical differences that do not fit one FP32 value exactly. Prefill has 55 such elements out of 1179648, separately in each mode. Fixed decode has 2 out of 147456. Only 529 / 576 prefill row pairs and 70 / 72 decode row pairs have entirely exact FP32 deltas. Thus, the complete Base2.6B FP32-store columns are not exact-delta encodings.

The CPU model needs 38 target-word corrections per Base2.6B prefill group and one for fixed decode. These corrections add 149 and 4 packed bytes. The totals with these additional words are 6314540 and 788653 bytes. Corrections reconstruct the sampled target words. They do not make each rounded FP32 difference an exact mathematical delta.

BF16 subtraction is also not generally exact here. Exact BF16 differences cover 842160 / 1179648 and 109527 / 147456 elements for Base1.4B prefill and decode. The corresponding Base2.6B counts are 863905 / 1179648 and 111835 / 147456. Prefill counts apply separately to each mode.

The checker decodes BF16 integer significands and binary exponents. It checks representability instead of assuming widened subtraction is exact. A separate rational-number oracle checks 2048 constructed finite pairs. Signed-zero corrections remain separate from numeric delta sparsity.

## Controls, provenance, and limits

Constructed controls pass for identical rows, one changed coordinate, one changed width-32 tile, a signed-zero change, and a dense change. The one-coordinate control detects 15 / 16 unchanged width-128 tiles. Its packed replacement costs 4 bytes instead of 4096. Thus, the screen can detect the requested positive premise.

Constructed subtraction controls also pass. For base `1` and target `2^-10`, the BF16 difference rounds to `-1`. Adding it to the base produces zero, not the target. For target `2^-30`, even the mathematical difference does not fit one FP32 value exactly. All three nonfinite-input rejection cases pass.

Raw capture SHA256 values are:

- Base1.4B: `31ea22ad59a5be626d0ef3d17073388755c3e1df72da85830a8296ed01f752e4`.
- Base2.6B: `6ae400c8b9ed08dcd95cede9c84d812909756fd4c556981f0c4029caa3f9739c`.

The artifact records raw paths, sanity-file hashes, checkpoint pins, and inspected source hashes. Both capture files and sanity files retain their hashes after the screen. The [script](../scripts/c06_sparse_fresh_delta_screen.py) runs on CPU and uses one CPU thread. Syntax checking and the exact-key coverage check pass. No GPU run, projection, package change, card change, or raw rewrite occurred.

The captured tensor is the MLP input of width 2048. It precedes the gate and up projections. See [OuroMLP.forward](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/models/ouro/modeling_ouro.py:167). It is not the width-5632 nonlinear intermediate before the down projection. No claim transfers to that uncaptured tensor.

The tested samples do not supply a sparse tile path for a cheaper fresh residual projection. This result does not reject every representation or every workload. Exact sparse FMA evaluation can change native accumulation and BF16 rounding. This audit does not establish projection parity, quality preservation, kernel latency, or numerical speedup. It also does not supply free base storage, base projections, gather work, merges, normalization, or nonlinear operations.

The source and CPU segment ran from `2026-10-19 16:01:42 UTC` to `2026-10-19 16:07:51 UTC`. The elapsed time was 369 seconds. Report drafting and lint followed this segment.
