# Lossless large-KV screen

## Evidence classes

The CPU codec measured encoded frame lengths and reconstructed every captured BF16 word exactly.
The read-demand calculations are structural models.
Neither result measures GPU HBM traffic, attention latency, memory admission, or E2E speedup.
No encoder, decoder, or compressed-attention kernel ran on a GPU.

## Input and controls

Each model contributes one controlled native TP1 request.
The capture contains selected keys only.
It does not contain values.
It selects three physical layers, three heads, four depths, 257 positions, and 128 dimensions.
The same codec processes every arm.
It uses reversible byte planes and CPU zlib level 1.
No model precision changes.

The independent-coded arm is essential.
Comparing only cross-loop coding with raw BF16 would overstate the value of recurrence.
An adaptive follow-up must also grant terminal-depth aliases to the baseline before coding unique payloads.

## Encoded capacity and read-demand model

Every ratio uses all four raw captured depth frames as its denominator.
The model assumes separate uncached reconstruction for each sequential depth.
It excludes metadata, decoder compute, scratch, append cost, and transaction granularity.

| Model | Independent coded capacity | Adjacent-XOR chain capacity | Common-base XOR capacity | Uncached chain read demand | Uncached common-base read demand |
|---|---:|---:|---:|---:|---:|
| 1.4B | 0.71707× | 0.67618× | 0.71488× | 1.72397× | 1.24077× |
| 2.6B | 0.68256× | 0.61228× | 0.65897× | 1.58297× | 1.15882× |

The adjacent chain stores fewer bytes than independent coding for these selected keys.
It also reads predecessors repeatedly under the stated uncached reconstruction model.
Lower capacity therefore does not establish lower bandwidth demand.
The model does not claim that every access misses hardware caches.
Actual retention requires a legal lifetime and physical measurement.

For one raw frame size S, the common-base path stores `b+d2+d3+d4` encoded bytes.
Separate reads across R4 require `4b+d2+d3+d4` bytes before added costs.
The chain requires `4b+3d2+2d3+d4` bytes under the same assumption.
Queries at different depths are not simultaneously ready merely because their weights match.

At 8192 positions, raw R1 keys across all physical layers occupy 0.1875 GiB per TP4 rank for 1.4B.
The corresponding 2.6B quantity is 0.375 GiB per TP4 rank.
These are key-only capacity calculations, not K+V traffic measurements.
Including values doubles those quantities under the actual MHA geometry.
The H100 L2 capacity is 50 MiB, shared with other traffic.
This capacity comparison neither proves universal cache eviction nor supplies free cross-depth retention.

## Prior boundary and disposition

Current ZipServ code already decodes lossless BF16 K/V inside attention tile loads.
Its inspected path lacks paged-KV and online-append support.
Generic on-read decoding is not a new execution axis.
See [the pinned code and source audit](lossless_kv_reopening_prior.md).

The large-KV regime is a valid change from C08's small-message test.
The current reopening still establishes no new distributed ownership benefit.
Native TP already retains its historical KV locally.
There is no native remote-KV transfer to remove in this comparison.

Disposition: no latency headroom under the tested uncached reconstruction model.
A capacity-only follow-up remains separate and unmeasured.
The evidence does not reject every lossless representation or every possible fused implementation.
It does not justify a new engine or decoder in this campaign.

## Reproduction

The script is `scripts/lossless_kv_screen.py`.
The versioned result is `artifacts/lossless_kv_screen_v1.json`.
It records input hashes, exact reconstruction, per-head results, and aggregate definitions.
The prior audit records source versions and read-path limits.
