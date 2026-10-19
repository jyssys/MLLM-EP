# Storage capture triangle: Base1.4B TP2

## Result

The saved v5 alias capture equals the saved v4 copies capture exactly on both ranks.
Both complete NPZ files pass direct byte comparison.
All three contained arrays also match exactly.
The equal relative-L2 summaries therefore reflect equal comparison inputs, not just coincidentally equal scalar metrics.

The v4 alias capture differs from both other captures.
Differences already occur in capture call 0, which the runner protocol maps to prefill.
They do not start only in later cached calls.
This audit identifies no kernel cause or new architecture result.

## Input and analyzer identity

The audit reads six explicit NPZ files across three jobs and two ranks.
All six are regular, non-symlink files with distinct filesystem identities.
Each recorded capture path resolves to its expected job-local file.
Each recorded file hash matches the actual file hash.
The [numerical artifact](../artifacts/storage_triangle_reanalysis_v1.json) retains complete paths, hashes, array identities, counts, and first differences.

The [same-arm analyzer](../scripts/compare_same_storage_repeat.py) reads each requested run's `parity_capture.json` at line 33.
It opens those records' NPZ paths at line 38.
The [storage analyzer](../scripts/compare_storage_controls.py) constructs separate alias and copies roots at lines 10–16.
It opens their recorded NPZ paths at line 63.
No stale input path appears in these inspected source routes.
This reanalysis also constructs expected paths independently instead of trusting the recorded paths.

The three frozen snapshots contain identical current versions of the runner, campaign worker, alias control, and storage analyzer.
The same-arm analyzer is not in those pre-run snapshots.
The imported sibling capture helper is also absent from all three snapshots.
Thus, the snapshots do not freeze every historical capture dependency.

All normalized engine configurations match after removal of the job-specific compile-cache path.
All workload hashes match.
The nine recorded cache roots are distinct.
Requested storage modes remain alias, copies, and alias respectively.
Capture counters report four stored logits calls and four teacher steps on every rank.
The helper saves at most four calls.
It does not record an independent total-call counter.
Each job contains 30 timing records and a separate four-token numerical capture.

## Triangular comparison

Counts below compare stored float32 words, not complete model states.
Logits and hidden counts are identical on both ranks.

| Pair | Equal NPZ files | Logit mismatches per rank | Hidden mismatches per rank | KV mismatches, rank 0 / rank 1 |
|---|---|---:|---:|---:|
| v4 alias / v4 copies | No | 151544 / 196608 | 7222 / 8192 | 11409 / 15360, 11452 / 15360 |
| v4 alias / v5 alias | No | 151544 / 196608 | 7222 / 8192 | 11409 / 15360, 11452 / 15360 |
| v4 copies / v5 alias | Yes, both ranks | 0 / 196608 | 0 / 8192 | 0 / 15360, 0 / 15360 |

For v4 alias against either other capture, the recorded logit relative L2 is `0.00791896400461355` on both ranks.
The one-thread CPU recomputation reproduces that scalar exactly.
It also reproduces the hidden scalar exactly.
One legacy rank-0 KV scalar differs slightly under this CPU reduction setting.
The [source ledger](../artifacts/storage_triangle_source_audit_v1.json) retains that discrepancy and a bounded CPU reduction control.
The legacy metric uses a float32 reference norm.
These summary details do not affect byte identity, mismatch counts, or first-difference coordinates.

Every ZIP member stores timestamp `1980-01-01 00:00:00`.
Identical container bytes therefore do not establish that somebody copied an earlier file.
Distinct paths, inodes, and capture records also do not prove the entire historical generation chain.

## First prefill differences

The runner disables chunked prefill and prefix caching.
Its four-token capture supplies one prefill result followed by three cached results.
The phase mapping below follows that protocol and the four recorded calls.
The NPZ contains no independent per-call position trace.

| Call | Protocol boundary | Logit mismatches / 49152 | Hidden mismatches / 2048 |
|---:|---|---:|---:|
| 0 | Last prompt token, position 1023 | 36661 | 1764 |
| 1 | Cached position 1024 | 36753 | 1820 |
| 2 | Cached position 1025 | 40590 | 1826 |
| 3 | Cached position 1026 | 37540 | 1812 |

This table applies to v4 alias against either other capture, on either rank.
The first logit difference is call 0, vocabulary index 2: `-6.625` versus `-6.71875`.
The first hidden difference is call 0, feature 0: `0.2099609375` versus `0.2353515625`.

KV samples cover positions 0, 1, 1023, 1024, and 1026 in all 96 logical slots.
They cover two local heads and 16 packed K/V features per slot and position.
Execution order is recurrent depth first, then physical layer.
In that order, the earliest differing sampled slot is depth 0, physical layer 2, with zero-based indices.
Its first differing prefill sample is at position 1023.
Rank 0 differs in global head 0, V feature 2: `-0.1884765625` versus `-0.1875`.
Rank 1 differs in global head 8, K feature 2: `0.224609375` versus `0.2265625`.

Matching samples in physical layers 0 and 1 do not prove equality of their complete KV tensors or intermediate arithmetic.
Array order differs from execution order.
The artifact retains both the lexicographic first difference and the first sampled slot in execution order.

## What remains unresolved

The hidden capture is the normalized input to `compute_logits` at four positions.
It is not a record of every layer intermediate.
The KV capture is a selected slice, not complete KV.
All four captured top-1 choices match, but teacher forcing supplies the generated tokens.
This observation is not natural-quality evidence.

The storage record contains an implementation-source hash, not loaded tensor-word hashes.
It records within-run equality between each original core and its three copies.
The ownership record contains tensor addresses, shapes, strides, dtypes, and byte counts.
Neither record verifies loaded weight words across these three runs.
A pinned checkpoint path does not replace that missing check.

The identity checks reject stale analyzer-path reuse as an explanation for the current three-file comparison.
The first-call comparison rejects a decode-only start.
The remaining causes cannot be ranked from this evidence.
Loaded-value differences, execution differences, and unrecorded capture-provenance faults remain unresolved possibilities, not findings.
The audit supplies no opaque-GEMM decisions or executed operand observations.
It does not attribute the difference to physical sharing, arithmetic order, compilation, synchronization, or HBM behavior.

No GPU command ran, and no raw file changed.
The source ledger records the actual work interval and source locators.
