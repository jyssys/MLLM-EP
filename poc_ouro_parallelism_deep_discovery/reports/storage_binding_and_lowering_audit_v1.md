# Storage controls: observed bindings and bounded lowering audit

## Result

Every intended lifted weight matches its expected metadata in the two observed Base1.4B P2 prefill jobs.
The check covers 1,928 weight operands and 11,568 metadata-field comparisons.
There are zero mismatches.

These observations concern two fresh evidence jobs, not the timed v4 processes.
Both observed calls have `cudagraph_runtime_mode=NONE` and 1024 prompt tokens.
They do not establish cached decode-kernel parameter addresses.
They do not establish the cause of the v4 numerical differences.

All eight completed v4 jobs also pass the saved-graph and cache-root checks.
The [complete provenance checkpoint](../artifacts/storage_provenance_v4_checkpoint4.json) retains those results.
All four v4 model/TP cells still have numerical differences.
No cell receives a numerical-parity PASS from this audit.

## Raw evidence and scope

The binding inputs are:

- `outputs/storage_binding_alias_base14_P2_v1/1.4B_R4_TP2/operand_binding.json`.
- `outputs/storage_binding_copies_base14_P2_v1/1.4B_R4_TP2/operand_binding.json`.

The numerical input is [storage_fully_isolated_checkpoint3.json](../artifacts/storage_fully_isolated_checkpoint3.json).
Its SHA256 is `92b568214c0ccb9d8fa073d01877087b89b180f61a0174ee17becf71018e6d18`.
The rank-0 lowering inputs belong only to the eight completed `fully_isolated_v4` jobs.

I did not inspect the running same-arm v5 repeat.
I did not wait for that repeat or interpret its output.
I did not execute a GPU command, modify a raw artifact, or modify an installed package.

## Binding recomputation: direct saved-record checks

The [CPU checker](../scripts/audit_storage_binding_records_cpu_v1.py) does not trust the recorded `CAPTURED_MATCH` flag.
It independently derives the expected parameter name from each weight placeholder.
It then compares the recorded operand against the loaded parameter snapshot.

The six compared fields are:

- Data pointer.
- Storage-base pointer.
- Storage offset.
- Shape.
- Stride.
- Dtype.

The checker also checks argument order and complete block coverage.
It checks every storage-owner name against the snapshot's pointer map.
It checks BF16 pointer arithmetic: `data_ptr = storage_data_ptr + 2 * storage_offset`.
It checks the four-depth execution groups against the same named snapshots.
The [binding ledger](../artifacts/storage_binding_records_audit_v1.json) retains all verdicts and source hashes.

| Arm | Rank | Total placeholders | Lifted weights checked | Field comparisons | Distinct observed weight storages | Result |
|---|---:|---:|---:|---:|---:|---|
| Alias | 0 | 295 | 194 | 1,164 | 194 | PASS |
| Alias | 1 | 295 | 194 | 1,164 | 194 | PASS |
| Copies | 0 | 871 | 770 | 4,620 | 770 | PASS |
| Copies | 1 | 871 | 770 | 4,620 | 770 | PASS |

Every checked weight is BF16 and has storage offset zero.
The alias arm has 192 block weights, one embedding weight, and one recurrent-normalization weight per rank.
The copies arm has 768 block weights and those same two shared-weight roles per rank.
The two jobs use different processes and addresses.
This audit does not compare absolute pointer values between processes.

Each rank has 192 corresponding block-weight groups across the four depths.
Every alias group has one storage across depths.
Every copies group has four distinct storages across depths.
All 768 recorded execution groups pass the recomputed checks.

The observed shapes per rank are:

| Shape | Alias count | Copies count |
|---|---:|---:|
| `[24576,2048]` | 1 | 1 |
| `[2048]` | 97 | 385 |
| `[3072,2048]` | 24 | 96 |
| `[2048,1024]` | 24 | 96 |
| `[5632,2048]` | 24 | 96 |
| `[2048,2816]` | 24 | 96 |

All observed vectors have stride `[1]`.
All observed matrices have row-major stride `[column_count,1]`.
The checker compares these strides against each named parameter's recorded stride.
It does not infer them only from shape.

## Observation boundary and limitations

Each record has `calls_seen=1`, no skipped calls, and `original_call_outcome=RETURNED`.
Each record identifies a 1024-token, nonuniform prefill descriptor.
Each job's guard and runner status reports success.
Each job returns the four frozen teacher token IDs.
These forced tokens are not a natural-quality result.

The source boundary is `VllmSerializableFunction.__call__`, at installed `vllm/compilation/caching.py:216–217`.
It forwards the lifted arguments to the existing `optimized_call`.
The process-local observer records metadata at that boundary.
The [probe contract](lifted_operand_binding_probe_contract.md) describes the observer.

The observations establish the intended weight-binding distinction for those two prefill invocations.
They do not prove downstream kernel addresses without an internal transformation.
They do not inspect weight values.
They do not observe the later cached decode replays.
They do not directly observe v4's timed processes, P4 binding jobs, or Base2.6B binding jobs.

## CPU negative checks

The checker modifies each of the six fields in memory, one field at a time.
It performs this check once for each arm.
All 12 injected errors produce the expected metadata mismatch.
No mutation reaches the raw JSON files.

The command is:

```bash
python poc_ouro_parallelism_deep_discovery/scripts/audit_storage_binding_records_cpu_v1.py --selftest
```

It returns four passing real-record verdicts and 12 detected negative checks.
This is a reproducible record-consistency check.
It is not a live numerical reproducer.
This task does not diagnose the numerical cause without that separate evidence.

## Rank-0 lowering comparison

The [lowering ledger](../artifacts/storage_v4_rank0_lowering_audit_v1.json) covers rank 0 in each v4 cell.
The [CPU inspector](../scripts/audit_storage_v4_lowering_cpu_v1.py) parses host graphs and embedded generated source.
It reads byte frames and pickle opcodes without loading saved objects.
It does not execute a compiled artifact.

All four rank-0 pairs have equal host operation counts and shape annotations.
Their normalized leaf functions also match.
Their collective/attention source sequences match.
This is a source comparison, not a measured device timeline.

Each alias job retains four compiled templates per rank 0.
Each copies job retains six.
The additional templates do not imply additional model operations.
The complete host graphs have the same operation counts.

| Model | Copies-only template | Matching alias template after normalization |
|---|---:|---:|
| Base1.4B | 25 | 1 |
| Base1.4B | 48 | 24 |
| Base2.6B | 49 | 1 |
| Base2.6B | 96 | 48 |

These mappings hold at P2 and P4, on inspected rank 0.
All 24 copies templates map to an alias template with the same normalized call AST and Triton compute-body list.
The [template ledger](../artifacts/storage_v4_rank0_template_pairs_v1.json) records every pair.
The normalized comparison renames local variables and symbolic dimensions.
It does not claim complete compiler or runtime equivalence.

Raw embedded source hashes differ across matched arms.
Direct examples include comments that name different job-local kernel paths.
The normalized call and compute-body checks do not erase the saved autotune differences below.

## Saved autotune differences

On rank 0, both P2 cells have matching autotune keys and non-timing values across all six mapped template pairs.
The `time_taken_ms` fields differ in saved records.
Those fields describe tuner bookkeeping, not measured decode latency.

Some P4 pairs also differ in `XBLOCK`, `num_warps`, and the Triton cache hash.
The saved Triton JSON files identify pointwise kernels with the corresponding warp counts.
The [autotune ledger](../artifacts/storage_v4_rank0_autotune_deltas_v1.json) records exact keys, paths, values, and hashes.

| Cell | Copies template indices | Saved alias choice | Saved copies choice | Kernel role |
|---|---|---|---|---|
| Base1.4B P4 | 1, 25, 96 | `XBLOCK=1024`, 4 warps | `XBLOCK=512`, 8 warps | Multiply/SiLU/slice |
| Base1.4B P4 | 24, 48 | `XBLOCK=1024`, 4 warps | `XBLOCK=512`, 8 warps | Fused add/cat/index/multiply/split/view |
| Base2.6B P4 | 0 | `XBLOCK=512`, 8 warps | `XBLOCK=1024`, 4 warps | Embedding/mask operations |
| Base2.6B P4 | 48, 96 | `XBLOCK=512`, 8 warps | `XBLOCK=1024`, 4 warps | Fused add/cat/index/multiply/split/view |

The detailed Triton metadata paths cover the common-index pairs.
The template ledger also records the mapped additional copies templates.
All listed saved choices use one stage.
The files record these launch-metadata differences.
They do not prove a numerical or timing cause.

This audit does not inspect every rank's autotune choices.
It does not measure the selected launch configuration of each timed replay.
It does not inspect opaque collective, attention, or vendor GEMM internals.

## Numerical evidence remains unresolved

The frozen v4 table reports these logit relative-L2 values.
Every rank within each cell has the listed value.

| Model | TP ranks | Captured logit relative L2 |
|---|---:|---:|
| Base1.4B | 2 | 0.00791896400461355 |
| Base1.4B | 4 | 0.006442085596647812 |
| Base2.6B | 2 | 0.013160730423628905 |
| Base2.6B | 4 | 0.009889388213631953 |

These checks cover four controlled teacher-forced positions.
They do not establish natural-task accuracy.
The binding PASS does not convert these numerical differences into a parity PASS.
Neither source similarity nor saved launch differences establish a numerical cause.
This audit makes no storage-benefit or latency-cause claim.

## Source ledger

| Source | Locator | SHA256 |
|---|---|---|
| Alias binding JSON | Two rank records | `b106dded657a4dfc913c7b7c1c1dd9cc8ba15177fb1e3b07bb53f866413cfd05` |
| Copies binding JSON | Two rank records | `711428cc1af67da4efe09b02878316a3a8bbb391cc57c6e60e8b3dbf2655d727` |
| [Binding checker](../scripts/audit_storage_binding_records_cpu_v1.py) | `check_row`, line 33 | `4abadd0454b7a94e431c632eee2f354f0e3516a584cb34c6acbaf2cfbe235bf8` |
| [Lowering inspector](../scripts/audit_storage_v4_lowering_cpu_v1.py) | `main`, line 34 | `2a7667f4925e131725df800a693f34dddd2421cb419a5b71fb8ca35dcb7f6fea` |
| [Shared source parser](../scripts/audit_storage_lowering_cpu_v1.py) | `normalized_function`, line 114. `artifact_summary`, line 149. | `cd8a2e3cca1ecfd54e857c5bb19367f6b8589383ded91c8399f860adb963367b` |
| [Observer core](../operand_binding.py) | `tensor_metadata`, line 56. `observed`, line 106. | `ab6640c68e9375556de013a8ed0522d5150d7557bb0ab8a0e424a3191f7b3254` |
| [Worker adapter](../operand_binding_worker.py) | `start_operand_binding_capture` | `4bd6dbd7182b5f3d0a931f38b4701f9c793c8261640b697f6a4f6a1260ec5ad9` |
| [Runner](../scripts/storage_binding_probe.py) | Frozen reference and one evidence request | `eb229557aa73706dfc7e7d1c394f4ea839e0d86b7bc0db0c9d6a236aed891d78` |

Both raw binding jobs record the observer, worker, and runner hashes above.
Their reference records identify the v4 configurations and passing provenance manifests.
Both consoles save AOT inside their separate job-local vLLM roots, at line 65.
Their ordinary compile-directory messages occur at line 61.

## Actual time

This segment began at 2026-10-19 13:43:18 UTC.
The final recorded time was 2026-10-19 13:54:56 UTC.
The elapsed wall time was 698 seconds.
The report lint returned zero findings.
The saved-record checks passed all four rank records and all 12 negative CPU checks.
No GPU execution occurred in this segment.
