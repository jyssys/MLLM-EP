# v4 storage provenance: checkpoint 2

## Result

Six completed v4 jobs pass the saved-graph and cache-root checks.
The snapshot time is 2026-10-19 13:29:05 UTC.
The [machine checkpoint](../artifacts/storage_provenance_v4_checkpoint2.json) records each check and exact path.
The queue still runs at this checkpoint.
Base2.6B P4 copies remains pending in this snapshot.

These results establish compiled operand counts and saved graph provenance.
They do not establish the runtime pointers bound to those operands.
They do not establish numerical parity or a storage-related timing effect.
All v1 and v2 failures remain unchanged.

## Completed exports

Each job has a new directory under `analysis/storage_provenance_v4/JOB/`.
Each directory contains `manifest.json` and one compressed host graph per rank.
The exporter checks source and gzip SHA256 hashes.
It does not retain binary compiler caches.

| Job suffix after `storage_` | Ranks checked | Weight operands per rank | Block operands by source depth | Result |
|---|---:|---:|---|---|
| `alias_base14_P2_fully_isolated_v4` | 2/2 | 194 | Depth 0: 192 | PASS |
| `copies_base14_P2_fully_isolated_v4` | 2/2 | 770 | Depths 0–3: 192 each | PASS |
| `copies_base26_P2_fully_isolated_v4` | 2/2 | 1538 | Depths 0–3: 384 each | PASS |
| `alias_base26_P2_fully_isolated_v4` | 2/2 | 386 | Depth 0: 384 | PASS |
| `alias_base14_P4_fully_isolated_v4` | 4/4 | 194 | Depth 0: 192 | PASS |
| `copies_base14_P4_fully_isolated_v4` | 4/4 | 770 | Depths 0–3: 192 each | PASS |

The two additional operands belong to the embedding and recurrent normalization.
Each model block contributes eight weight operands.
All checked graphs have the required layer coverage, weight roles, BF16 shapes, and source uses.
The observed distinction was absent from the invalid v2 pairs.
This distinction does not retroactively repair v2 attribution.

## Cache-root evidence

For each checked job, `cache_environment.json` records these roots before the vLLM import:

- `RUN/vllm_cache` for `VLLM_CACHE_ROOT`.
- `RUN/inductor_cache` for the initial `TORCHINDUCTOR_CACHE_DIR`.
- `RUN/triton_cache` for `TRITON_CACHE_DIR`.

`RUN` is that job's `<size>_R4_TP<ranks>` directory.
Every root differs between jobs.
The native driver sets these values before it imports vLLM.
Its earlier path-configuration functions do not import Torch or vLLM.

The consoles identify ordinary compile paths below `RUN/compile_cache`.
They identify saved AOT models below the job's `RUN/vllm_cache`.
The AOT hash suffix can match another arm's hash.
The enclosing cache roots remain different.
Each job has AOT model files for every expected rank.

Installed vLLM relocates the main Inductor cache below the selected AOT hash.
The actual location is:

```text
RUN/vllm_cache/torch_compile_cache/torch_aot_compile/HASH/inductor_cache
```

Every completed job has this directory.
The initially requested `RUN/inductor_cache` also exists.
The checkpoint distinguishes these two locations.
It does not mistake the initial environment value for the final AOT location.

The checker reads Triton group metadata under each job's root.
Every recorded child path stays inside that root and refers to an existing file.
Each P2 job has 45 group records and 315 recorded child paths.
Each checked P4 job has 49 group records and 343 recorded child paths.
The compiled-piece metadata also points only to existing files inside the job's ordinary compile cache.

The [cache-root checker](../scripts/audit_storage_cache_roots_cpu_v1.py) retains console line locators and source hashes.
Its path checks use resolved paths.
It exports newly completed jobs without overwriting earlier exports.
This audit does not claim isolation of every possible vendor-library cache.
It checks the requested vLLM, AOT, Inductor, Triton, and ordinary compile paths.

## Runtime pointer identity remains a separate check

Existing `storage_control.json` records distinct allocated copies and shared buffers after warmup.
Its allocation checks do not record the weights bound to the compiled call.
Saved graph arguments also do not directly record runtime pointers.

Installed `CUDAGraphWrapper` stores input addresses at `cuda_graph.py:279–282`.
It compares those addresses during replay only in DEBUG mode, at lines 347–355.
A FULL wrapper can receive only model inputs.
Therefore, its address list does not necessarily contain lifted weight operands.

The precise lifted-operand interface is `VllmSerializableFunction.__call__` at `caching.py:216–217`.
It forwards its arguments to `optimized_call`.
The opt-in [binding probe contract](lifted_operand_binding_probe_contract.md) describes the bounded follow-up.
No GPU binding capture occurred in this audit.

## Checks and limits

The CPU graph validator passed every completed job in the table.
The binding-name mapper also accepted the exported graph names for both model sizes and both arms.
The root checker passed every completed job in the table.
No GPU command ran in this audit.
No raw output, coordinator file, or installed package changed.

The parent agent must evaluate numerical parity separately.
The parent agent must not use these counts as a measured storage benefit.

## Source and time ledger

The queue source is `artifacts/storage_fully_isolated_queue_v4.json`.
The checkpoint records the exact queue snapshot hash.
The [first checkpoint](../artifacts/storage_provenance_v4_checkpoint1.json) remains unchanged.
It covers the first three completed jobs.

Installed source root:

```text
/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/
```

| Source | Locator | SHA256 |
|---|---|---|
| `vllm/compilation/decorators.py` | Lines 525–574 | `dd7167637a478aeaea3f2410a727ddefc702bda1d9bf632831c2a8b4811ebb1e` |
| `vllm/compilation/caching.py` | Lines 216–217 | `624426dfd4a86fdb9d0354a01a3733981e3008baf0cf141c9578e978f01e7fd7` |
| `vllm/compilation/cuda_graph.py` | Lines 128–135, 279–282, 347–355 | `cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412` |
| `triton/runtime/cache.py` | `FileCacheManager`, lines 36–55 | `0355c3e3452fa4500122244b8fff8864f22bc05c417a3d6f5dada9f2e92f8040` |
| `triton/knobs.py` | Cache directory setting, line 347 | `d84bda664f733f0547f9f4397c2dab11da1f9c58f974a840a3f1c61ea48c76fa` |

The combined audit and probe-work segment started at 2026-10-19 13:16:12 UTC.
The final recorded time was 2026-10-19 13:30:49 UTC.
The elapsed wall time was 877 seconds.
The combined CPU test run passed 25/25 tests.
Both new reports passed the prose lint with zero findings.
The time ledger covers both new reports, not two separate elapsed intervals.
