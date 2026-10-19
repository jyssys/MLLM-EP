# Full-core storage numerical audit

## Outcome

The completed P2 controls do not establish numerical parity.
Both sizes have different logits, hidden state, and sampled KV between arms.
All four captured top-1 tokens match on each rank.
That agreement is not a parity pass.

The logs establish a compiler-cache confound.
The copies process directly loaded compiled pieces from the same cache directory that the alias process used.
The audit does not establish an incompatible artifact assumption or prove that cache reuse caused the numerical differences.
The audit establishes no new source-level semantic defect in the corrected clone implementation.

Classification: `CONFOUNDED` storage counterfactuals with an unresolved numerical difference.
Parameter-storage assertions establish retained capacity and initial values.
They do not prove which weight addresses the compiled copies arm consumed.
No timing result from these runs establishes a pure residency effect.

## Scope and inputs

This audit reads only the four completed P2 job directories and their exact compiler-cache paths.
It also reads the campaign implementation and installed compiler/model source.
The audit executes CPU checks only.
It does not change frozen artifacts, compiled artifacts, GPU jobs, or implementation source.

Campaign root: `/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery`.
Installed package root: `/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages`.
Relative campaign and package paths below use those roots.

The frozen analyzer is `artifacts/storage_control_checkpoint1.json`.
Its SHA256 is `0a0841d9718649643a9da687835ec7aa8ac00fc9ab43da4735927ed935ab592c`.
The previous source review remains in `reports/full_core_alias_review.md`.
The coordinator corrected the earlier bound-method and loader-name blockers before these runs.

## Observed numerical differences

The frozen analyzer reports these aggregate relative L2 values.
Logits contain four positions and 49152 vocabulary values per position.
Hidden state contains four positions and 2048 values per position.
KV contains only the selected features of two local heads at five token positions.

| Model | Rank | Logits relative L2 | Hidden relative L2 | Sampled KV relative L2 | Top-1 matches |
|---|---:|---:|---:|---:|---:|
| 1.4B | 0 | 0.009118532903943813 | 0.016283988772556234 | 0.008595296869758776 | 4/4 |
| 1.4B | 1 | 0.009118532903943813 | 0.01628492064331092 | 0.009491920688950566 | 4/4 |
| 2.6B | 0 | 0.011880009914397456 | 0.02542241494253733 | 0.00724924809498797 | 4/4 |
| 2.6B | 1 | 0.011880009914397456 | 0.02542053162151021 | 0.007757717669907716 | 4/4 |

The first captured position is not an exact match.
Its small KL does not establish small vector error under every metric.
The following CPU calculation uses float64 for both norms.
The denominator is the alias logit vector's L2 norm.

| Model | First-position relative L2 | Unequal values | First-position KL from frozen analyzer |
|---|---:|---:|---:|
| 1.4B | 0.007438492243734547 | 38151/49152 | 5.944099782957087e-08 |
| 2.6B | 0.008330728446235683 | 38864/49152 | 1.3673992629764967e-07 |

The derived per-position logit relative L2 values are:

- 1.4B: `[0.007438492243734547, 0.008491669462517486, 0.010938643288745185, 0.010970174404585758]`.
- 2.6B: `[0.008330728446235683, 0.017526257147357525, 0.012739684180246583, 0.01288806043574088]`.

The logit arrays agree across ranks within each arm.
The alias hidden arrays differ across ranks at 1/8192 values for 1.4B and 8/8192 values for 2.6B.
Their maximum absolute cross-rank differences are 0.0078125 and 0.0625, respectively.
The copies hidden arrays agree across ranks for both sizes.
These observations do not identify a responsible kernel.

## Difference localization in sampled KV

Both arms sample token positions `[0,1,1023,1024,1026]`.
Both arms record the same cache names and block IDs `[513,513,450,449,449]`.
The first differing sampled logical slot is the same on both ranks.
Logical slot zero means physical layer zero in recurrent pass one.

| Model | Token positions | First differing sampled logical slot |
|---|---|---:|
| 1.4B | All five positions | 1 |
| 2.6B | 0 and 1 | 2 |
| 2.6B | 1023, 1024, and 1026 | 1 |

The selected slot-zero samples agree exactly for both sizes.
The differences therefore already exist in sampled KV from pass one.
Both arms execute the original parameter list in that pass.
Different values in later copied parameters alone cannot explain that localization.

This is not a first-divergent-operator measurement.
The capture samples only 16 features from two local heads per token and slot.
Unobserved slot-zero features or earlier intermediate activations can differ.
The capture cannot isolate attention, GEMM, normalization, collective arithmetic, or graph execution as the cause.

## Observed compiler-cache reuse

Each job has its own process and output directory.
The compiler directories are not process-specific.
Both arms use the same directory for a given model and rank.

| Model and arm | Rank-0 PID | Console evidence | Cache root suffix |
|---|---:|---|---|
| 1.4B alias | 2667739 | Lines 63 and 66 select the directory and compile range `(1,10240)`. | `37ba5182b0/rank_0_0/backbone` |
| 1.4B copies | 2675943 | Lines 57 and 61 select the same directory and directly load compiled graphs. | `37ba5182b0/rank_0_0/backbone` |
| 2.6B alias | 2685708 | Lines 63 and 66 select the directory and compile range `(1,10240)`. | `1cfae2feb0/rank_0_0/backbone` |
| 2.6B copies | 2705811 | Lines 59 and 61 select the same directory and directly load compiled graphs. | `1cfae2feb0/rank_0_0/backbone` |

The full compiler root is `/home/esjung/.cache/vllm/torch_compile_cache`.
Each size also has the corresponding `rank_1_0/backbone` directory.
The audit inspected only those four exact cache directories.

The saved metadata omits `OURO_CAMPAIGN_STORAGE_CONTROL` from its environment factors.
Both sizes record code hash `949a4ca77bfcf17541730743dfe76e92e7f82a07961967f1c9a5336e997033f7`.
Both record compiler hash `da15a9484b`.
The config hashes are `81c6ae5782` for 1.4B and `1eb5381ed0` for 2.6B.
The mode therefore does not produce a distinct automatic directory in these runs.

## What the cache lookup checks

Installed `vllm/compilation/backends.py:223–261` selects a saved handle using:

`(compile_range, graph_index, compiler_name)`.

That lookup does not compare the current FX graph or parameter alias structure.
`InductorStandaloneAdaptor.load` loads the selected artifact at `compiler_interface.py:416–446`.
It uses the current graph to determine tuple return handling.
It does not recompile that graph or compare its parameter identities at this boundary.

Torch's binary loader checks the artifact header and Torch version key.
See `torch/_inductor/standalone_compile.py:70–98`.
Its cache loader reconstructs a saved callable at lines 199–236.
These source boundaries do not provide a new whole-graph alias comparison against the copies trace.

This finding does not mean that every artifact-level alias check is absent.
`AOTDedupeWrapper.post_compile` can check expected duplicate input identity.
See `torch/_functorch/_aot_autograd/runtime_wrappers.py:1312–1325` and `1344`.
The audit does not prove that these saved pieces contain an incompatible duplicate-input assumption.
The audit did not load or execute the saved CUDA artifacts.

## Why cache reuse does not yet prove wrong-weight execution

The artifacts are piecewise compiled graphs, not a serialized whole-model alias forward.
The saved cache indices are `[0,1,24,96]` for 1.4B and `[0,1,48,192]` for 2.6B.
The fresh Dynamo trace still constructs the host graph and passes live operands to the compiled pieces.
Native compilation can reuse structurally equivalent pieces with different weight operands.

The saved readable graphs contain only depth-zero weight argument prefixes.
They contain 194 weight arguments for 1.4B and 386 for 2.6B, including embedding and loop norm.
Those files represent the initial alias trace.
The backend writes `computation_graph.py` only when it does not already exist.
See `backends.py:1259–1270`.
The shared directory therefore does not preserve a second readable copies graph for comparison.

A CPU syntax check found 193 host submodule calls for 1.4B and 385 for 2.6B.
No call repeats an argument name within its own argument list.
That check does not prove disjoint storage or compatible view aliases.
It only prevents an unsupported claim that visible duplicate weight arguments establish this failure.

The available files do not establish whether every compiled depth consumed its intended physical parameter copy.
They also do not establish that it consumed an incorrect parameter.
Both statements remain unresolved.

## CPU and source checks completed

The following checks passed without GPU execution:

- All eight NPZ captures match their recorded SHA256 values.
- Engine configuration and workload JSON match between arms for each size.
- Both captures use forced output IDs `[378,2988,314,216]`.
- Initial parameter-value, retained-capacity, and buffer checks are present in the successful runtime artifacts.
- Selected storage counts match between arms on both ranks.
- The existing three CPU clone regressions pass.

The common unique parameter storage is 5135407106 bytes per rank for 1.4B.
It is 10069481474 bytes per rank for 2.6B.
Those values establish capacity equality, not physical-copy execution or measured HBM traffic.

A separate CPU check extracts the installed `CompilerManager.load` function with Python AST tools.
It supplies two different graph/alias descriptors under the same lookup key.
The function selects the same saved handle for both descriptors.
The test uses a stub compiler.
It checks the vLLM lookup boundary only, not native artifact execution or GPU alias compatibility.

## Supported isolation control and next evidence

`CompilationConfig.cache_dir` is a supported setting at `vllm/config/compilation.py:463–466`.
A supplied directory bypasses automatic directory selection at `backends.py:1058–1078`.
The native backend adds the rank and model-prefix subdirectories.

For a new job, set `compilation_config['cache_dir']` to a new directory under that job's output root.
Keep separate roots for each arm, model, P, and repetition.
Do not delete or reuse the old directories.
The comparison must permit only this intended cache-directory difference between declared engine configurations.
An alternative is a job-scoped `VLLM_CACHE_ROOT`, set before import.

The vLLM directory does not by itself isolate every lower-level compiler cache.
For separate lower-level storage, also set job-scoped `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` before imports.
Installed `torch/_inductor/runtime/cache_dir_utils.py:14–18` and `31–32` support those settings.
This recommendation does not imply that ordinary lower-level, graph-keyed cache reuse is incorrect.

The next bounded control needs these results before timing interpretation:

1. Preserve a readable graph and compiler metadata for each isolated arm.
2. Check intended per-depth parameter operands and unchanged Attention/KV ownership.
3. Compare matched short-request logits, hidden state, and sampled KV.
4. Repeat a same-arm capture to measure process-to-process numerical variation.
5. If differences remain, capture the first divergent intermediate operation in a small diagnostic run.

Cache isolation alone is not a parity pass.
Top-1 agreement alone is not a parity pass.
The evidence cannot assign the differences to a semantic bug, compiler lowering, cache incompatibility, or process-to-process numerical variation.
No cause should replace the unresolved status without a discriminating check.

## Provenance and time

The four raw job roots are:

- `outputs/storage_alias_base14_P2_v1`.
- `outputs/storage_copies_base14_P2_v1`.
- `outputs/storage_alias_base26_P2_v1`.
- `outputs/storage_copies_base26_P2_v1`.

| Input or source | SHA256 |
|---|---|
| 1.4B alias console | `370f594e0f56f583a763c5c4d7e0c0e907e243cb08633613726620b7d62306a6` |
| 1.4B copies console | `395592f6db0d75f4b02d05fad49f07e4a837d59c6c04818557e3c87267a4dba8` |
| 2.6B alias console | `5cd9e80871bdd40d4e224ab2224f28898df15c9e21798effe2adbc56089ddc37` |
| 2.6B copies console | `2ad27024d21485bb3367ddef6335d8810474d2de26ce5b651274e6e539b56cf3` |
| `alias_control.py` | `3c11cde3e297be60d9fb14b5d6e15c13713051cccb964bef1abdaffcb458c8dd` |
| `scripts/compare_storage_controls.py` | `3585fb0d59eabc52aaebeb0ae8f6c8f983721dcc795c1bc3a7aa55e7f3bf554c` |
| `tests/test_storage_clone.py` | `cad6cb57501181348eae3c800f8ecfc501d7a33b49489316f7eb89592e60fb16` |
| `vllm/model_executor/models/ouro.py` | `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff` |
| `vllm/compilation/backends.py` | `fbd068047613b97f28dc58f98c77c64f8252a9fae2a88efae09b78e9e794bac5` |
| `vllm/compilation/compiler_interface.py` | `8d84e2864bf83f754bae9e9f77119e0287cca9f97f5fa5ec2535e07b5ad88944` |
| `vllm/config/compilation.py` | `267f4ca04fdcd2f0cf55c334782dd498c8051b9500ad088673d79518fc712db0` |
| `torch/_inductor/standalone_compile.py` | `21f7634f0524daa7582c3d86bfe2e70e117d80b085ead0233b516358bb9695c9` |
| `torch/_functorch/_aot_autograd/runtime_wrappers.py` | `6101332cb8d6549c3e662f518a7bb4dc932c3648c622a7c9b774c5ee59bdb82e` |

The rank-0 readable graph hashes are `7720e433f11c7ea1068ce7d7570ccab4f164130ade0678654f68c43172ae47ab` for 1.4B.
The corresponding 2.6B hash is `319efccc7323a7c6547f729b0ef457f6ea36b6cab8d13764187498da176c56e0`.
These shared cache files are mutable cache metadata, not independent process-specific records.
The hashes identify the files read during this audit.

Own segment start: `2026-10-19 12:33:16 UTC`.
Recorded end after checks: `2026-10-19 12:42:53 UTC`.
The recorded segment lasts 577 seconds.
This segment includes CPU analysis, source reads, and report writing only.
It excludes the completed GPU jobs and other agents' work.
The report linter found zero violations.
