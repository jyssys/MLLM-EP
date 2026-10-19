# Storage control: v2 graph and cache audit

## Result

All v2 alias-versus-copy attribution remains UNVALIDATED.
This includes the P2 bitwise equality results.
The jobs used separate ordinary compile directories, but both arms reused the same global AOT model cache.
The saved top-level weight arguments do not distinguish the intended arms.

The CPU inspection found no difference in generated arithmetic source between matched arms.
This finding does not identify the cause of the P4 numerical differences.
It does not establish a benefit, or absence of benefit, from parameter aliasing.
It does not establish a valid latency comparison.

This audit does not replace the initial [source review](full_core_alias_review.md).
It adds evidence about compiled execution provenance.
The [v1 numerical audit](storage_numerical_audit.md) remains unchanged.

## Scope and frozen evidence

The input is [storage_isolated_checkpoint3.json](../artifacts/storage_isolated_checkpoint3.json).
Its SHA256 is `1af10fc826bcecc9bf93a3bdeb730ae521b34e042a43e1d8889ec073aebea838`.
The new [machine ledger](../artifacts/storage_isolated_lowering_audit_v1.json) records all 12 matched rank pairs.
It retains graph hashes, weight counts, operation counts, and 56 compiled-piece comparisons.

I inspected only these job families:

- `outputs/storage_{alias,copies}_base14_P{2,4}_isolated_v2/`.
- `outputs/storage_{alias,copies}_base26_P{2,4}_isolated_v2/`.

Within each job, I inspected `console.log`, saved host graphs, cache metadata, and embedded generated source.
I did not open global AOT model artifacts.
I did not import Torch or load compiled objects for these CPU checks.
I did not run a GPU job or change a frozen artifact.

## Measured results: preserved, but not attributed

The frozen analyzer reports the following values.
The numerical checks cover four teacher positions, not natural-task quality.

| Model | TP ranks | Logit relative L2 | Captured hidden/KV result | Top-1 agreement |
|---|---:|---:|---|---|
| Base1.4B | 2 | 0 | Bitwise equal | 4/4 positions |
| Base2.6B | 2 | 0 | Bitwise equal | 4/4 positions |
| Base1.4B | 4 | 0.010242911645473862 | Different | 4/4 positions |
| Base2.6B | 4 | 0.014769949318441139 | Different | 4/4 positions |

P4 Base1.4B hidden relative L2 spans 0.018552859988291467 to 0.018557197011739805 across ranks.
Its sampled-KV relative L2 spans 0.006934039408304306 to 0.007607177327267034.
P4 Base2.6B hidden relative L2 spans 0.03128501976336032 to 0.031288005027181105.
Its sampled-KV relative L2 spans 0.008638682912350428 to 0.009596221388236246.

These results describe the jobs that ran.
They do not establish the requested physical-copy comparison.
Four matching top-1 values do not constitute numerical parity.

## Cache precedence: source-backed

Installed `vllm/compilation/decorators.py:525–574` selects whole-model AOT caching before ordinary tracing.
The AOT path uses `VLLM_CACHE_ROOT`, not the requested ordinary `compilation_config.cache_dir`.
The job metadata sets `VLLM_USE_AOT_COMPILE=true` and `VLLM_USE_MEGA_AOT_ARTIFACT=false`.

The lookup proceeds as follows:

1. `aot_compile_hash_factors()` hashes environment and vLLM configuration factors.
2. `_model_hash_key()` adds the vLLM version, forward-function name, and first source line.
3. The wrapper builds a global AOT directory from that hash.
4. The wrapper sets `local_cache_dir` to that directory.
5. The wrapper unconditionally replaces `TORCHINDUCTOR_CACHE_DIR` with its `inductor_cache` child.
6. The wrapper tries the saved rank-local `model` before it traces the current object.

The load path checks traced source contents at `decorators.py:265–280`.
It does not explicitly compare the current model's parameter-alias structure in this lookup code.
At lines 303–305, it disables loaded-function guards when `evaluate_guards=false`.
The v2 console configurations record this value as false.
This is not a claim that every PyTorch cache path lacks alias checks.
It describes this installed load path and these jobs.

Thus, separate ordinary compile directories did not isolate the whole-model cache.
Explicit job-level Inductor paths also did not override the AOT path's later assignment.

## Actual global AOT paths

The common path root is:

```text
/home/esjung/.cache/vllm/torch_compile_cache/torch_aot_compile/
```

Each hash below has the suffix `rank_<rank>_0/model`.
The consoles identify these paths directly.
The table does not infer them from ordinary cache directory names.

| Cell | Global AOT hash | Console evidence |
|---|---|---|
| Base1.4B P2 | `2dcb5f8fee4408408a6f452be00fb4a10c951500f4fca553807705e19129340b` | Alias lines 63, 65. Copies lines 63–64. Both load. |
| Base1.4B P4 | `45b313d913a80860cf957e2eded1841b95e0e1ade2c28a484911637e1237613b` | Alias lines 87–90. Copies lines 87–91. Both load. |
| Base2.6B P2 | `4d937b8512c97b49500565cae76af43793a364a86299d2e0bac43aa844b305f6` | Alias lines 63, 65. Copies lines 63–64. Both load. |
| Base2.6B P4 | `d72933cb6143771fb2b9f5332f81171294f57558d7a4f73e62aae32591a51ba3` | Copies line 96 saves rank 0. Alias lines 87–91 load all four ranks. |

All console paths are under the job families listed above.
The Base2.6B P4 copies console records compilation before the alias job loads the same hash.
This chronology supports a cache-provenance concern.
It does not prove the cause of the numerical difference.

## Host-graph provenance: direct CPU inspection

The inspected path is `compile_cache/rank_<rank>_0/backbone/computation_graph.py` within each run directory.
Each model block has eight weight arguments.
The embedding and recurrent normalization add two shared arguments.
The expected alias counts are 194 and 386 for Base1.4B and Base2.6B.
The expected copies counts are 770 and 1538.

| Cell | Alias graph weights | Copies graph weights | Depth-prefix counts in both graphs | Required distinction |
|---|---:|---:|---|---|
| Base1.4B P2 | 194 | 194 | Depth 0: 192 | Missing copies depths 1–3 |
| Base1.4B P4 | 194 | 194 | Depth 0: 192 | Missing copies depths 1–3 |
| Base2.6B P2 | 386 | 386 | Depth 0: 384 | Missing copies depths 1–3 |
| Base2.6B P4 | 1538 | 1538 | Depths 0–3: 384 each | Alias graph has copies-depth arguments |

The depth prefix is `L_self_modules_campaign_execution_modules_<depth>_`.
The two non-block arguments name `embed_tokens` and `norm`.
Every matched rank pair has a byte-identical readable host graph.
The ledger records each rank's SHA256.

Allocated copies do not prove that compiled execution consumes those copies.
These graph arguments also do not directly measure runtime addresses.
The intended arms require distinct graph provenance before numerical or timing attribution.

## Operation and shape comparison

All 12 matched rank pairs have equal operation counts and shape annotations.
Their normalized leaf functions and collective/attention callee sequences also match.
The following counts apply per rank in either arm, at P2 and P4.

| Host-graph quantity | Base1.4B | Base2.6B |
|---|---:|---:|
| Leaf graph classes | 193 | 385 |
| `vllm.all_reduce` | 193 | 385 |
| Linear calls | 384 | 768 |
| `unified_attention_with_output` | 96 | 192 |
| `rms_norm.default` | 196 | 388 |
| `fused_add_rms_norm.maybe_inplace` | 192 | 384 |

The collective tensor annotation is BF16 `[S,2048]` with group `tp:0`.
The inspected collective order is a source sequence, not a measured device timeline.

| Weight | P2 shape | P4 shape |
|---|---|---|
| QKV | `[3072,2048]` | `[1536,2048]` |
| Attention output | `[2048,1024]` | `[2048,512]` |
| Gate/up | `[5632,2048]` | `[2816,2048]` |
| Down | `[2048,2816]` | `[2048,1408]` |

## Embedded compiled-source comparison

The [CPU inspector](../scripts/audit_storage_lowering_cpu_v1.py) reads serialized byte frames and pickle opcodes.
It does not unpickle, import, or execute saved objects.
Installed Torch sources define the inspected serialization format.

All 56 matched compiled-piece pairs have byte-identical embedded generated source.
This includes Python wrappers, Triton source, and Triton decorators.
Their normalized runtime-call ASTs and Triton compute functions also match.
There is no observed source-level arithmetic or collective-order difference between these matched pieces.

Of the 56 outer serialized artifacts, 32 match byte for byte.
The other 24 belong to Base2.6B P4, with six pieces per rank.
I compared sorted entry keys and content hashes within their cache bundles.
Their `aot_autograd` and `inductor` entries match in all 24 pairs.
Their `autotune` entries differ for pieces 0, 1, 48, 49, and 96 on all four ranks.
Piece 192 has equal entries in every bundle kind.
Its outer serialization still differs.
This audit does not establish the reason for that outer difference.

Saved autotune differences do not prove different selected runtime arithmetic.
Equal source does not prove equal collective internals, GEMM internals, runtime addresses, or numerical results.
Neither observation establishes a latency cause.

## v3 provenance check and export

The new [validator/exporter](../scripts/validate_storage_graph_provenance_cpu_v1.py) uses only the Python standard library.
Its [CPU tests](../tests/test_storage_graph_provenance_cpu.py) pass: 6/6 tests.
The tests cover all four expected model/arm counts.
They also check incorrect arms, incorrect depths, unknown names, missing ranks, gzip integrity, and overwrite refusal.

The validator checks each saved top-level `forward` with an AST parser.
It checks counts, depth prefixes, layer coverage, eight roles per block, BF16 shapes, and source use of weight arguments.
Unknown generated naming causes failure.
The exporter retains host graphs as deterministic gzip files with SHA256 hashes.
It retains failures in `manifest.json` and returns a nonzero status.
It never exports binary compiler caches.
It refuses an existing output directory.

Example for a future job:

```bash
python poc_ouro_parallelism_deep_discovery/scripts/validate_storage_graph_provenance_cpu_v1.py \
  --cache-dir /absolute/job/path/compile_cache \
  --size 1.4B --mode copies --P 2 \
  --output-dir /absolute/new/evidence/path
```

The validator detects the four mismatched v2 arms in the provenance table.
A validator PASS describes saved host-graph provenance only.
It does not prove runtime addresses, numerical parity, cache isolation, or performance.

## Required evidence before attribution

1. Set a unique `VLLM_CACHE_ROOT` before imports for every v3 job.
2. Retain separate ordinary compile and Triton paths.
3. Record the actual AOT and forced Inductor paths from each job.
4. Export every rank's host graph and check the intended arm's provenance.
5. Check numerical parity with a clean comparator before timing attribution.
6. Keep all v1 and v2 raw evidence frozen.

The AOT path places its forced Inductor directory below the unique vLLM root.
Disabling AOT is another supported source-level option.
That option changes the compiler mode and needs a matched comparator.
This audit did not select or execute that option.

The P4 numerical cause remains unresolved.
The next check is graph provenance, not a full runtime rewrite.

## Source ledger

Installed source root:

```text
/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/
```

| Source under that root | Exact locator | SHA256 |
|---|---|---|
| `vllm/compilation/decorators.py` | Lines 255–262, 265–320, 515–574 | `dd7167637a478aeaea3f2410a727ddefc702bda1d9bf632831c2a8b4811ebb1e` |
| `vllm/compilation/caching.py` | `aot_compile_hash_factors`, line 565 | `624426dfd4a86fdb9d0354a01a3733981e3008baf0cf141c9578e978f01e7fd7` |
| `vllm/config/compilation.py` | `DynamicShapesConfig.evaluate_guards`, line 368 | `267f4ca04fdcd2f0cf55c334782dd498c8051b9500ad088673d79518fc712db0` |
| `torch/compiler/_cache.py` | Lines 23–39, 150–170 | `d7232d060ddc74d1cf150b25f071274aa19ad32ce317af0c96dc982ef905359a` |
| `torch/utils/_appending_byte_serializer.py` | `BytesWriter`, lines 21–44. `BytesReader`, lines 47–88. | `6ecf544ac68cfb679d1289b036e440444deaa6a33ac8cbdaba86f7fc40a11f8b` |

Campaign `alias_control.py` SHA256 is `3c11cde3e297be60d9fb14b5d6e15c13713051cccb964bef1abdaffcb458c8dd`.
The cache findings concern the installed source above.
They do not assert behavior for another vLLM version.

## Actual time

This audit segment started at 2026-10-19 12:58:50 UTC.
The final recorded time was 2026-10-19 13:14:29 UTC.
The elapsed wall time was 939 seconds.
The report lint returned zero findings.
The CPU tests passed 6/6 tests.
No GPU execution occurred in this audit segment.
