# Full-core alias control review

## Review outcome

The initial implementation had two blocking issues.
The coordinator corrected both issues during this review.
The final inspected source has no unresolved source-level blocker within its PP1, dense BF16, R4 scope.
This statement does not establish GPU correctness, memory fit, numerical parity, or a timing result.

The control compares repeated weight identity with depth-specific parameter storage.
Both arms retain the original core and three additional parameter copies.
It is a residency counterfactual, not a candidate engine or a new execution axis.

## Initial blockers and later resolutions

### 1. Copied normalization modules called the original parameter owner

The initial `clone_module` used `copy.copy(module)` and replaced its parameter dictionary.
That operation retained instance-bound methods from the original module.
Installed `CustomOp.__init__` stores `_forward_method` at `custom_op.py:133`.
Its `forward` calls that stored method at lines 135–136.

Installed `RMSNorm` inherits `CustomOp` at `layernorm.py:37`.
Its implementation reads `self.weight` at lines 83 and 91.
The shallow copy therefore allocated a private norm weight but called the original norm owner.
Identical values could hide this ownership error in a parity test.

The final implementation rebinds each instance-bound `MethodType` whose owner is the source module.
See `alias_control.py:27–31`.
The copied normalization dispatch now uses the copied module as `self`.
The CPU regression changes the copied weight and checks that only the copied output changes.
That regression passed.

Resolution status: corrected in source and checked with CPU objects.
Actual pretrained GPU execution remains outside this review's evidence.

### 2. Native loading checks could reject the new parameter names

The initial loader wrapper returned only the original checkpoint-loaded name set.
It registered the copied modules before returning that set.
Installed `DefaultModelLoader` checks initialized names after `model.load_weights` returns.
See `default_loader.py:427–445` and `447–469`.

The check defaults to enabled for nonquantized models with a returned name set.
Copied normalization parameters introduced new names outside the original returned set.
The installed check could reject those names before model execution.
Its quant-method exception does not exempt ordinary normalization weights.

The final wrapper records names before copy registration.
It adds only the newly registered, explicitly copied names to the original returned set.
It also checks their namespace.
See `alias_control.py:43–44` and `85–92`.
The original name set remains intact, so this change does not disable original missing-weight checks.

Resolution status: corrected at the inspected source boundary.
This review did not execute the pretrained loader on a GPU.

## Initialization and operation order

`native_baseline.py:55–58` selects the worker extension before `LLM` construction at line 60.
The worker-extension import calls `install` before worker construction.
Installed `worker_base.py:261–264` resolves that extension before constructing the worker at line 319.
This order places the model patch before core construction and checkpoint loading.

This timing matters for compilation.
Installed `TorchCompileWithNoGuardsWrapper.__init__` reads `self.forward` at `wrapper.py:127`.
It creates the compile wrapper at lines 147–154.
The patch therefore reaches the callable that the native wrapper captures.
The actual trace still needs the later model state and runtime initialization.

The final replacement forward is an ordinary function in `alias_control.py:8–15`.
It preserves these native operations:

1. Select input embeddings once.
2. Reset residual state at the start of each pass.
3. Execute each block in physical layer order.
4. Pass the same recurrent depth to each block.
5. Apply the original loop norm after each complete pass.

Native `ouro.py:367–374` uses the same order.
The replacement omits the native layer-list slice.
The final PP1 assertion at `alias_control.py:47` checks that this omission changes no executed layer.
PP greater than one remains unsupported by this control.

Embeddings, the loop norm, the declared gate, and the LM head remain outside the copied block lists.
The native R4 forward does not call the gate.
The control does not check adaptive execution or natural-generation quality.

## Parameter and buffer ownership

The alias arm selects the original layer list for every pass.
The copies arm selects lists 0, 1, 2, and 3 for passes 0, 1, 2, and 3.
Both arms register and retain the same four lists at `alias_control.py:73–75`.
Additional execution-list references do not create additional parameter storage.

The final assertions compare BF16 words through `int16` views.
They also compare parameter names and source-versus-copy storage pointers.
See `alias_control.py:62–68`.
This corrects the earlier numeric `torch.equal` assertion.
Numeric equality alone does not distinguish positive and negative zero.

Each clone allocates its own live parameter tensors.
The code does not copy the original vLLM parameter subclass or all loader metadata.
The inspected unquantized GPU linear path consumes the passed layer's weight tensor directly.
See `linear.py:214–228` and `layers/utils.py:338–344`.
This supports the declared fresh-load dense inference scope.
It does not establish support for quantization, LoRA, weight reload, or training.

The generic clone retains shared buffer tensors in a separate buffer dictionary.
The final implementation additionally shares the original parameter-free RoPE module itself.
See `alias_control.py:51–54` and `60–61`.
This choice handles a real native mutator.
Installed RoPE can replace `cos_sin_cache` when dtype or device changes outside compilation.
See `rotary_embedding/base.py:105–131`.

`storage_control_evidence` compares buffer names and storage pointers after engine initialization.
See `worker_probe.py:16–24`.
These checks provide runtime evidence only when the corresponding RPC returns successfully.
This review does not treat the presence of an assertion as a passed GPU check.

## Attention, static context, and KV

The copied wrappers share each original depth-indexed `Attention` object.
The clone function returns those objects without constructing replacements.
The runtime assertions check the same object identities at `alias_control.py:60`.

Installed `Attention` registers itself in `static_forward_context` during construction.
It rejects a duplicate prefix at `attention.py:443–446`.
Sharing the existing objects avoids duplicate registration and new KV owners.
Native KV binding uses that same context at `gpu_model_runner.py:7526–7530`.

Each copied `OuroAttention` wrapper still selects `attn[current_ut]`.
The depth and physical layer therefore select the same native cache slot as the alias arm.
The baseline checks `4*L` cache slots and the expected TP head ownership at `native_baseline.py:61–65`.
Those checks remain pending until the actual run returns their evidence.

The generic ownership inspector traverses registered wrapper modules.
Four parameter copies therefore produce `4*L` `OuroAttention` wrapper entries.
That count is not the `L` checkpoint layer identities or a new count of independent KV slots.
The final metadata states this distinction at `worker_probe.py:32`.
The contract now uses separate identity and wrapper terms.

## Compilation and graph limits

The initial replacement used a synthetic filename.
The native backend tolerates unreadable source paths but cannot hash their file contents.
See `backends.py:1043–1054`.
The final ordinary function removes that specific source-identity weakness.
The metadata also records its source text and implementation hash.

Both arms use native compilation and graph settings.
The command still needs `--async-scheduling` to meet the contract's enabled-async condition.
The storage-control flag alone does not enable that option.
Other baseline options must also match between arms.

The compiler sees different module identities and parameter alias structures in the two arms.
Equal mathematical operation order does not prove equal lowered kernels or graph partitioning.
The comparison therefore measures the complete same-layout identity counterfactual.
It does not isolate an L2 hit rate without suitable counters and matched kernel evidence.

Fresh processes prevent mutation of one live model from changing the other arm's captured pointers.
All copies exist before timing.
The setup memory fields precede later graph and serving allocations.
The later allocated, reserved, peak, and unique-storage fields must remain separate.
None of those capacity fields measures HBM traffic.

## Assertions and remaining evidence

The inspected source covers these stop conditions:

- R4, supported layer count, and PP1 bounds.
- Equal BF16 parameter words and distinct source-versus-copy storage.
- Shared original Attention and parameter-free RoPE objects.
- Original missing-weight checks plus explicitly initialized copied names.
- Cache-slot count, TP heads, and actual GPU UUIDs.
- Frozen teacher-token output for every timed request.
- Buffer ownership after engine initialization.

The CPU clone suite contains three tests.
All three passed in this review with `CUDA_VISIBLE_DEVICES=''` and Python `-B`.
The tests cover bound-method ownership, explicit Attention sharing, and buffer-container behavior.
They are not pretrained model evidence.

The four-position capture runs outside the clean timing lane.
It collects logits, hidden state, and sampled written KV.
Capture alone is not a comparison or a parity pass.
The final alias-versus-copies report must compare matching captures and retain every numerical difference.
Forced output tokens do not establish natural-generation parity.

The coordinator reported that the first Base 1.4B P2 alias run started during this review.
This review did not inspect its result or execute a GPU command.
No GPU correctness or performance status follows from that progress message.

## Source and time ledger

Campaign root: `/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery`.
Installed vLLM root: `/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm`.
The source references above use those roots.

| Reviewed campaign source | SHA256 |
|---|---|
| Initial `alias_control.py` | `770f08566485e356b026a39d0bf24603b961eaa2dd79a9039fdbcab17ccb1ecc` |
| Final `alias_control.py` | `3c11cde3e297be60d9fb14b5d6e15c13713051cccb964bef1abdaffcb458c8dd` |
| Final `worker_probe.py` | `b485d5d7e233e7daf1fb288da155642c94199ea416a584951cdc405abaa33a29` |
| Final control contract | `96d5805e4f7e0939f5a83b1e5bbc287d48d5d4b29fe41af805ae7b75647dc034` |
| `scripts/native_baseline.py` | `6c0824f346efb0d87d1bf9f500d4d7e3a6f0b6ee2bb8c542c0335635f7ca54ee` |
| `tests/test_storage_clone.py` | `cad6cb57501181348eae3c800f8ecfc501d7a33b49489316f7eb89592e60fb16` |

| Installed source | SHA256 |
|---|---|
| `model_executor/models/ouro.py` | `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff` |
| `model_executor/custom_op.py` | `4b4ba01623005d6a93698eaec007a934dd512bf9ec7241090927de819621de2e` |
| `model_executor/layers/layernorm.py` | `3460eba01758849027bb59b8592626406e33ab25b23d2def729bb0ec42afce67` |
| `model_executor/layers/linear.py` | `10b1ebae493158ca1074aef37a9f466f4e8e536c5a71d91c5b66062095ce3a44` |
| `model_executor/layers/utils.py` | `8d732d33cbc9ebcf18eba9ec3ef3ecedd46d0fc1405beff97d2854fadb7532fd` |
| `model_executor/model_loader/default_loader.py` | `5d120c07b8eb4d08ce1d4e9759b832a07086dcc78d0df4cefe9beb5c29b7de4e` |
| `model_executor/model_loader/base_loader.py` | `a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08` |
| `model_executor/model_loader/utils.py` | `01b9e0bfc7bd9a0c94251cfb1c01ee733c4f57ce3d8e1df8ee9d2aa4df70e543` |
| `v1/worker/worker_base.py` | `7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe` |
| `v1/worker/gpu_model_runner.py` | `81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0` |
| `model_executor/layers/attention/attention.py` | `04fa9a1acb71217c7b9815e193198f6b69f12019c84a8a989b8d7e87544f8bda` |
| `model_executor/layers/rotary_embedding/base.py` | `c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5` |
| `compilation/wrapper.py` | `c1b1fca679ea16aa07a696831a810c0d531da2bb0ea32dd6f1a95cdaef36de07` |
| `compilation/decorators.py` | `dd7167637a478aeaea3f2410a727ddefc702bda1d9bf632831c2a8b4811ebb1e` |
| `compilation/backends.py` | `fbd068047613b97f28dc58f98c77c64f8252a9fae2a88efae09b78e9e794bac5` |

Own review segment start: `2026-10-19 12:06:32 UTC`.
Recorded end after checks: `2026-10-19 12:13:33 UTC`.
The recorded segment lasts 421 seconds.
This segment includes source review, CPU checks, and this report only.
The coordinator made every implementation change described above.
This reviewer changed only this report.
The report linter found zero violations.
