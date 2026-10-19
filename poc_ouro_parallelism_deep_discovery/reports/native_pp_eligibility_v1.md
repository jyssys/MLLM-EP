# Installed native Ouro: PP eligibility

## Result

The inspected native Ouro implementation does not support stock vLLM pipeline parallelism above one rank.
This result applies to both pinned Base configs under native model resolution.
The installed package version is `0.26.0+cu129`.
The artifact records exact installed-source hashes, not an assumed upstream release state.

This is a stock-runtime support limit.
It is not proof that generic PP, tied-aware cyclic PP, or C14's legal physical trace is impossible.
It also does not establish a new architecture merely because an Ouro adapter is absent.
No GPU, model, engine, or native registry inspection process executes in this audit.
No installed source changes.

## Both pinned configs resolve to the same native class

| Base config | Revision | Architecture | Physical layers | Passes |
|---|---|---|---:|---:|
| Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | `OuroForCausalLM` | 24 | 4 |
| Ouro-2.6B | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | `OuroForCausalLM` | 48 | 4 |

Both configs use BF16 and have 16 attention and KV heads.
The registry maps `OuroForCausalLM` to `ouro.OuroForCausalLM` at `registry.py:185`.
Known native architectures retain that identity through `_normalize_arch` at lines 1192–1198.
The `auto` Transformers fallback at lines 1240–1247 requires all requested architectures to lack a native registration.
That condition is false here.
Explicit Transformers mode, external registry overrides, and modified classes are outside this native-path audit.

## Capability and configuration guard

The installed [class declaration](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:377) inherits `nn.Module` and `SupportsLoRA`.
It does not inherit `SupportsPP` or define a `supports_pp` marker.
Its constructor assigns an intermediate-tensor factory to the instance at lines 417–419.
That assignment does not supply the complete PP contract.

Inheritance alone is not the capability test.
The [interface predicate](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/interfaces.py:692) combines two checks:

1. Runtime protocol attributes, including the PP marker and intermediate factory.
2. A forward signature that accepts `intermediate_tensors`.

The native class accepts that keyword but lacks the PP attribute contract.
`_ModelInfo.from_model_cls` records the predicate result at `registry.py:812`.
`is_pp_supported_model` returns that recorded boolean at lines 1356–1362.
Lazy inspection can use a cache with a matching module hash or inspect the class in a subprocess at lines 894–989.
This audit neither invokes that inspection nor changes its cache.

The [parallel-config check](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/model.py:1240) tests PP>1 against this capability.
It raises `NotImplementedError` when support is false.
Its message says that the model lacks pipeline-parallel support and names the `SupportsPP` interface.
This is a source-derived eligibility conclusion, not a captured engine-start exception.

## Why `make_layers` is insufficient

The [helper](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/utils.py:798) uses PP rank and size to select physical layers.
It places `PPMissingLayer` modules outside the local interval.
Ouro calls that helper at `ouro.py:333–342`.
This proves partition-aware construction, not partition-correct execution.

The [actual forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:355) has the following source behavior:

- It accepts but never reads `intermediate_tensors`.
- It uses `inputs_embeds` or embeds `input_ids` locally.
- It has no first-rank or last-rank branch.
- It resets the residual and repeats the local layer interval for every recurrent pass.
- It applies the shared loop norm after each local interval.
- It returns the hidden tensor without a PP-stage intermediate return path.

The source also constructs embeddings, loop norm, gate, and head without first/last-stage ownership branches.
The declared return type and intermediate factory do not implement those missing transitions.

For a layer-stationary cyclic design, each pass must cross all physical owners before the loop norm and next pass.
The last owner must return the actual completed pass state to the next legal consumer.
The current local outer loop does not express that re-entry protocol.
It also does not transport residual versions, pass identity, or matching depth KV across stage boundaries.
Adding an interface flag or removing the guard cannot supply those semantics.
This audit does not bypass the guard or predict a particular resulting runtime exception.

## CPU source controls

The CPU check extracts the installed protocol definitions and predicate functions through Python ASTs.
It executes only those definitions with standard-library dependencies and tiny interface-only classes.
It imports neither Torch nor vLLM.
It loads no weights and constructs no engine.

Signature-only and factory-without-marker controls return false.
A declared structural PP control returns true.
A false-valued marker control also returns true because runtime protocol detection checks structure, not the `Literal[True]` value.
This bounded result explains why capability detection is not a semantic correctness proof.
It does not make native Ouro PP-eligible.

AST checks also establish the native class bases, unread intermediate input, and absent first/last-rank tests.
The [source artifact](../artifacts/native_pp_eligibility_v1.json) preserves the predicate results, source locators, package metadata, and hashes.
These are CPU/source contract checks, not a real PP execution test.

## Relation to C14

[C14's conceptual control](c14_cyclic_prefill_reopening_audit_v1.md) grants tied-aware physical owners, retained aliases, complete causal chunks, depth KV, and a correct cyclic schedule.
That constructed legal trace does not assert that the installed Ouro class supplies it.
This audit records the missing stock implementation boundary.

Three conclusions must remain separate:

1. Stock native Ouro lacks PP>1 support in the inspected installation.
2. A correctly implemented generic cyclic trace remains conceptually legal under its stated dependencies and state protocol.
3. Implementing missing support does not itself prove novelty, capacity gain, numerical parity, or latency benefit.

Any later PP investigation needs explicit entry/exit tensors, loop-norm placement, re-entry routing, depth-KV ownership, and head/gate behavior.
It also needs matched numerical and same-resource controls.
This audit builds none of that machinery and requests no GPU run.

## Actual work

Start: `2026-10-19 16:28:38 UTC`.
Validation cutoff: `2026-10-19 16:33:23 UTC`.
The observed segment spans 285 seconds, or 4 minutes 45 seconds.
Final timestamp insertion and handoff follow that cutoff.
The work includes local source/config inspection, package-metadata inspection, extracted CPU predicate controls, and report validation.
Concurrent worker intervals do not multiply campaign elapsed time.
Eight source/config hashes, 12 config fields, five CPU predicate controls, JSON parsing, and report lint passed.
