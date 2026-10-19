# One-call lifted-operand binding probe

## Scope

This probe records metadata for one existing `optimized_call` invocation.
It does not read tensor values, copy tensors, or synchronize a device inside model forward.
It is an evidence-only observer.
It is not a timing source.
It does not establish cached decode-kernel parameter addresses.

The default worker and `native_baseline.py` remain unchanged.
The new worker class is `poc_ouro_parallelism_deep_discovery.operand_binding_worker.OperandBindingProbe`.
The process must select this class explicitly.
Importing the metadata module does not install a wrapper.

## Exact interface

The installed call boundary is `VllmSerializableFunction.__call__`, at `vllm/compilation/caching.py:216–217`.
The function forwards its arguments to its existing `optimized_call` attribute.
The probe changes that attribute on one process-local object.
It does not change a class method or an installed package.

The worker exposes two RPC methods:

1. `start_operand_binding_capture()` arms the observer after model loading and compilation.
2. `finish_operand_binding_capture()` restores the original callable and returns the record.

The arm operation requires a finished AOT backend and an inspectable FX graph.
It reads placeholder order from that graph.
It rejects an incorrect arm, incomplete block coverage, or unknown weight naming.
It checks the expected 194/386 alias or 770/1538 copies weight operands.

The arm operation records all parameter names with `remove_duplicate=False`.
Thus, the map retains names that share storage.
It checks execution-module ownership across all four depths.
Alias mode requires one storage per corresponding block weight.
Copies mode requires four distinct storages per corresponding block weight.

## Observer behavior

The wrapper selects the first positional call with the correct arity and real weight tensors.
Fake tensors and unsupported call signatures do not qualify.
The wrapper forwards each skipped call unchanged.
It retains at most eight skip reasons.

For the selected call, it records:

- Placeholder order and argument index.
- Tensor data pointer and storage-base pointer.
- Storage offset, shape, stride, and dtype.
- Loaded parameter names that share the storage.
- The expected parameter name for each lifted weight.
- Whether all recorded metadata matches that expected parameter.

The wrapper restores the original callable before it invokes that callable.
It passes the same argument objects in the same order.
It returns the original result object.
It propagates an exception from the original call.
An observer metadata error does not replace the original execution.

The wrapper does not write a file, print a message, or call a tensor-value operation.
It does not add a collective, stream operation, or CUDA event.
Its Python metadata work can still perturb the observed request.
Therefore, the request must remain separate from clean timing.

## Record interpretation

`CAPTURED_MATCH` means every observed lifted weight matches the loaded parameter metadata.
Acceptance also requires `original_call_outcome=RETURNED` on every rank.
`CAPTURED_MISMATCH` retains the operands that differ.
`CAPTURE_ERROR` retains the observer error type.
`NOT_CAPTURED` means no suitable optimized call occurred before finish.

A FULL CUDA graph replay can bypass `optimized_call`.
Such a replay does not produce a new binding observation.
A prefill observation must not be described as proof of cached decode-kernel addresses.
The recorded forward-context mode identifies the observed path.
The probe does not use an eager fallback or compiler-mode change after a missing observation.

Matching pointers at this boundary do not prove every downstream kernel reads that address without an internal transformation.
The saved generated source and numerical evidence remain separate requirements.

## Standalone guarded runner

The runner is [storage_binding_probe.py](../scripts/storage_binding_probe.py).
Its arguments are `--size`, `--tp`, and `--arm`.
It requires the matching completed v4 arm and a passing graph-provenance manifest.
It copies that arm's configuration into a new evidence output directory.
It changes only the worker extension and ordinary compile-cache path in that configuration.

Before importing vLLM, it sets new job-local vLLM, Inductor, and Triton roots.
It sets `OURO_CAMPAIGN_STORAGE_CONTROL` to the requested arm.
It retains the v4 allocator setting, `expandable_segments:False`.
The AOT path can relocate Inductor below the new vLLM root.
The runner checks the pinned checkpoint path and guarded GPU UUIDs.
It also checks the expected depth-indexed KV slot count.

After loading, the runner retains ownership and storage-control metadata.
It arms the observer and sends one frozen 1024-token prompt.
The existing teacher hook requests four frozen output tokens.
The runner then finishes the observer and retains all rank records.
It does not record a timing result.

The runner retains configuration, cache roots, source hashes, the frozen request, generated token IDs, and status.
On failure, it retains a traceback and unsuccessful status.
If possible, it also retrieves the observer record after failure.
It does not overwrite an existing evidence run.

Example command, for the parent agent to schedule:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,6 --output outputs/storage_binding_alias_base14_P2_v1 --timeout 3600 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.storage_binding_probe \
  --size 1.4B --tp 2 --arm alias
```

This implementation task did not execute that command.

## CPU checks and source ledger

The metadata core and worker adapter pass 15 CPU tests.
The runner plan passes four CPU tests.
The tests use metadata-only stand-ins and worker-module stubs.
They do not import Torch or execute a GPU operation.

The tests cover metadata mismatch, alias names, restoration, one-call selection, skipped calls, and exception propagation.
They check all four model/arm operand counts.
They check worker ownership rejection and an unfinished backend.
They also check configuration preservation and the frozen request length.

| Source | Main locator | SHA256 |
|---|---|---|
| [operand_binding.py](../operand_binding.py) | `LiftedOperandBindingCapture`, line 72 | `ab6640c68e9375556de013a8ed0522d5150d7557bb0ab8a0e424a3191f7b3254` |
| [operand_binding_worker.py](../operand_binding_worker.py) | `start_operand_binding_capture`, line 10 | `4bd6dbd7182b5f3d0a931f38b4701f9c793c8261640b697f6a4f6a1260ec5ad9` |
| [storage_binding_probe.py](../scripts/storage_binding_probe.py) | `main`, line 41 | `eb229557aa73706dfc7e7d1c394f4ea839e0d86b7bc0db0c9d6a236aed891d78` |
| [Metadata and worker tests](../tests/test_operand_binding_cpu.py) | 15 CPU tests | `bcb76bbad38c06a358e66f568d145308c3e913ad2b1b73ebe339bb419664d640` |
| [Runner tests](../tests/test_storage_binding_runner_cpu.py) | Four CPU tests | `d3b09a9fdc039c32f86e950853cb8e98b60b569b1f9e00df4b9ae3d3b6557a34` |

The combined audit and probe-work segment started at 2026-10-19 13:16:12 UTC.
The final recorded time was 2026-10-19 13:32:18 UTC.
The combined segment used 966 seconds of elapsed wall time.
Both new reports passed the prose lint with zero findings.
The checkpoint reports and this contract use one time ledger.
