# CUDA runtime selection in the event diagnostic

Status: CPU regression passes. Native GPU recheck remains pending at this checkpoint.
This change affects the opt-in observer, not model arithmetic or installed packages.

## Failure and feedback loop

The constant-tensor event regression passed on physical GPU5.
The full native worker then rejected its loaded libraries before event insertion.
The error was `Exactly one already-loaded libcudart is required`.
Both native attempts remain in `outputs/native_graph_events_base14_P1_v1/` and `v2/`.
No event timing from those attempts is valid.

The live v2 worker mapped two different files:

- The Torch environment's `libcudart.so.12`, from CUDA 12.9.
- `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12.8.90`.

The mapping records show different inodes and no deleted-file suffix.
Import-only probes did not reproduce the second loaded runtime.
The minimized regression loads the same installed second library after Torch.
It requests no GPU context or kernel.
The old selector fails on that exact two-library pattern.
The [red record](../artifacts/runtime_library_repro_red_v1.json) preserves the error and relevant mappings.

## Ranked hypotheses and tests

| Rank | Hypothesis | Discriminating observation |
|---|---|---|
| 1 | The process legitimately maps two runtime versions | Two distinct files reproduce the strict selector failure without model execution |
| 2 | Multiple path strings refer to one loaded file | The actual mappings have different device/inode identities |
| 3 | A deleted library or ambiguous Torch owner causes selection failure | The actual files remain live, and the Torch CUDA library has one mapped path |

The first hypothesis explains the selector failure.
This audit does not identify which model-setup operation loads the second runtime.
That loader provenance is unnecessary to choose the observer's Torch-linked API safely.
The model and its existing runtime loading remain unchanged.

## Bounded correction

The observer opens the already-loaded Torch CUDA library with `RTLD_NOLOAD`.
It resolves `cudaRuntimeGetVersion` through that library's dependency scope.
It uses `dladdr` to identify the returned symbol's owning file.
The selected path must also occur in the current runtime mappings.
The observer still rejects deleted, absent, or ambiguous ownership.
It still requires CUDA 12.9 before it binds the graph API.
It does not choose the highest version or the first pathname.

The Linux manuals define the relevant [symbol lookup](https://man7.org/linux/man-pages/man3/dlsym.3.html),
[address ownership](https://man7.org/linux/man-pages/man3/dladdr.3.html),
and [already-loaded-object lookup](https://man7.org/linux/man-pages/man3/dlopen.3.html).

The same CPU regression now selects the original Torch runtime.
`torch.cuda.is_initialized()` remains false.
The [green record](../artifacts/runtime_library_repro_green_v2.json) preserves that result.
Additional CPU tests reject an unmapped selected path and a deleted runtime.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.runtime_library_resolver_regression \
  --output /absolute/new/result.json
```

The script refuses an existing output file.
The native recheck uses the original guarded event-diagnostic command in a new `v3` directory.
Its result must pass before any model event interval becomes usable.
This correction supplies no model speedup or candidate-quality result.

## Native recheck update

The v3 native worker passed runtime selection and reached actual graph-edge inspection.
That inspection returned CUDA error 402, `cudaErrorLossyQuery`.
The query omitted nondefault edge metadata.
This is a separate observer limitation, not another library-selection failure.
No model event interval exists from v3.
The next diagnostic will export typed edge metadata without changing graph dependencies.
