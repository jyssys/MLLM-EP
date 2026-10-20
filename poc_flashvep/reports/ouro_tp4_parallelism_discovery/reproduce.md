# Reproduction

Run commands from `/home/esjung/MLLM-EP-ouro-request-pp`.
Use the pinned environments. Do not install packages into another experiment's environment.
Every GPU launcher requires physical GPUs 4–7 to be free.
Stop only recognized project burners. Never signal an unrelated process.
Use a new label for each run. Do not overwrite preserved evidence.

## Inspect and check without GPU execution

```bash
git status --short
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_flashvep.ouro_tp4.test_evidence -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.analyze
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.make_reports
```

Analysis reads existing raw results. It writes only this new report directory.
Do not rerun `prepare.py` over the frozen preflight.

## GPU lifecycle

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease status
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease stop
```

The wrapper refuses to stop an unrecognized GPU job.
After experiments, restore the user's burner script:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease restore
```

## Exact recorded commands

`commands/*.json` contains each argv, cwd, selected environment variables, and source SHA256.
The paired completion file records the driver return code and process ID.
`logs/*.log` contains stdout, stderr, and failed assertions.
These records are authoritative for variants that changed during diagnosis.
The current TP source is the final FP32-accumulator diagnostic variant.
It must not be described as the exact source of the earlier BF16-reduction attempt.

To run a new component check, use a new label:

```python
from poc_flashvep.ouro_tp4.launch import tp
tp('new_operator_check', ['--mode', 'operators'])
```

To run a new short numerical check:

```python
from poc_flashvep.ouro_tp4.launch import tp
tp('new_short_check', ['--mode', 'correctness', '--smoke'])
```

To repeat a bounded PP control:

```python
from poc_flashvep.ouro_tp4.launch import execute, PYTHON
execute('new_layer_control', [PYTHON, '-u', '-m', 'poc_flashvep.ouro_tp4.mp_pp',
    '--arm', 'layer_pp4', '--indices', '0,1', '--label', 'new_layer_control'])
```

Use `loop_pp4` with the same settings for the other placement.
`--correctness` selects the C4/P128/O8 real-feedback probe instead of the hardware control.
The launcher supplies `CUDA_VISIBLE_DEVICES=4,5,6,7`.
It sets OMP/OpenBLAS thread counts to one and offline model access.
Do not use the unqualified TP serving benchmark. Its guard refuses the current failed C2 evidence.

## Profiler reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.profile_diagnostics --prefix new_
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.profile_analysis
```

Nsight exports SQLite locally. The committed `.nsys-rep` files can recreate those exports.
Use `--kill=none` to prevent profiler shutdown from terminating distributed workers after capture.
The `clean_tp_operator_diagnostic` attempt exited nonzero after all operator checks passed.
`clean2_tp_operator_diagnostic` repeats that check and exits normally.
Only the latter supplies the final TP operator timeline.

Native vLLM numerical reproduction uses `external_fidelity.py` through the isolated interpreter and scoped launcher.
Its initialization parameters remain in `external_engine_contract.json`.
It needs the local reference tensors from `native_reference.py`.
It does not run an optimized serving benchmark.
