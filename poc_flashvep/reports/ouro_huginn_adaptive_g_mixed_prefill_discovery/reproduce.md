# Reproduction

Run commands from `/home/esjung/MLLM-EP-github`.
Use `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python` for GPU workers.
Use `/home/esjung/anaconda3/bin/python` for the CPU dispatcher and analysis.
The package lock is in `validation/hardened_tests/packages.json`.

## Source reconstruction

The source pin is `4a350344027960a48384626aa646f1765fda6d4f`.
Create a separate RLT worktree only if `/home/esjung/external/ouro-huginn-g-mixed-rlt` does not exist.
Apply `provenance/hardened_common_group/runtime.patch` to that clean pin.
Do not apply the patch twice or overwrite another worktree.
The model paths and exact revisions are in `registration.json`.
Keep the original BF16 checkpoints outside Git.

## Resume and new attempts

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.campaign dp_retry mechanism
```

The dispatcher skips verified completed attempts.
It refuses to overwrite a failed or interrupted attempt.
It stops only verified owned burners and restores leased devices on exit.

For a new measured smoke attempt, register distinct IDs first:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.retry --plan smoke --suffix _repro_001
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.campaign smoke_repro_001
```

Choose a new suffix if that plan already exists.
Never remove frozen attempts to force a rerun.

## CPU exports

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.analyze
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.mechanism_analysis
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.dp_analysis
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.plots
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.report
```

Reports require all twelve clean confirmation starts.
Portable archives preserve raw outputs, failed logs, source snapshots, and the large CSV files.
Restore their relative paths in a separate evidence copy before reanalysis.
No model weights belong to those archives.
