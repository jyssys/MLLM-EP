# Reproduction

## Safety and scope

Use only physical GPUs4–7. Do not stop foreign GPU processes.
Keep existing results unchanged. Use new output directories for additional measurements.
The guard restores project burners after every run.

The pinned parent is `f1002a2f4baa6d1524e7e9b9bc8065bdc5c89a69`.
The branch is `flashvep/looped-recurrence-tp-4models-poc`.
The interpreter is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
The analysis interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
The latter supplies NumPy and Matplotlib. It never executes experiment CUDA kernels.

## Assets and frozen inputs

```bash
cd /home/esjung/MLLM-EP-looped-recurrence-tp
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_looped_recurrence_tp_4models.scripts.bootstrap
```

The bootstrap refuses to replace an existing audit. Use it only in a new, uninitialized collection directory.
For a clone with committed audit files, acquire weights without rewriting those files:

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from poc_looped_recurrence_tp_4models.contract import MODELS
for model in MODELS.values():
    snapshot_download(model['repo'], revision=model['revision'])
PY
```

The bootstrap resolves only the pinned checkpoint revisions. It stores weights in the external Hugging Face cache.
It does not commit model weights or update installed packages.
`results/source_and_model_audit.json` records runtime source hashes and model configuration hashes.
The committed frozen token file is sufficient for the hardware lane. Do not select new prompts from observed model outputs.

## Complete collection order

The original collection command was:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.run_campaign
```

This command skips existing output directories. It never replaces completed results.
For an independent repetition, use new directories with the guarded per-configuration command below.
Run the native arm before its matching tied arm. Replace `native` with `tied` in both command and output name.
The tied runner locates the paired reference through that name.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.gpu_session \
  --gpus 4,5,6,7 --timeout 2400 \
  --output results/independent_repetition/ouro_native_R4_TP4 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_recurrence_tp_4models.scripts.run_config \
  --model ouro --family native --r 4 --tp 4 \
  --output /home/esjung/MLLM-EP-looped-recurrence-tp/poc_looped_recurrence_tp_4models/results/independent_repetition/ouro_native_R4_TP4
```

Use `--gpus 4 --tp 1` for TP1. Use `--gpus 4,5 --tp 2` for TP2.
The full grid contains Ouro native/tied at R1/R2/R4 and TP1/TP2/TP4.
It also contains native Qwen3 and Qwen2.5 at TP1/TP2/TP4, each with `--r 1`.
Do not replace real Qwen checkpoints with synthetic weights.

## Separate graph and natural diagnostic

Use a new guarded output directory. This lane retains raw CUDA graphs for introspection.
Its times do not replace the primary clean timing policy.
The completed diagnostic queue uses tag `v4`. Earlier failed attempts remain preserved.
The queue includes the R2/TP4 short-prefill follow-up. It does not create a held-out dataset.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.run_postchecks --tag independent_diagnostic
```

The exact-primary-policy recheck waits for the original `v4` diagnostic queue to release the devices.
It repeats two calibration contrasts in reverse family order, without retaining raw graphs:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.run_same_policy_rechecks
```

This command refuses existing recheck directories. Preserve earlier evidence when you plan another repetition.

The natural-output audit waits for those four rechecks. It tests the observed TP4 math discrepancy without retaining raw graphs:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.check_natural_divergence
```

This audit reuses the original prompt. It captures selected logits under an identical forced prefix.
It does not change the primary worker code or numerical tolerance. It is not fresh held-out confirmation.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.gpu_session \
  --gpus 4,5,6,7 --timeout 1800 \
  --output results/diagnostics/ouro_native_R4_TP4 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_recurrence_tp_4models.scripts.run_control_diagnostic \
  --model ouro --family native --tp 4 \
  --output /home/esjung/MLLM-EP-looped-recurrence-tp/poc_looped_recurrence_tp_4models/results/diagnostics/ouro_native_R4_TP4
```

## Analysis and tests

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover -s poc_looped_recurrence_tp_4models/tests -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.analyze
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.audit_evidence
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.summarize_diagnostics
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.summarize_rechecks
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.check_natural_divergence --summarize
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.finalize_decision
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.make_figures
```

These scripts summarize the original `results/raw/*_v1` collection.
Independent repetitions retain their own directories and require explicit inclusion in a later analysis.
Do not merge them silently with the original cohort.

The Git commit excludes full parity arrays and large profiler events.
`results/heavy_evidence_manifest.json` records their external paths and SHA256 values.
On this server, restore missing copies before the detailed analysis:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_recurrence_tp_4models.scripts.archive_heavy --restore
```

The archive retains original evidence. It does not delete the raw files.
The selected natural logit arrays also use that external archive. Restore them before the natural-summary command.

## Evidence limits

Each primary configuration has eight requests at 1K and 16K. Each 32K confirmation has three requests.
Technical repeats do not increase the independent sample count.
The natural diagnostic is not GSM8K, HumanEval, or another benchmark score.
Kernel service is not an E2E percentage. No EP All-to-All exists in these dense TP runs.
