# Reproduction

## Scope

Use this study's branch and preserve existing evidence.
Do not rerun a label that already has a log or raw output directory.
Use a new label for an independent reproduction.
The `commands/` records preserve the exact original arguments, GPU mask, and source hashes.

The authoritative spec is `spec_frozen.md`.
The requested spec path was absent in this worktree. The source document came from the main repository.
The preflight records its original path and SHA256.

## Environments

Custom and HF interpreter:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python
```

Native vLLM and official CDB interpreter:

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python
```

Both environments use the frozen model revision from `arm_manifest.json`.
The CDB loader uses the pinned external source and isolated dependency paths.
It checks the source revision and the original gate tensors before generation.
The two interpreters have different PyTorch and Transformers versions.
Do not combine their timings into an engine-matched table without that qualification.

## CPU checks

```bash
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_flashvep.ouro_8arm.test_contract -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.scoring
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.paired_analysis
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.serving_analysis
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.plots
```

## Scoped GPU execution

The launcher checks ownership before each run.
It stops only identified project burners on the requested GPUs.
It refuses unrelated GPU jobs. It restores authorized burners after the experiment exits.
Never run two GPU campaigns simultaneously on the same devices.

The burner script's server default differs from this study's mask.
The launcher therefore sets `VLLM_UTILIZE_GPUS` explicitly.

Example independent Single probe:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -c 'from poc_flashvep.ouro_8arm.launch import execute; from poc_flashvep.ouro_8arm.contract import PYTHON; execute("reproduce_S4_probe",[PYTHON,"-u","-m","poc_flashvep.ouro_8arm.request_runtime","--arm","S4","--probe","--cap","8","--label","reproduce_S4_probe"],[4],timeout=600)'
```

Replace `S4` with `L4` or `O4` and use `[4,5,6,7]` for distributed PP.
Single always uses physical GPU4. Distributed logical ranks 0–3 map to physical GPUs4–7.

## Original staged campaigns

These commands resume successful completed stages. A failed existing label needs a new explicit retry label.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.campaign --stage static-probes
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.campaign --stage static-quality
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.campaign --stage dynamic-probes
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.campaign --stage dynamic-quality
```

Actual quality generation allows EOS and uses a 4096-token cap.
Synthetic or forced-exit probes do not count as task accuracy.
The custom PP prompt remains a whole request. No prompt chunk partition is used.

## User-approved 400-question evaluation

The latest user instruction replaces the complete 1,819-question target with 400 questions per setting.
The study retains all 64 GSM8K and 50 MATH pilot questions.
It adds 236 GSM8K and 50 MATH questions using metadata-only random sampling with seed 20261021.
The final target is 300 GSM8K plus 100 MATH questions for each valid arm.
This is a continuity pilot plus randomized extension. It is not a uniformly sampled full test set.

`evaluation_subset400.json` freezes IDs, cohort membership, selection rules, and input hashes.
Do not regenerate that existing manifest. The source module can reproduce it in a separate output workspace.
The extension command runs only 286 new questions per arm.
The `quality400_*` views join original complete pilot cohorts and new extension cohorts without duplicate samples.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.extension400
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.subset400_serving
```

Native TP1 and TP4 controls use the same 400-question target.
The separate official CDB production-oriented control retains its 114-question pilot.
No generation cap, gate, KV policy, or model precision changes with this scope reduction.
The original full-suite manifest remains available for future evaluation.

## Replay and profiling

The frozen shape selection is `replay_plan.json`.
Clean replay uses one warm-up and five measured repeats.
Heavy profiling uses a different label and a separate execution.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.replay_campaign
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_8arm.replay_campaign --trace
```

Only an actually completed command establishes a measured result.
A command listed here does not imply that its experiment completed.
Check the command's completion record and the final arm-status table.

## Payloads

Raw task JSON preserves prompt IDs, output IDs, text, stop reason, timing, and available depth traces.
Derived CSV and JSON preserve numerical seams, scoring, ownership, and timing summaries.
Git excludes model files, full tensor caches, and Nsight binary databases through the repository's existing rules.
Those local payloads remain available in `.tensor_cache/` and `profiles/`.
The measurement scripts regenerate them. The study does not delete those payloads.
