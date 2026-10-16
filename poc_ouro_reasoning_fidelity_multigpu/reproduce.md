# Reproduction

Run commands from this worktree's repository root.
Use only physical GPUs4–7.
Do not launch two overlapping scoped GPU jobs.
Each guard restores the project burner after success, failure, or timeout.
The launcher needs an explicit GPU override because its default mask is 2,3,6,7.

## Protocol and CPU checks

The input and protocol files already contain the frozen experiment.
Do not rerun a freeze script into an existing evidence directory.
Use a separate worktree and new output paths for a new campaign.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_ouro_reasoning_fidelity_multigpu.tests.test_contract -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.check_reference_cache_api --version 4.54.1
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.check_reference_cache_api --version 4.57.1
```

The isolated 4.54.1 packages use this target directory:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m pip install --target /home/esjung/external/ouro-reasoning-fidelity-assets/hf454_deps --no-deps transformers==4.54.1 tokenizers==0.21.4
```

This command does not change either shared environment.
Do not silently replace the pinned remote model code.
Source URLs and SHA256 hashes are in `sources/manifest.json`.
Downloaded third-party source pages stay local rather than being republished in Git.

## Guarded numerical probes

Use a fresh output suffix for every execution.
The following TP4 command is an example, not permission to overwrite `v1`.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session --gpus 4,5,6,7 --output results/native_fidelity_TP4_new --timeout 7200 -- /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.run_native_fidelity --tp 4
```

TP1 uses `--gpus 4 --tp 1`.
TP2 uses `--gpus 4,5 --tp 2`.
The HF reference uses one permitted GPU and no serving speed claim.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session --gpus 4 --output results/hf457_fidelity_new --timeout 7200 -- /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.run_hf_fidelity --reference-version 4.57.1
```

The optional MATH supplement uses `--gpus 5` and `--additional-only`.
Its runtime and GPU UUID remain explicit.
It provides reference states, not clean multi-request timing.

## Conditional quality evaluation

Do not bypass an unresolved consequential numerical failure.
The diagnostic run uses eight predeclared overlapping AIME questions.
The full run uses the frozen P1 protocol regardless of diagnostic scores.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session --gpus 4,5,6,7 --output results/quality_new --timeout 28800 -- /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.run_quality
```

The final report lists executed paths separately from these reproduction examples.
Missing runtime, quality, or systems evidence remains NOT_RUN.

## Executed campaign

The primary source lock has SHA256 `fcb626ccee0a403257117b06974649ae2462d3f5832ab14f8b94c59e0c99c25b`.
The actual full native output directory is `results/quality_v1`.
It contains 90 sampled AIME outputs at each of three caps.
It also contains 30 greedy AIME outputs at each cap and 100 greedy MATH outputs at cap8,192.
The MATH cap fraction is 9%, so the declared extension rule stops that lane.
No benchmark output changes after the diagnostic controls.

The original guarded collection command is:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session --gpus 4,5,6,7 --output results/quality_v1 --timeout 28800 -- /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.run_quality
```

This command records the executed path.
Do not execute it in a checkout that already contains that path.
Standalone probes accept a fresh `--output` path through the guard.
Analysis scripts also protect their existing derived artifacts.
Controllers with fixed output names require a fresh campaign directory with the frozen inputs, not an existing published result directory.

The remaining executed controls use these source modules:

| Control | Interpreter | Physical mask | Output |
|---|---|---|---|
| HF P0/P1 corroboration | HF 4.57.1 | 4 | `results/hf_prompt_corroboration_v1` |
| Failed history initialization | Native | 4,5,6,7 | `results/restart_same_history_default_v1` |
| Fixed-guard initialization retry | Native | 4,5,6,7 | `results/restart_same_history_default_v2` |
| Two core-MP0 probes | Native | 4,5,6,7 | `results/restart_same_history_MP0_v1` and `v2` |
| Two fixed-logit sampler probes | Native | 4 | `results/fixed_logits_sampler_v1` and `v2` |
| Small natural B1 audit | Native | 4 / 4,5 / 4,5,6,7 | `results/exploratory_B1_TP1_v1`, `TP2_v1`, and `TP4_v1` |

The history retry controller uses this command:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.queue_restart_history --default-output-name restart_same_history_default_v2
```

Its first attempt fails before generation because of a duplicate burner restore.
The retry changes guard cleanup only.
It retains the captured native engine arguments and the prescribed client history.
The MP0 flag is an isolated diagnostic, not the primary benchmark policy.
See `restore_race.md` and the source snapshots for both guard versions.

## Independent CPU score reproduction

Use a new empty extraction directory.
Do not extract over the preserved experiment directory.
The compressed archive retains raw token IDs, text, seeds, and finish reasons.

```bash
ouro_replay_dir=$(mktemp -d /tmp/ouro-score-replay.XXXXXX)
tar -xzf poc_ouro_reasoning_fidelity_multigpu/evidence/natural_generations_v1.tar.gz -C "$ouro_replay_dir"
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning --source "$ouro_replay_dir/results/quality_v1" --output "$ouro_replay_dir/score"
```

This command reuses the unchanged public grader from the parent experiment.
The parent commit contains its frozen answer metadata.
`results/reused_grader_provenance.json` checks that those references match this campaign.
The reference runtime needs Math-Verify 0.8.0 and the recorded CPU dependency directory.
The reproduced source-directory string changes, but the scores and bootstrap fields must match.

The two full CPU tokenizer audits use:

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.audit_output_decoding --version 4.54.1 --full-output
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_reasoning_fidelity_multigpu.scripts.audit_output_decoding --version 4.57.1 --full-output
```

These audit commands also refuse to overwrite their existing result files.
Each checks 260 frozen prompt arms and 462 output sequences.
They find no reference-decoding or final-answer-extraction difference.
The 462 sequences include 460 quality outputs and two continuity outputs.

## Evidence and final GPU policy

`evidence/manifest.json` records archive hashes and every included raw file.
Large state arrays remain in the experiment server's external evidence directory.
The unredacted GPU audit also remains local to protect other users' process identities.
Public environment records retain the assigned devices and the original audit hashes.
No original raw file is removed or rewritten.

After every model job ends, use the scoped restoration command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 bash /home/esjung/vllm-ep/run_utilize.sh
```

The final verification additionally uses `scripts.restore_burners` after existing restored burners finish their bootstrap.
It accepts only identified project burners and refuses an unidentified model context.
It verifies real burner work on all four assigned GPUs.
The launcher's default mask is not permission to use GPUs2–3.

The report's Git revision is available with:

```bash
git log -1 --format=%H -- poc_ouro_reasoning_fidelity_multigpu
```
