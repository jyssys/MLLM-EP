# Reproduction

## Frozen assets and runtime

The parent commit is `2481ef0f77dd029a9ac905ee749e9f0317becf02`.
Use branch `flashvep/looped-exact-depth-chunk-poc`.
The experiment directory is `/home/esjung/MLLM-EP-exact-depth-chunk/poc_looped_exact_depth_chunk_pipeline`.

The native interpreter is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
The CPU/reference interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
Dataset and evaluator dependencies reside outside both shared environments:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/pip install \
  --target /home/esjung/external/looped-exact-depth-chunk-assets/cpu_deps \
  pyarrow==20.0.0 math-verify==0.8.0
```

Use the committed dataset revisions, prompt IDs, and sampling contract.
The bootstrap refuses an existing environment freeze. Do not repeat it over committed evidence.
Download the exact model revision through Hugging Face if the external checkpoint cache is absent.
The pinned checkpoint is `ByteDance/Ouro-2.6B-Thinking@f1edd81e7ac41355db670500ceaf204e0f73af68`.

## Original collection

The original quality command used this guarded path:

```bash
cd /home/esjung/MLLM-EP-exact-depth-chunk
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.gpu_session \
  --gpus 4,5,6,7 --timeout 10800 --output results/quality_native_v3 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_exact_depth_chunk_pipeline.scripts.run_official_style_quality
```

The guard refuses an existing output directory. Choose a new output name for independent collection.
No old output needs deletion. The scorer defaults to the original `quality_native_v3` input.
For an independent quality run, score its new directory into a separate analysis directory:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning \
  --source poc_looped_exact_depth_chunk_pipeline/results/quality_native_reproduction \
  --output poc_looped_exact_depth_chunk_pipeline/results/quality_analysis_reproduction
```

Use `results/quality_native_reproduction` as the guard's new output name in that case.
The `v1` and `v2` attempts failed before quality generation because of tokenizer preflight checks.
Both failures remain preserved. The corrected path keeps all original frozen input IDs unchanged.

The physical queue waits for successful quality cleanup before it starts:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.run_physics_campaign
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.run_auxiliary_checks
```

The quality gate limits the physical plan. A weak baseline permits only the fresh native control and tiny diagnostic checks.
It blocks the broad stage-cost sweep and any candidate promotion.
The auxiliary checks wait for the complete physical queue. They never run concurrently with that queue.

## Analysis without new GPU execution

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover -s poc_looped_exact_depth_chunk_pipeline/tests -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.check_public_grader
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.pin_checkpoint_files
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.build_exact_dag
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.schedule_cost_model
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.summarize_native_counts
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.archive_large_evidence
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.make_figures
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_looped_exact_depth_chunk_pipeline.scripts.write_report
```

Raw quality outputs contain all token IDs, full text, stop reasons, seeds, and caps.
The scorer never runs generated code. Capped responses cannot count as completed correct answers.
This bounded commit includes the numerical arrays and heavy traces under `results/`.
Their combined size is 5,422,508 bytes. The HF component check uses arrays from the fresh native control.
The archive manifest also lists external backup copies and their SHA256 values.
The archive command can recreate those backup copies from the committed originals.

## GPU policy

Use only physical GPUs4–7. The guard checks ownership before it stops any project burner.
The launcher currently defaults to another GPU set. Always scope both variables:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 bash /home/esjung/vllm-ep/run_utilize.sh
```

The final restoration script verifies the four assigned devices. It does not stop foreign jobs.
Do not use a bare launcher command or change another worktree's staged files.
