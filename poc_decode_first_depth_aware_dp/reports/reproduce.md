# Reproduction

## Scope and immutable inputs

Run commands from the repository root.
Use only physical GPUs 4, 5, 6, and 7.
Do not run a GPU command without the scoped guard.
Do not replace the frozen source specification or original worktree changes.

| Input | Identity |
|---|---|
| Base checkpoint | `ByteDance/Ouro-1.4B` |
| Checkpoint revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Official CDB revision | `abdfae0adbdd12be8c6123ccb34eee98a38afb39` |
| Audited vllm-rlt revision | `9d601ceaf9af63121c742df5f584467e23630a0b` |
| Native interpreter | `/home/esjung/.venvs/cp-composition-vllm026/bin/python` |
| CPU/reference interpreter | `/home/esjung/.venvs/llada2-mini-reference/bin/python` |

`contract.py` records local paths and exact revisions.
The previous committed benchmark manifest supplies question text and answers.
Its path is `poc_async_recurrent_pp/outputs/manifests/data.json`.
The new quality manifests contain the actual input prompts and their hashes.
No remote dataset update changes those inputs during this run.
`artifacts/scoring_environment.json` records the exact inherited parser hashes and symbolic-scoring versions.
It records MathVerify 0.8.0, SymPy 1.14.0, and latex2sympy2-extended 1.10.2.
It also records antlr4-python3-runtime 4.13.2 and NumPy 2.2.6.

## Dependencies

The native interpreter supplies Torch 2.11.0+cu129.
The reference interpreter supplies Torch 2.8.0+cu128 and Transformers 4.57.1.
The CDB checkout remains unmodified.
Task-specific directories hold the additional CDB dependencies.

```bash
git clone https://github.com/kschwethelm/continuous-depth-batching /home/esjung/external/decode-first-cdb-upstream
git -C /home/esjung/external/decode-first-cdb-upstream checkout abdfae0adbdd12be8c6123ccb34eee98a38afb39
git clone https://github.com/ThinkFlowLab/vllm-rlt /home/esjung/external/decode-first-vllm-rlt-upstream
git -C /home/esjung/external/decode-first-vllm-rlt-upstream checkout 9d601ceaf9af63121c742df5f584467e23630a0b
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m pip install --no-deps --target /home/esjung/external/decode-first-depth-dp-assets/python_cuda129 --index-url https://download.pytorch.org/whl/cu129 flash-attn-3==3.0.0
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m pip install --no-deps --target /home/esjung/external/decode-first-depth-dp-assets/cdb_dependencies transformers==5.9.0 tokenizers==0.22.2
```

Do not execute these installation commands over an existing dependency target.
First check its versions and installation logs.
The commands document isolated setup, not a shared environment upgrade.

## Preparation and fixed policy

The following preparation commands refuse existing frozen output files.
They document the original run order.
Do not rerun them over this evidence directory.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.prepare
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.freeze_quality
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_decode_first_depth_aware_dp.scripts.dependency_check
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.source_audit
```

The policy remains q=0.5, minimum depth two, and maximum depth four.
It uses the trained original gate and `last_exited` KV propagation.
The fixed reference disables early exit within the same CDB engine.
GSM8K uses three canonical CDB CoT demonstrations.
MATH-500 uses the frozen zero-shot boxed-answer prompt.
Generation is greedy with a 4,096-token cap and natural EOS enabled.

## Physical execution and official comparison

Use a new output name for every GPU attempt.
The following command shape reproduces the successful execution diagnostic.
Its original output is `outputs/runtime_smoke_v6`.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.gpu_session --gpus 4 --output outputs/reproduce_runtime_01 --timeout 1800 -- /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_decode_first_depth_aware_dp.scripts.runtime_smoke
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.gpu_session --gpus 4 --output outputs/reproduce_reference_01 --timeout 1800 -- /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.reference_smoke --source poc_decode_first_depth_aware_dp/outputs/reproduce_runtime_01
```

The official comparison uses the original released HF implementation.
It retains the recorded numerical limits and reports discrete output disagreement separately.
The failed compatibility attempts remain in their original output directories.

## Quality pilot and continuation

The controller starts four independent B1 quality workers.
GPUs 4 and 5 run fixed-R4 shards.
GPUs 6 and 7 run adaptive shards.
This is not a live DP4 scheduler benchmark.
The guards restore burners when each worker ends.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.run_quality_pilot
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.quality_analyze --runs poc_decode_first_depth_aware_dp/outputs/quality_fixed_part0_v1 poc_decode_first_depth_aware_dp/outputs/quality_fixed_part1_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part0_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part1_v1 --output poc_decode_first_depth_aware_dp/artifacts/quality_pilot_analysis_v1
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.extend_quality
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.run_quality_extension
```

The continuation selects the next 16 metadata-ordered questions in each domain.
It preserves the first 16-question results and all prompts.
It does not tune the threshold, change the parser, or select questions by answer quality.
Its new protocol records the continuation time before any continuation inference.
The consumed continuation questions are no longer untouched held-out data.
`protocol_v3_addendum.json` explains an inherited metadata field in the frozen continuation manifest.
The original prompt recipe preceded all inference.
The continuation followed the first pilot and preceded its own inference.

Analyze all eight quality directories together with the v3 protocol.
Use a new analysis directory.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.quality_analyze --protocol poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json --runs poc_decode_first_depth_aware_dp/outputs/quality_fixed_part0_v1 poc_decode_first_depth_aware_dp/outputs/quality_fixed_part1_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part0_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part1_v1 poc_decode_first_depth_aware_dp/outputs/quality_fixed_part0_extension_v1 poc_decode_first_depth_aware_dp/outputs/quality_fixed_part1_extension_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part0_extension_v1 poc_decode_first_depth_aware_dp/outputs/quality_dynamic_part1_extension_v1 --output poc_decode_first_depth_aware_dp/artifacts/quality_pilot_analysis_v2
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.export_pilot --analysis poc_decode_first_depth_aware_dp/artifacts/quality_pilot_analysis_v2 --output poc_decode_first_depth_aware_dp/artifacts/pilot_export_v2
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover -s poc_decode_first_depth_aware_dp/tests -v
```

## Predefined longer-cap sensitivity

The primary comparison retains the common 4,096-token cap.
The follow-up selects the union of truncated question IDs from both policies.
It reruns both policies for each selected question with a common 8,192-token cap.
It does not select examples by correctness or replace the primary analysis.

The follow-up retains the same weights, gate, KV policy, prompts, and greedy sampler.
Its configured maximum request length is 10,240 tokens.
The existing 640-page allocator has 16 tokens per page.
The runner checks that the actual prompt plus output cap fits that capacity.
This is below the checkpoint's configured 65,536-position limit.
It is not a claim that all contexts up to that checkpoint limit passed validation.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.quality_cap_followup
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.analyze_cap_followup
```

The sensitivity report checks the shared token and depth prefixes.
Its output is `artifacts/quality_cap_sensitivity.json`.
It remains a quality diagnostic, not a DP scheduling benchmark.

## Decision, reports, and audits

Run these commands only after the completed primary and follow-up analyses exist.
The scripts refuse existing decision, report, and audit outputs.
The final GPU audit requires all experiment workers to finish and the scoped burners to become ready.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.finalize
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.final_audit --output poc_decode_first_depth_aware_dp/artifacts/final_audit.json
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.report_results
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.check_reports
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp.scripts.audit_publication --output poc_decode_first_depth_aware_dp/artifacts/publication_audit.json
```

Only this PoC directory belongs in the task commit.
Do not stage unrelated changes from another worktree.

## GPU restoration

The server script currently defaults to GPUs 2, 3, 6, and 7.
Do not execute it with that default for this task.
The guard sets both masks to its permitted idle subset.
For a verified idle four-GPU allocation, the explicit command is:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 bash /home/esjung/vllm-ep/run_utilize.sh
```

Do not start duplicate burners over an existing worker or experiment.
Do not signal an unknown GPU process.
The final audit checks burner ancestry and the original worktree's unchanged tracked diffs.
