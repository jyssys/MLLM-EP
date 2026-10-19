# Final reproduction index

## Scope and prerequisites

Run these commands from `/home/esjung/MLLM-EP-ouro-deep-discovery`.
Use the preserved server environments and checkpoint caches read-only.
The manifests pin the local source and model revisions.
They are not a portable container image or a complete binary build receipt.

The authorized device pool is physical GPUs 4/5/6/7 only.
Inspect ownership before a GPU command.
The campaign guard refuses unknown GPU contexts.
It stops only verified owned utilization workers and restores them after release.
Never stop the protected factorial experiment or another user's process.

Use a new output directory for every reproduction.
Do not overwrite a stored raw result, failed attempt, or analysis checkpoint.
Do not change a frozen protocol to improve an outcome.

## 1. Verify preservation and CPU regressions

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.checkpoint verify

CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.run_cpu_tests_isolated \
  --output artifacts/cpu_regression_reproduction_01.json
```

The isolated test runner starts one interpreter per module.
Some tests require a Torch-free import state.
A single combined unittest process does not preserve that condition.

## 2. Rebuild closed conventional baseline analysis

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_hybrid_cohorts \
  --output poc_ouro_parallelism_deep_discovery/analysis/native_hybrid_reproduction_01
```

The analyzer reads the explicitly named completed native, DP4, and hybrid directories.
It checks output work, ownership, timings, and guard completion.
It does not convert DP-parent wall into a native-API speedup.
For fresh GPU timing, reproduce the exact commands in those runs' `run_contract.json` files.
Use the frozen workload and fresh guard output names.

The long-context protocol is separate.
Its completed receipts are `outputs/native_longctx_base{14,26}_P{1,2,4}_v3/run_contract.json`.
Its aggregate KV reservation differs from the cohort protocol.
Do not merge the two protocols' absolute latencies.

The following CPU command rebuilds that archived context analysis.
This analyzer resolves `--output` relative to the campaign root.
The preceding hybrid analyzer instead uses its output path relative to the current working directory.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_long_context \
  --attempt 3 --output analysis/long_context_reproduction_01
```

Both hardware analyzers select named original directories.
A fresh guard output name does not redirect those selectors.
Analysis of new GPU outputs needs a separately reviewed selector or source-root change.
Rebuilding archived analysis does not become analysis of a new run.

The launch receipts store an inner command and a separate GPU mask.
The inner command alone omits the guard environment.
Use the recorded mask and the scoped wrapper together.
For example, this command form reproduces the Base 1.4B standalone TP2 cohort on physical GPUs4/6.
Inspect the frozen receipt before execution.
Use a fresh output name and preserve the original protocol.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,6 --output outputs/native_cohort_base14_P2_REPRO \
  --timeout 1800 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_cohort_control \
  --size 1.4B --tp 2 \
  --protocol poc_ouro_parallelism_deep_discovery/artifacts/native_cohort_protocol_v2.json
```

This example is a reproduction form, not a newly executed experiment.
The [denominator audit](reproducibility_denominator_audit_v1.md) pins all16 original hardware receipts and the endpoint limits.

## 3. Rebuild the staged quality extension

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.validate_baseline_extension_integrity \
  --output poc_ouro_parallelism_deep_discovery/artifacts/staged_integrity_reproduction_01.json

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_baseline_extension \
  --integrity poc_ouro_parallelism_deep_discovery/artifacts/staged_integrity_reproduction_01.json \
  --output poc_ouro_parallelism_deep_discovery/analysis/staged_quality_reproduction_01
```

The tokenizer validator uses the CDB Transformers 5.9.0 path.
The answer scorer uses the pinned reference environment and parser.
The post-launch integrity supplement changes no grading rule.
Only complete 64-request arms enter benchmark summaries.

The original four GPU commands remain in the per-run contracts.
Their source and launch freeze is `artifacts/baseline_extension_launch_contract_v1.json`.
The unchanged rows contain 32 requests per domain.
They are disjoint from the original calibration and held-out sets.

## 4. Native TP4 quality robustness

The fresh-cache v2 protocol is an additional baseline lane.
It is not a candidate or an independent held-out test.
Run the two model arms sequentially, never concurrently on the same four devices.
These are the executed command forms.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/native_quality_extension_base14_TP4_REPRO \
  --timeout 3600 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_quality_extension --size 1.4B

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/native_quality_extension_base26_TP4_REPRO \
  --timeout 3600 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_quality_extension --size 2.6B
```

The runner checks all 14 frozen sources before native imports.
It requires fresh job-local vLLM, Inductor, and Triton cache roots.
It rejects inherited model-control variables.
It uses the read-only ownership extension without teacher forcing.
Raw tokens, canonical decoded text, native API text, and completion metadata remain separate.

The validator intentionally names the original v2 output directories.
A new reproduction needs a separately reviewed directory-selection change and a new source manifest.
Do not edit that frozen selector during an active arm.
The original analysis command form is:

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.validate_native_quality_extension \
  --output poc_ouro_parallelism_deep_discovery/artifacts/native_integrity_reproduction_01.json

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_quality_extension \
  --integrity poc_ouro_parallelism_deep_discovery/artifacts/native_integrity_reproduction_01.json \
  --output poc_ouro_parallelism_deep_discovery/analysis/native_quality_reproduction_01
```

Native decoded-string stopping differs from staged token-suffix stopping.
No output-parity or speedup claim can ignore that distinction.

## 5. Rebuild structural screens and result tables

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.depth_affinity_extension_screen \
  --source poc_ouro_parallelism_deep_discovery/analysis/baseline_extension_complete_v1/summary.json \
  --output poc_ouro_parallelism_deep_discovery/analysis/depth_affinity_reproduction_01

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.build_result_tables_cd \
  --extension poc_ouro_parallelism_deep_discovery/analysis/baseline_extension_complete_v1/summary.json \
  --native-extension poc_ouro_parallelism_deep_discovery/analysis/native_quality_extension_complete_v1/summary.json \
  --output poc_ouro_parallelism_deep_discovery/analysis/result_tables_reproduction_01
```

The ownership screen is a declared unit-cost model, not a GPU speedup.
The tables retain separate quality, hardware, and headroom panels.
The optional native extension adds Table C2b without a candidate-performance cell.

The descriptive depth-quarter analysis is also CPU-only.
It uses completed trace length to define relative quarters.
It does not create a causal predictor or a scheduling policy.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.baseline_depth_profile \
  --output poc_ouro_parallelism_deep_discovery/analysis/depth_profile_reproduction_01
```

The following CPU commands rebuild the auxiliary composition and raw-output diagnostics.
Neither command executes a model or assigns a candidate speedup.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.function_composition_screen \
  --output poc_ouro_parallelism_deep_discovery/analysis/composition_reproduction_01.json

CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.backend_divergence_diagnostic \
  --output poc_ouro_parallelism_deep_discovery/analysis/backend_divergence_reproduction_01

CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.plot_context_controls \
  --output figures/context_plot_reproduction_01

CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.adaptive_alias_capacity_screen \
  --output poc_ouro_parallelism_deep_discovery/analysis/alias_payload_reproduction_01
```

The SVG renderer resolves its output relative to the campaign root.
It uses the archived context controls, not newly reproduced GPU outputs.
The current divergence script includes explicit EOS/terminal flags from diagnostic v2.
Its archived v1 output remains a historical checkpoint.

## Mechanical publication checks

Run this command only after the coordinator ledger records at least ten substantive hours.
The source-hash receipt must match the current CPU modules.
Use a fresh destination.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.validate_final_dossier \
  --output poc_ouro_parallelism_deep_discovery/artifacts/dossier_reproduction_01.json \
  --native-integrity poc_ouro_parallelism_deep_discovery/artifacts/native_quality_integrity_complete_v1.json \
  --native-quality poc_ouro_parallelism_deep_discovery/analysis/native_quality_extension_complete_v1/summary.json \
  --cpu-regression poc_ouro_parallelism_deep_discovery/artifacts/cpu_regression_isolated_v18.json \
  --staged-regrade poc_ouro_parallelism_deep_discovery/analysis/baseline_extension_regrade_v1/summary.json
```

This command checks metadata, hashes, result denominators, and local report links.
It does not certify scientific novelty, quality noninferiority, GPU release, or Git transport.
The publication scan preserves CSV CRLF and raw console whitespace.
The whole staged whitespace check therefore differs from the selected code/current-report check.
Do not rewrite frozen evidence to remove cosmetic warnings.

## 6. Failed attempts and source ledger

Both collective-pair runs remain NOT_ADMITTED.
Their original commands, source snapshot, cleanup stacks, and guard results remain in their raw directories.
The stopping rule allows no further retry in this campaign.
The first unisolated native extension also remains NOT_ADMITTED.
Its partial records never enter the complete v2 denominator.

Rebuild the source/evidence index only after the relevant files are stable.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.build_research_ledger \
  --label ledger_reproduction_01
```

The index counts observations and searches, not full-paper reads or successful experiments.
Incomplete and failed directories retain their explicit status.
The claim ledger preserves source scope and selected read extents.

The camera-ready recurrent-sampler PDF uses a separate temporary parser directory.
Its source receipt records the original PDF hash and parser version.
No installed native or CDB package changes for that read.
The PDF and extracted full text are not redistributed in this repository.

## 7. Burner restoration

The scoped guard restores released devices after each job.
Check the actual GPU contexts before any manual restoration.
Do not launch duplicate workers on a device that already has its owned burner.
The user script requires both overrides because its default mask can differ.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

This command is for released devices only.
Do not overlap it with a campaign model context or another user's work.

## 8. Final metadata checks

`scripts/validate_final_dossier.py` checks closed hashes, denominators, original-held-out metadata separation, links, CPU coverage, and recorded coordinator intervals.
It excludes concurrent worker durations.
It rejects normal completion before36000 recorded coordinator seconds.
It does not prove that every recorded interval was substantive.
The activity ledger supplies that attestation.

Its checks do not certify scientific novelty, task-quality noninferiority, candidate speedup, resource release, or Git transport.
Those claims require their separate evidence.
Use the final admitted native-integrity and native-quality paths when that extension closes.
