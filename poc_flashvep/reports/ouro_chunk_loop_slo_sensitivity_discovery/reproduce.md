# Reproduction

## Scope and safety

Use physical GPUs4–7 only. Each experiment child sees one GPU.
These workers do not run TP, PP, or a shared DP serving cluster.
The native-vLLM Ouro Loop PP project remains paused.
Do not stop another user's process.

The dispatcher checks the physical UUID and one-device CUDA visibility.
The resource wrapper stops only recognized user-owned burners.
Its exit handler restores those burners on GPUs4–7.
Do not run two resource wrappers at the same time.
The wrapper requires the preserved owner-checking lease helper in `/home/esjung/MLLM-EP-ouro-request-pp`.
That helper controls burners only. It does not resume PP research.
On another host, inspect local ownership before substituting a resource procedure.
The dispatcher itself does not terminate unrelated processes.

## Pinned environment

| Item | Value |
| --- | --- |
| Model | Base `ByteDance/Ouro-1.4B`, not Thinking |
| Model revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| RLT source | `4a350344027960a48384626aa646f1765fda6d4f` |
| Existing G2 adapter | `poc_flashvep/ouro_chunk_loop_prefill/patches/rlt_grouped_prefill.patch` |
| Python | `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`, version3.14.6 |
| Torch | `2.11.0+cu129` |
| Attention | FA4 `4.0.0b30` |
| Precision and depth | BF16. Static R4 for prefill and decode |
| New KV reserve | 18,432 global blocks. 54GiB per engine |
| Ordinary token budget | 1,024. The no-chunk control has a declared 16,384 exception |
| Client | In-process producer and token observer, not HTTP |

The isolated overlay uses native-vLLM packages through a `.pth` file.
Its inherited package metadata does not define a qualified native-vLLM environment.
Do not execute native-vLLM experiments in this overlay.
No model weights or binary compiler caches belong in the result commit.

## CPU audit and frozen registration

Run commands from `/home/esjung/MLLM-EP-github`.
The manifests retain prompt IDs, model settings, seeds, policy order, and arrival settings.
The registration files retain physical SLO values and their calibration hashes.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.audit
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.history
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m pytest -q poc_flashvep/ouro_chunk_loop_slo/test_metrics.py
```

Do not regenerate or replace frozen workloads with a newly selected sample.
The historical archive audit requires the preserved original evidence or its lossless archives.

## GPU queue

Run one manifest at a time.
The wrapper automatically uses four independent workers.
Each worker receives the next available job.
Each job retains its own engine warm-up, logs, identity, and CUDA context.

```bash
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/fixed_grid_manifest.json
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/repeated_shortlist_manifest.json
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/numerical_manifest.json
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/heldout_poisson_manifest.json
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/heldout_burst_manifest.json
bash poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/separate_profiling_manifest.json
```

The separate-profiling wrapper resolves registered midpoint and final-control extensions.
It then runs the GPU4 quiet policy control.
It restores the recognized burners before CPU analysis.
Do not edit a running shell wrapper.
The preserved transition failure illustrates that restriction.

Jobs with `completed.json` do not execute again.
Completed grid/traffic cells also prevent duplicate execution within an interrupted job.
A preserved `failure.json` requires review and a fresh corrective job ID.
The dispatcher returns a failure status for failed or unreviewed jobs.
Numerical diagnostics do not support partial-case resume within an incomplete probe job.

The explicit GPU4-alone calibration used a separate quiet-device control.
Its original command appears in `calibrate.sh`.
Other GPUs stayed idle during that interference control.

## Analysis and figures

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.analyse
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.controls
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.nsys_analysis
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.figures
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.report_tables
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.inventory
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.verify
```

Analysis uses 5,000 bootstrap draws and seed20261023.
Fixed-policy comparisons resample paired process blocks and cohorts.
Serving intervals resample engine blocks and request families.
Tokens and kernels are not independent inferential samples.

SLO thresholds are CPU views of the same measured requests.
A threshold sweep does not require another identical GPU generation.

## Lossless evidence

The archive manifest records source and compressed SHA256 values.
The packer does not remove local original evidence.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.pack
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.pack --verify
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.pack --unpack
```

Extraction refuses to overwrite different existing evidence.
Large local Nsight captures remain separate from their portable CSV exports.
Do not use heavy-profiler wall time as clean serving latency.

The clean RLT checkout must match the pinned revision and the recorded three-file adapter.
`run_manifest.json` includes the exact runtime diff and source hashes.
It also records reused experiment-module hashes and model/tokenizer file hashes.
The frozen specification is `inputs/authoritative_spec.md`.
Do not apply the runtime patch to an unrelated environment or restart the paused PP program.
