# Architecture-first H200 campaign

[Summary and evidence index](../reports/mllm_architecture_first_disaggregation_final_summary.md) · [Pre-run contract](../reports/mllm_architecture_first_disaggregation_discovery_poc_spec.md) · [Exact execution recipes](results/execution_ledger.json)

The campaign reuses the grouped-v2 Discovery200 manifest and its local image payloads. The compact cohort registry preserves request IDs, splits, exclusions and prompt hashes. Checkpoints are publisher safetensors pinned by repository, revision and payload hashes. Raw checkpoints, user-authored images, full KV and logs stay under `../h200_architecture_runs` on the H200 host.

## Reproduce cohort selection

Run from this repository root with the campaign Python environment.

```bash
campaign_namespace=poc_flashvep/mllm_architecture_first_disaggregation_discovery
campaign_module=poc_flashvep.mllm_architecture_first_disaggregation_discovery
campaign_python=../mllm_epd_runtime_env/bin/python
campaign_runs=../h200_architecture_runs
campaign_manifest=../h200_discovery_runs/dataset/grouped_v2/discovery200.json

"$campaign_python" -m "$campaign_module.cohorts" \
  --discovery200 "$campaign_manifest" --out "$campaign_runs/cohorts" \
  --registry "$campaign_namespace/results/cohorts.json"
```

Use the exact executed commands and device mappings in the run ledger for CogVLM2 and IDEFICS. The GPU lease requires the H200 host's existing authorized burner ownership and restores only the devices used by that run. Choose fresh result and lease paths when reproducing the trials. The recorded legacy compiler environment points to the GCC/G++ toolchain in `../mllm_epd_runtime_env/bin`. The online launch script sets those compiler paths explicitly.

The modified ERNIE processor is an offline, image-only overlay. Apply it to the downloaded pinned snapshot with `ernie_image_only_snapshot.py` before allowing its audited model-specific configuration or processor modules to import. Use the recorded overlay receipt when comparing native traces. The GPU trace requires native vLLM eager. Hook timings describe diagnostic host enclosures.

## Recreate figures

The plotting code never reads model weights, original images or credentials. A Python with matplotlib and NumPy can recreate the figures from the compact receipts.

```bash
python -m "$campaign_module.plot_campaign" \
  --results "$campaign_namespace/results" --out "$campaign_namespace/figures"
```

IDEFICS raw JSON, summarized physical-copy enclosures and correctness records are indexed from the [execution graph](../reports/mllm_modality_phase_execution_graph.md). Measured host boundaries, logical tensor payload, allocated storage and scheduler concurrency have separate meanings.
