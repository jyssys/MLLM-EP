# MLLM-specific disaggregation on H200

The campaign uses an existing verified asset checkout and runtime. The experiment order is common Native, Track B, then Track A. Read the [PoC contract](../reports/mllm_specific_disaggregation_discovery_poc_spec.md) and the [summary](../reports/mllm_specific_disaggregation_discovery_final_summary.md) for the measured scope and limitations.

Run project modules from the research worktree. The examples use environment variables for the verified Python, pinned asset checkout, runtime receipt, model snapshot, and external run directory. Raw model data and publisher images remain in the external workspace. The [dataset report](../reports/mllm_discovery200_dataset_profile.md) records source identities, cohort grouping and reconstruction.

```bash
package=poc_flashvep.mllm_specific_disaggregation_discovery
tier="$RUNS/dataset/grouped_v2/tier1_24.json"
manifest="$RUNS/dataset/grouped_v2/discovery200.json"

"$PYTHON" -m "$package.native_epd" --asset-repo "$ASSET_REPO" \
  --receipt "$DOWNLOAD_RECEIPT" --manifest "$manifest" --run-dir "$RUNS/native"

"$PYTHON" -m "$package.native_results" --run "$RUNS/native" --manifest "$manifest" \
  --asset-repo "$ASSET_REPO" --scorers "$RUNS/dataset/scorers" --out "$RUNS/native_analysis"

"$PYTHON" -m "$package.gpu_epoch" --run-dir "$RUNS/b1_b2_lease" --devices 0 \
  --wall-seconds 10800 -- "$PYTHON" -u -m "$package.track_b" \
  --model "$MODEL" --manifest "$tier" --run "$RUNS/track_b_tier1" \
  --teacher-reference "$RUNS/native/raw/discovery200_native_requests.jsonl" \
  --teacher-steps 32 --natural all

"$PYTHON" -m "$package.gpu_epoch" --run-dir "$RUNS/b3_lease" --devices 1,2 \
  -- "$PYTHON" -u -m "$package.dense_parallel" --snapshot "$MODEL" --manifest "$tier" \
  --out "$RUNS/dense_parallel.json" --id lmms-lab__DocVQA__56399 \
  --id TIGER-Lab__Mantis-Eval__wc_22 --id yubo2333__MMLongBench-Doc__658 \
  --id HuanjinYao__MMReason__896 --id AI4Math__MathVista__849 --real-activations

"$PYTHON" -m "$package.gpu_epoch" --run-dir "$RUNS/a_lease" --devices 6,5 \
  -- "$PYTHON" -u -m "$package.track_a" --model "$MODEL" --tier "$tier" \
  --native "$RUNS/native/raw/discovery200_native_requests.jsonl" --out "$RUNS/attention"

"$PYTHON" -m "$package.kv_results" --native "$RUNS/native_analysis/native_requests.json" \
  --manifest "$manifest" --model-config "$MODEL/config.json" --attention "$RUNS/attention" \
  --out "$RUNS/kv_analysis"

"$PLOT_PYTHON" -m "$package.kv_figures" --data "$RUNS/kv_analysis" \
  --attention "$RUNS/attention" --out "$RUNS/figures"
```

Pass a fresh external path for every run and analyzer output. The pinned files needed by the external benchmark scorers are in the source URI and digest table in `quality.py`. Local normalization for MathVista and MMReason has separate evidence from the publisher external-judge pipeline. The compact published case receipts and CSV/JSON reproduce the figures without publisher media or cached tensors.

The supplementary grouped Native epoch additionally passed `--full-qps 2.5` to measure the intermediate screening load. The actual launch arguments and failed CUDA Graph diagnostics are recorded in the [run ledger](results/run_ledger.json).

The GPU wrapper replaces only verified local burner workers on the nominated GPUs, holds the existing keepalive lease for the owned process group, and restores the burners at exit. Separate GPU timing epochs use separate run directories. Physical device IDs are recorded along with their UUIDs.
