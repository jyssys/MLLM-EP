# H200 late-bound correctness and transport experiments

Start from the research worktree. Reuse the pinned asset checkout, download receipt and existing runtime. Set `PYTHON`, `ASSET_REPO`, `DOWNLOAD_RECEIPT`, and `RUNS` to their local paths. `RUNS` is an external directory for raw artifacts. Each epoch uses a new child directory and the existing verified three-GPU lease.

```bash
"$PYTHON" -m poc_flashvep.mllm_epd_h200_modality_aware_dynamic_split.online \
  --asset-repo "$ASSET_REPO" --receipt "$DOWNLOAD_RECEIPT" \
  --run-dir "$RUNS/bi1_lead_control" --mode late10 --correctness \
  --serial-encode-control --natural-boundaries --natural-hold \
  --inline-redirect --lead-boundary-control --numeric-regime bi1

"$PYTHON" -m poc_flashvep.mllm_epd_h200_dynamic_correctness_slice_aware.transfer_ablation \
  --asset-repo "$ASSET_REPO" --receipt "$DOWNLOAD_RECEIPT" \
  --run-dir "$RUNS/same_cut_transport" --repeats 2

"$PYTHON" -m poc_flashvep.mllm_epd_h200_modality_aware_dynamic_split.online \
  --asset-repo "$ASSET_REPO" --receipt "$DOWNLOAD_RECEIPT" \
  --run-dir "$RUNS/stock_b8_bi1" --mode native-common --decode-cap 8 \
  --qps 5 --numeric-regime bi1 --stock-cut-trace --screen

"$PYTHON" -m poc_flashvep.mllm_epd_h200_dynamic_correctness_slice_aware.cut_slack \
  --run "$RUNS/stock_b8_bi1" --cohort native_common_b8_5 --source stock \
  --output "$RUNS/stock_b8_bi1/analysis/aligned"
```

`--stock-cut-trace` observes scheduler times and issued rows in a diagnostic epoch. The report distinguishes these from CUPTI kernel intervals. Offline alignment uses future observations after the run. Online policies use the current D state and frozen causal estimates.

The active runner accepts selected suffix bands or exact suffix token counts with `--qualified-ratios` and `--qualified-token-cuts`. Use the scope of the strict fixtures for the chosen numerical regime and transport. A new runtime variant, shadow result, or single-token transfer check has its own evidence scope.

Use `online_summary` to revalidate arrivals, completion receipts, and digest-checked client metrics. Use `campaign_receipt` to record the engine settings and source hashes. Use Matplotlib with `figures` and `transfer_figures` for the published plots. The report links the compact inputs that regenerate each figure.
