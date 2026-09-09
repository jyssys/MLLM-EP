# Results index

- `manifests/`: frozen request pools, exact-token features and randomized plans.
- `final_analysis/`: reduced, rank-deduplicated tables and decision documents.
- `profile/`, `screen/`, `analysis/`: 610 MB of raw/local execution data. These
  directories remain on the experiment host and are intentionally ignored by
  Git; the JSON manifests in every raw run record arguments, source hashes,
  timestamps and exit codes.

The main reproducible entry points are:

```bash
python poc_flashvep/mllm_moe_hierarchical_rebatching/analyze_runs.py ...
python poc_flashvep/mllm_moe_hierarchical_rebatching/summarize_poc.py \
  --results poc_flashvep/deepep_revalidation/results/mllm_moe_hierarchical_rebatching_poc_20260909_233848 \
  --out poc_flashvep/mllm_moe_hierarchical_rebatching
```

No raw observer wall time is used as clean speedup evidence.
