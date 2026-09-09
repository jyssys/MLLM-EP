# Result manifest

Root: `poc_flashvep/deepep_revalidation/results/mllm_moe_transient_branch_compression_poc_20260910_021423/`

| Subpath | Evidence |
|---|---|
| `capture_full/` | Nine-image raw TP2/DP2/EP4 capture, manifests, worker backend proofs, driver correctness. |
| `analysis_full/` | Pair atlas, compression policies, route-run atlas, reconstruction checks. |
| `centroid_full/` | Exact checkpoint branch replay and centroid/weighted-centroid oracle. |
| `contribution_full/` | Matched sharing, spatial, and zero-output skipping controls. |
| `stratified_pairs/` | Per-pair route Jaccard, router weights, spatial distance, input/output distances. |
| `quality_*_l44/` | One-layer downstream propagation controls. |
| `quality_*_6layers/` | Six-layer downstream propagation controls. |
| `FINDING_SUMMARY.json` | Machine-readable aggregate used by the report. |
| `matched_modality_pairs.csv` | Common-support Vision/Text control. |

The raw result root is approximately 939 MiB and remains an experiment
artifact rather than being added wholesale to Git. All compact analysis code,
decision documents, and the main report are versioned.
