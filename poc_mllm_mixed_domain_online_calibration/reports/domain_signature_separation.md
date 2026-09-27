# Domain signature separation

All separation metrics use the frozen held-out half; layerwise discovery values are in `artifacts/domain_signature_separation.json`.

| Signature | within cosine distance | between cosine distance | gap | NN same-domain | silhouette |
|---|---:|---:|---:|---:|---:|
| P_text | 0.1376 | 0.1976 | 0.0600 | 83.75% | 0.1799 |
| P_full | 0.3149 | 0.4449 | 0.1300 | 71.25% | 0.1732 |
| P_vision | 0.3884 | 0.5240 | 0.1356 | 68.75% | 0.1511 |
| P_separated | 0.2601 | 0.3530 | 0.0929 | 73.75% | 0.1639 |

The result tests routing-signature structure, not dataset-name labels. Positive between-minus-within and NN/silhouette quantify actual separation. `P_separated` preserves modality identity; its larger dimensionality is controlled by using the same cosine kNN family for all learned-signature predictors.
