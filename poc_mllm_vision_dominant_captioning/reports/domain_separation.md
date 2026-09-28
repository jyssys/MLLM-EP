
# Image-driven domain separation

| signature | within distance | between distance | gap | same-domain NN | silhouette |
|---|---:|---:|---:|---:|---:|
| P_text | 0.0145 | 0.0230 | 0.0085 | 76.000% | 0.0482 |
| P_vision | 0.3322 | 0.4488 | 0.1166 | 72.500% | 0.0795 |
| P_full | 0.2960 | 0.4107 | 0.1147 | 74.500% | 0.0966 |
| P_separated | 0.1447 | 0.1941 | 0.0494 | 71.500% | 0.0515 |

Actual decode-truth expert/rank separation: `{"expert": {"between_domain_cosine_distance": 0.05882457171777433, "between_minus_within": 0.014045650321354644, "nearest_neighbor_same_domain_rate": 0.54, "within_domain_cosine_distance": 0.04477892139641969}, "rank": {"between_domain_cosine_distance": 0.004229759259364075, "between_minus_within": 0.0008643268883368792, "nearest_neighbor_same_domain_rate": 0.495, "within_domain_cosine_distance": 0.0033654323710271957}}`.
Layer-wise values are preserved in `artifacts/domain_separation.json`.
