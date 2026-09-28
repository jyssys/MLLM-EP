# Structured K8→K6/K4 functional damage

- Exact marker audit: `{'all_pass': True, 'marker_count_distribution': {'3': 62, '4': 66}, 'probes': 128}`.
- Held-out K6 damage mean/median: 0.0111 / 0.0006.
- Held-out K4 damage mean/median: 0.0793 / 0.0026.
- Held-out structured damage mean/median: 0.0452 / 0.0021.

- dense_relation: n=4, K6 damage=0.0250, K4 damage=0.0672
- docci: n=5, K6 damage=0.0107, K4 damage=0.2491
- nocaps: n=5, K6 damage=0.0009, K4 damage=0.0018
- scene_layout: n=5, K6 damage=0.0007, K4 damage=0.0110
- textcaps: n=5, K6 damage=0.0209, K4 damage=0.0650

These are exact downstream-logit effects of semantic zero-weight interventions. They are not evidence that the physical sparse kernel skipped those rows.
