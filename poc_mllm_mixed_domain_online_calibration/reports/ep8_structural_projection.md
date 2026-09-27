# EP8 structural projection

Evidence label: **STRUCTURAL_EP8_PROJECTION_ONLY**. Experts/rank: 16. EP8 latency, TPOT, E2E, and speedup: **NOT_COMPUTED_PROHIBITED**.

- Requests: 80
- Rank max/mean: 4.5123
- Remote destination fraction: 93.74%
- Logical remote assignments: 1192208
- Logical dispatch+combine payload proxy: 9314.12 MiB (logical BF16 hidden bytes; not measured transport).

| Predictor | expert cosine | EP8 rank cosine | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|
| P_domain | 0.9526 | 0.9953 | 0.0164 | 100.00% |
| P_full | 0.9563 | 0.9957 | 0.0158 | 100.00% |
| P_separated | 0.9560 | 0.9956 | 0.0158 | 100.00% |
| P_text | 0.9574 | 0.9958 | 0.0155 | 100.00% |
| P_vision | 0.9546 | 0.9955 | 0.0160 | 100.00% |

Per-domain rank prototypes, active experts/rank, B4/B8/B16 locality, and causal changed-rank replay are machine-readable in `artifacts/ep8_structural_projection.json`. These results support only a structural interpretation such as “potentially stronger at higher EP.”
