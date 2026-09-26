# Frozen untouched simulator validation

| Target | Median APE | P90 APE | Sign accuracy | Spearman | Verdict |
|---|---:|---:|---:|---:|---|
| EP2 | 12.962% | 22.728% | 0.789 | 0.381 | SIM_UNVALIDATED |
| EP4 | 14.510% | 41.084% | 0.864 | 0.762 | SIM_UNVALIDATED |

Passing thresholds were median APE <=10%, P90 <=20%, sign >=90%, Spearman
>=0.8. Neither target passes. Full TTFT/fixed/natural E2E prediction therefore
remains UNVALIDATED even though physical baseline measurements exist.

Prediction interval coverage/normalized full width on the untouched test:
EP2 0.875 /
1.114;
EP4 1.000 /
1.133.
