
# Drop-one oracle headroom

Status: **MEASURED_HELDOUT_FUNCTIONAL_ORACLE**

Absolute-safe held-out path fraction: **72.66%**.
The frozen absolute-safe rule is target-token log-prob drop <=0.01 nat, JSD
<=1e-4, and no top-1 flip. Oracle selection sees true damage and is not deployable.

| Oracle skip | actual | safe precision | critical recall | mean damage | top1 flip among skipped |
|---:|---:|---:|---:|---:|---:|
| 10pct | 9.38% | 100.00% | 100.00% | 2.03096e-11 | 0.00% |
| 20pct | 19.53% | 100.00% | 100.00% | 1.00424e-09 | 0.00% |
| 30pct | 29.69% | 100.00% | 100.00% | 8.09448e-09 | 0.00% |
| 40pct | 39.84% | 100.00% | 100.00% | 2.92226e-08 | 0.00% |

This establishes whether low-damage paths exist; it does not establish that an
online estimator can identify them or that multi-path skipping preserves task quality.
