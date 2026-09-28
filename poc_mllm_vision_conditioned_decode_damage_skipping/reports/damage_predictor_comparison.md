
# Held-out damage predictor comparison

Status: **MEASURED_NO_FUTURE_LEAKAGE**

All families use the same fixed-alpha standardized ridge predictor. Fit,
normalization, frequency priors, and critical thresholds use calibration rows
only. Held-out requests are image-disjoint.

| Family | Spearman | AUROC | AUPRC | safe precision @20% | critical recall @20% | false-safe @20% |
|---|---:|---:|---:|---:|---:|---:|
| F0_GENERIC_LOCAL_ROUTER | 0.1851 | 0.6564 | 0.3819 | 68.00% | 85.19% | 16.00% |
| F1_MODES_STYLE_GENERIC | 0.2004 | 0.6670 | 0.3538 | 76.00% | 88.89% | 12.00% |
| F2_TEXT_PREFILL | 0.1925 | 0.6656 | 0.3546 | 80.00% | 92.59% | 8.00% |
| F3_FULL_PREFILL | 0.1847 | 0.6751 | 0.3532 | 76.00% | 88.89% | 12.00% |
| F4_VISION_PRIOR | 0.2027 | 0.6986 | 0.3748 | 80.00% | 92.59% | 8.00% |
| F5_VISION_SEPARATED | 0.2072 | 0.7008 | 0.3865 | 80.00% | 92.59% | 8.00% |

At the primary 20% nominal skip point:

- F5−F1 critical recall: +3.70 pp [-8.33, +10.00]
- F5−F1 safe precision: +4.00 pp [-8.00, +12.00]
- F5−F2 critical recall: +0.00 pp [-8.34, +8.11]
- F5−F2 safe precision: +0.00 pp [-8.00, +8.00]
- F5−F3 critical recall: +3.70 pp [-0.05, +12.50]
- F5−F3 safe precision: +4.00 pp [-4.00, +12.00]

Stage-B gate: **FAIL**. A point estimate is not
treated as vision-specific evidence when the request-cluster bootstrap CI includes zero.
