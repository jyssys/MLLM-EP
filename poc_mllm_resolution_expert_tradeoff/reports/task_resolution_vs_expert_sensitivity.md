
# Task-stratified resolution and expert sensitivity

The MMStar `ocr_document_proxy` stratum is a text-sensitive MMStar subset, not
an OCR benchmark. Task labels are analysis strata only and are not available
to an online selector.

| config | chart_numeric | ocr_document_proxy | scene_spatial |
|---|---|---|---|
| R025_K4 | 75.00 | 65.62 | 56.25 |
| R025_K6 | 75.00 | 68.75 | 68.75 |
| R025_K8 | 78.12 | 68.75 | 65.62 |
| R05_K4 | 87.50 | 71.88 | 59.38 |
| R05_K6 | 90.62 | 68.75 | 65.62 |
| R05_K8 | 90.62 | 75.00 | 62.50 |
| R0_K4 | 87.50 | 68.75 | 65.62 |
| R0_K6 | 84.38 | 75.00 | 68.75 |
| R0_K8 | 84.38 | 71.88 | 68.75 |

## Paired interaction

```json
{
  "R025_K4": {
    "bootstrap_95ci_pp": [
      -10.416666666666668,
      2.083333333333333
    ],
    "interaction_pp": -4.166666666666666
  },
  "R025_K6": {
    "bootstrap_95ci_pp": [
      -6.25,
      4.166666666666666
    ],
    "interaction_pp": -1.0416666666666665
  },
  "R05_K4": {
    "bootstrap_95ci_pp": [
      -6.25,
      2.083333333333333
    ],
    "interaction_pp": -2.083333333333333
  },
  "R05_K6": {
    "bootstrap_95ci_pp": [
      -8.333333333333332,
      4.166666666666666
    ],
    "interaction_pp": -2.083333333333333
  }
}
```
