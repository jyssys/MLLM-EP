# Refinement regime atlas

## Post-compaction sensitivity

| task | phase | fresh M | active experts | median M_e | p90 M_e | M_e<=4 | CV |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | early | 925.5 | 173 | 14.25 | 110.2 | 27.05% | 1.734 |
| GSM8K | middle | 419 | 147 | 9 | 57.2 | 35.15% | 1.607 |
| GSM8K | late | 35.5 | 71.5 | 2 | 8.7 | **68.49%** | 0.989 |
| HumanEval | early | 802 | 166 | 13 | 98.8 | 29.34% | 1.596 |
| HumanEval | middle | 172 | 122 | 5 | 29.4 | 46.90% | 1.275 |
| HumanEval | late | 15.5 | 45 | 2 | 5.0 | **79.43%** | 0.787 |

The core descriptive transition is real: fresh M and median M_e fall sharply,
while the fraction of active experts with at most four rows rises to 68--79%.
The proposed *middle-only* heterogeneity story is weaker. CV is highest early,
not middle, because hot experts make the upper tail extremely long. Early and
middle both contain tiny and `M_e>=16` experts; late is predominantly tiny.

Using `tiny>=25% AND large>=10%` as a permissive heterogeneity definition,
51.7%/52.2% of invocations and 59.5%/58.9% of modeled compacted critical expert
time qualify for GSM8K/HumanEval. With `tiny>=40% AND large>=5%`, the time mass
drops to 25.2%/31.5%. This is enough phenomenon mass to test, but not proof of
kernel headroom.

The measured dense runtime has the same direction but is less extreme: median
M_e is 15/15/5 across GSM8K early/middle/late and 14/9/3 for HumanEval; late
tiny fractions are about 45.7% and 64.1%.

Data: [REFINEMENT_REGIME_ATLAS.csv](../REFINEMENT_REGIME_ATLAS.csv) and
[HETEROGENEOUS_TIME_MASS.csv](../HETEROGENEOUS_TIME_MASS.csv). Figures 02--07
in [figures](../figures/) show M, median M_e, tiny fraction, CV, histograms and
expert-by-step heatmaps.

