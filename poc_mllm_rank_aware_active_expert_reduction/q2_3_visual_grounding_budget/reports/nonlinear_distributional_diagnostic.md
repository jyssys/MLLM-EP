# Exploratory nonlinear/distributional diagnostic

This was requested after partial held-out correlation was observed. It is
therefore exploratory and cannot alter the frozen verdict. All damage
thresholds, feature tertiles, and three-bin risk maps were nevertheless fit on
the 8 calibration requests only and evaluated on the 24 held-out requests.

| Feature | Original AUROC | Original AP | Damage-distribution JSD | Absolute causal-delta AUROC | Absolute causal-delta AP |
|---|---:|---:|---:|---:|---:|
| visual/text separated | 0.6053 | 0.5144 | 0.0033 | 0.5714 | 0.2333 |
| text energy | 0.5526 | 0.2912 | 0.0929 | 0.4524 | 0.1750 |
| full energy | 0.6368 | 0.5859 | 0.1264 | 0.3571 | 0.1452 |
| router removed mass | 0.4579 | 0.3743 | 0.1061 | 0.5952 | 0.2125 |

The visual score did not dominate the controls: full-prefill achieved the
higher original-condition AUROC/AP and much larger low-vs-high damage JSD;
router removed mass achieved the higher causal-delta AUROC. Thus nonlinear
distributional inspection does not provide a credible rescue signal.
