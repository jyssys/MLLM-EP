# Selective refinement freeze safety

All timings here are excluded: F0/F1 execute the dense forward and only replace outputs to isolate causal quality risk.

| Configuration | n | correct | accuracy delta (pp) | NFE delta (%) | W_active reduction (%) | parsed identity |
|---|---:|---:|---:|---:|---:|---:|
| p0_full_none_ratio100_parity8 | 32 | 31/32 | +0.00 | +0.0 | 0.0 | 1.000 |
| p1_random_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -28.8 | 42.1 | 1.000 |
| p2_confhigh_f0_r50_age1_sanity8 | 8 | 7/8 | -12.50 | -22.4 | 56.0 | 0.875 |
| p2_confhigh_f0_r75_age1_sanity8 | 128 | 122/128 | +3.12 | -6.5 | 26.3 | 0.945 |
| p2_confhigh_f0_r75_age2_sanity8 | 8 | 7/8 | -12.50 | -33.0 | 47.4 | 0.875 |
| p2_confhigh_f1_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -35.7 | 49.0 | 1.000 |
| p3_conflow_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -9.8 | 27.0 | 1.000 |
| p4_ageconf_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -28.1 | 42.8 | 1.000 |
| p5_epload_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -24.1 | 39.3 | 1.000 |
| p6_confep_l01_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -28.1 | 42.8 | 1.000 |
| p6_confep_l05_f0_r75_age1_sanity8 | 32 | 31/32 | +0.00 | -6.1 | 27.4 | 1.000 |
| p6_confep_l10_f0_r75_age1_sanity8 | 8 | 8/8 | +0.00 | -27.5 | 43.0 | 1.000 |
The promoted P2 gate used all 128 requests: baseline 118/128 versus selective 122/128. It corrected 5 baseline failures and regressed 1 baseline success (paired exact p=0.2188). The apparent score improvement is not statistically significant and is not claimed as a quality gain; the G2 PASS is based on the specified no-more-than-one-additional-wrong safety criterion, complete termination, and reduced NFE/work.


## Freeze staleness

| configuration | age | samples | expert Jaccard | EP4-rank Jaccard | EP8-rank Jaccard | confidence error | commit disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| p2_confhigh_f0_r75_age1_sanity8 | 1 | 41695 | 0.693 | 0.896 | 0.824 | 0.1968 | 0.0586 |
| p2_confhigh_f0_r75_age2_sanity8 | 1 | 1555 | 0.687 | 0.894 | 0.823 | 0.2067 | 0.0540 |
| p2_confhigh_f0_r75_age2_sanity8 | 2 | 420 | 0.611 | 0.866 | 0.772 | 0.1868 | 0.0405 |
| p2_confhigh_f1_r75_age1_sanity8 | 1 | 1921 | 0.802 | 0.934 | 0.890 | 0.1412 | 0.0125 |

F0 caches the routed-MoE branch output only. F1 caches each decoder layer's post-layer state for frozen current-block positions; attention still executes densely, so this is a semantic emulation rather than a sparse KV implementation.
