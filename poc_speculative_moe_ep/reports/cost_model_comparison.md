# Cost Model Comparison

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

Same standardized ridge alpha1, request-grouped5-fold, no target-specific tuning. Explicit acceptance in M0; actual dummy/executed expert work in M1; physical traffic in M2. These are explanatory models after verification, not deployable future predictors.

- M0: EP MAE=3.24619ms; R2=-0.00681; MoE MAE=4.09228; target MAE=4.36559.
- M1: EP MAE=3.28327ms; R2=-0.01243; MoE MAE=4.15052; target MAE=4.41299.
- M2: EP MAE=3.31209ms; R2=-0.01369; MoE MAE=4.15905; target MAE=4.40819.

M2−M1 MAE reduction -0.878%; R2 increment -0.00126; request-paired MAE improvement -0.028, 95% request CI [-0.070, 0.011], n=16ms.

Secondary nonlinear width-only control adds the **same gamma fixed effects to all three families**, preventing traffic features from merely encoding four batch widths: M2 MAE reduction -0.581%, R2 increment -0.00175.

Matched-pair status INSUFFICIENT_MATCHED_PAIRS, n=0; no null result is inferred from no eligible pairs. Expert-proxy vs measured EP best-depth disagreement 0.312, CI [0.125, 0.5625]. This can reflect known fixed-cost amortization rather than independently useful within-width rank structure. Formal gate {'mae': False, 'r2': False, 'depth_disagreement': True, 'matched_pairs': False}; independent rank-cost support=False.

Secondary **same-layer** matching avoids averaging away traffic variation: 2743 matched layer pairs, only 16 independent requests; exact gamma/rows/progress/user+executed expert count; traffic>=15%; no latency-based selection. Median EP change 0.1835%; request-cluster mean 0.623, 95% request CI [-0.016, 1.343], n=16%. These are not thousands of independent samples and do not meet the20% physical-time gate.
