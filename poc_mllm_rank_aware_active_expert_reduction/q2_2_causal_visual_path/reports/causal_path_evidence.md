# Q2-2A causal expert-path evidence

The frozen cohort contains 8 calibration and
24 fresh held-out requests. Every original and
same-domain counterfactual used the identical token-by-token teacher-forced
prefix at the same layer and decode step. The unit is `(request,layer,step,expert)`.

- Held-out paths: 192
- Natural counterfactual top-k retention: 86.328%
- Same-expert logical evaluations: 512 BF16 current-weight evaluations; runtime-selected validation max relative-L2 0.001292.
- PCS vs signed `D_visual` Spearman: **-0.058697**
- PCS vs generic `D_orig` Spearman: -0.261630
- High-minus-low mean signed `D_visual`: **0.003837**
- Request-bootstrap 95% CI: **[-0.009163, 0.022229]**
- Top-k membership flip rate: 11.979%
- Exact intervention marker audit: original=True, counterfactual=True.
- Free-running counterfactual ablation: N/A; it is not position/token aligned and was not used as causal evidence.

Gate: **FAIL**. The primary gate requires positive Spearman,
positive frozen-bin separation, and a request-bootstrap CI wholly above zero.
Paths absent from counterfactual top-k use the predeclared non-active marginal
convention `g_cf=0, c_cf=0, D_cf=0`; no substitute expert is used.
The frozen original-image continuation can be off-distribution under the
counterfactual image; teacher forcing is required here to isolate visual
content at identical token, position, layer, and expert coordinates.
