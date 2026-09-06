# Research tree

| Node | Family | Hypothesis | Evidence so far | Headroom | Prior-art risk | Children | Status |
|---|---|---|---|---:|---:|---|---|
| A | Partial completion | Early top-m expert contributions can produce a quality-safe provisional hidden state | Fresh live top-m curves fail for both modalities | 0% quality-gated | Medium (expert skipping) | A1, A2, A3 | CLOSED |
| B | Router-mass threshold | High cumulative router mass is a better stopping rule than fixed m | 95% mass fails text; 99% keeps all 8 | 0% quality-gated | Medium | B1, B2 | CLOSED |
| C | SpecMoE affinity | Affinity surrogate recovers missing expert output at low cost | Same-expert mean improves but remains outside gate | 0% quality-gated | High; affinity itself is prior art | C1 | CLOSED |
| D | Residual prediction | Missing weighted residual is predictable from hidden/router state | No evidence yet | UNKNOWN | Medium | D1 | CHEAP_PROBE |
| E | Cross-layer reuse | Prior-layer residuals enable earlier downstream compute | No evidence yet | UNKNOWN | Medium | E1 | UNTESTED |
| F | Spatial surrogate | Neighboring visual patches predict missing expert residuals | Same-expert visual neighbour is worse than mean/partial | 0% quality-gated | Medium | F1 | CLOSED |
| G | Confidence adaptive | Entropy/top-k gap predicts safe speculative depth | No confidence bin crosses quality/headroom gate | 0% quality-gated | Medium | G1 | CLOSED |
| H | Critical-path contribution | Late remote expert contributions have low quality cost but high overlap value | No evidence yet | UNKNOWN | Medium | H1 | CHEAP_PROBE |
| I | Verification/fallback | Verification can reject speculation cheaply enough | No implementation; nonlinear recompute risk | UNKNOWN | High | I1 | UNTESTED |
| J | Actual overlap | Provisional hidden readiness exposes meaningful downstream compute slack | Killed upstream: no quality-safe reduced-work boundary | 0% | High (speculative MoE overlap) | J1 | CLOSED |
| K | Speculative depth | S1/S2 scopes retain value after verification | Not measured | UNKNOWN | Medium | K1 | UNTESTED |
| L | MLLM specificity | Vision tokens tolerate partial completion beyond matched router controls | Fresh text fails at mass95; no transferable modality-safe point | 0% | Medium | L1 | CLOSED |
