# Active frontier

The queue remains deliberately broad until quality and direct headroom gates
close or promote a branch.  Nodes are not promoted from correlation alone.

| ID | Pending question | Cheap decisive test | Expected signal | Cost | Priority | Status |
|---|---|---|---|---:|---:|---|
| A1 | Does top-4 preserve weighted output? | Fresh top-m cosine/L2 by layer/modality | cosine >=.99, L2 <=.05 | 10m | 1 | CLOSED: FAIL |
| A2 | Does router mass improve early readiness? | 70–99% cumulative-mass sweep | same quality at smaller k | 10m | 2 | CLOSED: FAIL |
| B1 | Are errors concentrated in high-entropy tokens? | Stratify by entropy/effective-K | confidence-conditioned Pareto | 10m | 3 | CLOSED: no safe prefix |
| C1 | Does an affinity surrogate beat plain partial? | Offline same-expert/neighbor diagnostic | lower residual error | 15m | 4 | CLOSED: below gate |
| D1 | Can missing residual be predicted? | Ridge/nearest-neighbor on captured outputs | residual prediction | 15m | 5 | PENDING |
| F1 | Does spatial locality help vision tokens? | 2-D neighbor surrogate control | lower visual error | 15m | 6 | CLOSED: no |
| G1 | Does token confidence select safe depth? | confidence bins and threshold sweep | stable adaptive gate | 10m | 7 | CLOSED: no |
| H1 | Are late remote experts low-value/high-delay? | local-vs-remote contribution analysis | critical-path Pareto | 15m | 8 | PENDING |
| I1 | Is verification cheaper than recomputation? | one-layer analytical recompute budget | feasible fallback | 15m | 9 | PENDING |
| J1 | Is downstream slack material? | exact/provisional readiness timing | >=10% E2E oracle | 20m | 10 | CLOSED upstream quality gate |
| K1 | Does S1/S2 retain overlap? | scope cost model | positive slack | 15m | 11 | PENDING |
| L1 | Is any effect modality-causal after controls? | matched layer/entropy/mass | visual-only residual | 15m | 12 | CLOSED: no |
