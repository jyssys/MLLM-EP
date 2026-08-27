# Modality-dependent expert-placement trade-off and rank saturation

## Scope and provenance

This is an offline, route-artifact-only PoC. No model, vLLM, DeepEP, CUDA, or
GPU was initialized; the requested GPU set (physical 1–4) was therefore not
exposed. The source of truth is the existing exact Qwen3-VL capture:

`poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`

It contains 24 real-image requests and 24 paired text controls, each with 48
decoder layers and top-8 expert IDs. The captured configuration was BF16,
TP2/DP2/EP4, DeepEP high-throughput, DBO-off. This PoC only remaps expert IDs
to hypothetical rank labels; captured routing is never changed.

The validated current placement is linear, `rank = expert_id // 32` (128
experts, four ranks, exactly 32 experts/rank). Vision positions are the
contiguous `image_token_id=151655` rows; all other rows are labeled
text/non-vision. For the primary static-placement fit, all 48 traces are used;
`P_V_*` and `P_T_*` policies are fit only on their named modality.

## Fixed policy and metric definitions

For an expert-to-rank map `p`, rank load is the exact number of routed
assignments after remapping. For each token, `u` is the number of unique ranks
among its eight selected experts. Saturation is `S = mean(u)/4`; the sparse
communication-volume proxy is `sum(u-1)`. All placement maps have exactly 32
experts per rank.

Policies were fixed before inspecting outcomes:

* **P0:** current linear map.
* **P_load:** per-layer deterministic greedy load fit plus bounded pair-swap
  refinement on all routes.
* **P_sat:** per-layer deterministic co-activation grouping surrogate plus
  bounded pair-swap refinement. Co-activation is an explicit equivalent
  rank-coverage proxy, while true `u` is always evaluated separately.
* **P_joint(lambda):** normalized load-ratio versus co-activation surrogate,
  with the preregistered grid `lambda={0,.25,.5,.75,1}` (`lambda=1` is load
  weighted; `lambda=0` saturation weighted).
* **P_V_load/P_T_load** and **P_V_sat/P_T_sat:** the corresponding policies
  fit on Vision or Text controls only, then evaluated on both.

These are static heuristics, not exact placement oracles. The full per-layer
maps and optimizer labels are in `placement_assignments.json`.

## Stage 0 — Current placement characterization

Under P0, global and per-layer/request statistics are:

| modality | mean `u` | `S=mean(u)/4` | median `u` | p90 `u` | P(`u=4`) | P(`u≥3`) | global rank CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vision | 3.6331 | 0.9083 | 4 | 4 | 0.6511 | 0.9821 | 0.0228 |
| Text | 3.6246 | 0.9061 | 4 | 4 | 0.6444 | 0.9802 | 0.0250 |

At layer scope, mean rank-load CV is 0.1006 (Vision) versus 0.1816 (Text).
At request-layer scope it is 0.1623 versus 0.2302. Request-layer `u` spans
3.321–3.830 for Vision and 3.285–3.849 for Text; P(`u=4`) spans 0.382–0.834
and 0.321–0.853. Thus global Vision/Text saturation is nearly identical, but
instantaneous rank regimes vary substantially.

`current_saturation.csv` contains global, per-layer, per-request, and
request-layer rows with `u`, saturation, rank loads, max/mean, CV, and critical
rank. `current_expert_counts.csv` contains corresponding rank and expert
assignment counts.

## Stage 1 — Placement trade-off frontier

All-profile layer means are:

| placement | rank-load CV | max/mean rank load | mean `u` | P(`u=4`) | P(`u≥3`) |
|---|---:|---:|---:|---:|---:|
| P0 | 0.1204 | 1.1593 | 3.6289 | 0.6478 | 0.9811 |
| P_load | **0.0003** | **1.0003** | 3.6816 | 0.6944 | 0.9871 |
| P_joint(.75) | 0.0009 | 1.0010 | 3.6151 | 0.6387 | 0.9766 |
| P_joint(.5) | 0.0021 | 1.0024 | 3.5985 | 0.6242 | 0.9746 |
| P_sat | 0.6552 | 2.0464 | **2.8289** | **0.2337** | **0.6643** |

P_load reduces max/mean rank load by **13.72%** relative to P0, but its
unique-rank saturation rises by only **1.45%**. P_sat reduces saturation by
**22.04%**, but max/mean rank load worsens from 1.1593 to 2.0464 (**76.5%**).
The fixed lambda grid traces intermediate non-dominated points, so a clear
load/coverage frontier exists even though the extreme saturation endpoint is
not deployable as-is. Figure:
`figures/plot2_load_vs_saturation_frontier.png`.

## Stage 2 — Vision/Text placement conflict

Cross-evaluation layer means (lower is better):

| fitted placement | Vision load CV | Text load CV | Vision max/mean | Text max/mean | Vision mean `u` | Text mean `u` |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.1006 | 0.1816 | 1.1344 | 1.2434 | 3.6331 | 3.6246 |
| P_V_load | **0.0002** | 0.1609 | **1.0002** | 1.2104 | 3.6588 | 3.6651 |
| P_T_load | 0.0857 | **0.0001** | 1.1134 | **1.0001** | 3.6486 | 3.7139 |
| P_V_sat | 0.5761 | 0.4333 | 1.8805 | 1.6910 | **3.0396** | 3.2384 |
| P_T_sat | 0.2025 | 1.0074 | 1.3013 | 2.6550 | 3.3125 | **2.3144** |

P_V_load is near-optimal for Vision but P_T_load is **11.3%** worse in
Vision max/mean; P_T_load is near-optimal for Text but P_V_load is **21.0%**
worse in Text max/mean. The two load maps differ in 86–106 of 128 expert
assignments per layer (mean 96.6). Saturation-oriented maps also conflict:
Text mean `u` is 2.3144 under P_T_sat but 3.2384 under P_V_sat (39.9% higher).

This supports modality-dependent placement/co-activation structure, but not a
large current global Vision/Text saturation gap: P0 mean `u` differs by only
0.24%. The important conflict is that each modality's placement optimum
transfers poorly to the other. Figure:
`figures/plot3_modality_cross_evaluation.png`.

## Stage 3 — Calibration and held-out transfer

The split was fixed by paired request order:

* Fold A: first 12 image requests calibrate; last 12 evaluate.
* Fold B: last 12 calibrate; first 12 evaluate.

On held-out all-profile data:

| fold | P0 max/mean | P_load max/mean | P0 rank CV | P_load rank CV | P0 mean `u` | P_load mean `u` |
|---|---:|---:|---:|---:|---:|---:|
| A | 1.1632 | 1.0754 | 0.1245 | 0.0558 | 3.6280 | 3.6716 |
| B | 1.1519 | 1.0712 | 0.1136 | 0.0535 | 3.6329 | 3.6668 |

Aggregate P_load transfer improves held-out max/mean by 7.6%/7.0%, while
increasing saturation by about 1.0–1.2%. Transfer is modality-asymmetric:
Fold A improves held-out Vision max/mean by 4.2% and Text by 9.5%; Fold B is
essentially flat for Vision (+0.2% worse) but improves Text by 14.2%. P_sat
continues to lower `u` but has held-out aggregate rank CV 0.589/0.524.

Calibration-to-held-out policy ranking Spearman ρ (fold A/B) is 0.887/0.852
for rank CV, 0.852/0.796 for max/mean, and 0.937/0.979 for mean `u`.
Therefore the frontier is not caused by one half of the requests, although
modality-specific load transfer is not symmetric. Full records:
`calibration_transfer.csv`; figure:
`figures/plot4_calibration_transfer_saturation.png`.

## Stage 4 — Runtime-saturation motivation

No dynamic communication was implemented. Under the unchanged current map,
request-layer `u` and rank-load CV vary across requests and layers, and the
held-out fold results preserve this variation. A static map therefore does not
collapse runtime into a single communication regime. This is an offline
route-visible proxy, not a claim about measured DeepEP wall time.

## Gate and interpretation

`PLACEMENT_SATURATION_TRADEOFF: GO`

The characterization gate is met:

1. A clear Pareto frontier exists: P_load improves rank balance by 13.72%,
   whereas saturation-oriented maps trade rank balance for 22.04% lower
   unique-rank coverage.
2. Vision/Text optimum conflict is material: cross-modality load penalties are
   11.3% and 21.0% in max/mean, and saturation-optimal maps have a 39.9%
   cross penalty on Text.
3. Folded transfer shows that request/layer regimes and modality asymmetry
   remain on held-out requests; no single static map is simultaneously close to
   both modality-specific optima on all metrics.

The GO means that static placement and rank-saturation objectives conflict in
these route artifacts, justifying a bounded follow-up measuring post-router
communication adaptation. It does **not** establish a communication speedup,
nor justify changing expert placement, routing, or DeepEP in this PoC.

### Strongest positive evidence

P_load reaches essentially perfect rank balance, while P_sat lowers unique-rank
coverage by 22%; modality-specific load maps differ on most experts and incur
11–21% cross penalties. These effects persist in fixed first/last-12 transfer
folds.

### Strongest counter-evidence

Current Vision/Text global saturation is almost identical and P_load's actual
saturation penalty is only 1.45%. The P_sat endpoint obtains its coverage gain
by severe rank-load imbalance, so its proxy benefit may not translate to
communication latency. No CUDA timing, queueing, or collective behavior was
measured here.

## Artifacts and next single action

Result directory:

`poc_flashvep/deepep_revalidation/results/modality_placement_saturation_tradeoff_20260827_162835/`

It contains `current_saturation.csv`, `current_expert_counts.csv`,
`placement_frontier.csv`, `modality_cross_eval.csv`,
`calibration_transfer.csv`, `placement_assignments.json`, `summary.json`, and
four figures. Analysis code:
`poc_flashvep/modality_placement_saturation_tradeoff/analyze.py`.

**Next single action:** keep the current placement fixed and run one bounded
GPU trace replay measuring whether the observed load/saturation regimes produce
different DeepEP dispatch/combine latency before implementing any dynamic
communication mechanism.
