# Token-level modality placement/saturation recheck

## Scope

This is an offline, read-only reanalysis. No model, vLLM, DeepEP, CUDA, or GPU
was initialized; physical GPUs 1–4 were not exposed. The source is the existing
24 real-image Qwen3-VL route artifacts at:

`poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`

Only those 24 image-containing requests are used in every primary fit and
evaluation. Their paired text-control route files are excluded from the
primary analysis (they are not needed for this recheck). Each real-image
request is split exactly as requested: `Vision = token_id 151655`, and
`Text = every other token`. Route IDs are unchanged. The validated EP4 map is
`expert_id // 32`, with 128 experts and exactly 32 experts per rank.

## Fixed policies and definitions

For each decoder layer, placements have 32 experts per rank. `u` is the number
of unique EP ranks in a token's captured Top-8 expert set, `S=mean(u)/4`, and
the sparse communication proxy is `sum(u-1)`. `P0` is current linear
placement. `P_load` is a deterministic fixed-capacity load heuristic fit on
all real-image tokens. `P_V_load`/`P_T_load` fit only visual/non-visual tokens.
`P_V_sat`/`P_T_sat` use the fixed co-activation saturation surrogate.
`P_joint(0.5)` and `P_joint(0.75)` use the preregistered fixed lambda values,
fit on all real-image tokens. All are heuristics, not exact placement oracles;
no routing, replication, or token assignment is changed.

Held-out folds were fixed before evaluation: Fold A fits the first 12 requests
and evaluates the latter 12; Fold B reverses them. Paired text-control requests
follow their image pair only in the diagnostic code path; they are not part of
this primary recheck.

## Current P0 token-level saturation

Global and request-layer aggregates under the current linear placement are:

| source tokens within real-image requests | mean `u` | `S` | P(`u=4`) | P(`u≥3`) | rank-load CV | max/mean |
|---|---:|---:|---:|---:|---:|---:|
| Vision | 3.6318 | 0.9079 | 0.6499 | 0.9819 | 0.1014 | 1.1350 |
| Text/non-vision | 3.6747 | 0.9187 | 0.6882 | 0.9865 | 0.2102 | 1.2771 |

At request-layer scope, mean `u` is 3.6389 (Vision) and 3.6749 (Text), with
rank-load CV 0.1707 and 0.2346. Thus the corrected token-level analysis does
**not** support a claim that Vision has higher unique-rank saturation: Text is
slightly denser under P0. It does show substantially greater Text rank-load
imbalance. Global/layer/request-layer rows and per-expert counts are in
`current_saturation.csv`.

## Placement cross-evaluation

Layer means over the 24 real-image requests are:

| fit placement | Vision max/mean | Text max/mean | Vision `S` | Text `S` | Vision load CV | Text load CV |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 1.1350 | 1.2771 | 0.9079 | 0.9187 | 0.1014 | 0.2102 |
| P_V_load | **1.0002** | 1.2903 | 0.9149 | 0.9061 | **0.0002** | 0.2287 |
| P_T_load | 1.1225 | **1.0000** | 0.9080 | **0.9377** | 0.0935 | **0.0000** |
| P_V_sat | 1.9077 | 1.3943 | **0.7545** | 0.8775 | 0.5909 | 0.2957 |
| P_T_sat | 1.1612 | 3.0354 | 0.8281 | **0.4573** | 0.1215 | 1.1957 |
| P_joint(0.5) | 1.0094 | 1.3100 | 0.8952 | 0.8976 | 0.0077 | 0.2313 |
| P_joint(0.75) | 1.0095 | 1.3187 | 0.8986 | 0.8986 | 0.0078 | 0.2379 |

The modality-specific load maps are strongly different: the maps differ in a
mean **95.8/128 experts per layer** (range 85–105). Applying P_T_load to Vision
raises max/mean by **12.23%** relative to P_V_load; applying P_V_load to Text
raises it by **29.03%** relative to P_T_load. P_joint(0.5) is close to the
Vision load optimum (0.92% max/mean penalty) but is **31.00%** worse than the
Text load optimum; P_joint(0.75) is similarly **31.87%** worse on Text. It
therefore does not simultaneously solve both modalities.

The saturation endpoint is intentionally extreme: it lowers Vision `S` to
0.7545 and Text `S` to 0.4573, but produces severe rank-load imbalance. This
demonstrates an offline load/coverage trade-off, not a deployable policy.
Details are in `placement_frontier.csv` and
`modality_cross_eval.csv`; see `figures/plot2_token_modality_cross_eval.png`.

## Held-out transfer

The primary modality conflict survives source-request held-out evaluation,
especially for the Text objective:

| fold | metric/evaluation | P0 | P_V_load | P_T_load | P_joint(0.5) |
|---|---|---:|---:|---:|---:|
| A | Vision max/mean | 1.1547 | **1.1253** | 1.1393 | 1.1266 |
| A | Text max/mean | 1.3001 | 1.3296 | **1.1030** | 1.4149 |
| B | Vision max/mean | 1.1492 | **1.1380** | 1.1833 | 1.1598 |
| B | Text max/mean | 1.2722 | 1.3035 | **1.1018** | 1.2930 |

Relative to the modality-specific held-out optimum, using P_V_load instead of
P_T_load costs **20.5% (Fold A)** and **18.3% (Fold B)** on Text max/mean.
Using P_T_load instead of P_V_load costs 1.2% and 4.0% on Vision. The asymmetry
is itself evidence that the Text/non-vision footprint is a distinct regime,
while the Vision optimum transfers somewhat better in this sample. P_joint(0.5)
is not jointly near-optimal: its Text penalty versus P_T_load is 28.3%/17.4%
in the two folds, despite being close to the Vision load optimum in Fold A.

Full layer rows are in `heldout_transfer.csv`; the transfer figure is
`figures/plot3_heldout_transfer.png`.

## Interpretation and gate

`TOKEN_MODALITY_PLACEMENT: STRONG_GO`

The gate is satisfied at the offline characterization level:

1. Vision/Text token-level load-optimal maps are clearly different (mean
   Hamming distance 95.8 experts/layer).
2. Cross-modality penalties exceed 10%: 12.23% on Vision when using the
   Text-fitted load map and 29.03% on Text when using the Vision-fitted map.
3. The conflict transfers to held-out source requests: Text suffers 18.3–20.5%
   when the Vision-fitted map replaces the Text-fitted map.
4. P_joint(0.5) and P_joint(0.75) do not make both modalities near-optimal.

The previous modality claim is therefore **maintained in a corrected form**:
Vision and non-Vision tokens induce different load/placement optima, and a
single joint static map is not simultaneously optimal. However, a stronger
claim that Vision uniquely causes higher rank saturation is **not** maintained:
under P0, non-Vision/Text tokens have slightly higher `u` and saturation.

The result justifies a bounded GPU experiment measuring whether these
token-level load/saturation regimes map to real DeepEP dispatch/combine latency.
It does not justify dynamic placement or communication implementation yet.

## Strongest evidence and counter-evidence

**Positive:** modality-specific load maps differ on 85–105 of 128 experts per
layer, with 12.23%/29.03% cross penalties; the Text penalty remains 18–20% in
both held-out folds, and joint lambda policies fail to optimize both.

**Counter-evidence:** current global saturation is nearly dense and slightly
higher for Text (0.9187 versus 0.9079); Vision's cross penalty is small in one
held-out direction. The saturation-oriented maps achieve low `S` only by
creating very large rank imbalance, and no GPU latency/communication effect was
measured.

## Artifacts and next action

Result directory:

`poc_flashvep/deepep_revalidation/results/token_modality_placement_recheck_20260827_165021/`

Contains `current_saturation.csv`, `placement_frontier.csv`,
`modality_cross_eval.csv`, `heldout_transfer.csv`,
`placement_assignments.json`, source manifest, policy JSON, summary JSON, and
three figures. Analysis code:
`poc_flashvep/token_modality_placement_recheck/analyze.py`.

**Next single action:** with placement fixed, run one bounded GPU trace to test
whether the corrected token-level rank-load/saturation differences predict
actual DeepEP dispatch/combine latency; do not change placement or routing.
