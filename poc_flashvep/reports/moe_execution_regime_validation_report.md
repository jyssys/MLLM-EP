# MoE execution-regime validation sprint

Date: 2026-09-05 (KST)  
Branch: `flashvep/moe-execution-regime-validation`  
**FINAL STATUS: HOLD** (the historical sign flip is rejected; a robust high-M
fanout/expert–dispatch regime remains a generic MoE/EP research lead).  
Primary runtime: Qwen3-VL-30B-A3B-Instruct, BF16, vLLM 0.20.0 V1,
TP2/DP2/EP4/PP1, DeepEP high-throughput, Triton Unquantized Experts,
linear placement, eager mode, DBO off, prefix cache off.  Physical devices were
restricted to `CUDA_VISIBLE_DEVICES=1,2,3,4`; no other GPU was used by this sprint.

Measured live-run wall time was approximately 43 minutes (24 successful replay
windows plus two bounded initialization-failure attempts), corresponding to
approximately **2.9 four-GPU-hours**.  The wider wall-clock sprint elapsed from
13:22 to after the required 75-minute exploration window while analysis and
replication were interleaved.

## Executive result

**The historical M=128-versus-M=512 sign flip is not reproduced under the
pre-registered warmup and interleaved-order protocol.**  It was a runtime-state/
ordering artifact in the original block-ordered measurements.  A different,
reproducible phenomenon remains: at larger token batches (roughly M≥448), F4
destination fanout costs more than F1 even when total assignments, top-k,
aggregate EP-rank load, active experts, and activation tensor are held fixed.

At M=1024 the 30-repetition Qwen3-VL replication gives F4/F1 increases of
31.34% expert, 26.03% dispatch, 12.59% combine, and 19.54% critical wall; the
paired expert effect is positive in 29/30 repetitions (96.7%).  The active×fanout
replication gives critical-wall penalties of 15.37%, 18.61%, and 24.32% for
A8/A16/A32.  The signal is also present at M=512 in layers 4/24/44
(expert +13.71/+14.19/+15.92%).

This is a **generic MoE/EP execution-regime candidate**, not yet an
MLLM-specific phenomenon: the Qwen3-30B text control has the same direction
(M=512 expert +16.08%; M=1024 expert +32.40%).  Real Qwen3-VL route transfer
confirms that natural routes occupy the high-fanout region (mean F≈3.4–3.8),
but does not provide a matched F1/F4 causal pair.

## Protocol and invariants

- Global model initialization and warmup, followed by five per-shape warmups.
- Case order was deterministically shuffled; paired F1/F4 cases were interleaved.
- Primary runs used 10, 20, or 30 repetitions; medians are primary and paired
  per-repetition ratios are reported where available.
- Every successful observation reports `correctness.passed=true`,
  `route_identity=true`, and `token_partition_identity=true`.
- Controlled routes preserve M, top-k=8, total assignments (8M), aggregate
  rank assignments, and BF16 captured hidden activations.  They are explicitly
  synthetic route-shape diagnostics, not claims about a changed model router.
- H8 uses verbatim real Qwen3-VL route IDs with shape-compatible validated
  layer-24 activations; it is labelled route-transfer evidence.

## H1 — exact M×fanout regime

The interleaved boundary sweep (A16, 10 repetitions per point) produced:

| M | F4/F1 expert | F4/F1 dispatch | F4/F1 combine | F4/F1 critical wall |
|---:|---:|---:|---:|---:|
| 64 | +5.47% | −1.55% | +15.49% | +1.31% |
| 96 | −0.09% | +3.21% | −11.48% | −0.50% |
| 128 | −2.75% | −6.71% | +8.81% | −2.68% |
| 160 | +0.57% | +11.51% | +1.59% | +1.20% |
| 192 | −0.98% | −6.64% | +3.59% | −5.24% |
| 256 | +2.78% | −0.98% | +3.85% | −2.34% |
| 320 | +7.96% | +16.61% | +20.93% | +5.35% |
| 384 | +7.09% | +7.18% | +30.15% | +1.15% |
| 448 | +15.11% | +10.94% | +2.80% | +5.17% |
| 512 | +15.14% | +28.61% | −10.51% | +4.34% |
| 640 | +26.00% | +13.78% | −21.02% | +5.24% |
| 768 | +28.87% | +21.11% | −24.64% | +7.63% |
| 1024 | +30.62% | +39.70% | +8.76% | +19.24% |

The transition is gradual from approximately M=320 through M=1024.  There is
no discontinuity at 127→128, 255→256, or 511→512.  A targeted 30-repetition
M=1024 run measured expert +31.34% (bootstrap 95% CI for the paired median
[29.22%, 34.09%]), dispatch +26.03%, combine +12.59%, and wall +19.54%.

Therefore:

- `SIGN_FLIP_REPRODUCED = NO`.
- The robust replacement finding is a high-M fanout penalty/regime transition.
- First-use confound is removed for the paired result; block-order outliers are
  retained as evidence that state can still perturb absolute measurements.

## H2 — active experts versus fanout

With active experts held fixed, M=512 F4/F1 expert effects in the 30-repetition
run were +9.80% (A8), +14.07% (A16), and +23.77% (A32); dispatch effects were
+9.95%, +14.10%, and +14.85%.  At M=1024 they rose to +28.10%, +31.83%, and
+37.85% expert and +26.76%, +33.91%, and +39.82% dispatch for A8/A16/A32.
Critical wall at M=1024 was +15.37%, +18.61%, and +24.32%.

This is an interaction rather than an active-expert-only effect: fanout is
small at M=128 (expert +2.04–5.94%, wall +0.55–3.16%) and grows with both M and
active local experts.

## H3 — local expert versus DeepEP decomposition

The local-only diagnostic uses the exact receive layout from a real DeepEP
dispatch, but times only the local Triton expert invocation.  At M=1024,
F4/F1 was +40.49% local expert, +30.31% standard DeepEP expert, +19.59%
dispatch, +2.68% combine, and +12.17% critical wall (20 interleaved
repetitions).  At M=512 the corresponding local/DeepEP expert effects were
+22.74%/+15.86%.

The local measurement has a separate timing path and is diagnostic rather than
a replacement runtime.  Taken together, the data support an
**expert-kernel + dispatch/packing interaction**.  Combine can move in the
opposite direction and cancel part of the expert/dispatch penalty; therefore
assignment counts alone are not a cost model.

## H4 — alignment/tile boundaries

The fine sweep around 128, 256, and 512 found no power-of-two discontinuity.
The F4/F1 expert effect grows smoothly (approximately +12–19% at M=496–513
and +23% at M=520–528 in the 30-repetition focus).  The evidence supports a
broader workload/kernel regime boundary, not a single tile threshold.

## H5 — layer persistence

At M=512/A16, F4/F1 expert was +13.71% (layer 4), +14.19% (layer 24), and
+15.92% (layer 44), with dispatch +15.31%, +20.00% (30-rep layer-24 run),
and +20.06%, respectively.  M=128 responses were near zero or mixed in all
three layers.  A separate layer-44 M=1024 retry failed during engine
initialization and is recorded as blocked; no value was imputed.

## H6 — sender→destination geometry

At fixed M, F2, A16, and exactly balanced aggregate rank load, pair-concentrated
versus cyclic destination geometry was effectively null: cyclic/concentrated
expert +0.35% and wall +0.90% at M=512, and expert −2.58% and wall −1.56% at
M=1024 (20 repetitions each).  Thus the observed high-M effect is not explained
by this traffic-matrix permutation alone.

## H7 — per-expert distribution shape

At fixed M, fanout=4, A16, and rank load, interleaved uniform versus skewed
per-expert distributions were weak: skew/uniform expert −0.65% (M=128) and
+3.66% (M=512); wall −0.46% and +1.77%.  Distribution shape alone is not a
strong explanation in the tested controls.

## H8 — real-route transfer

The route-transfer set contains astronaut, motorcycle, deep-field, retina,
model-card, method, and coffee/rocket samples at M=128/512.  Natural routes
have mean fanout approximately 3.44–3.78 and active experts 95–127.  Critical
wall spans vary from approximately 1.35 to 2.73 ms at M=128 and 1.38 to
2.43 ms at M=512 in layer 24; layer 44 includes a `method_M512` combine outlier
(4.76 ms).  These data show that production-like visual routes live in the
high-fanout regime and exhibit route/sample variance, but there is no matched
natural F1/F4 pair, so causal transfer is `NOT_ESTABLISHED`.

## H10 — generic Qwen3 control

The same interleaved controlled shape on Qwen3-30B-A3B text-only gives F4/F1
expert −1.15% at M=128, +16.08% at M=512, and +32.40% at M=1024 (the latter
also has dispatch +33.21% and wall +2.14%).  Thus the high-M regime is not
MLLM-specific in this first cross-model check.  It is best classified as a
generic MoE/EP execution phenomenon whose visual-route prevalence remains an
open MLLM question.

## Hypothesis scoreboard

| Hypothesis | Status | Effect / evidence | Q1 Dense? | Q2 MoE no-EP? | Q3 EP required? | Q4 Text-only? |
|---|---|---|---|---|---|---|
| H1 exact M×fanout sign flip | NO_GO (reframed) | Sign flip absent; high-M penalty robust | n/a | unknown | yes in tested path | same direction in Qwen3 |
| H2 active×fanout interaction | GO | M512 +9.8–23.8% expert; M1024 +15–24% wall | likely | unknown | likely | same direction |
| H3 local vs DeepEP | GO | M1024 local +40.5%, DeepEP expert +30.3%, dispatch +19.6% | unknown | unknown | distributed phase contributes | not checked local |
| H4 alignment/tile | NO_GO | no power-of-two discontinuity | n/a | n/a | n/a | n/a |
| H5 layer persistence | GO | M512 +13.7–15.9% across layers 4/24/44 | unknown | unknown | likely | not checked layers |
| H6 traffic geometry | NO_GO | ≤2.6% expert/wall | n/a | n/a | no isolated signal | n/a |
| H7 distribution shape | NO_GO | ≤3.7% expert, ≤1.8% wall | n/a | n/a | weak | n/a |
| H8 real-route transfer | HOLD | natural F≈3.4–3.8; no matched causal pair | unknown | unknown | unknown | n/a |
| H9 TP/EP beyond token count | BLOCKED | not run; topology sweep intentionally deferred | unknown | unknown | unknown | n/a |
| H10 generic Qwen3 check | GO (genericity) | M512 +16.1%, M1024 +32.4% expert | unknown | likely | yes in DeepEP path | effect persists |

Status labels distinguish a research lead from a production method decision;
no routing, placement, scheduler, pruning, or kernel optimization was implemented.

## Final causal judgement

| Field | Result |
|---|---|
| `SIGN_FLIP_REPRODUCED` | **NO** |
| `SIGN_FLIP_BOUNDARY` | No 1.0 crossing under interleaving; gradual high-M onset around M≈320–448, strong by M≥512 |
| `FIRST_USE_CONFOUND_REMOVED` | **YES** for paired primary results |
| `PRIMARY_CAUSE` | **INTERACTION** (local expert execution plus DeepEP dispatch/packing; combine cancellation) |
| `ACTIVE_EXPERT_EFFECT` | Amplifies fanout at M=512/1024; not sufficient alone |
| `FANOUT_EFFECT` | Near-zero at M=128, +14–38% expert/dispatch and up to +24% wall at M=1024 |
| `ALIGNMENT_BOUNDARY` | No isolated power-of-two/tile jump; gradual regime |
| `LAYER_EFFECT` | M512 response persists across early/mid/late representative layers |
| `TRAFFIC_GEOMETRY_EFFECT` | Null in balanced pair-concentrated/cyclic controls |
| `REAL_ROUTE_TRANSFER` | **NOT_TESTED causally**; route-transfer prevalence only |
| `TP_EP_GEOMETRY_SIGNAL` | Not tested in this sprint (H9 blocked) |
| `GENERIC_OR_MLLM_SPECIFIC` | **GENERIC MoE/EP** in current evidence; visual routes likely operate in high-F regime |

## Best new research question

Why does increasing destination fanout create a large local expert and dispatch
penalty only after a workload-size/active-expert threshold, while combine often
improves and masks it at lower M?  The cheapest decisive next experiment is a
kernel-name and grouped-GEMM tile trace for the M=320–1024 transition, paired
with a matched natural-route bin at F≈3–4 and an actual F1/F4 control.  This is
an execution-cost model question, not a scheduler proposal.

## Most important negative and novelty boundary

The strongest negative is that balanced sender→destination geometry and
per-expert distribution shape do not explain the effect.  The most surprising
positive is the order-controlled M1024 penalty (expert +31%, wall +19.5%) and
its replication on generic Qwen3.  This makes the phenomenon potentially
important for MoE/EP kernel/packing design, but weakens an MLLM-only novelty
claim.  A contribution would need to explain the interaction and demonstrate
that real visual routing reaches a distinct cost regime beyond generic text
traffic.

## Do not pursue now

Do not implement RL, token pruning/merging, dynamic routing, expert placement,
or production scheduling from this dataset.  Do not claim the old sign flip or
MLLM-specificity.  Do not use the real-route outlier as a causal result.

## Artifacts

- Specification: `poc_flashvep/reports/moe_execution_regime_validation_specs.md`
- Checkpoint: `poc_flashvep/reports/moe_execution_regime_validation_checkpoint_t45.md`
- Anomalies: `poc_flashvep/reports/moe_execution_regime_validation_anomalies.md`
- Results: `poc_flashvep/deepep_revalidation/results/moe_execution_regime_validation_20260905_132158/`
- Consolidated scoreboard/figures: `.../consolidated/`
- Environment: `.../environment_manifest.json`
