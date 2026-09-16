# Phase 3A: static threshold → EP coupling smoke test

## Verdict

The static coupling is **positive but overwhelmingly NFE-driven**.

- Threshold 0.80 preserved the bounded GSM8K score (31/32) while reducing
  actual true-EP2 routed-MoE replay time by 42.58% and simulated EP4
  routed-MoE time by 41.98%.
- Threshold 0.60 reduced those costs by 55.36% and 55.15%, respectively, but
  lost two additional correct answers (29/32, -6.25 percentage points).
- Rank imbalance and fanout were essentially invariant. EP-aware prediction
  explained the measured EP2 delta only marginally better than NFE alone.

This is evidence for a useful static 0.80 operating point on this smoke
cohort, not evidence for an EP-specific controller. No early/middle/late
policy or dynamic controller was implemented.

## Protocol integrity

The same fixed GSM8K IDs 0--31, checkpoint, four-shot prompt, tokenizer,
temperature 0, block length 32, maximum 32 refinement steps per block, EOS
behavior, generation cap 2,048, and per-ID RNG seeds were used throughout.
Thresholds 0.95, 0.80, and 0.60 each ran an independent official-HF rollout.
No trajectory was obtained by re-thresholding the 0.95 trace.

Every rollout produced a separate compact trace containing mask/acceptance
state and every routed layer's top-8 expert IDs. Hook-on/off parity was checked
on the first request of each rollout. All outputs terminated without remaining
MASK tokens.

For each trace, every routed-MoE invocation was replayed through a real
DP1/TP1/SP1/EP2 DeepEP path on physical GPUs 0 and 1. Each component is the
sum of per-invocation medians from three repetitions. This is an **all-
invocation true-EP2 route replay**, not production serving latency and not a
full-model EP2 rollout. It measures the physical EP path for the real route
shape with controlled BF16 expert weights/activations.

Cost accounting remains strictly separated:

1. EP dispatch;
2. routed expert compute;
3. EP combine;
4. replicated-state bridge all-gather.

Only 1--3 are reported below. Category 4 is absent from the replay and is not
counted as EP communication or EP speedup.

## Quality and refinement trajectory

| Threshold | Accuracy | Mean NFE | Total NFE | Blocks | Accepted / forward | Accepted positions | Output tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 31/32 (96.875%) | 98.25 | 3,144 | 232 | 2.20 | 6,903 | 6,398 |
| 0.80 | 31/32 (96.875%) | 57.34 | 1,835 | 206 | 3.31 | 6,071 | 5,598 |
| 0.60 | 29/32 (90.625%) | 44.28 | 1,417 | 207 | 4.31 | 6,103 | 5,636 |

Threshold 0.80 lowered total NFE by 41.63%. Threshold 0.60 lowered it by
54.93%. The mean accepted-token count per refinement forward increased
monotonically as expected.

The 0.95 and 0.80 parsed answers were identical on all 32 requests, even
though only one full token sequence was exactly identical. Both missed only
ID 12. At 0.60, IDs 11 and 16 additionally changed from correct to incorrect;
30/32 parsed answers remained identical to 0.95. Thus the first visible
quality loss occurs at 0.60 in this cohort.

This is a smoke test: one sample is 3.125 percentage points, so equality at
31/32 is not a statistically precise zero-loss claim.

## Actual true-EP2 route replay

Aggregate time over every routed layer/invocation in the 32-request traces:

| Threshold | EP dispatch | Routed expert | EP combine | Routed-MoE stage | Stage reduction vs 0.95 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 9.894 s | 66.723 s | 7.355 s | **84.059 s** | — |
| 0.80 | 5.662 s | 38.388 s | 4.170 s | **48.267 s** | **42.58%** |
| 0.60 | 4.451 s | 29.776 s | 3.270 s | **37.522 s** | **55.36%** |

At 0.80, dispatch/expert/combine fell by 42.77%/42.47%/43.30%. At 0.60,
they fell by 55.01%/55.37%/55.54%. There is no component-specific reversal.

The calibrated EP2 simulator underpredicted the absolute replay totals by
8.20%, 7.02%, and 7.58% at thresholds 0.95, 0.80, and 0.60. This is within the
previous held-out stage-error gate, and the nearly constant bias cancels in
the threshold deltas. Actual replay, rather than the calibrated estimate, is
therefore used for the EP2 reduction headline.

## EP2 physical shape

The EP2 structural projection uses the measured true-EP2 ownership/source
mapping and the independently calibrated communication/compute models.

| Threshold | Mean max/mean load | Mean load CV | Logical remote bytes | Mean token fanout |
|---:|---:|---:|---:|---:|
| 0.95 | 1.1283 | 0.1283 | 398.09 GB | 1.9690 |
| 0.80 | 1.1303 | 0.1303 | 225.22 GB | 1.9692 |
| 0.60 | 1.1300 | 0.1300 | 174.90 GB | 1.9691 |

Remote bytes decreased by 43.43% at 0.80 and 56.06% at 0.60. These reductions
are 1.79 and 1.13 percentage points larger than the respective NFE reductions,
but load geometry did not materially improve: max/mean changed by only about
0.2%, while fanout was effectively constant.

## Simulated EP4

All values below are `SIMULATED-EP4-EP2-CALIBRATED` routed-MoE stage values,
not measured EP4 and not full BCT/E2E.

| Threshold | EP dispatch | Routed expert | EP combine | Routed-MoE stage | Stage reduction vs 0.95 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 9.941 s | 48.490 s | 5.495 s | **63.926 s** | — |
| 0.80 | 5.747 s | 28.169 s | 3.173 s | **37.088 s** | **41.98%** |
| 0.60 | 4.445 s | 21.769 s | 2.455 s | **28.668 s** | **55.15%** |

| Threshold | Mean max/mean load | Mean load CV | Logical remote bytes | Mean token fanout |
|---:|---:|---:|---:|---:|
| 0.95 | 1.3511 | 0.2540 | 943.58 GB | 3.1033 |
| 0.80 | 1.3537 | 0.2554 | 533.70 GB | 3.1034 |
| 0.60 | 1.3522 | 0.2549 | 414.47 GB | 3.1033 |

Actual EP2 and simulated EP4 have the same fastest-to-slowest ranking:
0.60, 0.80, 0.95. The EP4 reductions track EP2 closely; higher virtual EP
degree does not reveal a threshold-dependent load/fanout advantage in this
vanilla workload.

## NFE-only versus EP-aware explanation

Using the measured EP2 stage reduction as the target:

| Predictor | Mean absolute error over 0.80/0.60 reduction |
|---|---:|
| NFE-only ratio | 0.688 percentage points |
| EP-aware calibrated model | **0.522 percentage points** |

EP-aware wins, but by only 0.166 percentage points. At 0.80 the measured EP2
stage reduction exceeds the NFE reduction by 0.95 point; at 0.60 it exceeds it
by 0.43 point. The corresponding simulated EP4 excess is only 0.35 and 0.22
point. Therefore NFE/refinement-count reduction explains nearly all of the
threshold effect, with only a small additional payload/shape correction.

## Answers to the Phase-3A questions

1. **Does threshold↓ cause NFE↓?** Yes: -41.63% at 0.80 and -54.93% at
   0.60 relative to 0.95.
2. **Where does quality visibly decrease?** At 0.60 in this cohort. Threshold
   0.80 retained the same 31/32 score and the same parsed answer on 32/32.
3. **Does EP cost follow NFE?** Almost exactly. Actual EP2 receives less than
   one percentage point of additional reduction; EP4 receives less than 0.4
   point. Rank geometry/fanout is essentially unchanged.
4. **Which predictor explains actual EP2 changes better?** EP-aware is
   numerically better (0.522 vs 0.688 point MAE), but the incremental value is
   small; NFE is the dominant explanatory variable.
5. **Is simulated EP4 ranking consistent with actual EP2?** Yes, exactly.

## Decision and next step

Threshold 0.80 is a strong static smoke point: it retains bounded quality and
removes about 42% of routed-MoE work. However, this experiment does **not**
establish an EP-specific optimization opportunity because the gain is almost
entirely generic NFE reduction. The appropriate next experiment is a larger
quality validation of static 0.80 before considering any policy; this report
does not run that validation or a dynamic controller.

Machine-readable results are in
`reports/phase3a_threshold_ep_coupling_summary.json` and `.csv`. Figures are
in `reports/figures/phase3a_threshold_ep/`.
