# Four dLLM–MoE–EP Hypotheses: Discovery Summary

## Outcome

No hypothesis clears a method-implementation gate. H1 and H2 expose real structure worth retaining as characterization; H3 and H4 have essentially zero physical headroom.

| Rank | Hypothesis | dLLM-specific signal | EP-specific consequence | True EP2 support | Simulated EP4 headroom | Verdict |
|---:|---|---|---|---|---:|---|
| 1 | H1 block-persistent straggler | same-block match exceeds matched random by 13.35 pp | early signature beats global-majority rank by only 2.51 pp | critical-rank match 12/12; high/low persistence contrast not reproduced | structural predictability, no latency-removal oracle | **HOLD** |
| 2 | H2 mask-state specialization | same-position top-k Jaccard 0.231 P50 | full-load max/mean effect 0.167% P50, 1.864% P99 | aggregate structural audit only | max 5.62%, not reproducible in tail percentiles | **HOLD** |
| 3 | H4 complementary batching | early signatures repeat, but shuffled histories are almost as similar | pair gain 0.098%; group-of-4 loses 0.044% | selected replay gain 0.20%, similar-hot also ties | <1% | **NO-GO** |
| 4 | H3 co-activation placement | early placement tracks an oracle with no value | full-future full-stage P50 -0.028%, max 0.301% | not needed to reject oracle | <1%; migration dominates | **NO-GO** |

## Cohort and trace integrity

- Model: `inclusionAI/LLaDA2.0-mini`, revision `dad945cac317da394b390f82c7b40691d8a881ed`, BF16.
- Decode anchor: threshold 0.95, block length 32, maximum 32 refinement steps/block, EOS early stop.
- Cohort: fixed GSM8K IDs 0–127; requests 0–63 discovery/train and 64–127 held out.
- Trace collection used GPU0/GPU1 as independent single-GPU workers; it is not labeled EP2. Only the selected-state DeepEP replay is physical true EP2.
- Accuracy: 118/128 = **92.19%**, consistent with the already reproduced full-test 91.81% baseline.
- NFE: mean 113.95, median 73, P90 162.3, P95 212.1, P99 1,339.8, max 1,411. The two 65-block/NFE>1,000 long-tail requests were retained.
- Aggregate v2: 277,134 layer/refinement invocations (=14,586 NFE × 19 routed layers).
- Heavy audit: 16 retained full-row requests × 32 sampled invocations = 512 checks. Exact current routes, five-way position classes, expert histograms, EP2 rank loads, and EP4 rank loads all passed at **100%**.

The audit found and fixed one conversion defect before analysis: prompt-tail positions in the first physical generation block were initially labeled DECODED. EP routing/rank-load arrays were unaffected, but H2 state labels were corrected and the complete trace was rebuilt.

## What was known before this PoC

The prior infrastructure physically established a DP1/TP1/SP1/EP2 isolation harness:

1. EP dispatch,
2. owner-local routed expert compute,
3. EP combine.

It also calibrated the EP2 stage predictor to median APE 3.87% and P90 APE 9.40%. The replicated-state bridge all-gather remains a separate cost category and is excluded here. Existing EP4/EP8 results were virtual ownership/traffic/compute projections, not physical measurements. Average structural max/mean imbalance had been observed, but persistent critical-rank behavior had never been tested.

## What this PoC newly establishes

### H1: persistence is real, but mostly global

Simulated EP4 layer-local adjacent critical-rank match is 99.32% versus 84.80% matched random. The block-bootstrap difference is +13.35 pp, CI [+12.30, +14.36] pp. Yet cross-block match is already 95.57% and a single global-majority rank predicts 94.14%; refinements 1–2 reach 96.65%. Thus repeated refinement adds only a small increment over a generic/global expert-popularity hotspot.

The initial compute envelope was out of range for some selected states. Extending it from 96 to 512 measured heavy-trace shapes expanded local assignments from 8,725 to 11,792 and produced 12/12 structural critical-rank matches in true EP2. Both selected high- and low-persistence cohorts nevertheless measured 100% adjacent persistence, so their directional contrast was not reproduced.

### H2: route identity shifts, destination geometry mostly does not

MASKED-versus-DECODED expert JSD is 0.562 P50 and exceeds the prompt/prior-block generic control by 0.357 (block-bootstrap CI [0.346, 0.369]). The same position retains only 0.231 top-k Jaccard at P50, while destination-rank-set Jaccard is 0.750 P50 and 1.0 P90. After the state rows are returned to the full vanilla invocation, their median EP4 max/mean impact falls to 0.167%; even P99 is 1.864%.

### H3: placement cannot convert co-activation into savings

The global static permutation reduces max assignment load 5.69% but increases remote bytes 14.30% and fanout 14.68%, making the full stage 0.038% worse. The block-local full-future oracle has -0.028% P50 and only 0.301% maximum full-stage gain. Moving 185 expert IDs/layer costs at least 22.11 GB and 68.50 ms. There is no oracle headroom to amortize.

### H4: signature stability does not imply batching value

Early-to-future load cosine is 0.999999 P50, but shuffled-history cosine is also 0.999940; the mean advantage is only 0.000198. Complementary pairs gain 0.098% in simulated EP4 and 0.20% in selected true EP2, while similar-hot pairs are also marginally faster than random. The intended ordering is absent.

## Common simulated EP4 tails

| Metric | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Max/mean rank load | 1.285 | 1.553 | 1.624 | 1.887 | 1.935 |
| Critical-rank expert compute (ms) | 0.826 | 0.893 | 0.893 | 0.893 | 0.893 |
| Remote bytes | 15,777,792 | 27,082,752 | 30,060,544 | 33,374,208 | 36,851,712 |
| Mean destination fanout | 3.100 | 3.194 | 3.245 | 3.321 | 3.343 |
| Routed-MoE stage (ms) | 1.092 | 1.208 | 1.226 | 1.246 | 1.260 |

These are `SIMULATED-EP4-EP2-CALIBRATED` component predictions, not production-serving E2E measurements.

## Decision

H1 is the strongest candidate only in the sense of structural evidence. It does not yet support a paper-level method because global expert/rank popularity explains most of the predictability. H2 is a clean dLLM-specific routing observation without meaningful physical EP impact. H3 and H4 should be stopped.

```text
H1_BLOCK_PERSISTENT_STRAGGLER:
HOLD

H2_MASK_STATE_SPECIALIZATION:
HOLD

H3_COACTIVATION_PLACEMENT:
NO-GO

H4_COMPLEMENTARY_BATCHING:
NO-GO

STRONGEST_DLLM_EP_OBSERVATION:
Within-block critical-rank identity is highly persistent, but after global-majority and cross-block controls the incremental dLLM-specific signal is modest and not yet method-level.

BEST_FOLLOWUP_METHOD_IF_ANY:
NONE; if H1 is revisited, first perform physical EP4 validation with global expert-popularity normalization rather than implementing a scheduler.

WHAT_EXISTING_TRACE_ALREADY_PROVED:
True EP2 ownership/dispatch/owner-compute/combine and calibrated component timing, plus simulated average EP4/EP8 structural imbalance; it had not proved persistent stragglers.

WHAT_THIS_POC_NEWLY_PROVED:
H1 persistence and H2 mask-state routing specialization exist structurally, but H3 placement and H4 complementary batching have sub-1% headroom and no hypothesis currently justifies method implementation.

DO_NOT_IMPLEMENT_METHOD_AUTOMATICALLY:
true
```
