# Complementary Request Rebatching — Offline Oracle/Characterization

## Scope and provenance

This is an artifact-only analysis. No model, vLLM, DeepEP, CUDA, or GPU was
started during this PoC (the requested physical GPU set is therefore not
exposed). The source of truth is the existing 24-pair Qwen3-VL route capture:

`poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`

The copied manifest is `source_workload_manifest.json` in the result directory.
It contains 24 real-image requests and their 24 text controls, each with 48
layers and top-8 expert IDs. The scheduling population is the 24 vision
requests in manifest order; paired text requests are diagnostic controls only.
The artifact configuration was TP2/DP2/EP4, BF16, DeepEP high-throughput,
DBO-off. This run itself performed no execution.

Expert placement was taken from the validated prior EP4 linear placement:
`rank = expert_id // 32` (128 experts, four ranks). For each token, the eight
expert IDs were mapped to ranks and counted. Vision positions are exactly the
contiguous `image_token_id=151655` positions used by the prior capture code;
all other positions are labeled non-vision/text. Every 2,304 request-layer
row satisfies `R = V + T` exactly (reported in `capture_integrity.json`).

## Fixed analysis policy

The required windows plus two larger diagnostics were evaluated:

| window `(B_s,B_e)` | full windows | covered requests |
|---|---:|---:|
| (8,4) | 3 | 24 |
| (12,4) | 2 | 24 |
| (12,6) | 2 | 24 |
| (16,8) | 1 | 16 |
| (24,8) | 1 | 24 |
| (24,12) | 1 | 24 |

Windows are disjoint and contiguous in manifest order. The eight-request
remainder for `(16,8)` is not duplicated; it is explicitly reported as
uncovered. Every method has the same request set and therefore the same total
assignment count. FIFO is manifest order. Random uses 1,000 fixed-seed random
partitions per configuration. Oracle-L minimizes each layer independently;
partitions up to 10,000 are exhaustively enumerated, otherwise a deterministic
greedy construction with pair swaps is used. Oracle-F chooses one partition per
window from the 48-layer aggregate and holds it fixed across layers. The
objective is exactly the requested sum of per-group maximum rank loads.

## Footprint and modality characterization

The 24 vision requests contain 1,152 request-layer observations. Mean (median)
vision assignment fraction is **93.44% (93.60%)**, with range 83.72–99.03%.
The paired text controls have `V=0` and `T=R` by construction. In vision
requests, `argmax(V)` equals `argmax(R)` in **94.27%** of request-layer rows.
`argmax(T)` equals `argmax(R)` in **38.19%**, so the small non-vision tail can
still change the selected critical rank (with ties resolved by numpy's first
index rule). These are descriptive diagnostics; modality is not used by the
primary grouping objective.

The exact footprint and all per-request/layer fields are in
`per_request_layer_rank_footprint.csv`. `modality_characterization.csv` also
contains total-R Oracle-L V/T decompositions and counterfactual V-only/T-only
grouping diagnostics. The latter are not proposed as a scheduler.

## Oracle result

The primary ratio is `FIFO window cost / method window cost` (higher is better):

| `(B_s,B_e)` | Oracle-L | Oracle-F | random median | random p5 | random best |
|---|---:|---:|---:|---:|---:|
| (8,4) | 1.0048× | 1.0003× | 0.9974× | 1.0008× | 1.0023× |
| (12,4) | 1.0129× | 1.0003× | 0.9990× | 1.0063× | 1.0091× |
| (12,6) | 1.0177× | 0.9997× | 1.0124× | 1.0151× | 1.0160× |
| (16,8) | 1.0044× | 0.9974× | 0.9975× | 1.0014× | 1.0029× |
| (24,8) | **1.0231×** | 0.9998× | 1.0085× | 1.0177× | 1.0203× |
| (24,12) | 1.0150× | 0.9990× | 1.0087× | 1.0122× | 1.0133× |

Oracle-L median across the six configurations is **1.0139×**; the strongest
configuration is `(24,8)` at **1.0231×**. Across layer-level totals, the
median Oracle-L ratio is 1.002–1.010 for five of six configurations; only the
`(24,8)` configuration has 8.33% of layers at or above 1.07×, and no
configuration has any layer at or above 1.15×. The largest layer ratio is
1.1141×. Thus the effect is not a broad 15% headroom phenomenon.

Random is a meaningful control: its best trial reaches 1.0203× for `(24,8)`,
close to the Oracle-L 1.0231×. In several settings random median is at or
below FIFO (ratio below 1), showing that arbitrary grouping is not reliably
helpful. The oracle's extra benefit over the best random trial is only about
0.15–0.37 percentage points in these aggregates.

Oracle-F is essentially neutral (median approximately 0.9995×), indicating
that the useful complementarity is predominantly layer-local rather than a
stable request-level grouping signal across all 48 layers. This is a result,
not a post-hoc grouping change.

The generated figures are:

1. `figures/plot1_fifo_random_oracle_headroom.png` — all fixed configurations,
   including random median/p5 and both oracle variants.
2. `figures/plot2_request_rank_footprint_heatmap.png` — a fixed representative
   layer-0 request/rank heatmap, selected by the largest aggregate Oracle-L
   configuration `(24,8)`.
3. `figures/plot3_modality_critical_rank.png` — V/T assignment share and
   critical-rank agreement.

## Answers to the research questions

1. **Do complementary footprints exist?** Yes, but the exact-R improvement is
   modest. Layer-local partitions can reduce the sum of max-rank group loads by
   roughly 0.4–2.3% in the tested windows.
2. **How much oracle headroom?** Median Oracle-L is 1.0139× and best is 1.0231×.
   This is well below the preregistered 1.07× HOLD boundary.
3. **Is it better than random?** Sometimes, but only slightly. Oracle-L is
   close to random-best in the strongest case; random median is not reliably
   better than FIFO.
4. **Is complementarity layer-local?** Yes. Oracle-F is approximately neutral,
   while Oracle-L has small gains; the request grouping signal changes with
   layer.
5. **Does vision dominate total footprint?** Yes: mean vision assignment share
   is 93.44% (median 93.60%).
6. **Can a small text tail change critical rank?** Yes. In vision requests,
   text/non-vision `argmax(T)` agrees with total `argmax(R)` in only 38.19% of
   rows, despite the small mean assignment share. Tie behavior is recorded in
   the artifact; this statistic should not be read as a latency claim.
7. **Is modality-aware scheduling needed?** The V-only/T-only counterfactuals
   do not show a large additional opportunity over total-R grouping. For
   example, in `(24,8)`, total-R Oracle-L cost is 1,734,753, V-only grouping
   evaluated on total R is 1,735,576 (0.05% worse), while T-only grouping is
   1,787,196 (3.02% worse). The evidence favors using the exact total footprint
   if a follow-up is attempted; it does not justify a modality-weighted method.
8. **Does this justify a Route-Then-Batch GPU PoC?** Not under this gate. The
   offline upper bound is below 1.07× and is close to random-best. A GPU/runtime
   PoC should not be justified as a likely large win from this artifact alone.

## Gate and limitations

`COMPLEMENTARY_REBATCH: NO-GO`.

The primary gate is a median Oracle-L speedup of at least 1.15× for GO,
1.07–1.15× for HOLD, and below 1.07× for NO-GO (or complementarity confined
to a few observations). The measured median is 1.0139× and no configuration
reaches a 1.15× layer-level ratio. The strongest `(24,8)` result is still only
2.31% and is nearly matched by the best random partition.

This is an assignment-load oracle, not a CUDA latency prediction: it does not
include kernel shape, communication, queueing, or request arrival timing. The
`(16,8)` configuration covers 16 of 24 requests under the fixed full-window
policy. Text controls are paired diagnostics rather than a second scheduling
population. These limitations make the conclusion conservative: even with
perfect per-layer route visibility and no execution overhead, the measured
headroom is small.

## Reproducibility artifacts

Result directory:

`poc_flashvep/deepep_revalidation/results/complementary_rebatch_oracle_20260827_150130/`

It contains `per_request_layer_rank_footprint.csv`, `grouping_results.csv`,
`grouping_layer_detail.csv`, `modality_characterization.csv`, policy/manifest/
integrity JSON, summary JSON, and the three figures. The analysis entry point
is `poc_flashvep/complementary_rebatch_oracle/analyze.py`.

**Next single action:** do not start a GPU rebatching prototype; if this line is
revisited, first obtain a new artifact with a larger, arrival-order-preserving
request trace and test whether the <3% exact-R headroom survives realistic
window coverage before considering any runtime implementation.
