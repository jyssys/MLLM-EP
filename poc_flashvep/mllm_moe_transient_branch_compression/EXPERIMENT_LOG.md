# Experiment log

## 2026-09-10 — initialization

- Created and read the working spec on branch `flashvep/mllm-moe-transient-branch-compression-poc`.
- Verified that only an owned utilization process occupies physical GPUs 4–7; left it running during CPU-only audit.
- Audited the installed vLLM MoE path and existing branch-output capture.
- Started a streaming CPU pilot over the previous 24-image raw capture, explicitly excluding pre-image control tokens from the Text control.

## 2026-09-10 — live capture and exact replay

- Two initial smoke launches used the generic conda environment and stopped
  before model execution because the DeepEP package was absent.  Retried in the
  pinned `flashvep-deepep-v020` environment; no method conclusion uses the
  failed launches.
- Successful full-token smoke: one real image, layers 4/12/24/36/44/47,
  TP2/DP2/EP4 DeepEP HT.  Four worker proof files independently identify
  `DeepEPHTPrepareAndFinalize`, `DeepEPHTAll2AllManager`, `TritonExperts`, EP
  ranks 0--3, and physical visible set 4--7.
- Successful full capture: nine real images, three content categories, three
  image edges, all prompt tokens, six layer strata.  Stock output remained
  unchanged and both DP drivers completed with identical one-token greedy IDs.
- Exact checkpoint replay on physical GPU 4 validated that captured raw expert
  outputs are not instrumentation reconstructions: 27 sample/layer rows have
  median rel-L2 0, worst p99 rel-L2 `1.56e-8`, and minimum cosine 0.9999983.
- Exact centroid/router-weighted-centroid evaluation found lower error than an
  existing-token medoid but no median strict pass for any grouping/cap.  Even
  cap-2 contiguous groups have affected-token combined-update rel-L2 about
  32--48% for centroid variants.

## 2026-09-10 — quality propagation controls

- Ran three matched HF quality-only diagnostics in parallel on physical GPUs
  5/6/7: impossible output-oracle sharing, contribution skipping, and spatial
  contiguous sharing at a 5% visual-row budget.
- At layer 44, all four 8-token greedy outputs matched, but the output oracle's
  next-route agreement was only 97.83% and logit rel-L2 was 6.85%; the
  preregistered 99%/1% propagation screen failed.
- Across layers 4/12/24/36/44/47, all eight short greedy outputs again happened
  to match, while output-oracle logit rel-L2 rose to 23.00% and route agreement
  collapsed to 41.93%.  Contribution skipping and spatial sharing were slightly
  worse but qualitatively identical.  This is a mechanistic quality failure,
  not evidence that eight short answers establish safe quality.
- Restarted the requested utilization workload on physical GPUs 4--7 after the
  live diagnostics.  CPU full-atlas, contribution, and stratified analyses
  continue while it runs.

## 2026-09-10 — stratified controls and final gates

- Completed 358,291-pair stratified analysis. Leave-one-request-out regression
  shows only 0.034% median held-out RMSE improvement from route Jaccard after
  hidden distance and layer/expert/policy controls; router weights add 0.419%
  and spatial distance 0.035%.
- Among 14,727 visual pairs with route Jaccard >=0.75, median output rel-L2 is
  0.606 and no pair meets the 1% strict branch tolerance. H09/H10 therefore do
  not provide a missed deployable selector.
- Closed 27 causal questions and completed the four required forced-rethink
  checkpoints. Full-output sharing, sub-expert sharing, contribution weighting,
  layer subsets, exact route packing, codebooks, and cheap repair all fail an
  independent geometry, quality, algebraic, prior-art, or direct-E2E gate.
- Applied the spec's permitted early-stop rule. Additional GPU repetitions,
  Kimi transfer, and a one-invocation DeepEP prototype are not justified after
  median output-oracle reduction 0%, maximum 0.0283%, and a 3.26% ceiling for
  eliminating every expert computation.
