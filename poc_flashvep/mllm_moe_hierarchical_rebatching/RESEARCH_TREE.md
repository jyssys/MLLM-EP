# Causal research tree

All nodes below were tested with fresh GPU measurements or a diagnostic derived
from those measurements. Threshold variants are not counted as distinct
hypotheses.

## H01 — CPU preprocessing order masquerades as GPU grouping benefit — CLOSED

- EXPECTED: vision grouping improves GPU BCT.
- OBSERVED: the first harness interleaved rendering and submission; after a
  common pre-submit barrier, the apparent 7–8% advantage disappeared/reversed.
- FAILED ASSUMPTION: frontend work was outside the comparison.
- NEW SYSTEM FACT: offline BCT must begin only after the closed cohort is ready.
- BCT IMPLICATION: old smoke result excluded.
- NEXT CHILDREN: H02, H03.

## H02 — DP membership, not hierarchy, explains the large order effect — CLOSED

- EXPECTED: policies differ after equal global work.
- OBSERVED: rank-strided P0 assigned 16,627 vs 29,266 prompt tokens
  (max/mean 1.275). Actual-token balancing makes all policies 22,947 vs 22,946.
- FAILED ASSUMPTION: equal global totals imply equal DP critical work.
- NEW SYSTEM FACT: order can silently change DP membership.
- BCT IMPLICATION: unbalanced results are not hierarchy evidence.
- NEXT CHILDREN: H03, H04.

## H03 — Small warmup is sufficient for the 128-request execution shape — CLOSED

- EXPECTED: two small requests warm the relevant path.
- OBSERVED: independent b128 restarts showed policy CV up to 17%; two complete
  plan warmups changed winners and reduced some, but not all, variance.
- FAILED ASSUMPTION: kernel/JIT/allocator state transfers across shape scale.
- NEW SYSTEM FACT: full-shape warmup is mandatory.
- BCT IMPLICATION: first-case results excluded.
- NEXT CHILDREN: H12.

## H04 — Vision-only bucketing is a robust BCT win — CLOSED

- EXPECTED: grouping compatible grids reduces encoder waste.
- OBSERVED: P1 is best at b64 and in one un-warmed b128 run, but is 7.3% slower
  than P0 in the warmed primary run.
- FAILED ASSUMPTION: encoder-local efficiency dominates whole-job order.
- NEW SYSTEM FACT: vision benefit is workload/state dependent.
- BCT IMPLICATION: no robust direct gain.
- NEXT CHILDREN: H13.

## H05 — LM-length grouping is robustly beneficial — CLOSED

- EXPECTED: similar LM lengths improve packed attention.
- OBSERVED: it wins at b16 but is 46.3% slower at warmed b128 and 8.6% slower
  at fixed-16 generation.
- FAILED ASSUMPTION: length sorting improves continuous-batch admission at all scales.
- NEW SYSTEM FACT: a large sorted cohort can create adverse scheduler waves.
- BCT IMPLICATION: simple length buckets are not a safe hierarchy base.
- NEXT CHILDREN: H08, H20.

## H06 — Text/image split provides a general stage-compatible grouping — CLOSED

- OBSERVED: not best for max-token-1; best by 2.8% for fixed-16 and 5.4% for
  natural max-32, both below gate and both with multi-token output mismatch.
- FAILED ASSUMPTION: modality is the dominant grouping variable.
- NEW SYSTEM FACT: output phase changes ranking, but weakly.
- BCT IMPLICATION: incremental-only at best.
- NEXT CHILDREN: none; headroom <8%.

## H07 — Single/multi-image split helps only vision — REJECTED

- OBSERVED: it is the attention and MoE winner in all 3 observer restarts and
  the vision winner in one restart.
- FAILED ASSUMPTION: image-count grouping is encoder-local.
- NEW SYSTEM FACT: coarse request composition co-shapes LM scheduler steps.
- BCT IMPLICATION: one simple grouping already aligns attention and MoE.
- NEXT CHILDREN: H14.

## H08 — Optimal grouping is invariant to batch size — REJECTED

- OBSERVED: b16 winner is LM length, b32 global, b64 vision, and the b128 winner
  is not stable until full-shape controls.
- FAILED ASSUMPTION: one-dimensional batch size captures scheduler geometry.
- NEW SYSTEM FACT: grouping interacts with admission/chunking scale.
- BCT IMPLICATION: explains variation, but does not create module hierarchy headroom.
- NEXT CHILDREN: H20.

## H09 — Randomization is a strong robust baseline — CLOSED

- OBSERVED: random is competitive in some runs but has no stable advantage and
  observer stage totals vary substantially.
- FAILED ASSUMPTION: mixing alone regularizes all module shapes.
- BCT IMPLICATION: keep as a null control, not a method.
- NEXT CHILDREN: none.

## H10 — Total tokens alone predict BCT — REJECTED

- OBSERVED: all b128 policies contain exactly 45,893 prompt tokens and balanced
  DP loads, yet warmed medians range 1.129–1.651 s.
- FAILED ASSUMPTION: order does not affect internal scheduling waves.
- NEW SYSTEM FACT: sequence order/state matters beyond total volume.
- BCT IMPLICATION: real phenomenon, but a simple adverse-order warning rather
  than evidence for cross-module rebatching.
- NEXT CHILDREN: H20.

## H11 — Vision, attention and MoE prefer three distinct groupings — REJECTED

- OBSERVED: attention and MoE select single/multi in 3/3 observer restarts.
  Vision differs in only 2/3, with small direct incremental mass.
- FAILED ASSUMPTION: each module has an orthogonal compatibility objective.
- NEW SYSTEM FACT: LM attention and expert execution co-vary under the same
  scheduler-step geometry.
- BCT IMPLICATION: central causal chain fails.
- NEXT CHILDREN: H15–H18.

## H12 — Observer-heavy timing is a valid clean BCT source — REJECTED

- OBSERVED: per-policy observer stage CV is 15–45% and winners can differ from
  clean BCT; CUDA rows remain useful only for attribution.
- FAILED ASSUMPTION: detailed observation is free.
- NEW SYSTEM FACT: clean/observer evidence must remain separate.
- BCT IMPLICATION: no observer BCT used as headline speedup.
- NEXT CHILDREN: none.

## H13 — Vision-optimal grouping materially conflicts with LM-optimal grouping — CLOSED

- OBSERVED: choosing the vision winner in addition to the LM winner saves a
  median 1.37% direct BCT, maximum 4.57% in three restarts.
- FAILED ASSUMPTION: encoder compatibility has large critical-path mass after
  the strongest LM grouping.
- BCT IMPLICATION: below 8% no-go gate.
- NEXT CHILDREN: none.

## H14 — Expert routing adds independent hierarchy value — CLOSED

- OBSERVED: route HHI and rank max/mean correlate with diagnostic MoE time, but
  the full attention→MoE split adds a median 0.50% over independent stage batching.
- FAILED ASSUMPTION: route statistics imply schedulable direct BCT headroom.
- NEW SYSTEM FACT: route cost is real but aligned with the attention grouping here.
- BCT IMPLICATION: MoE-specificity threshold (5%) fails.
- NEXT CHILDREN: none; route-only rebatching is prior NO-GO.

## H15 — BatchGen-style attention→MoE combine has large residual headroom — CLOSED

- OBSERVED: P3 median oracle 0.00%, maximum 1.69%.
- FAILED ASSUMPTION: current b128 MoE batches remain too fragmented after simple grouping.
- BCT IMPLICATION: strongest generic LM baseline is already near the module lower envelope.
- NEXT CHILDREN: none.

## H16 — Independent vision/LM stage batching is insufficient — CLOSED

- OBSERVED: P4 optimistic direct oracle median 1.37%, maximum 4.57%.
- FAILED ASSUMPTION: stage-only systems leave a large internal LM conflict.
- BCT IMPLICATION: even the stage-only oracle has little work to remove.
- NEXT CHILDREN: none.

## H17 — Full hierarchy beats stage-only MLLM batching — CLOSED

- OBSERVED: P5 over P4 median incremental value 0.50%, below the required 5%.
- FAILED ASSUMPTION: expert grouping is orthogonal to attention grouping.
- BCT IMPLICATION: generic stage separation is the most defensible classification.
- NEXT CHILDREN: none.

## H18 — Full hierarchy beats BatchGen-style LM-only batching — CLOSED

- OBSERVED: encoder incremental value median 1.37%, below the required 8%.
- FAILED ASSUMPTION: an additional encoder boundary has material BCT value.
- BCT IMPLICATION: no MLLM-specific successor.
- NEXT CHILDREN: none.

## H19 — Yield memory/copy changes the decision — CLOSED

- OBSERVED: one global BF16 hidden state is 187.98 MB; even an optimistic HBM
  copy is 0.125 ms per boundary. Adding two lower-bound copies reduces the
  median oracle from 1.37% to 1.35%.
- FAILED ASSUMPTION: checkpoint cost is the limiting gate.
- NEW SYSTEM FACT: lack of removable work, not checkpoint bytes, kills the idea.
- BCT IMPLICATION: a real implementation can only be worse.
- NEXT CHILDREN: none.

## H20 — Output length makes hierarchy material — CLOSED

- OBSERVED: fixed-16 best-vs-global is 2.8%; natural max-32 is 5.4%.
- FAILED ASSUMPTION: decode-tail composition creates >8% benefit.
- NEW SYSTEM FACT: ranking changes weakly with output mode.
- BCT IMPLICATION: below gate; output correctness also fails.
- NEXT CHILDREN: H21.

## H21 — Reordering preserves exact greedy output — REJECTED

- OBSERVED: max-token-1 warmed run agrees for 128/128 requests, but fixed-16
  agrees for 105/128 full sequences (126/128 first tokens) and natural max-32
  for 107/128 sequences.
- FAILED ASSUMPTION: distributed BF16 execution is bitwise order invariant.
- NEW SYSTEM FACT: future work needs fixed-prefix logits/tolerance validation.
- BCT IMPLICATION: multi-token runs excluded from positive evidence.
- NEXT CHILDREN: none; performance gate already failed.

## H22 — A minimal prototype is warranted — CLOSED BY GATE

- OBSERVED: P7 feasible lower bound median 1.35%, max 6.40%.
- FAILED ASSUMPTION: the oracle reaches 10–12%.
- BCT IMPLICATION: implementation and Kimi promotion are forbidden by spec.
- NEXT CHILDREN: none.
