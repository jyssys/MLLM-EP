# Ranked discoveries

## 1. Expert-row shape, not pair count alone, explains fused-expert latency

**Evidence strength: high; method headroom: low.** Across datasets, row-shape features reduce held-out expert-latency RMSE by 38.5% raw and 45.2% after p99 trimming. EP-level fanout/load features add less than 0.5% further RMSE reduction. This is a robust physical fact, but the current fused runner already groups a wave's rows by expert, leaving only 2.85--3.31% perfect E2E headroom.

## 2. Coarse routing geometry is stable while exact work identity is volatile

**Evidence strength: high; exact-reuse value: negative.** Rank-load cosine remains near 0.999 and destination-rank-set identity rises to about 74% late, while exact top-k set identity remains only 7--18%. This suggests delta metadata might be compressible, but not that expert computation can be reused. The entire router attribution is only 6.4--7.2% before implementation cost.

## 3. Live-row compaction exposes severe tiny-expert geometry

**Evidence strength: modeled counterfactual; novel headroom: low.** After hypothetical exact liveness compaction, late <=4-row fractions reach 68.5% and 79.4%. Yet perfect removal of the resulting fragmentation is only 2.12% and 1.13% clean E2E. This is a useful interaction result for Epoch-like systems, not a paper-level successor on this substrate.

## 4. Natural late-phase fragmentation is real but mostly a ready-pool effect

**Evidence strength: high falsification.** Natural tiny-expert fraction rises 20.7--37.6 points, but fixed M=1024 reverses the tiny-group direction. The originally suspected liveness-driven broad-support/tiny-group paradox does not survive the matched control.

## 5. Confidence/router mismatch is absent

**Evidence strength: high negative.** Token confidence rises while router entropy falls and top-k probability mass rises. No child method based on increased routing diffusion at high confidence is justified.

## Research implication

The system reveals a real cost-model mismatch, not a large removable-work mismatch. The best new perfect oracle is below 3.4%, and the post-Epoch residual below 2.2%. The correct classification is therefore `CHARACTERIZATION-SIGNAL`.

