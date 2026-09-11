# Final decision

## POSITIVE-CONTROL-ONLY

TEAM's official positive direction reproduced and survived true EP2, but no
new large, independent, feasible residual opportunity survived the fused
substrate control and prior-art attack.

## Required questions

1. **Did TEAM's official trend reproduce?** Yes. Three clean bounded restart
   pairs achieved 2.335x, 1.714x, and 1.832x speedup (median 1.832x). NFE fell
   41.7% and active-expert events 50.8% in the structural trace.
2. **What differs from the official setup?** This is a bounded GSM8K/HumanEval
   subset on H100 FP16, not the full A100 four-benchmark OpenCompass run. The
   released generator/model code and settings are preserved; only the harness
   import surface is minimized. MATH and MBPP were not completed.
3. **Did benefit survive true EP2?** Yes. The reference TP1/DP1/EP2 path had a
   median 1.289x across three restart pairs and 1.551x at the warm paired point.
   The vLLM-fused control retained a smaller 1.179x advantage.
4. **How much physical work does TEAM remove?** It reduced model/MoE call count
   by 41.7% in Stage A and 33.3% in the matched EP2 trace, and active-expert
   events materially. It did **not** reduce every metric: speculative execution
   increased token-expert assignments and EP2 remote payload in captured runs.
5. **What is dominant after TEAM?** On the fused control, attention is the
   largest individually attributed stage (30.44%); whole MoE is 40.84%, split
   across expert 10.27%, router 9.69%, prepare 9.43%, combine 6.49%, dispatch
   3.41%, and wrapper residual 1.54%.
6. **Which costs scale and which remain?** Forward/call count, preparation, and
   collective launches scale down; speculative width can increase assignment
   and byte volume; router and per-call framework costs remain on surviving
   work.
7. **Is there an independent >=5% or >=10% oracle?** The best independent
   perfect oracle is early speculative-branch commitment at 6.28%. There is no
   credible >=10% oracle, and the 6.28% excludes prediction/verification cost.
8. **Is it novel against TEAM/Epoch/DES/REFLEX/TIDE/DICE?** No candidate combines
   material headroom and a clean gap. Route-plan reuse collides with Epoch,
   expert-budget/set changes with REFLEX/DES/TEAM, overlap with DICE/generic EP,
   and early commitment is adjacent to speculative dLLM work.
9. **Is EP4 validation justified?** No. EP4 could increase communication share,
   but the complete free-communication EP2 bound is only 9.90%, and no novel
   executable mechanism survived. That is insufficient evidence to spend an
   EP4 campaign.

## Evidence boundary

- Request speedups: clean timing.
- Stage shares: same-device CUDA events from observer-heavy runs.
- Fused control: one bounded prompt; systems-cost falsification, not a full
  quality benchmark.
- EP backend: reference NCCL sparse A2A, not production DeepEP.
- Candidate numbers: perfect upper bounds unless explicitly labeled measured.

## Recommendation

Keep this branch as a reproducible TEAM+true-EP2 positive-control substrate.
Do not pursue speculative branch reuse, route-plan delta reuse, or EP4 backend
optimization as a paper direction from these results.  A future study should
start from a new measured workload where a production backend leaves at least
10% *feasible* request-level residual, rather than extending these weak oracles.
