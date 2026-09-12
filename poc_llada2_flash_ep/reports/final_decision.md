# Final decision

## Label

**CHARACTERIZATION-ONLY**

The 100B substrate establishes true capacity-relevant EP4 and several useful
scaling facts, but no novel candidate has a credible request-level oracle of
8% or more after strongest-static, prior-art, fixed-cost, and feasibility
controls. The strongest novelty-eligible bound is 5.34%.

## Required questions

1. **Official topology:** dense/routed/shared TP4, EP1. The official four-GPU
   command is not true sparse EP.
2. **True EP4 fit:** yes. Dense TP4 plus routed EP4 fits at 56.86 GiB/rank,
   with 64 complete experts/rank and no CPU offload.
3. **Topology comparison:** the executable faithful paths are TP4 and the
   bounded routed-EP4 bridge. TP2×EP2 is unsupported in this runtime. TP/EP
   winner changes across configurations and restarts; no robust global EP win.
4. **Economic crossover:** nominal EP gains appear at submitted batch 8+, but
   change sign across restarts. With each topology's best static granularity,
   TP4 is faster at batch 16; no stable crossover is established.
5. **Low versus high batch:** low batch is startup/fragmentation dominated;
   moderate waves amortize it; oversized waves regress. The dominant fix is
   the existing mini-batch knob.
6. **Denoising phase:** yes structurally—live/physical rows fall from 82.88%
   early to 13.31% late—but physical MLP cost remains. This is Epoch's space.
7. **Temporal persistence:** coarse load is highly stable (lag-1 cosine
   0.9942), while exact top-k sets are only 29.11% stable. This predicts state
   but does not create a large removable-cost oracle.
8. **Strongest oracle:** fresh-work compaction is 25.00% but collides with
   Epoch. The strongest eligible candidate is perfect load balance at 5.34%.
9. **Training-free and TEAM-independent:** the evaluated oracles are, but none
   is large enough to promote.
10. **Complementary to Epoch:** no large one was found. Exact residuals are
    below 5.34%.
11. **Faster backend:** the replication residual falls from 2.99% to
    2.24/1.50/0.75% as communication cost becomes 0.75/0.5/0.25×.
12. **EP8 follow-up:** not justified as a method validation. It may be useful
    for deployment characterization on a quieter isolated node, not for the
    present paper direction.

## One-sentence reason

After correcting the static microbatch baseline and excluding Epoch's already
claimed liveness compaction, the largest credible new E2E upper bound is only
5.34%, below the 8% research gate.
