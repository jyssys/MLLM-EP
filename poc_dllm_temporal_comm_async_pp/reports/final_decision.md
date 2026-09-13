# Final decision

## Track A

**NO-GO**

The temporal premise is verified—late-phase lag-1 same-destination hit exceeds
91% and hidden activations are similar—but the calibrated payload-sensitive
request oracle is only 0.37--0.72%.  Codec overhead makes net gain
non-positive.  CompactFusion also makes the broad delta-communication framing
non-novel.

## Track B

**CHARACTERIZATION-SIGNAL**

The ideal PP ceiling is large (about 2.13x request speed), boundary stability is
strong near layer 8, and periodic synchronization can restore coarse bounded
scores.  However, the best policy maintaining baseline score on both tasks has
only a 1.184x worst-task analytical upper, large sequence divergence, and
substantial NFE drift.  No live PP speedup was measured, and the runtime lacks
an exact PP request substrate.

## Overall recommendation

**Stop both as primary directions; do not combine them.**  Track B is the only
one worth retaining as a characterization artifact, not as a promoted method.
It should be reopened only if an exact LLaDA2 PP runtime becomes available and
a larger evaluation demonstrates a common >=1.3x-at-99% Pareto point.

Evidence boundary:

- observed actual: clean EP4 BCT, routing/cacheability, codec cost, DeepEP
  payload surface, boundary similarity, final trajectories, benchmark outputs and
  scores, NFE, PP group/weight capacity;
- analytical only: PP speed ceilings and NFE-adjusted policy speed upper;
- not observed: compressed DeepEP E2E, exact sequential PP E2E, live async PP
  speedup.
