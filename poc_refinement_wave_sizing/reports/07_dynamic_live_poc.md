# Live RAWS Prototype

**NOT RUN BY SPEC GATE.**

The feasibility-aware perfect dynamic oracle is 0.650%, below the mandatory 5%
controller gate and below the preferred 8--10% threshold. A live controller cannot
outperform its zero-cost perfect oracle. Implementing one would add shape
observation, repartitioning, synchronization, and possibly CUDA-graph/workspace
switching costs while risking quality changes from altered BF16 execution order.

There is therefore no measured live RAWS speedup and no claim of preserved live
dynamic output quality. The absence of a prototype is a gated negative result, not
an environment failure.
