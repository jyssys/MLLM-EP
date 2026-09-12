# Epoch-Like Compaction Sensitivity

This is offline sensitivity analysis only; Epoch was not implemented. The model
uses the measured decision-live ratio and the parent PoC's observer-light 53.06%
MLP share. Only the modeled MLP portion contracts; attention/other time remains.
For each compacted ready set, the same feasibility-aware partition oracle is run.

| runtime assumption | best dynamic gain over its own best static |
|---|---:|
| current dense-block runtime | 0.650% |
| hypothetical live-row-compacted runtime | **0.224%** |

The best compacted static policy is mini32. The phase aggregate is early mini16,
middle/late mini32, but per-wave dynamic selection saves only 0.224%. Compaction
does not reveal a latent RAWS opportunity; it makes large waves cheaper while
preserving the benefit of aggregation.

Evidence: [`DYNAMIC_ORACLE.csv`](../DYNAMIC_ORACLE.csv).
