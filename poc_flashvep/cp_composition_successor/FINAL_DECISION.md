# Final decision

## Status

`PARTIAL_ENVIRONMENT_BLOCKED`

No track is promoted:

1. Track C has the strongest clean measurements but its 15.42% oracle is entirely a trivial fleet-concurrency rule; matched content does not change the decision.
2. Track A exhibits a 14.46% combined-versus-spec E2E signal and 4× KV capacity, but fails the required greedy correctness gate and directly overlaps active upstream work.
3. Track B's proposed zero-transition ownership is already implemented in current source and working single-axis runs show 0% clean transition mass. The combined topology remains environment-blocked in the newest driver-compatible runtime.

`ALL_THREE_NO_GO` would overstate evidence because PCP4+DCP4 never reached serving. `FOUND_PROMISING_HOLD` would overstate research value because every surviving measurable signal is killed by correctness, headroom, triviality, or upstream collision.

## What remains useful

- The fixed-fleet PCP crossover is operationally useful: all GPUs per request at concurrency one, independent replicas under concurrent load.
- DCP provides a real capacity lever, but the measured exact combined-spec path needs upstream correctness and performance work rather than a distinct successor paper.
- If revisited after a driver/runtime refresh, only Track B's combined correctness/runtime should be rerun; the zero-transition novelty premise should not be revived without evidence of new transition work.
