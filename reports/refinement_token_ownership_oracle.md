# Refinement-coupled token ownership oracle

Labels are `SIMULATED-EP4-EP2-CALIBRATED` and `SIMULATED-EP8-EP2-CALIBRATED`. O1/O5 use future information. O2/O3 use only the first one/two refinements. Expert compute is held fixed.

| target | O1 current-block bytes | O1 whole bytes | O2 current-block bytes | O3 current-block bytes | O2 whole-stage gain |
|---|---:|---:|---:|---:|---:|
| EP4 | 3.628% | 0.060% | 1.464% | 1.684% | 0.000% |
| EP8 | 3.662% | 0.060% | 1.244% | 1.528% | 0.006% |

Even the impossible full-future oracle saves only about 3.6% of current-block remote rows, and about 0.06% after prompt/prior-row dilution. Early policies save about 1--2% of current-block bytes, far below the 15% suggested signal and before migration.
