# Decode-depth oracle

Status: **NOT_RUN because Stage 0 does not qualify scheduling claims**.

An oracle would use future token depths that do not exist at arrival time.
That information would define a retrospective upper bound, not an online policy input.
The spec requires a validated live CDB cost model before a realistic oracle claim.
Neither a depth sum nor a unique request-length statistic supplies that model.

| Planned policy | Evidence in this PoC |
|---|---|
| Round-robin, queue-aware, admissible length-aware, strong-cost placement | No live serving baseline |
| Perfect depth-only oracle | NOT_RUN |
| Perfect true-length-only oracle | NOT_RUN |
| Perfect depth plus true length | NOT_RUN |
| Depth-homogeneous grouping | NOT_RUN |
| Depth-aware joint microbatch optimization | NOT_RUN |
| Zero-cost migration bound | NOT_RUN |

| Cohort | Strongest generic baseline | Length-only gain | Depth-only incremental gain | Combined gain |
|---|---|---|---|---|
| Planned serving cohort | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |

The missing depth-specific gain is not a measured 0% gain.
The study does not claim that ordinary queue or length scheduling already eliminates it.
It also does not claim that local early-exit work savings establish extra cross-GPU scheduling headroom.

The intended action remains initial assignment before prefill.
Every request would retain one resident GPU throughout decode.
Future decoded tokens, true EOS lengths, and post-prefill hidden states would not become causal arrival features.
No simulator, oracle search, or scheduler implementation bypasses the Stage-0 gate in this study.
