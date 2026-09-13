# Phase-aware intra-MoE pipeline

## Tested policy space

- Same-state dispatch(next) + expert(current)
- Same-state expert(current) + combine(previous)
- Same-state three-stage complete-wave pipeline
- Ordered cross-state dispatch/combined communication + expert compute

The same generic three-stage structure is positive across every sampled
physical state. Cross-state two-stage efficiency changes, but no different
phase-specific operation order wins. A three-stage cross-state scheduler was not
built because:

- the direct same-request schedule is illegal under denoising dependencies;
- the strongest generic independent-wave upper is 4.35--4.59% using sampled
  medians, and 4.84--6.73% even if the best observed saving is assigned to every
  wave;
- prior RAWS results show that manufacturing waves by splitting mini32 is
  counterproductive; and
- TBO/COMET/StreamEP/X-Stage already occupy the generic implementation space.

## Decision

The demonstrated incremental request-level benefit of a refinement-phase-aware
pipeline over the best generic pipeline is **0%**. This is not a claim that an
unmeasured controller can never alter throughput; it is the scope-correct gate
for this PoC. A new live controller would be unsupported by the measured
headroom.
