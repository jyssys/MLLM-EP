# Timeline analysis

## First-order timeline

The instrumented fused TEAM path retains the dependency order:

`router → route preparation → A2A dispatch → local fused experts → A2A combine`

Across 768 MoE calls, summed same-device event time was 323.5, 314.8, 113.8,
342.8, and 216.6 ms respectively.  The event-based request was 2.48% faster
than the independent clean request, consistent with ordinary run variation; no
observer tax is treated as removable work.

This PoC did not capture a full Nsight Systems trace.  Therefore it does **not**
claim exact kernel-level idle-gap ownership, stream overlap, or NCCL kernel
residency.  The causal evidence is limited to same-device stage events and
actual A2A work conservation.

## Kernel-shape diagnostic

TEAM created three recurring decoder shapes:

| Physical rows | Calls | Mean decoder-layer time |
|---:|---:|---:|
| 32 | 192 | 3.869 ms |
| 64 | 48 | 3.823 ms |
| 128 | 528 | 4.267 ms |

The 4x row increase from M=32 to M=128 adds only 0.397 ms per layer on the
fused substrate.  This is the central economic fact: speculative branches add
logical work, but batched GPU execution makes their incremental wall cost much
smaller than their assignment count suggests.

Even a perfect future oracle that reduced all 528 M=128 layer calls to the M=32
mean would save only 209.7 ms, or 6.28% of clean request latency.  An actual
early-commit policy would also pay prediction, verification, and control costs.

## Cross-branch reuse diagnostic

Among 362,496 traced TEAM assignments:

| Relation | Assignments | Fraction |
|---|---:|---:|
| Same logical position and expert in another branch | 184,876 | 51.001% |
| Exactly identical expert input | 107 | 0.0295% |
| Expert input within 1% relative distance | 4,111 | 1.134% |
| Expert input within 5% relative distance | 13,310 | 3.672% |

Thus the tempting timeline hypothesis—execute identical routed work once and
fan it out—fails.  The routes repeat, but speculative hidden states almost
always differ.  Exact reuse has negligible mass; approximate reuse would become
a quality-changing method and still starts from a small close-input fraction.

## Interpretation

No large independent communication bubble or branch-serialization window is
demonstrated.  The largest observed avoidable block was the Python expert loop,
and an existing fused primitive removed it.  The remaining timeline supports a
positive TEAM substrate, not a new EP scheduling contribution.

Figure: `plots/12_team_fused_timeline.png`.
