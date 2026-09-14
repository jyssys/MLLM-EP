# Kernel headroom oracle

## Definitions

- O0: fastest valid existing path per actual compacted shape. LL wins every
  nonempty shape, so this is effectively static LL.
- O1: zero control cost; keep only measured required remote payload movement.
  This is deliberately impossible/optimistic.
- O2: payload lower bound plus 20 us total irreducible issue/progress cost.
- O3: payload lower bound plus 50 us total (25 us/direction), capped by O0.
  This is the credible but aggressive RefineEP target—roughly 2.5x below the
  observed tiny-LL dispatch+combine kernel floor.

Request mapping removes the measured current dispatch/combine and expert
component, then adds compact-expert predictions plus the selected compact
communication path. This is a post-compaction analytical sensitivity, not a
measured Epoch or integrated RefineEP result.

## Request results

| task | current measured | post-compact normal | O0 existing LL | O1 impossible | O2 lower | O3 credible |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 5925.0 ms | 4882.8 | 4630.9 | 4427.3 | 4466.4 | 4524.9 |
| HumanEval | 7261.0 ms | 5979.6 | 5616.0 | 5369.4 | 5420.8 | 5498.0 |

| task | O0 gain vs post-compact normal | O1 extra vs O0 | O2 extra vs O0 | O3 extra vs O0 |
|---|---:|---:|---:|---:|
| GSM8K | 5.16% | 4.40% | 3.55% | **2.29%** |
| HumanEval | 6.08% | 4.39% | 3.47% | **2.10%** |

The total O3 reduction versus the current dense request is 23.63%/24.28%, but
that number mostly belongs to hypothetical liveness compaction and existing
LL. It must not be called RefineEP gain. The incremental third-path quantity is
only 2.29%/2.10%.

Even O1—the impossible path with no layout, launch, staging, synchronization,
or irreducible protocol cost—stays below the 5% kill gate on both tasks. A
heroic 25 us two-direction control floor gives only 3.34%/3.25%; at 75 us it
falls to 1.23%/0.95%.

## Decision

The requested >=12% CUDA promotion gate fails by more than 5x. The correct
label is `NO-KERNEL-HEADROOM`; implementing a fixed-contract CUDA path cannot
be justified by the measured envelope.
