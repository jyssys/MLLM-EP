# TEAM: single GPU → EP2 → EP4

## Result

TEAM is the one faithfully reproduced positive control.  Its algorithmic
benefit is real on one GPU, survives weakly on EP2, and reverses on EP4 in the
exact reference EP implementation.

| Setting | Baseline | TEAM | TEAM speedup | Evidence |
|---|---:|---:|---:|---|
| Single GPU | restart-matched | restart-matched | **1.832× median** | three official-code restarts |
| EP2 fused-local control | 3.936 s | 3.338 s | **1.179×** | true EP2; one bounded fused control |
| EP4 restart 1 | 2.466 s | 2.735 s | **0.902×** | clean |
| EP4 restart 2 | 2.632 s | 3.674 s | **0.716×** | clean |
| EP4 restart 4 | 2.632 s | 3.205 s | **0.821×** | clean |
| EP4 median | 2.632 s | 3.205 s | **0.821×** | restart is statistical unit |

A third attempted TEAM run hung in NCCL progress for more than five minutes
after its paired baseline completed.  The task-owned process group was killed;
the run is explicitly excluded, not imputed.

## Algorithmic work versus EP work

| Metric | Single | EP2 | EP4 |
|---|---:|---:|---:|
| Baseline → TEAM NFE / forward change | 24→14 (−41.7%) | 24→16 (−33.3%) | 14→11 (−21.4%) |
| Token-expert assignments | **+11.5%** | **+31.5%** | **+32.14%** |
| Remote assignments | N/A | **+33.61%** | **+32.11%** |
| Remote hidden bytes | N/A | **+33.61%** | **+32.11%** |
| Destination fanout | N/A | 2→2 | 4→4 |
| Mean active local experts | N/A | 28.28→23.48 | 14.79→13.12 |

TEAM's temporal/spatial policy reduces the number of model invocations and
local active-expert events, but its four-way speculative exploration creates
larger physical forwards.  On EP4, 40.56% of TEAM's selected branch rows are
above a hypothetical 32-row non-speculative forward.  Routes already touch all
four ranks, so smaller expert support does not reduce rank fanout.

## Stage direction

The EP4 structural trace is observer-heavy and not a latency headline.  It
nevertheless localizes the mismatch:

- dispatch sum changes only +2.1% despite +32.1% payload;
- expert sum drops 21.2%, consistent with fewer calls and fewer active experts;
- combine sum rises 49.3%;
- Python route preparation becomes grossly observer-sensitive.

The faithful conclusion is not that EP communication alone explains every
millisecond of the reversal.  It is that reduced NFE is traded for larger,
fully fanned-out distributed forwards, and that trade crosses from beneficial
to harmful between single GPU, EP2, and EP4 on this reference substrate.

## Quality boundary

The previous official single-GPU suite had GSM8K 3/4 vs 3/4 at 128 tokens and
the 256-token boundary pair 2/2 vs 2/2.  One 128-token HumanEval completion was
truncated under TEAM (2/4 vs 1/4), while the longer boundary rerun aligned.
EP2 preserved the selected-expert/trajectory semantics in the bounded check.
EP4 outputs in the 32-token timing anchor have matching semantic prefixes but
are too short for answer accuracy; this report does not claim EP4 benchmark
quality from that anchor.

## Answer to the TEAM questions

1. Speedup degrades sharply with EP degree: 1.832× → 1.179× → 0.821×.
2. Speculation amplifies remote assignments by roughly one third on EP2/EP4.
3. EP4 traffic is amplified, but amplification itself is similar to EP2; the
   larger topology exposes more startup/synchronization sensitivity.
4. Fanout saturates rather than grows beyond the baseline: 4 ranks in both.
5. Expert work falls; combine/reference-layout overhead and enlarged forwards
   erase the benefit.
6. This pathology is specific to the policy's optional speculative work
   creation, not merely to fixed required routing.
