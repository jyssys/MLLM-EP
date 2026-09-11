# Final decision

## Label

**EP2 NO-GO**

## One-sentence reason

Although physical work scaling changes the dispatch/expert composition and
DeepEP HT improves clean request latency over Naive by 6.01%, real dInfer
refinement does not shrink the EP payload within a block, one backend wins all
active-ratio phases, and the zero-cost per-forward oracle adds only 0.144%
request-level benefit over best static.

## Gate summary

| Gate | Result |
|---|---|
| True EP2 | PASS: TP1/DP2/EP2, 32 disjoint experts/rank, remote dispatch/combine |
| Correct same-topology backends | PASS: bit-exact audit output and identical request tokens |
| Stage-composition change | PASS: expert share increases by ~10--11 pp from M=2 to 256 |
| Production backend | PASS: DeepEP HT real kernels verified |
| Meaningful backend crossover | FAIL: none over M=2--256 or early/middle/late |
| Real denoising traverses physical M regimes | FAIL within block; M changes only with cache/block window |
| Perfect dynamic MoE-local oracle | FAIL: 0.174% |
| Amdahl/request oracle | FAIL: 0.144%, optimistic and zero switching cost |
| Simple active-state policy | FAIL: degenerate always-HT, 0% oracle recovery |
| Switching cheap enough | FAIL economically; source path is construction-time and oracle is already negligible |

## What changed and what did not

Measured runtime-cost reduction:

- DeepEP HT versus Naive: 6.01% clean median request latency improvement.

Logical work reduction:

- None. Routing, top-k, expert computation, and dense model positions are held
  fixed.

Excluded opportunity:

- Skipping already-resolved positions could be large, but that is the
  Epoch-style live/fresh-work problem, not communication-backend adaptation.

## EP2-to-EP4 transfer risk

The following observations likely transfer qualitatively: fixed startup costs,
expert share increasing with payload, and dense dInfer forwarding of resolved
positions. None of the quantitative backend conclusions can be promoted to
EP4: peer count, experts/rank, fanout, message size, imbalance, HT/LL support,
and crossover must all be remeasured.

Because this is negative at EP2, no EP4 work is recommended for this specific
adaptive-backend direction. If EP4 is later tested for another reason, the
minimum decisive check is current-runtime HT versus a working LL/AGRS path on
the physical M values produced by an actual fresh-worklist runtime.

## Recommendation

Do not implement a backend switcher or threshold controller. Use DeepEP HT as
the best measured static request path on this pinned substrate. Revisit only if
both prerequisites change: (1) a runtime exposes a physically sparse fresh
worklist, and (2) EP4 shows a reproducible backend crossover whose zero-cost
request oracle exceeds 5%.
