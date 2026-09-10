# Final decision

## NO-GO

The system bottleneck is real, and EP-tail-aware removal is a stronger
max-rank-work selector than semantic-only removal at the same assignment
budget. The research direction still fails because the selection advantage
does not survive the quality and direct-E2E gates.

| Gate | Result | Evidence |
|---|---|---|
| A — bottleneck reproduction | **GO** | Expert compute is 35.8--39.7% of clean TTFT at 8K/16K; optimistic load-tail mass is 9.5--10.6%. |
| B — heterogeneous top-k accuracy | **NO-GO** | Vision routing is flatter than text. In a nine-image all-layer diagnostic, 10% P3 retains the first token but only 6/9 four-token sequences, versus 9/9 for P2. |
| C — EP awareness | **Mechanism positive, quality-matched NO-GO** | At 10% budget, P3 reduces max-rank assignments 19.48% versus P2 8.12% across 54 fresh image-layer rows, but removes more route mass, has worse replay error, and causes earlier greedy divergence. |
| D — GPU replay/economic mapping | **NO-GO** | Three-content median P3 projection is 6.94% at 10% removal; 8.30% at 20% requires ~19.7% local relative-L2. |
| E — E2E integration | **Not entered by design** | Gate D did not justify a full variable-k runtime. |

## One-sentence conclusion

**Qwen3-VL on four H100s has a material MoE/EP bottleneck, but removing
vision-token expert assignments according to the current EP tail cannot turn
that bottleneck into a safe 8--10% request-level opportunity.**

## What was learned

1. Critical-rank-aware selection has real causal leverage: at 5% and 10%
   assignment budgets it gives 3.02x and 2.40x the max-rank reduction of
   semantic-only selection.
2. That leverage conflicts with semantic safety because the hot rank does not
   exclusively hold negligible branches.
3. The model's vision routes are less concentrated than its text routes
   (top-6 mass 78.72% versus 85.23%), undermining vision-only top-k reduction.
4. Large operator speedups appear only after local output error is already
   severe; the all-layer control further shows 10% P3 changes the four-token
   greedy sequence on 3/9 samples while P2 changes 0/9.
5. The remaining idea is crowded by MACS, MoDES, AnyExperts, and ReaLB and
   does not demonstrate a superior quality-matched systems tradeoff.

## Recommendation

Do **not** implement dynamic variable-k or run a large task benchmark for this
candidate. Do not market the route-oracle geometry result as a method. If the
broader bottleneck is revisited, preserve expert contributions and reduce
their hot-rank cost rather than dropping them; that would be a different
research question and must be checked against ReaLB and MACS first.

## Evidence boundary

- Observed request result: clean fixed-top-k TTFT only.
- Observed operator result: exact captured-layer DeepEP replay.
- Analytical result: count-proportional load-tail upper bound.
- Projected result: Amdahl-adjusted TTFT from measured MoE share.
- Observed correctness result: full-model first-token logits and four-token
  greedy agreement under weight masking; all eight experts still execute.
- Not performed: full-model variable-k performance, large benchmark accuracy,
  or Kimi transfer. These were correctly gated off rather than silently
  treated as positive evidence.
