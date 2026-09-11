# Epoch / TEAM / REFLEX / DES work-removal audit

## Bottom line

The largest measured opportunity—stop running stable decoded positions through
route/dispatch/expert/combine—is not residual novelty.  Epoch explicitly makes
the fresh token-expert list the physical EP payload, and TEAM's DCD also caches
decoded positions.  A successor must therefore remove work *inside Epoch's
fresh lane* or remove a distinct hot-path cost.

## One-to-one work matrix

| Work item | Stock dInfer EP2 | Epoch | TEAM | REFLEX | DES | Truly residual after Epoch? |
|---|---|---|---|---|---|---|
| Router/gating | all physical positions every forward | current gate/top-k for compact fresh positions; support only is cached | decoded positions bypass it; masked/newly accepted are gated, cold tokens may be rerouted | still computes router ranking; changes number of selected experts | router logits feed sequence coreset and constrained top-k | only fresh-lane gate work |
| Token-expert computation | all M×8 pairs | stable decoded cached; live/new/refresh fresh | decoded cached; hot speculative branches add work; cold work remains but is constrained | selected-pair count reduced/reallocated by refinement state | top-k per token retained inside a smaller coreset | only fresh-lane pairs not safely reusable |
| Activated experts | natural top-k union | Expert Atlas bounds block-local support | LAC restricts cold tokens to experts activated by new/hot tokens | no explicit unique-expert objective | explicitly minimizes sequence-level unique-expert coreset | no, support/activation reduction is prior art |
| Dispatch payload | dense routed rows | FreshLane dispatch sends compact fresh rows only | paper/code target activation work; released reference is not an EP communication runtime | not optimized; selected-pair reduction may indirectly reduce payload | not its primary savings; same top-k pair count can remain | only fresh-lane payload |
| Combine payload | dense routed results | compact fresh results only, then merge cached decoded outputs | no source-level EP combine implementation in official repo | not directly optimized | not directly optimized | only fresh-lane payload |
| Routing/layout metadata | rebuilt every forward | caches block support, ownership and descriptors; hot path still updates liveness, compaction and fresh route | Python masks and expert masks rebuilt for computed tokens | per-token k allocation and top-k still built | coreset voting plus constrained routing added | hot fresh-plan delta only |
| Accepted/dead positions | physically recomputed | stable decoded skipped; newly decoded refreshed once; M=5 periodic refresh | decoded cached after delayed first committed computation; paper reports refresh-free | assigns role-dependent k; does not cache output | not a liveness cache | no, already Epoch/TEAM |
| Live positions | fully recomputed | gate, route and routed outputs stay current | hot/cold treatments modify work/route | expert budget is refinement-state aware | route constrained to shared coreset | yes, but only if a new safe redundancy exists |
| Inter-iteration reuse | none in tested cache-off path | block plan/support and stable-decoded MoE output cache | decoded hidden/KV state cache | confidence history informs allocation, not output reuse | no temporal output reuse | fresh-live output/metadata only |

## Epoch: adversarial reading

Epoch's relevant contract is narrower and more complete than “cache decoded
tokens.” It separates two clocks:

1. Stable block structure (expert support, sequence ownership, cache
   descriptors) is planned/cached.
2. Values that can affect a live decode decision are refreshed on the
   iteration clock.

Its Algorithm 2 defines stable decoded positions as the intersection of current
and previous decoded masks.  Newly decoded positions must be computed once
under their committed identity.  Outside a periodic full refresh (default
M=5), only live plus newly decoded positions enter the compact buffer.  Gate
logits, top-k, dispatch, expert compute, combine, and cache update all operate on
that same fresh list.  Stable decoded outputs are merged back into the full
logical shard before downstream work.

Therefore all of the following are already claimed by Epoch and excluded here:

- accepted/dead-position expert-output reuse;
- physical removal of those rows from dispatch and combine;
- expert-support reuse across iterations;
- a block-scoped plan and bounded refresh;
- compact decision/LM-head work for live positions.

The exact residual is: fresh-lane router values, fresh top-k choices, fresh
token-expert outputs, per-iteration live/compact indices, and any payload those
fresh pairs require.  Epoch's own invariant explicitly recomputes every live
position's router and routed-expert output. [Epoch paper](https://arxiv.org/abs/2609.09748)

No official Epoch repository was linked from the paper as of 2026-09-11, so
this is a paper-algorithm audit rather than a code audit.

## TEAM: paper and source agree on the coarse boundary

TEAM combines:

- Delayed Caching for Decoded Tokens (DCD): compute masked and newly accepted
  positions, then reuse older decoded representations;
- Speculative Exploration for Hot Tokens (SEH): explore multiple high-value
  candidate branches;
- Limited Activation for Cold Tokens (LAC): obtain the expert set from
  newly-accepted/hot tokens and constrain cold-token routing to that set.

The official source at commit
[`e9c502e`](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM/commit/e9c502e5753ce79f660371e2fb4a8666f66cae75)
sets `compute_mask = ~decoded_index`, copies cached hidden states for decoded
rows, runs gate/top-k only on `compute_hidden_states`, and optionally masks the
router to `necessary_experts` for limited tokens
(`modeling_sdar_moe.py:372-492`).  The driver constructs decoded and
expert-limit masks from the current block and prior confidence history.

Important evidence boundary: this released implementation is a reference
PyTorch expert loop, not a true-EP dispatch/combine manager.  It proves semantic
work selection, not physical multi-GPU payload elimination. [TEAM paper](https://arxiv.org/abs/2602.08404),
[official code](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM)

## REFLEX

REFLEX preserves the default router ranking but assigns a variable expert count
per token from refinement role and lagged confidence progress.  It therefore
removes/reallocates token-expert pairs and reports roughly 15% fewer selected
pairs.  The paper explicitly excludes attention, router scoring, shared
experts, and dispatch overhead from its pair-count metric.  It neither caches
fresh branch outputs nor reuses a dispatch plan.  It is direct prior art for
“refinement-dependent partial expert refresh” if that phrase merely means
changing k. [REFLEX paper](https://arxiv.org/abs/2608.01784)

No official repository was linked from the paper as of the audit date.

## DES

DES changes routing support rather than temporal freshness.  It constructs a
sequence-level expert coreset from per-token router saliency, then forces every
token's top-k selection to come from the coreset.  The goal is fewer unique
expert weight loads; top-k pair count can remain unchanged.  Its custom kernel
fuses coreset selection work.  It does not reuse the same token-expert output
between iterations. [DES paper](https://arxiv.org/abs/2602.00879)

No official repository was linked from the paper as of the audit date.

## Novelty boundary used in this PoC

Only these candidates were allowed to survive the audit:

1. exact/delta reuse of fresh-lane routing/layout state;
2. reuse of temporally stable branch values *within* the fresh lane;
3. layer/iteration-selective refresh of fresh-lane branches.

Changing expert count/support, caching stable decoded positions, or merely
compacting their dispatch payload was not counted as a new opportunity.
