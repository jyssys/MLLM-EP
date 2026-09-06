# RETHINK checkpoint after fifteen live hypotheses

The fifteenth hypothesis in the campaign is H23 (one-image versus two-image
request composition), following the completed H03–H12, H14 and H15 blocks.
This checkpoint is intentionally conservative: the only large signal so far is
H06's last-completion spread under heterogeneous output lengths.

## Current assumptions challenged

- Multimodal composition is not automatically a systems variable: H15 and the
  one/two-image control must be interpreted through scheduled-token and request
  completion metrics, not image presence.
- A DP-rank request partition is not sufficient to create a distributed
  critical-path effect. Equal-work randomized controls H03–H05 have small
  request-level effects.
- More asynchronous work is not necessarily more useful work. The H06 wave
  makespan effect may represent completion semantics and queueing rather than
  a model-kernel opportunity.

## New questions from this checkpoint

1. **H42 queue-residence attribution:** when a request p99 grows, is the added
   time before the first model step, between serving steps, or inside a sampled
   MoE event? Join arrival/first-token/finish times to step cadence.
2. **H43 pinned versus unpinned DP:** does supported `X-data-parallel-rank`
   pinning itself create a persistent scheduling artifact compared with the
   same request pool left to the load balancer?
3. **H44 prompt reuse state:** equal shape and output work with repeated versus
   changing prompt content may exercise different allocator/graph/cache state;
   test only if the current API exposes a clean cache-off control.
4. **H45 metadata cadence:** measure the per-layer `make_from_list` host
   allocation/copy time and its correlation with host execute gaps. This is a
   source-derived tax candidate, not a claim that allocation is the root cause.

The campaign will continue through H28 and then run H36/H37 and these
source-derived diagnostics. A candidate must survive direct request-level and
critical-step metrics; wave-only effects are not promoted.
