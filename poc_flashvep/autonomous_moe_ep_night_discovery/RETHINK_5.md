# Forced research rethink after five distinct tests

## Current evidence

- H01 measurement trust: deferred nonblocking observation costs about 3.27%
  E2E, so absolute cross-process comparisons below 5% are unsafe.
- H03 DP work partition: same total work, balanced versus skewed token sizes,
  about 0.2% direct wave effect.
- H04 phase placement: same request multiset, wave effect about 1%; sign is
  not stable enough to call a penalty.
- H05 idle/dummy participation: an early 5.7% local signal collapsed to about
  1–2% after long randomized sampling; sampled MoE stage and request wave
  effects disagree.
- H06 output-length churn: equal total requested decode tokens but uniform
  versus heterogeneous per-request output lengths currently gives about 90%
  wave-makespan difference while request median E2E differs by under 1%.

## Assumption audit

I was repeatedly assuming that a large sampled MoE or wave-level effect is a
user-visible request-latency opportunity. That assumption is false unless the
same logical request or throughput metric carries the excess. I was also
assuming that DP-idle cost would be visible without measuring the RPC dummy
path, which the first wrapper missed.

## New non-cosmetic children

| ID | New question | Decisive control | Why it is not cosmetic |
|---|---|---|---|
| H37 | Does equal aggregate decode work with heterogeneous per-request lengths create a real throughput/completion-spread cost even when median request E2E is unchanged? | same exact prompt pool, output-token sum, compare uniform/heterogeneous; report last-completion, tokens/s, p99 request | tests whether H06 is a scheduling mass phenomenon rather than a MoE-stage artifact |
| H38 | Does per-layer Python-list→CPU→GPU expert metadata copy account for a recurrent fraction of the 48-layer step? | instrument `ExpertTokensMetadata.make_from_list`; compare host/device copy spans and a storage-reuse diagnostic | source exposes an explicit TODO and it occurs at every MoE layer |
| H39 | Does phase placement alter Triton/DeepEP kernel regime through batch-shape history, explaining H04's small sign-reversed effect? | same work and route, sample kernel identity and shape transitions | distinguishes a kernel-regime interaction from DP communication imbalance |
| H40 | Are attention and MoE tails a shared whole-GPU resource event rather than EP-specific debt? | matched current shape, co-tail test plus text-only/dense attention control | determines whether EP-specific method space exists |
| H41 | Is the H06 wave effect caused by streaming response completion semantics rather than backend execution? | compare server step/token throughput and request-level completion under same output sum | prevents promoting an API-observation artifact |

## Next priority

Complete H06 with at least 100 randomized waves, then test H07–H12. Promote H37
only if throughput and p99 request effects survive; instrument H38/H39 in a
separate worker restart; keep H40 as a generic-runtime falsification control.
