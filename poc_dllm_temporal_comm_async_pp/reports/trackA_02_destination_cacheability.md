# Track A2 — destination cacheability

The cache key used by the oracle is
`(request, block, position, layer, destination-rank)`.  A prior activation is
valid only when the same logical row reaches that destination again.  Exact
expert identity is recorded separately because same-rank/different-expert is
still transport-cacheable but requires distinct expert routing metadata.

| Lag | GSM8K same rank | GSM8K same expert | HumanEval same rank | HumanEval same expert |
|---:|---:|---:|---:|---:|
| 1 | 84.65% | 78.80% | 81.03% | 72.80% |
| 2 | 74.68% | 66.61% | 68.86% | 59.21% |
| 4 | 57.51% | 47.50% | 51.32% | 40.93% |

For a periodic full refresh K, the effective cacheable fraction is bounded by
the lag-1 hit rate multiplied by `(K-1)/K`.  At K=16 this is 79.36% on GSM8K
and 75.96% on HumanEval.  Because FP8/INT8 delta is half the BF16 payload, raw
dispatch-byte saving is at most 39.68% and 37.98%, respectively—not 2x over the
whole request and not a combine-byte saving.

Rank changes, first occurrences, block changes, and periodic syncs are full
refreshes.  Request completion invalidates all entries.  Cache storage for a
full production design was not allocated because the economic gate failed.
