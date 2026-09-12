# Rank-local refinement-view oracle

## Exactness boundary

Each physical EP rank runs a no-communication approximate view using only its
resident experts, with local top-k and renormalization. The generated request
never consumes that approximate result in the diagnostic: the exact global EP
forward runs immediately afterward and remains the sole source of target
tokens. This isolates proposal quality and real draft cost without silently
changing the baseline output.

An agreement signal alone cannot make an approximate token exact. Any live
method must still use an exact global verifier, and E2E benefit requires that
one verifier validate multiple local refinement steps. The oracle therefore
charges dense attention, local expert work, proposal scoring/exchange, the
global verifier, and fallback; it also labels independent one-step survival as
an optimistic proxy rather than a proven trajectory verifier.


## Measured result

One exact EP4 refinement forward was paired with one observer-only rank-local
draft forward for 72 refinement states. Dense attention and the rest of the
decoder remain necessary on every rank-local view, so the local draft cost was
not small: 66.3%, 69.7%, and 68.4% of the exact global forward for local
top-k=1, 2, and 4 respectively.

| Local top-k | Rank-0 logit cosine | Rank-0 top-5 overlap | Four-view union hit | 4/4 coverage / precision | 3/4 coverage / precision | Next accepted-position match |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.537 | 3.51% | 3.75% | 0% / N/A | 0% / N/A | 18.40% |
| 2 | 0.763 | 25.84% | 31.32% | 0.55% / 61.64% | 4.40% / 50.83% | 32.99% |
| 4 | 0.835 | 45.89% | 54.21% | 5.95% / 88.86% | 20.88% / 56.03% | 46.18% |

The best high-precision signal—four-rank agreement at local top-k=4—covers only
5.95% of masked positions. Three-rank agreement increases coverage but has only
56.0% precision. The four-view proposal union misses the exact token in 45.8%
of cases even at local top-k=4. Thus physical rank partitions are not adequate
independent refinement views, and agreement cannot safely avoid a global
forward at useful frequency.

**Decision: KILL (0% feasible request gain).** K=1 always adds a draft before
the same verifier. K>1 is considered in the separate multi-step oracle, but the
high draft cost and low validity probability already violate the primary gate.
