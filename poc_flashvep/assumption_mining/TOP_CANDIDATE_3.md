# Candidate 3: state-aware workspace/communication contract

## Counterfactual

DeepEP buffer ownership, vLLM KV state and CUDA allocator state are managed by
separate lifecycles. A co-designed contract could reserve/reuse workspace or
schedule communication based on resource state.

## Adversarial result

Common regime transitions were observed, but no intervention-specific request
mass was isolated. Metadata outliers reach 13.9 ms only rarely, and the
communication-SM lower envelope is <=2.10%. Simple preconditioning/buffering is
an engineering fix unless a repeated >=15% request-level mass is found.

**Status: UNKNOWN / DEFERRED.** This remains a possible future observability
project, not a paper candidate in the current evidence set.
