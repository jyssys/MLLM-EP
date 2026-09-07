# Candidate 2: engine-owned cross-request slack

## Counterfactual

vLLM's worker owns stream switching, DeepEP handles and model execution while
the engine supplies a step descriptor. A counterfactual engine-level contract
would expose communication debt/ready state and let independent requests use
otherwise idle slack without changing model math.

## Why it was considered

It is a boundary between scheduler and distributed runtime, and could lead to
deadline-aware admission, slack lending, or queue-credit methods. This is more
structural than choosing `num_sms`.

## Adversarial result

The common-state and turnover experiments move both A/B arms together. The
surviving request-level effects are <=2.5%; large effects are wave-completion
semantics. Existing serving systems already study request admission/locality,
and the current source does not expose a high-mass state signal. Without a
direct oracle, a GPU patch would be speculative.

**Status: UNKNOWN / DEFERRED, not promoted.** The next decisive measurement is
a no-hook request trace with allocator/KV/clock state joined to the critical
path; it is not authorized by this branch's analytical gate.
