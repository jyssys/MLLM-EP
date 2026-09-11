# Candidate oracle competition

All percentages below are request-level **perfect upper bounds**.  They are not
measured policy gains.  Bounds assume zero selector overhead and, where stated,
future information.

| Candidate | Best setting | Perfect E2E oracle | Gate | Why it does not advance |
|---|---|---:|---|---|
| A. Cost-aware re-ranking | TEAM | 4.01% | kill | removes all optional-row communication while unrealistically holding NFE fixed |
| B. EP-budgeted work | REFLEX | 3.83% | kill | maps 6.05% fewer pairs onto the full MoE share; port quality already fails |
| C. Utility / marginal EP cost | TEAM/REFLEX | 4.01% | kill | bounded by the same physical work mass |
| D. Topology-aware action selection | DES | 3.90% | kill | generous elimination of one of four rank contacts; fanout actually stayed four |
| E. TEAM EP-aware width | TEAM | 4.01% | kill | assumes no NFE/acceptance penalty from reducing speculation width |
| F. Perfect future joint decode–EP | TEAM | **6.28%** | weak | prior measured early-commitment bound; future knowledge and no verification cost |
| All EP communication disappears | LLaDA | 15.60% | impossible ceiling | required transport; not a policy candidate |

## Candidate E detail

TEAM's EP4 trace contains 227,328 selected branch assignments.  Relative to 32
rows × top-8 for each of 528 calls, 40.56% are optional speculative rows.  The
clean fused EP2 bound for eliminating every dispatch+combine span is 9.90% of
request time.  Even granting that this EP communication fraction transfers and
that every optional row can be removed without increasing NFE gives only
`9.90% × 40.56% = 4.01%`.

This upper bound is below the 5% implementation gate.  A live width-1/2/3/4
sweep would also conflate a tuning knob with a research method and can only be
worse once lost acceptances are counted, so it was not implemented.

## Communication durability

The TEAM-width bound scales to 4.01/3.01/2.01/1.00% when communication cost is
multiplied by 1.0/0.75/0.5/0.25.  The opportunity is therefore not durable to a
2× faster Epoch/DeepEP-like communication path.

## Overlap

Perfect removal of all EP communication is at most 9.90% on the faithful fused
TEAM EP2 residual and 15.60% on the LLaDA reference EP4 anchor.  No independent
per-request slack was demonstrated: router→dispatch→expert→combine is a strict
dependency, and moving it under speculative work changes the TEAM decision
trajectory.  The previous timeline screen found no independent overlap window
above 5%.  No overlap prototype was justified.

## Decision

No A–F candidate reaches 8%; none reaches the 12% strong gate; and no candidate
generalizes with quality-valid evidence to two policy families.  Per the
working contract, no complex controller or runtime method was built.
