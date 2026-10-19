# Fixed-feature attention summary screen

Candidate: C12. Evidence: `ANALYTIC_BOUND` with executed rational constructions.
This screen does not use pretrained Ouro states or GPU kernels.
It does not establish a latency or task-quality result.

## Mechanism under test

The candidate stores a query-independent summary of committed keys and values.
It then answers later queries through a fixed number of linear features.
This could reduce state traffic only if the summary stays small and accurate.
Shared recurrent weights do not, by themselves, provide such a summary.

[Performer v4](https://arxiv.org/html/2009.14794v4), Section 2.2, distinguishes the approximate feature-map attention in Equation 4.
Section 2.3 expresses softmax through a random-feature expectation.
A finite sampled feature map is not automatically exact softmax.
This is a generic neighboring mechanism, not a new Ouro result.

## Universal exactness counterexample

Choose keys `k_j=j` and possible queries `q_i=i*log(2)`.
The unnormalized kernel has entries `A_ij=2^(i*j)`.
Its columns form a Vandermonde matrix with distinct bases `2^j`.
Therefore its determinant is nonzero for every finite N.
Its rank is N.

The CPU script checks N=2 through N=8 with exact rational elimination.
It also checks the corresponding Vandermonde determinant product.
Positive row normalization preserves rank, so the softmax matrix also has rank N in this construction.

Suppose an exact implementation factors all these query responses through m fixed linear features.
The resulting attention map has rank at most m.
It therefore cannot represent this construction when m is less than N.
One constant-size linear summary does not support arbitrary history lengths and arbitrary value vectors under that universal claim.

The queries describe possible reads of one committed historical set.
They need not represent simultaneous ready Ouro tokens.
The argument concerns the summary's response family, not available model concurrency.

## Limits

This result does not reject every nonlinear data structure, distribution-specific representation, or approximate feature method.
It does not prove a lower bound for every attention algorithm.
It does not characterize the rank of actual Ouro query/key distributions.
It is a real-arithmetic argument, not a native BF16-kernel equivalence proof.

A useful approximation still needs mode-matched free-running quality and a measured critical-path opportunity.
The earlier small attention fraction belongs to its measured predecessor regime.
This report does not transfer that fraction to every context or model.

## Evidence and decision

The valid artifact is `artifacts/attention_summary_rank_v2.json`.
The script is `scripts/attention_summary_rank.py`.
The initial artifact retains an invalid source-window end timestamp.
Its metadata correction file identifies the problem and replacement.
The rational numerical results agree across both executions.

Disposition: reject the universal exact fixed-feature claim.
No approximate attention engine is justified by this screen.
No new parallelism or achieved speedup is claimed.
