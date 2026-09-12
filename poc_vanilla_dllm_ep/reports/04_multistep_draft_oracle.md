# Verification-coalesced multi-step local drafting

This report is gated on the one-step rank-view diagnostic. K=1/2/4/8 estimates
must not be described as exact execution unless a bidirectional diffusion
trajectory verifier is actually implemented and validated. The initial oracle
uses measured local/global forward cost and compounds observed agreement
coverage and precision; this is deliberately favorable to the candidate.


The most favorable measured local view (top-k=4) costs 0.684 exact-forward
equivalents. Ignoring trajectory-verification implementation cost, K=4 leaves
only 0.263 global-forward equivalents of raw saving *if all four local steps are
valid*. The observed 4/4 agreement coverage×precision is 5.29% per step;
compounding this already-optimistic independent proxy gives a K=4 valid segment
probability of `7.83e-6`. The resulting perfect model-forward reduction is
`5.15e-7` (0.000052%). K=8 falls further to `1.17e-11` despite a larger raw
saving.

Using 3/4 agreement does not rescue the method: coverage×precision is 11.70%,
too small to amortize four local forwards, and its 56.0% precision is not an
exact acceptance signal. A real bidirectional trajectory verifier would add
cost and cannot improve this oracle.

**Decision: KILL.** No multi-step live prototype is justified.
