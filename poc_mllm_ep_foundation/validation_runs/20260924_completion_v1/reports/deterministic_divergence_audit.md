# Deterministic divergence audit

The original 12-request cross-runtime generated-sequence comparison remained
2/12 exact and was not overwritten. The frozen 192-request TP1-vs-TP2 control
is 90.104% exact-token and 96.875% parsed-answer agreement, with input metadata
hashes identical for every request. ChartQA deterministic accuracy is identical
(84.375%); MMStar differs by +1/96 for TP2. The paired overall delta is
+0.521pp, cluster-bootstrap 95% CI [-1.042,+2.604]pp.

The image pipeline had zero empty outputs and zero prompt leakage, and
blank/wrong-image controls changed the answer appropriately. A two-request
first-token gate matched stock TP4, true EP2, and true EP4 exactly. The
remaining full-sequence divergence is consistent with accumulated BF16/TP/EP
reduction-order and kernel differences, but margin-conditioned teacher-forced
layer localization remains incomplete. Status: NUMERICAL_PARITY_PARTIAL.
