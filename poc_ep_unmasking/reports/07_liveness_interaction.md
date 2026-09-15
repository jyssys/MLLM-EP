# 07 — Vanilla versus liveness-aware execution

Current vanilla dInfer performs one physical row per block position for each
ready request, including finalized positions. A representative measured
seven-request invocation used 224 rows despite some requests having fewer
MASK positions. Therefore earlier unmasking cannot be credited with a
fresh-row saving in the measured vanilla runtime.

An Epoch-like idealized live-only worklist would remove finalized positions.
However, with the required **same transfer count** and unchanged top-k=8,
the two alternatives leave the same number of positions live at the next
step. Their aggregate expert-assignment count is identical:

`A_(t+1) = 8 × L × |L_(t+1)|`, for `L` routed layers.

The identity of the live positions changes, so per-expert group sizes,
destination-rank loads, and remote fraction can differ. That is a
shape/composition opportunity, not reduction of total routed MoE rows or
useful expert FLOPs. The current one-swap rank-load screen and previous
matched EP4 replay find little economic room for this shape-only effect.

Epoch is a conceptual sensitivity baseline here, not a measured baseline or
implemented compaction path. Consequently the post-Epoch additional E2E
oracle and actual compacted speedup are **not measured**. Both must be shown
before any claim that early finalization eliminates physical EP work.
