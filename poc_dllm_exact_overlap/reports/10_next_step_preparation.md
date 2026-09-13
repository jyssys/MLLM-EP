# Next-timestep input-independent preparation

## Source audit

DeepEP buffers and static workspace persist after initialization. What remains
on each denoising step is dominated by input-dependent work:

- router logits and grouped top-k;
- exact destination counts and offsets;
- received-row layout and per-expert counts;
- the combine handle tied to that dispatch;
- expert output storage and reverse mapping.

Receive pre-posting and asynchronous events are already part of the DeepEP
execution contract. Recreating static descriptors every step was not observed as
a separately measurable critical-path component. Exact next-step routing cannot
be prepared while step `t` runs because step `t+1` hidden state and selected
experts depend on the current decision.

## Oracle

Making all identified input-independent recurring preparation free does not
expose a measured component approaching 5% E2E. The candidate is killed before
prototype. Speculative or stale route preparation would no longer be exact and
is outside this PoC.
