# Dormant-Live EP Expert Refresh PoC

Discovery-first study of whether still-masked tokens that are far from their
future acceptance frontier carry removable routed-MoE EP work in
LLaDA2.0-Flash 100B.  All GPU evidence is restricted to physical GPUs 4--7.

The study separates three evidence classes:

- measured clean request/runtime evidence;
- observer-heavy structural and causal evidence;
- analytical oracles, including the post-Epoch residual.

No row-count or byte-count reduction is called a measured speedup.
