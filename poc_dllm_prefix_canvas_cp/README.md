# Prefix–Canvas CP discovery

Verdict: **NO-GO for the tested exact four-GPU CP prototype**.

Native attention/KV cost increased with prefix length. The communication-inclusive
candidate lost to optimized native attention and an exact local control.
Stage C did not run. No measured E2E improvement or quality result exists.

- [Report](report.md): measurements, estimates, limits, and stop decision.
- [Attention contract](attention_contract.md): actual native layout and cache semantics.
- [Prior boundary](prior_overlap.md): sources and novelty limits.
- [Reproduction](reproduce.md): commands and environment requirements.
- [Verdict JSON](results/final_verdict.json): machine-readable decision.
- [Completion audit](results/completion_audit.json): preservation checks and GPU state.

Only physical GPUs 4, 5, 6, and 7 ran experiments. The native baseline used
TP4/EP1/DP1/PP1/CP1. The component candidate reused those four ranks for CP.
It did not create an independent TP4 × CP4 group.
