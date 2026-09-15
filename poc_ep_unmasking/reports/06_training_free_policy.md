# 06 — Training-free policy gate

P1 weighted confidence/EP score, P2 near-tie lexicographic, and P3 constrained
subset optimization were not implemented. No RL training was performed.

The discovery gate requires a causal, quality-safe full-trajectory oracle
before spending implementation effort. That gate was not reached. Thus the
fraction of oracle gain recovered by a future-free policy is **unknown**,
not 0%. The empty `UNMASK_POLICY_SWEEP.csv` row states `NOT_RUN_EARLY_GATE`.

If a future decoder configuration creates substantially more semantic slack,
P2 would be the smallest interpretable test. That is a different experiment,
not a positive result of this PoC.
