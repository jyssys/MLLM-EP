# Track B4 — error propagation

The intervention establishes the following causal chain.

1. **Boundary error is non-trivial.** Lag-1 rel-L2 grows from about 0.08 at
   boundary 8 to 0.27/0.33 at boundary 24.
2. **Without synchronization it accumulates.** PP4 W0/no-refresh changes every
   GSM8K sequence and 87.5% of HumanEval sequences, increases NFE by 52/39, and
   drops bounded score from 5→3 and 6→1.
3. **Periodic exact refresh partially resets task error, not trajectory
   divergence.** W4/K2 restores bounded scores on both tasks, but only
   12.5%/37.5% of sequences are exact and HumanEval needs 14 more forwards.
4. **Late-only staleness is safer but has little mass.** W4/K4 late-only keeps
   both coarse task scores and changes NFE by +5/-2, yet its request speed
   upper is only 1.05--1.06x.
5. **Middle states are less forgiving.** Middle-only operation changes about
   69--94% of sequences and can increase NFE; the quality/latency trade is not
   monotonic in nominal phase.

These results show some dLLM self-correction at task level, but also the key
difference from image diffusion: stale hidden state changes discrete token
acceptance, which changes the number and identity of subsequent refinement
steps.  Current-step similarity does not predict final trajectory identity.

Accepted-token event counts were not exported for every policy.  NFE,
sequence identity, benchmark pass agreement, and final benchmark execution are
the available trajectory evidence; this limitation is retained in the final
decision.
