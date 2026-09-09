# Surprise log

1. **The largest early “vision grouping” gain was CPU submission overlap.** A
   barrier after preprocessing removed it.
2. **Global equality did not imply DP equality.** Rank-stride assignment made
   P0 27.5% imbalanced by prompt tokens, enough to reverse rankings.
3. **LM-length sorting becomes actively harmful at batch 128.** It is 46.3%
   slower than the warmed global baseline despite exact global and DP totals.
4. **Attention and MoE do not conflict in the measured regime.** The same
   single/multi policy wins both in all three observer restarts.
5. **The apparent hierarchy problem collapses after headroom accounting.** The
   full optimistic oracle median is only 1.37% direct BCT.
6. **Grouping can change long greedy continuations.** First-token agreement is
   almost complete, but 23/128 fixed-16 sequences differ after autoregressive
   propagation; those runs cannot be used for a speedup claim.
