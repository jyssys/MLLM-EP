# Structural comparison of Token and Loop partitions

Evidence class: STRUCTURAL_ACCOUNTING. This section contains no measured speedup or measured FLOP count.

Let T denote prompt tokens, R recurrent loops, n P-side loops, and k P-side prefix tokens.
The remaining recurrent token rows are `T(R-n)` for Loop Split and `(T-k)R` for Token Split.
These counts are equal when `k/T = n/R`.

Equal recurrent rows do not imply equal causal-attention work.
A full causal prompt has `T(T+1)/2` query-key pairs per recurrent attention layer.
The D-side suffix has `T(T+1)/2 - k(k+1)/2` pairs per recurrent attention layer.
Loop Split assigns full-prompt attention to only R-n loops. Token Split assigns suffix attention to every loop.
Thus a suffix can contain a larger fraction of attention pairs than its fraction of token rows.

This accounting excludes Prelude, Coda, softmax details, kernel padding, and migration overhead.
It does not predict the critical path. Attention kernels can differ in arithmetic intensity, memory access, and launch cost.
The experiment must report actual device time and permit independent token-prefix tuning.
Equal split ratios alone are not equal-compute controls.

## Novelty limit

The attention difference is a generic depth-versus-token partition property.
It does not prove an advantage unique to recurrent weight sharing.
A logical-layer split at the same recurrent boundary is an equivalence control.
Both original PD workers already retain the full model, so shared weights do not automatically reduce residency in this comparison.

## Required measurements

- Report actual P and D execution costs at each candidate boundary.
- Separate recurrent rows from causal-attention pairs and measured kernel work.
- Compare transport costs at full-PD and partial boundaries.
- Test Decode SLO for each offload size.
- Do not convert structural row reductions into latency speedups.

The CPU accounting implementation is `poc_flashvep/fluidtoken_loop_split/split_accounting.py`.
