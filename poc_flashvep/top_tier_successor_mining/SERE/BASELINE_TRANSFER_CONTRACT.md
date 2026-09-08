# SERE confirmatory transfer contract

Primary confirmatory table: `quality/sere_fineweb_400x128_official_norm/similarity.pt`,
400 FineWeb-Edu sequences ×128 tokens, original BF16 Frobenius norm arithmetic.
Earlier FP32-accumulation pilot remains labeled exploratory. No held-out answers
are used to choose similarity values or thresholds.

The EP port uses the unchanged official CUDA rerouter, original top-8 weights,
top-S primary union and similarity threshold. It neither drops tokens nor merges
duplicate assignments. Primary union covers the complete **DP-local scheduled
batch**, before TP sequence partitioning. TP route AllGather restores the union
after sequence parallelism. Independent DP engines do not share primary sets.
This is an explicit deployment adaptation from the paper's single-device setup,
not an unmentioned global-EP union or an algorithmic contribution.

Quality comparisons use natural EOS and the same short-answer instruction,
official full-prediction task scoring, identical held-out IDs, paired policies,
and clustered confidence intervals by image. HF replica/padded-cohort results
are mathematical transfer probes, not original vLLM evaluation reproductions.
The actual vLLM/DeepEP runs are needed to confirm both quality and direct E2E.

Two timing contracts must stay separate:

- Natural completion: user-visible request E2E/TTFT/ITL, answer quality and
  output length. Extra generated work is counted, not silently truncated away.
- Fixed output length: decode step/kernel efficiency only. Ignore-EOS changes
  the stopping contract and cannot establish quality-matched natural E2E gain.

Trivial-fix attacks are S=4 and rho=.7. If they recover quality, quantify their
retained original speedup before calling the underlying failure nontrivial.
Small-batch quality alone is not a failure of the paper's intended batch regime.
