# Cross-state / phase-pair overlap matrix

The fresh diagnostic retained three real layer-16 states per workload and
replayed complete communication from one state beside exact owner-expert compute
from another. Each ordered pair has five warmups and 30 measured repetitions.

| dataset | pair family | median saving across ordered cross-state pairs | best saving | worst saving |
|---|---|---:|---:|---:|
| GSM8K | combine(A) + expert(B) | 0.060 ms | 0.120 ms | 0.032 ms |
| GSM8K | dispatch(A) + expert(B) | 0.054 ms | 0.068 ms | 0.049 ms |
| HumanEval | combine(A) + expert(B) | 0.076 ms | 0.109 ms | 0.030 ms |
| HumanEval | dispatch(A) + expert(B) | 0.061 ms | 0.072 ms | 0.043 ms |

All expert outputs match the isolated reference with minimum cosine
0.99999988 and relative L2 0. Cross-state shape clearly changes overlap
efficiency, so the resource profile is not perfectly stationary.

However, this is not the desired dLLM-specific win:

1. every pair still requires independent complete waves;
2. the best cross-state two-stage saving (0.120/0.109 ms) does not exceed the
   generic three-stage complete-wave medians (up to 0.140/0.186 ms);
3. the generic service upper is already below 8%; and
4. phase alone is not a sufficient shape key because physical rows are
   non-monotonic with iteration.

Therefore cross-state variation is a characterization signal, not evidence that
phase-aware pairing beats a resource-blind complete-wave pipeline. Raw medians
and p10/p90 are in `PHASE_PAIR_OVERLAP_MATRIX.csv`.
