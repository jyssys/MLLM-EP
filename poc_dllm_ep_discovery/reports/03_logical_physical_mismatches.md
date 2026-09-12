# Logical-to-physical mismatch analysis

## A. Liveness decreases faster than physical latency

| dataset | live reduction | expert latency reduction | dispatch reduction | combine reduction | whole measured MoE reduction |
|---|---:|---:|---:|---:|---:|
| GSM8K | 80.0% | 23.4% | 25.7% | 41.6% | 21.2% |
| HumanEval | 78.1% | 42.5% | 23.3% | 49.1% | 24.4% |

The mismatch is large, but it does not satisfy the predeclared example gate of `>=50% logical reduction and <=20% latency reduction`. More importantly, eliminating dead/live work is the core territory of Epoch, so this is context rather than a new candidate.

## B/C. Support and rows per expert

- Natural execution: active support falls late, p50 rows/expert collapses 15.75→5 (GSM8K) and 15→3 (HumanEval), and tiny groups grow strongly.
- Fixed M=1024: lower liveness modestly broadens support but does not grow tiny groups and changes whole-MoE latency by about 1% or less.

The initially attractive three-way paradox (`live rows down / support up / tiny groups up`) is therefore not supported after controlling physical M.

## D/G/H. Work count versus expert shape

Expert latency is not determined by pair count alone. Among adjacent within-dataset/layer/rank observations matched to within 5% pair count, 1,178 pairs differ by at least 10% in expert latency; the stored top 1,000 have median ratio 1.153, p95 1.461 and p99 1.927. The largest 2--3.5× cases resemble previously observed history-dependent runtime tails. They are not attributed to fragmentation.

The robust model result is cleaner: after dropping only group-p99 observations, count+layer gives 0.05347 ms leave-dataset-out RMSE; adding active-expert and row-shape statistics reduces it to 0.02929 ms (-45.2%) and raises R² from 0.9223 to 0.9767. Adding EP rank-CV/fanout/remote-fraction improves RMSE only another 0.47%. This establishes that local expert-row shape matters to expert kernel time, but not that a scheduler can cheaply change it.

## E. Confidence versus router diffusion

The proposed paradox is absent. GSM8K live-token confidence rises 24.7% early-to-late while router entropy falls 2.5% and top-k probability mass rises 21.4%. HumanEval follows the same entropy/top-k direction. Refinement confidence and router concentration improve together rather than diverge.

## F. Overlap without exact reuse

Exact reuse is not supported. No representative layer/phase has any matched row with MoE-output relative L2 below `1e-4`. Layer-1 output cosine is high (roughly 0.97--0.98) but relative L2 is still 0.13--0.19; middle-layer output cosine is only about 0.67--0.80 with relative L2 0.61--0.82. Exact-set route match is also only 5--18% on GSM8K and about 7% on HumanEval. Approximate reuse would require a quality method and collides with existing temporal-compute-allocation work.

## Decision on the discovered mismatch

The only predeclared strong-signal gate that passes is: **shape predicts expert latency materially better than work count**. The more novel opposite-direction and post-compaction-residual gates fail. This is sufficient to generate candidate oracles, not sufficient to justify an implementation.

