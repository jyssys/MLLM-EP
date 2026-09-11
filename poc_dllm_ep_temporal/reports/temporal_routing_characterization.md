# Temporal routing characterization

## Dataset and observer tax

- 30 measured prompts after two warmup prompts; fixed prompt length 64,
  generation/block length 64, threshold 0.8, early stop enabled.
- Three independent clean restarts and three traced restarts: 90 measured
  request executions in each mode.
- 3,792 unique `(request, block, iteration, layer)` records and 11,376 raw rows
  across traced restarts.
- NFE/request: median 5, range 4–34.
- Clean request-latency p50 by restart: 189.52, 215.61, 187.83 ms.
- Traced p50: 213.95, 213.47, 213.16 ms.
- Median observer tax: **12.64%**. Route structure uses traced data; request
  economics use clean timing.

The intended target was 32–64 measured requests. GPU return shortened the
measured set to 30, so this dataset is slightly below the target. It is still
substantially more than a single-request persistence anecdote, and no positive
claim is promoted from it.

## Measured EP stage distribution

Medians are taken per logical record after first taking the median across three
traced restarts.

| Metric | p50 | p90 | p99 |
|---|---:|---:|---:|
| Dispatch | 0.604 ms | 1.111 ms | 2.115 ms |
| Expert | 0.329 ms | 0.473 ms | 0.509 ms |
| Combine | 0.230 ms | 0.378 ms | 0.419 ms |
| Whole MoE | 1.221 ms | 1.722 ms | 2.869 ms |
| Iteration wall | 37.909 ms | 40.613 ms | 50.740 ms |

The expert stage is 14.33% of the clean request-time sum under the deliberately
optimistic accounting used for the E2E upper bounds.

## Temporal signal

| Horizon | Expert-load cosine | Rank-load cosine | Same critical rank | Top-5% expert retention | Hot ≥1.5× mean retention |
|---:|---:|---:|---:|---:|---:|
| t+1 | 0.9607 | 0.9967 | 81.55% | 72.74% | 82.00% |
| t+2 | 0.9359 | 0.9945 | 75.53% | 65.15% | 75.90% |
| t+4 | 0.9313 | 0.9944 | 73.66% | 61.30% | 73.30% |
| t+8 | 0.9245 | 0.9939 | 72.16% | 57.15% | 70.29% |

The temporal premise is real: routing and physical rank load remain highly
persistent across denoising iterations.

## What kills the two systems opportunities

Persistence is accompanied by low request diversity. Aggregate request-rank
profiles have pairwise cosine p10/p50/p90 of 0.9986/0.9995/0.9999. Rank 3 is
dominant for 29/30 requests; the remaining request is rank-2 dominant. Thus
requests do not expose complementary physical vectors for 1B.

Max/mean imbalance is nonzero (median 1.203, p90 1.414), but it is not a measured
latency predictor on this fixed-size naive-EP path. After trimming the slowest
1%, `T_MoE ~ max_rank_assignments` has slope -0.000244 ms/assignment and
R²=0.0017. The empirical positive slope used for economic mapping is therefore
zero. A perfectly proportional expert-time model is reported separately only
as a hypothesis-favorable upper bound.
