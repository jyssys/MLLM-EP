# dLLM MoE EP temporal opportunity PoC

## Executive decision

Both research directions are **NO-GO** on the measured official dInfer +
LLaDA-MoE four-H100 substrate.

The most useful scientific result is a separation between predictability and
economic value. Consecutive denoising iterations are exceptionally similar
(t+1 expert-load cosine 0.9607, rank-load cosine 0.9967, hot-expert retention
82.0%), so a causal predictor can recover 61.6% of the best H=8 replica oracle.
Nevertheless, the costed oracle is only 0.746% of clean request time. Even an
impossible unlimited, zero-copy replica system is bounded at 2.63%.

Request rank vectors are similarly predictable but not complementary: median
pairwise cosine is 0.9995 and 29/30 requests share the same dominant rank. A
future-aware batching search produces 0.166% optimistic E2E, while the causal
EMA policy produces 0.049%. Even fractional perfect rank equalization—stronger
than any request scheduler—has only a 2.36% request upper bound.

## What was built and measured

- An isolated branch and dInfer environment; no changes to the existing MLLM
  PoCs.
- A source/runtime audit proving that stock four-GPU dInfer has sharded experts
  but no A2A dispatch manager.
- A two-hunk dInfer patch and initialization-order correction producing a real
  TP1/DP4/EP4 `NaiveAll2AllManager` path with 16 experts/rank.
- Low-overhead route identity and stage timing for
  `(request, block, iteration, layer, expert, rank)`.
- Three clean and three traced restarts, all-pair 12 MiB P2P copy tests, temporal
  characterization, replication leases, batching policies, and 16 figures.
- Unit tests for work conservation, ownership, complementary selection, and
  copy-cost accounting.

## Key numbers

| Result | Value |
|---|---:|
| Logical route records | 3,792 |
| Clean/trace exact output matches | 90/90 |
| Trace observer tax | 12.64% |
| Whole-MoE p50 | 1.221 ms |
| Expert p50 | 0.329 ms |
| Expert share of clean request sum | 14.33% |
| One expert | 12 MiB |
| P2P copy p50 | 0.05547 ms |
| Visible overlapped copy cost | 0.04684 ms median |
| 1A H=8 perfect / causal | 0.746% / 0.459% E2E |
| 1A impossible absolute upper bound | 2.6266% E2E |
| 1B future-aware / EMA | 0.166% / 0.049% E2E |
| 1B impossible fractional upper bound | 2.3557% E2E |

## Artifact map

- `environment_snapshot.md`: pinned software/hardware/runtime configuration.
- `runtime_audit.md`: stock-path falsification, true-EP patch, communication
  proof, and correctness caveat.
- `temporal_routing_characterization.md`: traces, stage distributions,
  persistence, diversity, and observer tax.
- `predictive_transient_replication.md`: 1A cost model and absolute kill bound.
- `rank_complementary_batching.md`: 1B policies and absolute kill bound.
- `prior_art_boundary.md`: adversarial primary-source comparison.
- `final_decision.md`: independent decisions, evidence boundary, limitations,
  and recommendation.

The raw result root is `results/dllm_moe_ep_20260911_151500/`.
