# H4 — Cross-Request Complementary EP Batching

## Verdict

**NO-GO.** Early signatures are numerically stable, but combining complementary signatures does not lower the calibrated routed-MoE stage by a meaningful amount.

## Setup

- The online signature uses only refinements 1–2; no future refinement information enters grouping.
- Groups never contain two blocks from the same request.
- Random, similar-hot, and complementary groups consume the same held-out set, so total work is fixed.
- Pair size 2 and group size 4 are evaluated under simulated EP4.
- Three random, three similar, and three complementary full-row states were replayed through true EP2.

## Signature predictability

| Metric | Result |
|---|---:|
| Early signature → future rank-load cosine P50 | 0.9999992 |
| Shuffled-history cosine P50 | 0.9999401 |
| Mean cosine advantage over shuffled | 0.000198 |
| Future critical-rank accuracy | 98.93% |

The raw stability is high, but shuffled unrelated blocks are nearly as similar. This means the signal largely reflects a global EP-load shape rather than a distinctive request/block signature that a scheduler can exploit.

## Simulated EP4 grouping

| Group size | Complementary vs random | Similar-hot vs random | Interpretation |
|---|---:|---:|---|
| 2 | **0.098% faster** | 0.114% faster | no expected ordering; below noise/headroom gate |
| 4 | **0.044% slower** | 0.027% slower | no benefit |

Random pair cost was evaluated across ten seeds. Its P50 was 127,026 ms aggregate, with P90 127,156 ms; policy differences are smaller than random grouping variation.

## True-EP2 selected-state replay

Mean selected-state routed-MoE stage (`dispatch + critical expert + combine`) was:

| Policy | Mean stage |
|---|---:|
| Random | 2.0963 ms |
| Similar-hot | 2.0939 ms |
| Complementary | 2.0921 ms |

Complementary is directionally 0.20% below random, but similar-hot is also 0.11% below random. The physical replay therefore validates that all three are effectively tied, not a complementary-batching mechanism.

## AR-MoE control and interpretation

Load-aware batching is generic MoE. Repeated dLLM refinement would be useful only if its early signature supplied incremental, block-specific information and at least 5% routed-stage reduction. Neither condition holds. A serving scheduler should not be implemented from this result.
