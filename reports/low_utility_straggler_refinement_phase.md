# Refinement-phase analysis

| target | axis | phase/bin | invocations | bottom-25 enrichment | difference (pp) | bottom-25 ATTRIBUTED_EXCESS share |
|---|---|---|---:|---:|---:|---:|
| EP4 | block_relative | early | 92,663 | 0.9731x | -0.5361 | 0.3390% |
| EP4 | block_relative | middle | 85,956 | 0.9725x | -0.6965 | 0.4333% |
| EP4 | block_relative | late | 98,515 | 0.9787x | -0.6394 | 0.5154% |
| EP4 | mask_ratio | gt75 | 72,390 | 0.9722x | -0.5305 | 0.3230% |
| EP4 | mask_ratio | 50_75 | 65,094 | 0.9780x | -0.5134 | 0.4035% |
| EP4 | mask_ratio | 25_50 | 66,709 | 0.9717x | -0.7729 | 0.4672% |
| EP4 | mask_ratio | le25 | 72,941 | 0.9784x | -0.6679 | 0.5306% |
| EP8 | block_relative | early | 92,663 | 1.0061x | +0.1201 | 0.3073% |
| EP8 | block_relative | middle | 85,956 | 0.9922x | -0.1972 | 0.3923% |
| EP8 | block_relative | late | 98,515 | 0.9718x | -0.8472 | 0.4654% |
| EP8 | mask_ratio | gt75 | 72,390 | 1.0064x | +0.1212 | 0.2927% |
| EP8 | mask_ratio | 50_75 | 65,094 | 1.0079x | +0.1834 | 0.3652% |
| EP8 | mask_ratio | 25_50 | 66,709 | 0.9813x | -0.5088 | 0.4224% |
| EP8 | mask_ratio | le25 | 72,941 | 0.9641x | -1.1073 | 0.4792% |

This phase view tests whether repeated dLLM refinement adds a pattern beyond generic MoE tail pruning.
