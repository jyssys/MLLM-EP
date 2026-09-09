# Hierarchical schedule oracles

## Evidence contract

Clean runs provide BCT. Observer runs provide same-device CUDA durations for
vision, attention, router, DeepEP dispatch/expert/combine and whole LM forward.
Nested stage sums are not equated with wall-clock BCT. For each observer
restart, the calculation selects the policy with minimum measured
`vision + llm_forward` as its static module reference, computes only the module
work that could disappear under exact regrouping, and divides that saving by
the same restart's recorded GPU BCT.

The 128-request matched pool contains 45,893 actual prompt tokens, 47,600,528
pixels and 128 images. Every policy uses exactly the same requests and global
marginals. DP prompt-token loads are 22,947 versus 22,946 at batch 128.

## Oracles

- P3 BatchGen-style LM-only: encoder and attention keep a common grouping;
  MoE may regroup after attention.
- P4 independent stage: vision may regroup, while attention and MoE share one
  LM grouping.
- P5 full hierarchy: vision, attention unit and MoE independently choose their
  measured best grouping.
- P7 feasible lower bound: P5 plus the ideal HBM copy-time lower bound for two
  BF16 hidden checkpoints. It is deliberately favorable to the proposal.

| Restart | Static ref | Vision winner | Attention-unit winner | MoE winner | P3 direct oracle | P4 direct oracle | P5 direct oracle | P7 lower bound |
|---:|---|---|---|---|---:|---:|---:|---:|
| 0 | single/multi | text/image | single/multi | single/multi | 0.00% | 1.37% | 1.37% | 1.35% |
| 1 | single/multi | global | text/image | single/multi | 1.69% | 4.57% | 6.42% | 6.40% |
| 2 | single/multi | single/multi | text/image | single/multi | 0.00% | 0.00% | 0.50% | 0.47% |
| **median** | — | — | — | — | **0.00%** | **1.37%** | **1.37%** | **1.35%** |

The ideal checkpoint for one global BF16 hidden tensor is 187,977,728 bytes
(45,893 × 2,048 × 2). At an optimistic 1.5 TB/s it takes 0.125 ms; real
metadata, allocation, ordering and scheduler waiting can only reduce P7.

## Incremental value

- Encoder/vision value over BatchGen-style LM-only: median **1.37%**.
- MoE-specific value over independent vision/LM stage batching: median
  **0.50%**.
- Full-hierarchy optimistic direct BCT oracle: median **1.37%**, maximum
  **6.42%** across three observer restarts.

All values are below the 8% no-go threshold. P7 is below the 10–12% prototype
gate. No production scheduler, native BatchGen port or Kimi validation is
permitted by the working contract.

Machine-readable values are in `HIERARCHICAL_ORACLES.csv`.
