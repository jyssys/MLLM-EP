# Physical EP2 / EP4 and topology scaling

- EP2: **TRUE_DEEPEP_EP2_MEASURED**, 48 balanced requests, 64 experts/rank.
- EP4: **TRUE_DEEPEP_EP4_MEASURED**, 160 requests, 32 experts/rank.
- Common EP2/EP4 expert histogram cosine: 0.4457 across 48 requests. Rank identity is topology-specific and is not equated across EP sizes.
- Bounded deterministic topology parity: exact token sequence 20.83%; exact output text 20.83%. No policy changes model semantics; differences, if any, are audited as topology/BF16 numerical behavior.

| physical evidence | P_separated expert cosine | rank cosine | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|
| EP2 physical discovery subset | 0.7020 | 0.9925 | 0.0487 | 66.60% |
| EP4 physical heldout | 0.9560 | 0.9972 | 0.0212 | 100.00% |

Prior frozen same-question/same-token causal image-swap expert IDs were remapped without latency modeling:

| topology | changed expert | changed destination-rank | events |
|---|---:|---:|---:|
| EP2 | 5.30% | 2.11% | 31440 |
| EP4 | 5.30% | 3.32% | 31440 |
| EP8 | 5.30% | 4.12% | 31440 |

Rank-signature domain-separation gaps:

| evidence | P_text | P_full | P_separated |
|---|---:|---:|---:|
| ep2_physical_subset | 0.0011 | 0.0020 | 0.0021 |
| ep4_physical_heldout | 0.0049 | 0.0075 | 0.0076 |
| ep8_structural_heldout | 0.0110 | 0.0173 | 0.0167 |

This directly tests frozen H2/H3: finer expert ownership can turn more expert changes into rank changes. Physical EP2/EP4 prediction metrics are in `artifacts/ep2_ep4_scaling.json`; EP8 remains structural.
