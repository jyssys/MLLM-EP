# Layer × refinement-phase stability

## Representation result

There is no late-refinement identity region.

| task / phase | median layer rel-L2 update | median input/output cosine | median MoE/total update-norm ratio |
|---|---:|---:|---:|
| GSM8K early | 0.3519 | 0.9694 | 0.8370 |
| GSM8K middle | 0.3486 | 0.9672 | 0.8635 |
| GSM8K late | 0.3494 | 0.9639 | 0.8520 |
| HumanEval early | 0.3765 | 0.9628 | 0.8704 |
| HumanEval middle | 0.3587 | 0.9651 | 0.8661 |
| HumanEval late | 0.3570 | 0.9624 | 0.8702 |

Across all 192 `(task, layer, phase)` cells, zero layers had median relative-L2
update below 0.05. Absolute attention/MoE norms fall late because the ready set
shrinks, but the update relative to the current representation does not.
Routed and shared output norms remain of the same order throughout refinement.

The complete values are in `LAYER_PHASE_STABILITY.csv`; maps are:

- `figures/01_layer_phase_hidden_update.png`
- `figures/02_layer_phase_moe_update.png`
- `figures/03_layer_phase_attention_update.png`
- `figures/04_layer_phase_output_cosine.png`

## Logical phase state

Decision-live ratio collapses as expected:

| task | early | middle | late |
|---|---:|---:|---:|
| GSM8K | 0.904 | 0.445 | 0.181 |
| HumanEval | 0.783 | 0.459 | 0.172 |

This logical progress does not make the layer transformation closer to an
identity. The result is therefore not the desired `late -> small update -> safe
skip` chain. It also prevents treating hidden cosine as a necessity oracle.

## Similarity is not decision necessity

Spearman correlation between routed-MoE causal sensitivity and layer update
magnitude is weak: rel-L2 is 0.002 on GSM8K and -0.157 on HumanEval; MoE update
norm is 0.255 and 0.034. Output cosine is similarly weak (-0.088 and 0.152).
Thus even the modest geometric layer variation is not a reliable selector for
the decoding decision.

The correlations are in `STABILITY_DECISION_CORRELATION.csv`. The low-overhead
cost map is `LOW_OVERHEAD_LAYER_PHASE_COST.csv` and
`figures/07_layer_phase_latency_map.png`.
