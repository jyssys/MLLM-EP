# Frozen mixed-domain cohort

The cohort was selected and ordered before any routing result was read. It contains 160 image-hash-disjoint requests, 40 per domain, with a frozen 20/20 discovery/held-out split and frozen IID/domain-shift streams. Routing and answer correctness were not selection inputs.

| Domain | N | discovery / heldout | mean output tokens | median | max | >=64 | length-capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| visual_math_geometry | 40 | 20 / 20 | 154.1 | 192 | 192 | 40 | 21 |
| chart_plot_reasoning | 40 | 20 / 20 | 97.0 | 92 | 192 | 34 | 3 |
| structured_diagram_layout | 40 | 20 / 20 | 135.6 | 135 | 192 | 35 | 12 |
| general_visual_reasoning | 40 | 20 / 20 | 105.1 | 96 | 192 | 32 | 4 |

`structured_diagram_layout` is a semantically distinct MMStar diagram/code-sequence cohort because no local DocVQA/OCRBench asset was available. It is not claimed as an OCR benchmark reproduction.

## Realized balanced warm-up

- nominal 0% — chart_plot_reasoning: 0/40 (0.0%), general_visual_reasoning: 0/40 (0.0%), structured_diagram_layout: 0/40 (0.0%), visual_math_geometry: 0/40 (0.0%)
- nominal 1% — chart_plot_reasoning: 1/40 (2.5%), general_visual_reasoning: 1/40 (2.5%), structured_diagram_layout: 1/40 (2.5%), visual_math_geometry: 1/40 (2.5%)
- nominal 5% — chart_plot_reasoning: 2/40 (5.0%), general_visual_reasoning: 2/40 (5.0%), structured_diagram_layout: 2/40 (5.0%), visual_math_geometry: 2/40 (5.0%)
- nominal 10% — chart_plot_reasoning: 4/40 (10.0%), general_visual_reasoning: 4/40 (10.0%), structured_diagram_layout: 4/40 (10.0%), visual_math_geometry: 4/40 (10.0%)
- nominal 20% — chart_plot_reasoning: 8/40 (20.0%), general_visual_reasoning: 8/40 (20.0%), structured_diagram_layout: 8/40 (20.0%), visual_math_geometry: 8/40 (20.0%)

Natural generation used one frozen image-grounded reasoning prompt, deterministic decoding, EOS enabled, and `max_new_tokens=192`. Short-answer domains were not artificially extended.
