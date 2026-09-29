# RQ3 — Layered Prefill on EP

Baseline type: **LAYERED_EP_PHYSICAL_REPLAY**. This is not native end-to-end Layered serving. Every active-layer datum is a physical full-prompt TRUE-EP4 DeepEP call; the contiguous groups are composed according to the pinned upstream scheduler.

The inspected upstream rule selects **4 stages × 12 layers** for 2,029 tokens. This contradicts the spec's stated eight-stage expectation; the code's `<=2048` comparison unambiguously selects four.

| Metric | Layered replay | CP512 physical |
|---|---:|---:|
| ExpertReloadAmp | 1.000 | 3.821 |
| injected expert-execution excess | 4.903 ms | 0.737 ms |
| mixed expert-execution total | 28.499 ms | 121.927 ms |
| injected MoE excess | 306.884 ms | 79.535 ms |
| MNCD dispatch | 34.472 ms | 29.232 ms |
| CIDD | 97.048 ms/stage | 37.794 ms/victim iteration |
| replay response peak inflation | 62.893% | measured 24.247% |

Reused physical size sweep:

| visual size | prompt tokens | full-prompt dispatch/layer | evidence |
|---|---:|---:|---|
| low | 175 | 1.347 ms | REUSED_MEASURED_TRUE_EP4 |
| medium | 721 | 1.979 ms | REUSED_MEASURED_TRUE_EP4 |
| high | 2029 | 3.431 ms | REUSED_MEASURED_TRUE_EP4 |

The comparison uses matched scheduling quanta: a Layered stage contains 12 measured full-prompt layer collectives, while CP512 reports the sum across all affected layers in its worst victim iteration. Per-layer CIDD remains available in the machine-readable artifact. RQ3 verdict: **LAYERED_LEAVES_EP_COLLECTIVE_GAP**.
