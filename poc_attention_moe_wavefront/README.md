# Attention–MoE Wavefront Phase-0

This directory is intentionally independent of `poc_flashvep/`.  It contains
the Phase-0 measurement, oracle analysis, tests, and reports governed by
[`SPEC.md`](SPEC.md).

The study uses physical GPUs 4–7 only and does not implement a production
scheduler or a custom kernel.  A naive physical split is gated on a median
Amdahl-adjusted ideal TTFT oracle of at least 15%.

## Result

Phase 0 is **NO-GO**. The optimistic zero-cost ceiling is 15.47% median TTFT,
but empirical exact fragment costs are about 2.41x the unsplit stages,
modality-boundary O3 is only 1.59%, and the physical concurrent path is both
slower and numerically non-equivalent on the largest workload. See
[`reports/final_report.md`](reports/final_report.md).

The pure analysis can be reproduced in the local worktree from the preserved
worker traces with:

```bash
python -m poc_attention_moe_wavefront.scripts.analyze_baseline \
  --root poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500 \
  --runs \
    poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500/baseline \
    poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500/baseline_v2 \
    poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500/baseline_v3
python -m poc_attention_moe_wavefront.scripts.analyze_naive \
  --root poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500
python -m poc_attention_moe_wavefront.scripts.analyze_split_scaling \
  --root poc_attention_moe_wavefront/results/wavefront_phase0_20260910_214500
```

Worker-level event JSON and logits remain under the local result root but are
gitignored because they occupy roughly 260 MiB. The aggregate stage traces,
oracle tables, correctness summary, figures, and decision inputs are committed.
