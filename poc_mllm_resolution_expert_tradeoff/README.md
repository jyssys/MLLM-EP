# Resolution / expert-count trade-off (Part 2)

This directory contains the training-free Qwen3-VL resolution-by-visual-expert
factorial PoC built on the Part 1 MLLM EP foundation.

- Final report: `reports/final_summary.md`
- Machine-readable summary: `artifacts/20260925_part2_v1/summary.json`
- Consolidated 1,440-result record: `artifacts/20260925_part2_v1/results.jsonl`
- Frozen candidates: `artifacts/20260925_part2_v1/frozen_policy.json`

The final verdict is `GENERIC_ONLY`: the resolution/expert trade-off exists,
but the EP-aware selector did not improve on the strongest generic control.
The Part 1 foundation remains reusable. Runtime wave directories and burner
logs are intentionally excluded; compact analysis, route replay, figures, and
reports are included.
