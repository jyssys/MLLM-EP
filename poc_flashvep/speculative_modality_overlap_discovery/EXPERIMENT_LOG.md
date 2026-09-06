# Experiment log

| Time | Experiment | Live GPU | Result | Decision |
|---|---|---:|---|---|
| 2026-09-06 | Branch/spec audit; previous functional hook inspection | 0m | Existing 24-image Qwen3-VL capture and analyzer identified | Fresh capture required |
| 2026-09-06 | GPU process hygiene | 0m | Only old burn worker was terminated; GPUs 1-4 idle | Safe to launch bounded capture |
| 2026-09-06 | DeepEP live serving smoke (conda env) | 0.5m | vLLM initializes but `has_deep_ep=False`; no claim made | Switched to validated DeepEP venv |
| 2026-09-06 | DeepEP live serving without hook | 0.5m | `DeepEPHTAll2AllManager` and `DeepEPHTPrepareAndFinalize` active; no raw tensors | Hook timing issue isolated |
| 2026-09-06 | Fresh Qwen3-VL partial-output capture | 2.0m | 6 real images, 3 layers, 108 raw files; exact reconstruction valid | Partial/mass quality sweep |
| 2026-09-06 | Partial top-m and router-mass sweep | 0m | top1–top7 fail; mass95 fails text; mass99 retains all 8 | Speculative overlap gate killed |
| 2026-09-06 | Output-affinity/spatial diagnostic | 0m | Same-expert mean improves but remains outside quality gate; spatial worse | No finalist |
