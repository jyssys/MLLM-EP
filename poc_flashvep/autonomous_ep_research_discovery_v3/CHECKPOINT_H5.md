# Checkpoint H5 — 2026-09-06 20:30 KST

- Wall interval since first fresh run: ~3 h 10 min.
- New live controls: text warmup, text→vision, text↔vision alternation,
  text-only shape alternation, high-resolution vision matched warmup,
  Qwen3 dense replication, low-latency backend startup probe.
- Strongest positive: text→vision transition keeps T_MoE around 2.4 ms vs
  1.20 ms after matching warmup; alternating text/vision stays around 2.12 ms.
- Strongest negative: text-only shape alternation is normal (1.21–1.25 ms),
  and matching high-resolution vision warmup removes the penalty. This is a
  shape/state preconditioning issue, not a new orthogonal scheduling variable.
- Open anomaly: GPU DVFS ramp and multimodal cache/stream state are correlated;
  no clock-controlled evidence is available. Dense Qwen3 fixed-text is also
  consistently ~2.1 ms, but needs a matched protocol before any general claim.
- Live GPU time: all completed controls used physical GPUs 1–4 only; GPUs
  0/5/6/7 were not touched.
- Next plan: update aggregate atlas, residuals, prior-art/triviality gate and
  final report. No additional large experiment is justified unless it can
  separate DVFS from shape state without changing the validated runtime.
