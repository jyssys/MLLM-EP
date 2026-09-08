# MoDES — paper audit

Final reproduction/status update: official calibration and live transfer are
complete within REPRODUCTION.md's boundary;pending wording below is historical.

[Paper](https://arxiv.org/abs/2511.15690), CVPR 2026. Main text and appendix read.
Status: PAPER UNDERSTOOD; baseline calibration and transfer pending.

## Exact intended algorithm

Calibrate output KL after skipping each MoE layer, separately by modality in the
released code. Normalize importance across layers; multiply by each selected
expert's normalized routing weight. Drop assignments below tau_text or tau_visual,
without renormalizing retained weights. Search threshold pairs along a monotone
skipping frontier to minimize answer-distribution KL at a target skip ratio.
Original settings: 1024 GQA samples, 100 rectified-sigmoid grid points.

## What has already been shown

Three MLLM families, 13 image/video tasks; Qwen3-VL and Kimi are original baselines,
not novel transfers in themselves. General vision-vs-text tolerance and early-layer
sensitivity are the central contribution; they cannot be repackaged as a successor.
Dataset/size/grid ablations exist. Qwen's 97.33% figure is **average task-score
retention**, not raw accuracy and not 97.33% on every task. ChartQA, for example,
falls from 85.08 to 78.84 at the aggressive operating point in the paper itself.

## Efficiency scope

Performance evaluation uses 8 H200, speed evaluation one H200. The fast router
filters invalid assignment sentinels before grouped GEMM; it is not equivalent to
zeroing weights after executing every expert. Public code labels the Qwen path
simulated and advertises the fast CUDA implementation as pending.

## Untested assumptions worth controlled screening

- Additive one-layer calibration must rank importance after many layers are jointly
  modified; frozen-route monotonicity does not automatically establish monotonicity
  after downstream hidden states/routing change.
- One global modality/layer factor is sufficient across answer requirements and
  real task composition; larger or domain-specific calibration is the first attack.
- Assignment skip fraction predicts the quality/speed trade-off under distributed
  EP, where communication and rank occupancy may persist after skipping.

These are questions only. No failure is established by source inspection.
