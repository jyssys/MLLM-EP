
# Baseline readiness

- Status: **PASS_REUSED_VALIDATED_FOUNDATION**
- Model revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- Shape: BF16, d=2048, m=768, E=128, K=8, 48 MoE layers.
- Measured tensor shape: gate/up `[128, 2048, 1536]`, down `[128, 768, 2048]`.
- Exact BF16 expert bytes: **9,437,184 bytes (9 MiB)**.
- EP2 evidence: TRUE `TP1-DP2-EP2`, 64 experts/rank, `DeepEPHTPrepareAndFinalize` + `TritonExperts` on physical GPUs 2,3.
- EP4 evidence: TRUE `TP2-DP2-EP4-PP1`, 32 experts/rank, the same DeepEP sparse path on physical GPUs 2,3,6,7.
- EP8 scope: `STRUCTURAL_EP8_PROJECTION_ONLY`; no latency, TPOT, E2E, or speedup is inferred.

Part-1 public-quality status remains the frozen foundation status; this PoC did not rerun MMStar/ChartQA. No MoDES GMLG/DMT/global-layer/semantic score is used by any candidate policy.
