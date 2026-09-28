
# Baseline readiness

Status: **PASS_REUSED_VALIDATED_FOUNDATION**

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- dtype / shape: BF16, d=2048, m=768, 48 sparse MoE layers, 128 experts, top-k=8
- TRUE EP4 path: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank
- TRUE EP2 path: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank
- frozen source cohort: `poc_mllm_mixed_domain_online_calibration/artifacts/manifests/frozen_domain_pool.json`

The Part-1 public quality baseline and the mixed-domain EP2/EP4 path proofs were
integrity-checked and reused. Full MMStar/ChartQA evaluation was not repeated.
This PoC collected new branch-output and functional intervention evidence only;
no simulator latency is used.
