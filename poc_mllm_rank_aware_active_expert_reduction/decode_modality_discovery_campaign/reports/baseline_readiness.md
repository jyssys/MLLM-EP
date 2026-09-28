# Baseline readiness

- Status: **READY**.
- Base commit: `7e809abb60a8991961a17d71b0b458392bb01cfb`; model revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Model contract: BF16, d=2048, m=768, E=128, K=8, 48 layers.
- Physical evidence path: GPUs 2,3,6,7, TRUE TP2-DP2-EP4-PP1, DeepEP HT, 32 experts/rank.
- Rich traces and exact interventions use current checkpoint weights. Offline output-vector replay and structural rank accounting are not latency measurements.
