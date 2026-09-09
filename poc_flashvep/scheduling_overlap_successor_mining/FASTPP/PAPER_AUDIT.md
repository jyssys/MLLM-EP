# FastPP paper audit

Source: https://www.usenix.org/system/files/osdi26-hwang.pdf, OSDI 2026; complete technical body pp.2–15 read. Primary Qwen2.5-32B/14B BF16, four PCIe A100 40GB, SGLang 0.4.1/torch2.5.1/CUDA12.4.

Greedy dynamic chunks use latency slack; ALP combines offline non-attention profiling and online RLS attention correction. Both reduce prefill-induced pipeline bubbles. Delay scheduling redistributes suspended/running decode requests to reduce decode-batch imbalance. Original goodput is maximum arrival rate satisfying p90 TTFT/TPOT limits; conditional SLO-compliant CDFs must not replace all-request reporting here.

Adversarial boundaries:

- Token-only latency is not the full baseline: prefix×prefill, prefill², decode context and a residual are already modeled.
- Better predictor MAPE does not necessarily improve E2E (§5.2.6).
- Decode rebalancing targets dense kernel boundaries; applicability to routed expert costs must be measured.
- Suspended requests can have worse p99 TPOT even when aggregate completion improves (§5.2.4).
- Paper's PCIe-versus-TP result is not expected to transfer unchanged to NVLink H100. Compare native PP policy ablations before any MoE claim.

No failure established. Native Qwen3/PP×EP source feasibility must be verified separately.
