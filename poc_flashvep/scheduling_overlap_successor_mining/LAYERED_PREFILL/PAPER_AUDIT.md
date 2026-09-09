# Layered Prefill paper audit

Source: https://arxiv.org/abs/2510.08055v2, MLSys 2026; technical body pp.1–12 read, references inspected. Local PDF hash recorded in COMMON manifest.

The mechanism replaces repeated token-chunk traversal with contiguous layer-group prefill while decoding traverses every layer each iteration. Its opportunity is repeated sparse expert-weight traffic under tight inter-token limits, not arbitrary dense-model acceleration. Original primary setting is BF16 Qwen3-30B-A3B/GPT-OSS-20B, TP2 on two NVLink H100s, Poisson ShareGPT/arXiv arrivals. The paper reports request-level latency, SLO attainment and energy.

Important adversarial findings:

- §4.4 already adapts group count to input length, approximately ceil(L/512), and permits joint chunking. An entirely static group-count strawman is invalid.
- §5.9 already reports convergence with large chunks/relaxed SLOs; §5.10 sweeps group count; §5.13 documents dense-model regression.
- §4.1 relies on homogeneous sequential decoder layers. Unequal group execution costs remain a testable assumption, not an established failure.
- Expert-load bytes are a routing-derived accounting proxy, not automatically measured DRAM traffic.
- Fixed SLO comparison is required: original per-request attainment requires every TBT within its limit, unlike a TPOT-only criterion.

No transfer failure or successor headroom established yet.
