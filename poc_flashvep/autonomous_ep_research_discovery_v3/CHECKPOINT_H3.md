# Checkpoint H3 (~1 h 30 min)

- Strongest new causal control: reducing `max_num_batched_tokens` 8192→4096 at c8 raises matched-M T_MoE p50 1.155→2.133 ms, mainly dispatch/event wait, and request p50 901→1,278 ms.
- Triviality attack: requests/s proxy remains 0.52→0.50, so the effect is a known token-budget trade-off and does not provide a non-trivial oracle.
- Strongest negative: fixed-tail and fanout candidates remain below direct E2E headroom gate; fanout adds no predictive information.
- Queue update: C1b marked TRIVIAL_ENGINEERING; C1/K1 retained as diagnostic only. No finalist promoted.
