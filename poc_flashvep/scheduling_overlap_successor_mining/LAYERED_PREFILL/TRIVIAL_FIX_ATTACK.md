# Existing-knob adversarial control

Two independent graph screens used three randomized restarts per configuration:
chunk512/cap4/cap16, then chunk2048/cap4/cap12/cap24 with identical full-trace
warmup. Unsupported cap8 was not invented. Each run retains actual chosen stage
counts, request arrivals/emissions and all-token TBT compliance.

Best static cap4 is already near the tested per-workload lower envelope. Tight
TBT prefers cap12 or cap16, looser TBT cap4/chunk; this is a known SLO tradeoff,
not a new workload-sensitive execution principle. Simple existing choices leave
no demonstrated material residual in this screen. We do not calculate a
“fraction of MLLM failure recovered” when no native MLLM failure was measured.

See ORACLES.md for finite-space versus universal-oracle limits and raw roots.
