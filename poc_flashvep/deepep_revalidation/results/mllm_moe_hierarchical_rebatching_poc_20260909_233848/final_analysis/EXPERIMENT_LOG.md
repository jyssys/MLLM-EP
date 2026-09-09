# Experiment log

## Environment and safety

- Physical GPUs: 4/5/6/7 only; H100 80GB, driver 570.211.01.
- Other user's processes on GPUs 2/3 were observed and never touched.
- Validated runtime: Qwen3-VL-30B-A3B-Instruct BF16, vLLM 0.20.0,
  TP2/DP2/EP4, `DeepEPHTPrepareAndFinalize`, `TritonExperts`, DBO off.
- Clean request wall time and observer-heavy CUDA diagnostics are separate.
- The initial `smoke_clean_r0` used the wrong interpreter and did not reach the
  target runtime; it is retained in `GPU_TIME_LOG.csv` and excluded.

## Sequence

1. Prepared frozen real-image pools (32 and 128 requests) from the prior
   verified Qwen3-VL trace manifest.
2. Ran a six-policy batch-16 smoke. Found and removed frontend preprocessing /
   submission overlap.
3. Repeated corrected smoke in three independent engine restarts.
4. Captured single-request-pair and batched observer traces.
5. Swept batch sizes 16/32/64/128. Found policy-dependent DP membership.
6. Rebuilt all waves with actual prompt-token-balanced, policy-independent DP
   assignment; asserted the matched global marginals.
7. Repeated b128 in independent restarts; found full-shape first-use variance.
8. Added two complete-plan warmups and collected a clean three-repeat b128 run.
9. Collected three independent warmed observer restarts and reduced logical
   operations by rank-critical same-device CUDA duration.
10. Ran fixed-16 and natural-EOS/max-32 output controls. Both remained below
    the headroom gate and failed full greedy-sequence agreement.
11. Computed P3/P4/P5/P7 module-mixing oracles and applied the hard gates.

Meaningful target-runtime wall time is 31.71 minutes (2.114 four-GPU-hours),
excluding burn and the failed interpreter launch. The sprint stopped GPU data
collection because the optimistic oracle itself was below 8%, not because a
fixed time quota was reached.
