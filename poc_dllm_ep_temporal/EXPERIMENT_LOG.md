# Experiment log

All live GPU work used only physical GPUs 4, 5, 6, and 7. GPUs were released
before final analysis; no burn was restarted.

## 2026-09-11 — substrate audit

1. Pinned parent HEAD `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` and
   dInfer HEAD `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
2. Audited stock TP4/DP1/nominal-EP4. Found 16 expert weights per rank but no
   all-to-all manager, dispatch, or combine call.
3. Applied the two-hunk expert-group patch and initialized the physical world
   before the DP-aware vLLM config. Verified TP1/DP4/EP4, 16 experts/rank,
   broadcast dispatch, local fused experts, and all-reduce combine.
4. Saved per-rank audit JSON, Chrome profiler traces, hidden boundaries, and
   logits for one-GPU, stock, and true-EP paths.

## 2026-09-11 — copy and temporal measurements

1. Measured all 12 directed 12-MiB expert P2P copies with 10 warmups and 50
   repetitions, alone and concurrent with destination-GPU compute.
2. Ran three clean and three traced engine restarts. Each restart used two
   warmup prompts and 30 measured prompts with a fixed 64-token input shape and
   64-token diffusion block.
3. Verified conservation and output-repeatability invariants. Measured trace
   observer tax and used clean timing for economic denominators.

## 2026-09-11 — CPU-only analysis after GPU return

1. Collapsed restart timing by median while retaining one route trajectory per
   logical record.
2. Computed t+1/t+2/t+4/t+8 temporal persistence, layer autocorrelation,
   request-vector diversity, and load-to-latency calibration.
3. Evaluated costed non-overlapping H=1/2/4/8 replica leases and current-state
   causal decisions.
4. Evaluated fixed-budget FCFS, random, future-aware offline search,
   previous-iteration, and EMA batching.
5. Added implementation-independent absolute kill bounds: zero-cost unlimited
   replication and fractional perfect rank equalization.
6. Regenerated all 16 required figures and reran unit tests.
