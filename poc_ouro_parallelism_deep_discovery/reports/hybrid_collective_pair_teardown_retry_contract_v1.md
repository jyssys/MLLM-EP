# Collective-pair teardown retry

## Preserved failure

The first P2 run completed all six timing cells.
Both workers then remained alive without rank-local cleanup records for more than four minutes.
The coordinator terminated only the verified owned torchrun launcher.
The outer guard completed scoped cleanup and restored the owned burners.
The run is NOT_ADMITTED.

The [abort decision](../artifacts/hybrid_collective_pair_P2_v1_abort_decision.json) identifies the exact process and source.
The complete [v1 source](../artifacts/hybrid_collective_pair_probe_v1.py) retains its original SHA256.
The raw timing and termination logs remain unchanged.
The evidence does not identify which native cleanup call blocked.

## One bounded retry

The retry changes teardown only.
It destroys the explicit NCCL subgroup, destroys Gloo WORLD, and then closes CustomAllreduce.
This order follows the inspected vLLM coordinator's groups-before-device-communicator lifecycle.
CustomAllreduce passes its cached rank into buffer release.
It therefore does not require a live metadata group during close.

Each cleanup operation records CPU-only BEFORE and AFTER timestamps.
A one-shot faulthandler dump records Python stacks after 30 seconds if cleanup remains blocked.
Those timestamps are cleanup diagnostics, not communication completion or cross-GPU timing.
No new CUDA query enters failure logging.

The original row shapes, payloads, warmups, 31 trials, 16 pairs, correctness checks, and arm order remain unchanged.
The retry uses a fresh output directory.
No model, engine, kernel, or policy changes.
The outer timeout remains bounded by 1800 seconds.

Admission requires successful outer-guard completion and every rank's error-free cleanup record.
A successful retry would establish completion under the amended lifecycle.
It would not prove the cause of the first stall.
If this retry also blocks, stop this component investigation without an engine rewrite.
